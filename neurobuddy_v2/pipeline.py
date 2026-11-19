"""V2 orchestration. One recording bundle, one explainable route, one portable report."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import hashlib
import mne

from . import __version__
from .intake import discover,load_eeg,sha256,RecordingBundle,safe_extract_zip
from .events import load_events
from .routing import decide
from .analysis import shared,general,oddball,motor,sleep
from .large_recordings import analyze_large, requires_streaming
from .player import make_payload
from .reporting import save_report,schema_check


def process(source: Path | str, output: Path | str, *, player_frames: int = 1500,
            large_threshold_mb: int = 256, chunk_seconds: int = 60,
            sleep_staging: str = "if_missing"):
    source=Path(source);output=Path(output)
    if source.is_file() and source.suffix.lower()=='.zip':
        with tempfile.TemporaryDirectory(prefix='neurobuddy-staging-') as tmp:
            staged=safe_extract_zip(source, Path(tmp))
            return process(staged, output, player_frames=player_frames,
                           large_threshold_mb=large_threshold_mb, chunk_seconds=chunk_seconds,
                           sleep_staging=sleep_staging)
    bundles=discover(source)
    if not bundles:
        raise ValueError('No supported EEG recording found. Provide .set/.edf/.bdf/.vhdr/.fif (and required companions).')
    delivered=[]
    for b in bundles:
        target=output/b.name if len(bundles)>1 else output
        target.mkdir(parents=True,exist_ok=True)
        errors=[];warnings=list(b.warnings)
        dec={'task_family':'UNKNOWN','subtype':None,'evidence':[],
             'confidence_tier':'not_classified','analysis_readiness':{},'status':'NOT_STARTED','conflicts':[]}
        status=b.status
        ev=[]
        shared_metrics={};specialized={};quality={};figs={};figure_tables={};player=None;raw=None
        record={'file':b.signal.name}
        try:
            if b.status=='NEEDS_REVIEW':
                warnings+=b.conflicts
            else:
                raw=load_eeg(b)
                record.update({'duration_s':round(raw.n_times/raw.info['sfreq'],4),
                               'sample_rate_hz':float(raw.info['sfreq']),
                               'n_channels':len(raw.ch_names),'eeg_channels':sum(t=='eeg' for t in raw.get_channel_types())})
                ev,ew=load_events(b,raw);warnings+=ew
                decision=decide(b,ev,ew);dec=decision.as_dict();status=decision.status
                if status=='READY':
                    # For non-EEGLAB imports with only external event TSV, attach
                    # validated annotations so the reused IONM event audit sees them.
                    if not len(raw.annotations) and ev:
                        valid=[x for x in ev if x.alignment=='verified_within_recording']
                        if valid:
                            raw.set_annotations(mne.Annotations(
                                onset=[x.onset_s for x in valid],
                                duration=[x.duration_s or 0.0 for x in valid],
                                description=[x.raw_value for x in valid]))
                    print(f'[V2] {b.signal.name}: classified {dec["task_family"]} ({dec["subtype"]})')
                    large = requires_streaming(raw,b,threshold_mib=large_threshold_mb)
                    if large:
                        print(f'[V2] Large EEG detected, using bounded-memory streaming (chunks of {chunk_seconds}s)',flush=True)
                    base=(analyze_large(raw,target,chunk_seconds=chunk_seconds) if large else shared(raw,target))
                    filtered=base.pop('_filtered'); channels=base.pop('_eeg_channels')
                    figs.update(base.pop('figures'))
                    figure_tables.update(base.pop('figure_tables',{}))
                    shared_metrics={'channel_statistics':base.pop('channel_statistics'),
                                    'ionm_event_audit':base.pop('ionm_event_audit'),
                                    'preprocessing':base.pop('preprocessing')}
                    quality={'ionm_quality':base.pop('ionm_quality'),'ionm_artifacts':base.pop('ionm_artifacts'),
                             'ionm_outliers':base.pop('ionm_outliers')}
                    warnings+=base.pop('warnings')
                    routes={'GENERAL':lambda:general(raw,filtered,ev,target,channels),
                        'ODDBALL':lambda:oddball(raw,filtered,ev,target,channels),
                        'MOTOR':lambda:motor(raw,filtered,ev,target,channels,dec['subtype']),
                        'SLEEP':lambda:sleep(raw,filtered,ev,target,channels,
                                              staging_policy=sleep_staging,chunk_seconds=max(300,chunk_seconds))}
                    special=routes[dec['task_family']]()
                    specialized={dec['task_family'].lower():special['metrics']}
                    figs.update(special['figures']);figure_tables.update(special.get('figure_tables',{}));warnings+=special['warnings']
                    player=make_payload(raw,ev,max_frames=player_frames)
                    if not player.get('topomap_available'):
                        warnings.append('No reliable scalp-wide montage; player uses channel-level fallback.')
                    status='COMPLETE' if all(x for k,x in dec['analysis_readiness'].items()
                         if k in ('shared_eeg',) ) else 'PARTIAL'
                    if dec['task_family']=='SLEEP':
                        # Classification readiness from annotations is only a
                        # preliminary decision. An estimated hypnogram can be
                        # computed later, without overriding human scores.
                        origin=special['metrics'].get('stage_origin','unscored')
                        dec['analysis_readiness']['hypnogram']=origin!='unscored'
                        dec['analysis_readiness']['automated_staging']=bool(
                            special['metrics'].get('automatic_staging',{}).get('available'))
                        dec['analysis_readiness']['spindles']=bool(
                            special['metrics'].get('sleep_event_detection',{}).get('method'))
                        dec['analysis_readiness']['slow_waves']=dec['analysis_readiness']['spindles']
                        if not dec['analysis_readiness']['hypnogram']:
                            status='PARTIAL'
                    if dec['task_family']=='ODDBALL' and not dec['analysis_readiness']['erp']:
                        status='PARTIAL'
                    if dec['task_family']=='MOTOR' and not dec['analysis_readiness']['motor_erd']:
                        status='PARTIAL'
        except Exception as exc:
            if b.status == 'INCOMPLETE' and dec['task_family'] == 'UNKNOWN':
                # Keep task identity when an externally stored .fdt is missing.
                # Classification is separate from whether EEG analysis can execute.
                recovered = decide(b, [], []).as_dict()
                dec = {**recovered, 'status':'INCOMPLETE',
                       'analysis_readiness':{k:False for k in recovered['analysis_readiness']}}
            errors.append({'code':'PROCESSING_ERROR','message':str(exc)})
            status=b.status if b.status in ('INCOMPLETE','FAILED') else 'FAILED'
        finally:
            if raw is not None: raw.close()
        figlist=[]
        for label,filename in figs.items():
            p=target/filename
            if p.is_file():figlist.append({'title':label,'file':filename,'sha256':sha256(p),'mime':'image/png'})
        digest=sha256(b.signal)
        report={
          'schema_version':'2.0.0','pipeline_version':__version__,
          'report_id':'nb-'+digest[:16], 'recording_id':digest[:24],
          'processing_status':status,'input_manifest':b.manifest(),
          'classification':dec,'routing':{'selected':dec['task_family'],'subtype':dec.get('subtype'),
                  'shared_eeg_ionm_executed':bool(shared_metrics)},
          'recording':record,'quality':quality,'shared_metrics':shared_metrics,
          'specialized_metrics':specialized,'visualizations':{'figures':figlist,
             'brain_activity_player':{'available':bool(player),'asset':'player_data.json' if player else None,
                'topomap_available':player['topomap_available'] if player else False,
                'html_entrypoint':'report.html'}},
          'statistics':{},
          'events':{'count':len(ev),'normalized':[e.as_dict() for e in ev]},
          'warnings':sorted(set(warnings)), 'errors':errors,
          'provenance':{'created_utc':datetime.now(timezone.utc).isoformat(),
            'mne_version':mne.__version__,'task_adapter_ruleset':'eegmmi-physionet-run-map-v1',
            'model_classification_used':False,'clinical_validation':False},
        }
        schema_check(report)
        archive=save_report(target,report,player,figs,figure_tables)
        delivered.append((report,archive))
        print(f'[V2] {b.signal.name}: {status} → {archive}')
    return delivered
