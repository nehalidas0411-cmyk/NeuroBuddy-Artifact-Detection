"""Athlete-level longitudinal and test-versus-baseline analysis.

This module does not calculate EEG features. It consumes *unchanged* single-record
reports from the four-route pipeline and compares only compatible scalar metrics.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import zipfile
from typing import Any, Callable, Sequence

from . import __version__
from .interactive_views import heatmap_html, sleep_replay

MIN_BASELINE_RECORDINGS = 3
ABNORMAL_Z = 3.0  # Strictly MORE than 3 SD; exactly 3 is Within range.


class ComparisonError(ValueError):
    """User-facing reason a baseline comparison cannot be performed."""


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_is_comparable(name: str) -> bool:
    """Only physiological/descriptive features, not test significance or sample sizes.

    All excluded values remain in individual EEG reports; nothing is removed from
    the existing analysis. This is strictly about validity of longitudinal z-scores.
    """
    label = name.lower().split('.')[-1]
    if label in {'metrics_available', 'finite_fraction', 'time_verified', 'confidence', 'sample_count',
                 'p', 't', 'z', 'f', 'df', 'dof', 'cohens_d', 'z_score', 'n', 'n_trials'}:
        return False
    if (label.startswith(('n_', 'num_', 'number_of_', 'count_', 'total_events', 'hit_count',
                          'miss_count', 'false_alarm_count', 'correct_rejection_count'))
            or label.endswith(('_count', '_counts', '_epochs', '_trials', '_samples', '_p_value',
                                '_p', '_uncorrected', '_corrected', '_statistic', '_confidence', '_index'))):
        # _index intentionally excludes ordinal labels, not validated indices.
        return False
    if 'p_value' in label or label in {'p_uncorrected','p_corrected','p_fdr','t_statistic',
                                      'trial_index','sample_rate_hz'}:
        return False
    return True


def _flatten_numeric(tree: Any, prefix: str, out: dict[str,float]) -> None:
    if isinstance(tree, dict):
        for key, value in tree.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            if isinstance(value, dict):
                _flatten_numeric(value, path, out)
            elif _number(value) and _metric_is_comparable(path):
                out[path] = float(value)
    # Lists are intentionally not interpreted as comparable scalar statistics.


def scalar_metrics(report: dict) -> dict[str, float]:
    """Metric paths are stable across baseline/test reports and preserve channels."""
    found: dict[str, float] = {}
    channel_stats = report.get('shared_metrics', {}).get('channel_statistics', {})
    for channel, stats in channel_stats.items():
        if not isinstance(stats, dict) or stats.get('metrics_available') is False:
            continue
        _flatten_numeric(stats, f'shared.channel.{channel}', found)
    # The original single-EEG engine also reports IONM-style channel QC and
    # event-locked trial summaries. Reuse their scalar measurements verbatim.
    for channel, qc in report.get('quality', {}).get('ionm_quality', {}).items():
        if isinstance(qc, dict):
            _flatten_numeric(qc, f'shared.quality.{channel}', found)
    event_audit = report.get('shared_metrics', {}).get('ionm_event_audit', {})
    for trial in event_audit.get('trial_statistics', []):
        if not isinstance(trial,dict):
            continue
        code, ch = trial.get('event_code'), trial.get('channel')
        if not isinstance(code,(str,int)) or not isinstance(ch,str):
            continue
        safe_code = re.sub(r'[^a-zA-Z0-9_-]+','_',str(code))
        _flatten_numeric(trial, f'shared.ionm_event.{safe_code}.{ch}', found)
    family = report.get('classification', {}).get('task_family', 'UNKNOWN').lower()
    specialized = report.get('specialized_metrics', {}).get(family, {})
    if isinstance(specialized, dict):
        _flatten_numeric(specialized, f'specialized.{family}', found)
    return found


def _signal_digest(report: dict) -> str:
    manifest = report.get('input_manifest', [])
    signals = [v for v in manifest if v.get('role') == 'signal']
    if len(signals) != 1 or not signals[0].get('sha256'):
        raise ComparisonError('Each report must identify exactly one source EEG recording with SHA-256 provenance.')
    return signals[0]['sha256']


def _bids_subject(report: dict) -> str | None:
    name = report.get('recording', {}).get('file', '')
    matched = re.search(r'(?:^|_)sub-([A-Za-z0-9]+)(?:_|\.)', name)
    return matched.group(1).lower() if matched else None


def _comparison_identity(report: dict) -> tuple[str,str|None]:
    cla = report.get('classification', {})
    return cla.get('task_family','UNKNOWN'), cla.get('subtype')


def _protocol_issues(base: dict, test: dict) -> list[str]:
    issues = []
    if _comparison_identity(base) != _comparison_identity(test):
        issues.append(f'task family/subtype mismatch: {_comparison_identity(base)} vs {_comparison_identity(test)}')
    bs = base.get('recording', {}).get('sample_rate_hz')
    ts = test.get('recording', {}).get('sample_rate_hz')
    if _number(bs) and _number(ts) and not math.isclose(float(bs),float(ts),rel_tol=0,abs_tol=1e-6):
        issues.append(f'sampling-rate mismatch: {bs} vs {ts} Hz')
    bc = set(base.get('shared_metrics',{}).get('channel_statistics',{}))
    tc = set(test.get('shared_metrics',{}).get('channel_statistics',{}))
    if bc and tc and bc != tc:
        issues.append('EEG channel name sets differ; montage/referencing must be harmonized')
    bp = base.get('shared_metrics',{}).get('preprocessing',{})
    tp = test.get('shared_metrics',{}).get('preprocessing',{})
    for key in ('psd_method','filtered_quality','raw_stats'):
        if bp.get(key) != tp.get(key):
            issues.append(f'analysis-method mismatch for {key}: {bp.get(key)!r} vs {tp.get(key)!r}')
    if base.get('pipeline_version') != test.get('pipeline_version'):
        issues.append('pipeline versions differ; reprocess every recording with the same version')
    bs, ts = _bids_subject(base), _bids_subject(test)
    if bs and ts and bs != ts:
        issues.append(f'participant BIDS IDs differ: sub-{bs} vs sub-{ts}')
    return issues


def metric_unit(path: str) -> str:
    name = path.rsplit('.', 1)[-1].lower()
    if name.endswith(('_relative', '_ratio', '_fraction', '_rate')):
        return 'ratio'
    if 'latency_ms' in name or name.endswith('_ms'): return 'ms'
    if 'reaction_time_s' in name or name.endswith('_s'): return 's'
    if 'duration_min' in name: return 'min'
    if 'percent' in name or 'erd_ers' in name: return '%'
    if 'amplitude_uv' in name or name.endswith('_uv'): return 'µV'
    if 'frequency' in name or name == 'spectral_edge': return 'Hz'
    if 'power' in name or 'variance' in name or name.endswith('_activity_v2'): return 'V² (or route-defined)'
    if name in {'mean','standard_deviation','rms','peak_to_peak','min','max','median','mad'}: return 'V'
    if name == 'spectral_entropy': return 'nats'
    if name == 'zero_crossings': return 'count'
    return 'as reported'


def build_comparison(baseline_reports: Sequence[dict], test_report: dict,
                     *, z_threshold: float = ABNORMAL_Z,
                     minimum_baselines: int = MIN_BASELINE_RECORDINGS) -> tuple[dict, dict]:
    """Pure aggregation. Reuses immutable single-EEG report dictionaries.

    Returns (longitudinal_baseline, comparison). No arithmetic alters source
    EEG files or the calculations of any of the four pipeline routes.
    """
    if len(baseline_reports) < minimum_baselines:
        raise ComparisonError(f'At least {minimum_baselines} distinct baseline recordings are required.')
    if not math.isfinite(z_threshold) or z_threshold <= 0:
        raise ComparisonError('The z-score threshold must be a positive finite number.')
    all_reports = list(baseline_reports) + [test_report]
    for n, report in enumerate(all_reports):
        if report.get('processing_status') != 'COMPLETE':
            raise ComparisonError(f'Recording {n+1} is {report.get("processing_status")}; all inputs must complete successfully.')
        if not report.get('shared_metrics',{}).get('channel_statistics'):
            raise ComparisonError(f'Recording {n+1} has no channel statistics.')
    digests = [_signal_digest(r) for r in all_reports]
    if len(set(digests)) != len(digests):
        raise ComparisonError('Baseline recordings must be distinct from each other and from the test EEG (duplicate signal SHA-256).')
    issues=[]
    for i, r in enumerate(baseline_reports):
        issues += [f'Baseline {i+1}: {msg}' for msg in _protocol_issues(r,test_report)]
    if issues:
        raise ComparisonError('Incompatible EEG recordings: ' + '; '.join(sorted(set(issues))))

    baseline_maps=[scalar_metrics(r) for r in baseline_reports]
    testing_map=scalar_metrics(test_report)
    if not any(baseline_maps):
        raise ComparisonError('No finite comparable numeric EEG measurements were available.')
    baseline_keys=set.union(*(set(m) for m in baseline_maps))
    union_keys=sorted(baseline_keys | set(testing_map))
    longitudinal=[];comparison=[]
    for key in union_keys:
        values=[m[key] for m in baseline_maps if key in m]
        valid_baseline=len(values)==len(baseline_maps)
        mean=statistics.mean(values) if valid_baseline else None
        sd=statistics.stdev(values) if valid_baseline and len(values)>=2 else None
        unit=metric_unit(key)
        cv=(100*sd/abs(mean) if sd is not None and mean is not None and mean!=0 else None)
        baseline_row={'metric':key, 'unit':unit, 'n':len(values),
                      'mean':mean, 'sd':sd, 'cv_percent':cv,
                      'baseline_values':values if valid_baseline else None,
                      'status':'COMPLETE' if valid_baseline else 'MISSING_BASELINE_METRIC'}
        longitudinal.append(baseline_row)
        x=testing_map.get(key)
        z=None
        if not valid_baseline:
            status='Not comparable';reason='Metric missing in one or more baseline reports'
        elif x is None:
            status='Not comparable';reason='Metric unavailable in test EEG'
        elif sd == 0:
            status='Not comparable';reason='Zero baseline SD, z-score undefined even if test equals baseline mean'
        else:
            z=(x-mean)/sd
            status='Abnormal' if abs(z)>z_threshold else 'Within range'
            reason=('Test outside ±3 sample SD of baseline' if status=='Abnormal' else
                    'Test within ±3 sample SD of baseline') if z_threshold == 3 else (
                    f'Test outside ±{z_threshold:g} SD' if status=='Abnormal' else f'Test within ±{z_threshold:g} SD')
        comparison.append({'metric':key,'unit':unit,'baseline_n':len(values),
                           'baseline_mean':mean,'baseline_sd':sd, 'cv_percent':cv,
                           'test_value':x,'z_score':z,'lower_bound':mean-z_threshold*sd if sd is not None else None,
                           'upper_bound':mean+z_threshold*sd if sd is not None else None,
                           'status':status,'reason':reason})
    now=datetime.now(timezone.utc).isoformat()
    meta={'schema_version':'1.0.0','created_utc':now,'pipeline_version':test_report.get('pipeline_version'),
          'task_family':_comparison_identity(test_report)[0],
          'task_subtype':_comparison_identity(test_report)[1],
          'baseline_report_ids':[r['report_id'] for r in baseline_reports],
          'test_report_id':test_report['report_id'],
          'baseline_recording_count':len(baseline_reports),
          'method':'arithmetic mean and sample SD (ddof=1) across per-recording scalar metrics',
          'provenance':{'baseline_signal_sha256':digests[:-1],'test_signal_sha256':digests[-1]},
          'cautions':['Three baseline sessions yield an unstable SD estimate; these flags are within-athlete statistical deviations, not clinical abnormalities.',
                      'Only identically named channel/condition metrics are compared.',
                      'Different experimental protocols, electrode references, artifacts and recording times may affect interpretation. Reference and montage compatibility cannot be fully verified from current report metadata.']}
    longitudinal_result={**meta,'report_type':'longitudinal_baseline', 'metrics':longitudinal}
    compare_result={**meta,'report_type':'test_vs_longitudinal_baseline',
                    'z_threshold':z_threshold,'abnormal_rule':'abs(z) > threshold',
                    'metrics':comparison,'summary':{
                       'total':len(comparison),'within_range':sum(r['status']=='Within range' for r in comparison),
                       'abnormal':sum(r['status']=='Abnormal' for r in comparison),
                       'not_comparable':sum(r['status']=='Not comparable' for r in comparison)}}
    return longitudinal_result,compare_result


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(row[k]) if isinstance(row.get(k),(list,dict)) else row.get(k)
                             for k in fields})


def _text(v: Any) -> str:
    if v is None:return 'N/A'
    if isinstance(v,float):return f'{v:.8g}'
    return str(v)


def _table(rows: list[dict], keys: list[str]) -> str:
    heading=''.join('<th>'+html.escape(k.replace('_',' ').title())+'</th>' for k in keys)
    body=[]
    for r in rows:
        cells=[]
        for k in keys:
            value=r.get(k)
            css=' class="abnormal"' if k=='status' and value=='Abnormal' else ''
            cells.append(f'<td{css}>'+html.escape(_text(value),quote=True)+'</td>')
        body.append('<tr>'+''.join(cells)+'</tr>')
    return '<div class="scroll"><table><thead><tr>'+heading+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'


_COMPARATIVE_CSS = '''
:root{font:14px/1.5 system-ui,-apple-system,sans-serif;color:#e8f2f7;background:#081c2a}
main{max-width:1350px;margin:auto;padding:30px 20px}h1{font-size:28px}h2{font-size:19px}
details{background:#102c3e;border:1px solid #24465a;border-radius:12px;padding:18px;margin:18px 0}
summary{font-weight:700;font-size:19px;cursor:pointer}section{margin:20px 0}
.scroll{overflow:auto;max-height:620px}table{border-collapse:collapse;font-size:12px;min-width:100%;white-space:nowrap}
td,th{text-align:left;border-bottom:1px solid #305267;padding:9px}
th{position:sticky;top:0;background:#15384b}a{color:#79d2ef}
.abnormal{color:#ff9d9d;font-weight:800}.muted{color:#a9c4d2}.small{font-size:12px}
'''


def _html_doc(title: str, inner: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{_COMPARATIVE_CSS}</style></head><body><main><h1>{html.escape(title)}</h1>{inner}</main></body></html>'


def _write_json(path:Path, value:dict) -> None:
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def _identity_slug(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]+','_',value).strip('_')[:70] or 'recording'


def run_athlete_workflow(baselines: Sequence[str | Path], test: str | Path, output: str | Path,
                         *, team_id: str, athlete_id: str,
                         process_one: Callable | None = None,
                         min_baselines: int = MIN_BASELINE_RECORDINGS,
                         z_threshold: float = ABNORMAL_Z,
                         large_threshold_mb: int = 256, chunk_seconds: int = 60,
                         sleep_staging: str = 'if_missing', player_frames: int = 1500) -> dict:
    """Frontend-ready orchestration, exactly one existing EEG analysis per recording.

    Use `process_one` injection for deterministic fixture tests. In production,
    the same neurobuddy_v2.pipeline.process() function is called each time.
    """
    from .pipeline import process
    run = process if process_one is None else process_one
    files=[Path(p).expanduser().resolve() for p in baselines]
    test=Path(test).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    if len(files) < min_baselines:
        raise ComparisonError(f'Choose at least {min_baselines} baseline recordings.')
    if not team_id.strip() or not athlete_id.strip():
        raise ComparisonError('Team and Athlete IDs must be provided by the frontend.')
    if len(set(files+[test])) != len(files)+1:
        raise ComparisonError('The same EEG path cannot be used as multiple baselines or as the test EEG.')
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for index, path in enumerate(files,1):
        dest=output/'baseline'/f'{index:02d}_{_identity_slug(path.stem)}'
        dest.mkdir(parents=True,exist_ok=True)
        delivered=run(path,dest,large_threshold_mb=large_threshold_mb,
                      chunk_seconds=chunk_seconds, sleep_staging=sleep_staging,
                      player_frames=player_frames)
        if len(delivered)!=1:
            raise ComparisonError(f'Baseline {index} must produce exactly one EEG report; received {len(delivered)}.')
        results.append((delivered[0][0],dest))
    testdir=output/'test'
    testdir.mkdir(parents=True,exist_ok=True)
    delivered=run(test,testdir,large_threshold_mb=large_threshold_mb,
                  chunk_seconds=chunk_seconds, sleep_staging=sleep_staging,
                  player_frames=player_frames)
    if len(delivered)!=1:
        raise ComparisonError('Testing EEG must produce exactly one report.')
    test_report=delivered[0][0]
    try:
        baseline,comp=build_comparison([r for r,_ in results],test_report,
                                       z_threshold=z_threshold,minimum_baselines=min_baselines)
    except ComparisonError as exc:
        _write_json(output/'comparison_error.json',{'processing_status':'NOT_COMPARABLE',
                    'team_id':team_id,'athlete_id':athlete_id,'message':str(exc),
                    'baseline_reports':[str((p/'report.json').relative_to(output)) for _,p in results],
                    'test_report':'test/report.json'})
        raise
    for obj in (baseline,comp):
        obj['team_id']=team_id
        obj['athlete_id']=athlete_id
    basedir=output/'baseline'
    _write_json(basedir/'longitudinal.json',baseline)
    _write_json(output/'comparative_analysis.json',comp)
    _csv(basedir/'longitudinal.csv',baseline['metrics'],
         ['metric','unit','n','mean','sd','cv_percent','baseline_values','status'])
    keys=['metric','unit','baseline_mean','baseline_sd','test_value','z_score','status',
          'baseline_n','lower_bound','upper_bound','cv_percent','reason']
    _csv(output/'comparative_analysis.csv',comp['metrics'],keys)
    basetable=_table(baseline['metrics'],['metric','unit','n','mean','sd','cv_percent','status'])
    comp_table=_table(comp['metrics'],keys)
    (basedir/'longitudinal.html').write_text(_html_doc('Neuro-buddy Longitudinal Baseline',
        f'<p class="muted">{len(results)} distinct recordings. Sample SD (N-1). CV is exploratory.</p>{basetable}'),encoding='utf-8')
    (output/'comparative_analysis.html').write_text(_html_doc('Neuro-buddy Comparative Analysis',
        f'<p class="muted">Abnormal when |z| &gt; {z_threshold:g}. Numerical table remains authoritative.</p>{heatmap_html(comp)}{comp_table}'),encoding='utf-8')
    baseline_links=''.join(f'<p>Baseline {i}: <a href="{html.escape(str((p/"report.html").relative_to(output)),quote=True)}">Open full EEG report</a> · <a href="{html.escape(str((p/"report.json").relative_to(output)),quote=True)}">JSON</a></p>'
                           for i,(_,p) in enumerate(results,1))
    baseline_replays=''.join(sleep_replay(r, f'nb-baseline-sleep-{i}') for i,(r,_) in enumerate(results,1))
    test_replay=sleep_replay(test_report,'nb-test-sleep')
    page=_html_doc('Neuro-buddy Athlete Comparative Report',f'''
<p>Team: {html.escape(team_id)} · Athlete: {html.escape(athlete_id)} · {html.escape(str(comp['task_family']))} · {html.escape(str(comp['task_subtype']))}</p>
<p class="muted">All baseline and testing scans use the same Neuro-buddy single-EEG engine. Results are exploratory within-athlete deviations, not clinical diagnoses.</p>
<details open><summary>BASELINE</summary>{baseline_links}<h2>Longitudinal file</h2>
<p><a href="baseline/longitudinal.json">Baseline JSON</a> · <a href="baseline/longitudinal.csv">CSV</a></p>{basetable}{baseline_replays}</details>
<details open><summary>TEST EEG</summary><p><a href="test/report.html">Open complete single-EEG report with statistics, graphs and Brain Activity Player</a> · <a href="test/report.json">JSON</a></p>
<p>Test file: {html.escape(str(test_report.get('recording',{}).get('file','unknown')))}</p>
<p>Task: {html.escape(str(comp['task_family']))} · Single-EEG status: {html.escape(str(test_report.get('processing_status')))}</p>{test_replay}</details>
<details open><summary>COMPARATIVE ANALYSIS</summary><p class="muted">Baseline mean, sample SD, Test value, Z-score, and Abnormal / Within range. Strict |z| &gt; {z_threshold:g}; SD=0 is not comparable.</p>
<p><a href="comparative_analysis.json">Comparison JSON</a> · <a href="comparative_analysis.csv">CSV</a></p>{heatmap_html(comp,'nb-athlete-heatmap')}{comp_table}</details>
<p class="muted small">Longitudinal SD estimated from {len(results)} sessions; with three sessions estimates are unstable. Match task protocol, montage, sampling frequency, and preprocessing.</p>''')
    (output/'athlete_report.html').write_text(page,encoding='utf-8')
    structure={'schema_version':'1.0.0','report_type':'athlete_comparative_package',
               'team_id':team_id,'athlete_id':athlete_id,
               'pipeline_version':__version__,'processing_status':'COMPLETE',
               'task_family':comp['task_family'],'task_subtype':comp['task_subtype'],
               'baseline_reports':[{'report_id':r['report_id'],'recording_id':r.get('recording_id'),
                     'report_json':str((p/'report.json').relative_to(output)),
                     'report_html':str((p/'report.html').relative_to(output))} for r,p in results],
               'baseline_longitudinal':'baseline/longitudinal.json',
               'test_report':{'report_id':test_report['report_id'],'report_json':'test/report.json',
                              'report_html':'test/report.html'},
               'comparative_analysis':'comparative_analysis.json',
               'html_entrypoint':'athlete_report.html','z_threshold':z_threshold,
               'comparison_summary':comp['summary']}
    _write_json(output/'athlete_report.json',structure)
    file_manifest={}
    for f in sorted(output.rglob('*')):
        if f.is_file() and f.name not in ('athlete_report.zip','manifest.json'):
            rel=f.relative_to(output).as_posix()
            if '..' in Path(rel).parts: raise AssertionError('Unsafe report archive path')
            h=hashlib.sha256()
            with f.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
            file_manifest[rel]={'sha256':h.hexdigest(),'bytes':f.stat().st_size}
    _write_json(output/'manifest.json',{'package_format':'neurobuddy-athlete-v1',
        'json_entrypoint':'athlete_report.json','html_entrypoint':'athlete_report.html',
        'includes_original_eeg':False,'assets':file_manifest})
    with zipfile.ZipFile(output/'athlete_report.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(output.rglob('*')):
            if f.is_file() and f.name != 'athlete_report.zip':
                z.write(f,f.relative_to(output).as_posix())
    return structure


def run_from_manifest(manifest_path: str | Path, output: str | Path, **options) -> dict:
    """Stable machine-to-machine entrypoint: manifest paths resolved relative to JSON."""
    p=Path(manifest_path).expanduser().resolve()
    content=json.loads(p.read_text(encoding='utf-8'))
    paths=content.get('baseline_inputs',[])
    if not isinstance(paths,list) or not isinstance(content.get('test_input'),str):
        raise ComparisonError('Manifest must provide baseline_inputs array and test_input path.')
    def absolute(path):
        x=Path(path).expanduser()
        return (x if x.is_absolute() else p.parent/x).resolve()
    return run_athlete_workflow([absolute(x) for x in paths],absolute(content['test_input']),output,
                     team_id=str(content['team_id']),athlete_id=str(content['athlete_id']),**options)


def run_team_manifest(manifest_path: str | Path, output: str | Path, **options) -> dict:
    """Process a Team roster, retaining strict per-athlete baseline separation.

    Each athlete gets exactly the same 3-dropdown report as athlete mode. The
    team index links to those reports; no metrics are pooled across athletes.
    """
    source = Path(manifest_path).expanduser().resolve()
    data = json.loads(source.read_text(encoding='utf-8'))
    team_id = data.get('team_id')
    athletes = data.get('athletes')
    if not isinstance(team_id,str) or not team_id.strip() or not isinstance(athletes,list) or not athletes:
        raise ComparisonError('Team manifest requires team_id and nonempty athletes array.')
    athlete_ids = [a.get('athlete_id') for a in athletes if isinstance(a,dict)]
    if len(athlete_ids) != len(athletes) or any(not isinstance(v,str) or not v.strip() for v in athlete_ids):
        raise ComparisonError('Every athlete must have a valid athlete_id.')
    if len(set(athlete_ids)) != len(athlete_ids):
        raise ComparisonError('Duplicate athlete_id in Team roster.')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    def resolve(path):
        p = Path(path).expanduser()
        return (p if p.is_absolute() else source.parent/p).resolve()
    summaries=[]
    for athlete in athletes:
        athlete_id = athlete['athlete_id']
        dest = output/'athletes'/_identity_slug(athlete_id)
        try:
            baseline_paths = [resolve(p) for p in athlete['baseline_inputs']]
            test_path = resolve(athlete['test_input'])
            result=run_athlete_workflow(baseline_paths,test_path,dest,team_id=team_id,
                                         athlete_id=athlete_id,**options)
            summaries.append({'athlete_id':athlete_id,'processing_status':'COMPLETE',
                              'report_html':str((dest/'athlete_report.html').relative_to(output)),
                              'report_json':str((dest/'athlete_report.json').relative_to(output)),
                              'summary':result['comparison_summary']})
        except (ComparisonError,KeyError,TypeError,FileNotFoundError) as exc:
            dest.mkdir(parents=True,exist_ok=True)
            summaries.append({'athlete_id':athlete_id,'processing_status':'FAILED',
                              'error':str(exc)})
    report={'schema_version':'1.0.0','report_type':'team_comparative_index',
            'team_id':team_id,'pipeline_version':__version__,
            'processing_status':'COMPLETE' if all(r['processing_status']=='COMPLETE' for r in summaries) else 'PARTIAL',
            'athletes':summaries,'cross_athlete_averaging_performed':False}
    _write_json(output/'team_report.json',report)
    links=[]
    for record in summaries:
        if record['processing_status']=='COMPLETE':
            links.append({'Athlete':record['athlete_id'],'Status':'COMPLETE','Report':record['report_html']})
        else:
            links.append({'Athlete':record['athlete_id'],'Status':'FAILED','Report':record['error']})
    # Explicit links; table fields escaped separately.
    html_rows=[]
    for a in summaries:
        target=(f'<a href="{html.escape(a["report_html"],quote=True)}">Open athlete report</a>'
                if a['processing_status']=='COMPLETE' else html.escape(a['error']))
        html_rows.append(f'<tr><td>{html.escape(a["athlete_id"])}</td><td>{html.escape(a["processing_status"])}</td><td>{target}</td></tr>')
    page=_html_doc('Neuro-buddy Team Processing',
        f'<p>Team: {html.escape(team_id)}. Independent Athlete/Team workflows; no cross-athlete baselines.</p>'
        +'<table><thead><tr><th>Athlete</th><th>Status</th><th>Report</th></tr></thead><tbody>'
        +''.join(html_rows)+'</tbody></table>')
    (output/'team_report.html').write_text(page,encoding='utf-8')
    with zipfile.ZipFile(output/'team_report.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.name!='team_report.zip':z.write(p,p.relative_to(output).as_posix())
    return report
