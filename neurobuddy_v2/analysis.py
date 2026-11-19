"""Scientific analysis routes. Reuses original EEG+IONM metric functions unchanged."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from . import legacy_combined as legacy
from .figure_tables import (shared_tables, oddball_erp_table, oddball_rt_table,
                            motor_table, hypnogram_table, sleep_band_table)


def _plot(out: Path, filename: str, fig):
    fig.tight_layout()
    fig.savefig(out / filename, dpi=120)
    plt.close(fig)
    return filename


def _safe(x):
    x = float(x)
    return x if np.isfinite(x) else None


def shared(raw, out: Path):
    """Shared existing metrics, quality audit, artifact flags, statistical tests.

    Requires modest-size recordings; V2.1 should implement out-of-core for overnight
    data. We record this limitation and avoid claiming that this is streaming.
    """
    raw.load_data()
    eeg_ch = [name for name, typ in zip(raw.ch_names,raw.get_channel_types()) if typ=='eeg']
    cfg = legacy.Settings()
    filtered, lowpass = legacy.prepare_filtered(raw, eeg_ch, cfg)
    eeg_stats = {ch: legacy.eeg_statistics(raw.get_data(picks=[ch])[0],raw.info['sfreq'])
                 for ch in eeg_ch}
    quality = legacy.quality_audit(filtered,eeg_ch,cfg)
    outliers = legacy.outlier_audit(filtered,eeg_ch,quality,cfg)
    artifacts = legacy.artifact_audit(raw,filtered,eeg_ch,cfg)
    # Original IONM event contrasts are exploratory and not substituted for ERP/ERD.
    events, event_ids, pairs, source = legacy.extract_events(raw,cfg)
    epochs, counts = legacy.make_epochs(filtered,raw,events,event_ids,cfg,quality,artifacts)
    records = legacy.event_response_records(epochs,quality,cfg)
    trials = legacy.trial_statistics(records)
    tests = legacy.statistical_comparisons(records,cfg)
    flags = legacy.evidence_summary(quality,artifacts,records,cfg)
    figures = legacy.save_graphs(out,raw,filtered,eeg_ch,eeg_stats,quality,records,tests,cfg)
    figure_tables = shared_tables(raw,filtered,eeg_ch,eeg_stats,quality,records,tests,cfg)
    return {
        'channel_statistics':eeg_stats,
        'ionm_quality': quality, 'ionm_artifacts': artifacts, 'ionm_outliers':outliers,
        'ionm_event_audit': {'source':source,'counts':counts,'marker_pairs':pairs,
                              'trial_statistics':trials,'tests':tests,'screening':flags},
        'preprocessing': {'raw_stats':'unfiltered EEG','filtered_quality':f'{cfg.high_pass_hz}-{lowpass} Hz',
                          'artifact_removal_performed':False,
                          'line_noise_assessed_on':'unfiltered EEG'},
        'warnings':['IONM grading and SNR proxy are unvalidated heuristics, not a diagnosis.',
                    'Artifact flags are not artifact removal.',
                    'EEG-only fatigue/stress/recovery cannot be diagnosed from these metrics.'],
        'figures': figures, 'figure_tables': figure_tables,
        '_filtered': filtered,
        '_eeg_channels':eeg_ch
    }


def general(raw, filtered, events, out, channels):
    if hasattr(filtered, 'hjorth'):
        return {'metrics': {'exploratory_hjorth': filtered.hjorth}, 'figures': {},
                'warnings': ['Hjorth parameters are descriptive and computed with bounded streaming windows; no fatigue diagnosis.']}
    rows = {}
    for ch in channels:
        x = filtered.get_data(picks=[ch])[0]
        d = np.diff(x)
        var = np.var(x)
        dx = np.var(d)
        mobility = np.sqrt(dx/var) if var>0 else None
        complexity = None
        if mobility and len(d)>1 and dx>0:
            complexity = np.sqrt(np.var(np.diff(d))/dx) / mobility
        rows[ch] = {'hjorth_activity_v2':_safe(var), 'hjorth_mobility':_safe(mobility) if mobility is not None else None,
                    'hjorth_complexity':_safe(complexity) if complexity is not None else None}
    return {'metrics':{'exploratory_hjorth':rows},'figures':{},
            'warnings':['Hjorth parameters are descriptive; no direct stress or fatigue diagnosis.']}


def _epochs(filtered, selected, tmin, tmax, baseline):
    sf = float(filtered.info['sfreq'])
    valid = []
    seen = set()
    for e in selected:
        if e.alignment != 'verified_within_recording': continue
        i=int(round(e.onset_s*sf)) + filtered.first_samp
        if i in seen or e.onset_s+tmin<0 or e.onset_s+tmax>filtered.n_times/sf: continue
        valid.append((i,e.semantic_label));seen.add(i)
    labels = sorted(set(v for _,v in valid))
    if not valid or not labels:
        return None, {}
    ids = {lab:n+1 for n,lab in enumerate(labels)}
    arr=np.array([[i,0,ids[lab]] for i,lab in sorted(valid)],dtype=int)
    if hasattr(filtered, 'raw') and hasattr(filtered, 'sos'):
        # A WindowFiltered view cannot be passed to mne.Epochs (which expects
        # an MNE BaseRaw). Read only event-centered EEG windows, retaining no
        # more than 96 MiB of event epochs in RAM. A selection cap must be
        # recorded by the caller if it affects statistical inference.
        n_samp=int(round((tmax-tmin)*sf))+1
        cap=max(2,int((96*1024*1024)//max(1,n_samp*len(filtered.ch_names)*8)))
        requested_epochs=len(arr)
        if len(arr)>cap:
            # Keep a balanced deterministic subset across each condition.
            grouped={label:[row for row in arr if row[2]==ids[label]] for label in labels}
            selected=[]
            while len(selected)<cap and any(grouped.values()):
                for lab in labels:
                    if grouped[lab] and len(selected)<cap:
                        selected.append(grouped[lab].pop(0))
            arr=np.array(sorted(selected,key=lambda x:x[0]),dtype=int)
        pieces=[];rows=[]
        for row in arr:
            sample=int(row[0]-filtered.first_samp)
            start=sample+int(round(tmin*sf))
            stop=start+n_samp
            if start<0 or stop>filtered.n_times:continue
            item=filtered.get_data(start=start,stop=stop)
            if item.shape[1]!=n_samp:continue
            pieces.append(item);rows.append(row)
        if not pieces:return None,{}
        filtered.epoch_sampling_summary = {
            'requested_epochs':requested_epochs, 'included_epochs':len(pieces),
            'memory_cap_bytes':96*1024*1024,
            'selection':'deterministic balanced subset by condition' if requested_epochs>cap else 'all in-range event windows',
            'subsampled':requested_epochs>cap}
        info=mne.create_info(filtered.ch_names,sf,['eeg']*len(filtered.ch_names))
        ep=mne.EpochsArray(np.stack(pieces),info,
                           events=np.asarray(rows,dtype=int),event_id=ids,
                           tmin=tmin,baseline=baseline,verbose='ERROR')
        ep.info['description']='Bounded event-window extraction from lazy EEG source'
        return ep,ids
    obj=mne.Epochs(filtered,arr,event_id=ids,tmin=tmin,tmax=tmax,baseline=baseline,
                   preload=True, reject_by_annotation=True,verbose='ERROR')
    return obj,ids


def oddball(raw, filtered, events, out, channels):
    rows = [e for e in events if e.family=='oddball' and e.semantic_label in ('target','standard','distractor')]
    counts = dict(Counter(e.semantic_label for e in rows))
    warnings=[];figs={};tables={}
    if hasattr(filtered,'sos'):
        warnings.append('Long-recording ERP calculated from bounded event windows; legacy full-record epoch rejection is unavailable.')
    if not {'target','standard'}.issubset(counts):
        return {'metrics':{'event_counts':counts,'erp':None},'figures':figs,'figure_tables':tables,
                'warnings':['Oddball conditions insufficient for target-versus-standard ERP.']}
    erp,ids = _epochs(filtered, [e for e in rows if e.semantic_label in ('target','standard')],-.2,.8,(-.2,0))
    metrics={'event_counts':counts,'erp':None,'behavior':None}
    if hasattr(filtered,'epoch_sampling_summary'):
        metrics['event_epoch_sampling']=filtered.epoch_sampling_summary
        if filtered.epoch_sampling_summary['subsampled']:
            warnings.append('ERP input trials were capped for memory safety: see event_epoch_sampling for counts.')
    if erp is not None and all(s in erp.event_id for s in ('target','standard')):
        target=erp['target']; standard=erp['standard'];
        if len(target)>0 and len(standard)>0:
            ch = 'Pz' if 'Pz' in erp.ch_names else 'Cz' if 'Cz' in erp.ch_names else erp.ch_names[0]
            ct=erp.ch_names.index(ch)
            tx=target.get_data(copy=False)[:,ct,:]*1e6
            sx=standard.get_data(copy=False)[:,ct,:]*1e6
            a=np.mean(tx,axis=0);b=np.mean(sx,axis=0)
            t=erp.times
            pwin=(t>=.25)&(t<=.5)
            nwin=(t>=.15)&(t<=.3)
            difference=a-b
            # A preselected electrode and 250-500 ms mean-amplitude window
            # avoids presenting post-hoc peak-picking as confirmatory inference.
            targ_amplitude=tx[:,pwin].mean(axis=1) if pwin.any() else np.array([])
            std_amplitude=sx[:,pwin].mean(axis=1) if pwin.any() else np.array([])
            comparison=None
            if len(targ_amplitude)>2 and len(std_amplitude)>2:
                stat,pvalue=stats.ttest_ind(targ_amplitude,std_amplitude,equal_var=False)
                pooled=np.sqrt(((len(targ_amplitude)-1)*np.var(targ_amplitude,ddof=1)+
                               (len(std_amplitude)-1)*np.var(std_amplitude,ddof=1))/
                              (len(targ_amplitude)+len(std_amplitude)-2))
                d=(float(np.mean(targ_amplitude)-np.mean(std_amplitude))/pooled) if pooled>0 else None
                comparison={'test':'Welch independent trial mean-amplitude t-test',
                            't':_safe(stat),'p_uncorrected':_safe(pvalue),
                            'cohens_d':_safe(d) if d is not None else None,
                            'reference_electrode':ch,'latency_window_ms':[250,500],
                            'exploratory':True}
            metrics['erp']={'electrode':ch,'n_target_epochs':len(target),'n_standard_epochs':len(standard),
                            'target_vs_standard_trial_test':comparison,
                            'p300_difference_mean_uv':_safe(difference[pwin].mean()) if pwin.any() else None,
                            'p300_difference_peak_uv':_safe(difference[pwin].max()) if pwin.any() else None,
                            'p300_difference_peak_latency_ms':_safe(1000*t[pwin][np.argmax(difference[pwin])]) if pwin.any() else None,
                            'n200_difference_mean_uv':_safe(difference[nwin].mean()) if nwin.any() else None,
                            'note':'ERP peaks at the selected electrode; polarity and latency depend on reference and paradigm.'}
            fig,ax=plt.subplots(figsize=(9,4))
            ax.plot(t*1e3,a,label=f'Target (n={len(target)})')
            ax.plot(t*1e3,b,label=f'Standard (n={len(standard)})')
            ax.plot(t*1e3,difference,label='Target minus standard',ls='--')
            ax.axvline(0,color='black',lw=0.7);ax.axhline(0,color='gray',lw=.6)
            ax.axvspan(250,500,color='orange',alpha=.10)
            ax.set(xlabel='Time from stimulus (ms)',ylabel='EEG potential (µV)',title=f'Oddball ERP at {ch}')
            ax.legend()
            figs['Oddball ERP target vs standard']=_plot(out,'oddball_erp.png',fig)
            tables['Oddball ERP target vs standard']=oddball_erp_table(t,a,b,difference,metrics['erp'])
    else:
        warnings.append('No sufficient in-range, quality-screened trials for ERP.')
    # Derive RT from verified timestamped response events rather than ambiguous response_time units.
    responses = sorted(e.onset_s for e in events if e.semantic_label=='response' and e.alignment=='verified_within_recording')
    stimuli = sorted([e for e in events if e.semantic_label in ('target','standard','distractor')],key=lambda e:e.onset_s)
    hits,misses,false_alarms,correct_rejections = 0,0,0,0
    rts=[]; j=0
    for i,s in enumerate(stimuli):
        nxt=stimuli[i+1].onset_s if i+1<len(stimuli) else float('inf')
        while j<len(responses) and responses[j] < s.onset_s: j+=1
        responded = j<len(responses) and responses[j]<min(nxt,s.onset_s+1.5) and responses[j]>=s.onset_s+0.12
        if responded:
            if s.semantic_label=='target': hits+=1;rts.append(responses[j]-s.onset_s)
            else: false_alarms+=1
            j+=1
        elif s.semantic_label=='target': misses+=1
        else: correct_rejections+=1
    if responses:
        # Log-linear correction prevents infinite d-prime with near-perfect performance.
        from scipy.stats import norm
        hn=hits+misses;fn=false_alarms+correct_rejections
        hrate=(hits+.5)/(hn+1) if hn else None
        frate=(false_alarms+.5)/(fn+1) if fn else None
        dprime=float(norm.ppf(hrate)-norm.ppf(frate)) if hrate is not None and frate is not None else None
        metrics['behavior']={'d_prime_loglinear_corrected':dprime,
          'hit_count':hits,'miss_count':misses,'false_alarm_count':false_alarms,
          'correct_rejection_count':correct_rejections,'mean_reaction_time_s':_safe(np.mean(rts)) if rts else None,
          'reaction_time_sd_s':_safe(np.std(rts)) if len(rts)>1 else None,
          'reaction_time_source':'stimulus_response_onset_difference','rt_matches':len(rts),
          'notes':'Responses matched to first eligible stimulus window; requires paradigm-level verification.'}
        if rts:
            fig,ax=plt.subplots(figsize=(7,3));ax.hist(np.array(rts)*1000,bins=min(20,max(4,len(rts)//5)))
            ax.set(xlabel='Response time (ms)',ylabel='Responses',title='Matched target response latencies')
            figs['Reaction times']=_plot(out,'oddball_reaction_times.png',fig)
            tables['Reaction times']=oddball_rt_table(rts,metrics['behavior'])
    else:
        warnings.append('No timestamped response events; reaction-time statistics unavailable.')
    if any(e.original_response_time is not None for e in events):
        warnings.append('Original response_time column excluded because its JSON units are inconsistent; onset-based RT used when possible.')
    return {'metrics':metrics,'figures':figs,'figure_tables':tables,'warnings':warnings}

def motor(raw, filtered, events, out, channels, subtype):
    cues=sorted([e for e in events if e.family=='motor' and e.semantic_label in
                 ('left_hand','right_hand','both_hands','both_feet') and e.alignment=='verified_within_recording'],key=lambda e:e.onset_s)
    sf=float(raw.info['sfreq'])
    features=[]
    for e in cues:
        if e.onset_s<1.6 or e.onset_s+2.8 >= raw.n_times/sf:continue
        # PSD bandpower from pre-cue rest (-1.5,-.25) vs postcue (+.5,+2.5).
        before=filtered.get_data(picks=[ch for ch in ('C3','C4') if ch in channels],
                                  start=int((e.onset_s-1.5)*sf),stop=int((e.onset_s-.25)*sf))
        after=filtered.get_data(picks=[ch for ch in ('C3','C4') if ch in channels],
                                 start=int((e.onset_s+.5)*sf),stop=int((e.onset_s+2.5)*sf))
        names=[ch for ch in ('C3','C4') if ch in channels]
        for i,ch in enumerate(names):
            for band,(low,high) in {'mu':(8.,12.),'beta':(13.,30.)}.items():
                f,p=signal.welch(before[i],sf,nperseg=min(len(before[i]),int(sf)))
                f2,p2=signal.welch(after[i],sf,nperseg=min(len(after[i]),int(sf)))
                a=legacy.band_power(f,p,low,high)
                b=legacy.band_power(f2,p2,low,high)
                if a>0 and np.isfinite(b):
                    features.append({'cue':e.semantic_label,'channel':ch,'band':band,
                                     'erd_ers_pct':_safe(100*(b-a)/a)})
    counts=dict(Counter(e.semantic_label for e in cues))
    by_group={}
    for cue in sorted(counts):
        by_group[cue]={}
        for ch in ('C3','C4'):
            for band in ('mu','beta'):
                vals=[r['erd_ers_pct'] for r in features if r['cue']==cue and r['channel']==ch and r['band']==band]
                if vals: by_group[cue][f'{ch}_{band}_mean_erd_ers_pct']=_safe(np.mean(vals))
    figs={};tables={}
    if features:
        conditions=sorted(counts)
        fig,ax=plt.subplots(figsize=(8,4))
        x=np.arange(len(conditions))
        for ix,ch in enumerate(('C3','C4')):
            vals=[by_group[c].get(f'{ch}_mu_mean_erd_ers_pct',np.nan) for c in conditions]
            ax.bar(x+ix*.35,np.nan_to_num(vals,nan=0.0),.35,label=f'{ch} mu ERD/ERS')
        ax.set_xticks(x+.175,conditions);ax.axhline(0,color='gray',lw=.7)
        ax.set(ylabel='Change from pre-cue baseline (%)',title='Motor imagery mu ERD/ERS (negative = desynchronization)')
        ax.legend()
        figs['Motor mu ERD ERS']=_plot(out,'motor_mu_erd_ers.png',fig)
        tables['Motor mu ERD ERS']=motor_table(conditions,by_group,features,subtype)
    return {'metrics':{'subtype':subtype,'cue_counts':counts,'per_condition':by_group,
                       'trial_features':features,
                       'definition':'100*(postcue_bandpower - precue_bandpower)/precue_bandpower',
                       'pre_cue_s':[-1.5,-.25],'post_cue_s':[.5,2.5]},
            'figures':figs,'figure_tables':tables,'warnings':['Motor imagery is not measured physical exertion or physiological recovery.',
                'ERD/ERS must be interpreted with reference, electrode geometry and task timing in mind.']}


def sleep(raw, filtered, events, out, channels, *, staging_policy="if_missing", chunk_seconds=120):
    # Preserve auxiliary/unknown scoring intervals as explicit gaps, so an
    # undocumented code never becomes the previous valid sleep stage.
    ordered=sorted([e for e in events if e.family=='sleep' and e.alignment=='verified_within_recording'],
                   key=lambda e:e.onset_s)
    stage_labels={'W','N1','N2','N3','REM'}
    # Resolve same-onset auxiliary/stage annotations without inventing extra time.
    chosen_by_onset={}
    for e in ordered:
        old=chosen_by_onset.get(e.onset_s)
        if old is None or (old.semantic_label not in stage_labels and e.semantic_label in stage_labels):
            chosen_by_onset[e.onset_s]=e
    ordered=sorted(chosen_by_onset.values(),key=lambda e:e.onset_s)
    # Infer a final zero-duration epoch only if all prior scoring onsets
    # establish an unambiguous fixed cadence (e.g. the documented 30-s PSG scoring).
    cadence=None
    gaps=[b.onset_s-a.onset_s for a,b in zip(ordered,ordered[1:]) if b.onset_s>a.onset_s]
    if len(gaps)>=3:
        median_gap=float(np.median(gaps))
        if median_gap>0 and np.mean(np.abs(np.asarray(gaps)-median_gap)<=max(.02,.01*median_gap))>=.95:
            cadence=median_gap
    last_duration_inferred=False
    intervals=[]
    for i,e in enumerate(ordered):
        if e.duration_s is not None and e.duration_s>0:
            end=e.onset_s+e.duration_s
        else:
            end=ordered[i+1].onset_s if i+1<len(ordered) else (e.onset_s+cadence if cadence is not None else None)
            if i==len(ordered)-1 and end is not None:
                last_duration_inferred=True
        if e.semantic_label in stage_labels and end is not None and end>e.onset_s and end<=raw.n_times/raw.info['sfreq']+.5:
            intervals.append((e.onset_s,end,e.semantic_label))
    # Automated stage inference is an optional YASA add-on. Expert-provided
    # stage scores ALWAYS remain the primary source where supplied.
    from .sleep_advanced import auto_stage, microstructure, plot_sleep_extensions
    stage_source='provided_scoring' if intervals else 'unscored'
    staged_model=None
    extension_warnings=[]
    if staging_policy not in ('if_missing','always','off'):
        raise ValueError('staging_policy must be if_missing, always or off')
    if staging_policy=='always' or (staging_policy=='if_missing' and not intervals):
        try:
            staged_model=auto_stage(raw)
        except Exception as exc:
            staged_model={'available':False,'reason':f'Automated sleep staging failed: {exc}'}
        if staged_model.get('available'):
            if not intervals:
                intervals=staged_model['intervals']
                stage_source='automated_yasa_unreviewed'
            else:
                # Model hypotheses are reported separately and do not override
                # verified external stages or change the scored hypnogram.
                extension_warnings.append('Automated stage comparison did not replace provided PSG sleep scores.')
        else:
            extension_warnings.append(staged_model.get('reason','Automatic staging unavailable'))
    by_stage={k:sum(b-a for a,b,s in intervals if s==k) for k in ('W','N1','N2','N3','REM')}
    tst=sum(by_stage[k] for k in ('N1','N2','N3','REM'))
    transitions = sum(x[2]!=y[2] for x,y in zip(intervals,intervals[1:]))
    figs={};tables={};warnings=[]
    if intervals:
        stage_y={'N3':0,'N2':1,'N1':2,'REM':3,'W':4}
        fig,ax=plt.subplots(figsize=(11,4))
        for a,b,s in intervals:
            ax.plot([a/3600,b/3600],[stage_y[s]]*2,lw=3,color={'W':'#8d9da3','REM':'#b04468','N1':'#6fb5d9','N2':'#2c7aac','N3':'#1f3e6c'}[s])
        fig_label='Automated sleep hypnogram (unreviewed)' if stage_source.startswith('automated') else 'Scored hypnogram'
        ax.set_yticks(list(stage_y.values()),list(stage_y));ax.set(xlabel='Hours from recording start',ylabel='Sleep stage',title=fig_label)
        figs[fig_label]=_plot(out,'sleep_hypnogram.png',fig)
    else:
        warnings.append('No usable stages available; supplied scores or successful YASA automatic staging are required.')
    # Stage-specific EEG PSD on a representative EEG channel. Each scored interval
    # contributes one comparable, bounded window; metrics are descriptive.
    eeg_channels = [ch for ch,typ in zip(raw.ch_names,raw.get_channel_types()) if typ=='eeg']
    reference = next((ch for ch in ('C3','C4','Cz') if ch in eeg_channels), eeg_channels[0] if eeg_channels else None)
    stage_samples = {stage: [] for stage in ('W','N1','N2','N3','REM')}
    sf=float(raw.info['sfreq'])
    if reference:
        for a,b,stage in intervals:
            if b-a<2:continue
            start=int(a*sf);stop=min(raw.n_times,int(min(b,a+30)*sf))
            sig=raw.get_data(picks=[reference],start=start,stop=stop)[0]
            if len(sig)<int(2*sf):continue
            f,psd=signal.welch(sig,fs=sf,nperseg=min(len(sig),int(4*sf)))
            bands={}
            for label,(lo,hi) in {'delta':(.5,4),'theta':(4,8),'sigma':(11,16),'beta':(13,30)}.items():
                bands[label]=legacy.band_power(f,psd,lo,hi)
            stage_samples[stage].append(bands)
    stage_psd={stage:{band:_safe(np.mean([x[band] for x in vals])) for band in ('delta','theta','sigma','beta')}
               for stage,vals in stage_samples.items() if vals}
    if stage_psd:
        fig,ax=plt.subplots(figsize=(9,4));names=list(stage_psd);width=.18;x=np.arange(len(names))
        for i,band in enumerate(('delta','theta','sigma','beta')):
            ax.bar(x+(i-1.5)*width,[stage_psd[s][band] or 0 for s in names],width,label=band)
        ax.set_xticks(x,names);ax.set(ylabel='Welch band power (V²)',title=f'Stage-specific EEG band power at {reference}')
        ax.legend();figs['Stage band powers']=_plot(out,'sleep_stage_bandpowers.png',fig)
    metrics={'stage_intervals':[{'start_s':float(a),'end_s':float(b),'stage':stage} for a,b,stage in intervals],
             'stage_origin':stage_source,'stage_bandpower_v2':stage_psd,'stage_psd_channel':reference,
             'stage_psd_window_s':30,'stage_psd_max_samples_per_interval':1,
             'final_epoch_duration_inferred_from_verified_cadence':last_duration_inferred,
             'verified_scoring_cadence_s':cadence,
             'time_by_stage_s':by_stage,'total_sleep_time_s':tst if intervals else None,
             'stage_transitions':transitions if intervals else None,
             'sleep_efficiency':None,'sleep_efficiency_unavailable_reason':'Lights-off/on or time-in-bed boundaries not verified',
             'scored_intervals':len(intervals),'last_stage_without_duration_excluded':bool(ordered and len(intervals)<len(ordered))}
    if intervals:
        tables[fig_label]=hypnogram_table(intervals,metrics)
    if stage_psd:
        tables['Stage band powers']=sleep_band_table(stage_psd,reference,metrics)
    # Microstructure detection uses at most two EEG channels and short windows
    # plus overlap; original raw stays lazily backed for >800 MB .fdt files.
    try:
        micro=microstructure(raw,intervals,chunk_seconds=chunk_seconds) if intervals else {
            'available':False,'reason':'No valid sleep stages; event detection requires verified or estimated N2/N3'}
    except Exception as exc:
        micro={'available':False,'reason':f'Sleep event detectors failed: {exc}'}
    if micro.get('available'):
        metrics['spindle_summary']={k:v for k,v in micro['spindle'].items() if k!='events'}
        metrics['spindle_events']=micro['spindle']['events']
        metrics['slow_wave_summary']={k:v for k,v in micro['slow_wave'].items() if k!='events'}
        metrics['slow_wave_events']=micro['slow_wave']['events']
        metrics['sleep_event_detection']={'method':'YASA spindle/slow-wave detectors',
            'version':micro['yasa_version'],'channels':micro['channels'],
            'N2_N3_hours':micro['scored_n2_n3_hours'],
            'windows_skipped':len(micro['skipped_windows']),
            'chunk_seconds':micro['chunk_seconds']}
        metrics['sleep_detection_skipped_windows']=micro['skipped_windows']
        extension_warnings.extend(micro['warnings'])
    else:
        metrics['sleep_event_detection']={'available':False,'reason':micro.get('reason')}
        extension_warnings.append(micro.get('reason','Sleep spindle and slow-wave detection unavailable'))
    if staged_model:
        metrics['automatic_staging']={k:v for k,v in staged_model.items() if k not in ('intervals','epoch_rows')}
        if staged_model.get('available'):
            metrics['automatic_stage_epochs']=staged_model['epoch_rows']
            if stage_source=='provided_scoring':
                pairs=[]
                for row in staged_model['epoch_rows']:
                    mid=row['onset_s']+15
                    manual=next((lab for a,b,lab in intervals if a<=mid<b),None)
                    if manual is not None:
                        pairs.append((manual,row['stage']))
                if pairs:
                    labels=('W','N1','N2','N3','REM')
                    ref=np.array([labels.index(a) for a,b in pairs]);pred=np.array([labels.index(b) for a,b in pairs])
                    observed=float(np.mean(ref==pred))
                    fr=np.bincount(ref,minlength=5)/len(ref);fp=np.bincount(pred,minlength=5)/len(pred)
                    expected=float(np.dot(fr,fp))
                    kappa=(observed-expected)/(1-expected) if expected<1 else None
                    metrics['automatic_vs_scored_comparison']={
                        'matched_30s_epochs':len(pairs),
                        'agreement_fraction':observed,
                        'cohens_kappa':_safe(kappa) if kappa is not None else None,
                        'note':'Exploratory agreement with provided scores; does not validate clinical use'}
    extension_figs,extension_tables=plot_sleep_extensions(out,staged_model,micro)
    figs.update(extension_figs);tables.update(extension_tables)
    warnings+=extension_warnings
    return {'metrics':metrics,'figures':figs,'figure_tables':tables,'warnings':warnings}
