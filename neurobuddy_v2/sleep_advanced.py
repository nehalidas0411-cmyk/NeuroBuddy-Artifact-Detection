"""Research-grade optional sleep extensions for Neuro-buddy.

YASA provides the published classifier and event detection methods. Crucially,
large EEGLAB/FDT recordings stay lazy: only small, selected-channel windows
are loaded for event detection; staging constructs a reduced-channel 100-Hz
recording, not a full preload of the original overnight PSG.

Outputs remain *algorithmic estimates* and must be reviewed against manual PSG
scoring. Scored stage labels always take precedence for sleep architecture.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
import re

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from .figure_tables import spec

SLEEP_CODES = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
YASA_VERSION_RANGE = '>=0.8.1,<0.9'


def _load_yasa():
    try:
        import yasa
        return yasa
    except ImportError:
        return None


def _safe_number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def _stage_text(value):
    label = str(value).upper().strip()
    return {'WAKE': 'W', 'NREM1': 'N1', 'NREM2': 'N2', 'NREM3': 'N3',
            'R': 'REM'}.get(label, label)


def _eeg_candidates(raw):
    """Select EEG, not an EOG/ECG/EMG channel mislabeled as EEG."""
    names = []
    for ch, typ in zip(raw.ch_names, raw.get_channel_types()):
        if typ != 'eeg' or re.search(r'(?i)EOG|EMG|ECG|EKG|CHIN|AIR|RESP|FLOW|OXY', ch):
            continue
        names.append(ch)
    def rank(name):
        upper = name.upper()
        preferences = ('C4-M1', 'C3-M2', 'C4-A1', 'C3-A2', 'C4', 'C3', 'CZ',
                       'F3', 'F4', 'FZ', 'PZ')
        for idx, item in enumerate(preferences):
            if upper == item or upper.startswith(item+'-') or upper.startswith(item+' '):
                return idx
        return len(preferences)
    return sorted(names, key=rank)


def _aux_candidates(raw, kind):
    names=[]
    for ch,typ in zip(raw.ch_names,raw.get_channel_types()):
        if kind == 'eog' and (typ=='eog' or re.search(r'(?i)EOG|LOC|ROC',ch)):
            names.append(ch)
        if kind == 'emg' and (typ=='emg' or re.search(r'(?i)EMG|CHIN|MENTALIS',ch)):
            names.append(ch)
    return names


def _read_resampled(raw, picks, start_s, end_s, target_hz=100):
    """Read a *bounded* window, downsample selected channels to target_hz.

    For source sampling rates at least target_hz, scipy resample_poly provides
    anti-aliasing. Timing is referenced to recording start, not wall clock.
    """
    sf=float(raw.info['sfreq'])
    start_idx=max(0,int(round(start_s*sf)))
    stop_idx=min(raw.n_times,int(round(end_s*sf)))
    if stop_idx <= start_idx:
        return np.zeros((len(picks),0),dtype=float)
    arr=raw.get_data(picks=list(picks),start=start_idx,stop=stop_idx)
    if not np.isfinite(arr).all():
        raise ValueError('Non-finite EEG data in selected sleep analysis window')
    if abs(sf-target_hz)<1e-6:
        return arr
    frac=Fraction(target_hz/sf).limit_denominator(10000)
    return signal.resample_poly(arr,frac.numerator,frac.denominator,axis=-1)


def auto_stage(raw, *, max_hours=18, target_hz=100):
    """YASA automated staging from reduced, *unfiltered* selected PSG channels.

    Whole-night YASA inference needs the complete reduced-channel time series
    for contextual features. The large original .fdt remains lazily backed;
    memory is O(duration * 100 Hz * <=3 channels), not O(all EEG channels).
    """
    yasa=_load_yasa()
    if yasa is None:
        return {'available':False, 'reason':
            'YASA sleep extras not installed. Run: python -m pip install -r requirements-sleep.txt'}
    picks=_eeg_candidates(raw)
    if not picks:
        return {'available':False,'reason':'No documented EEG channel suitable for automated staging'}
    sf=float(raw.info['sfreq'])
    total=float(raw.n_times/sf)
    if sf<80 or total<300:
        return {'available':False,'reason':'Automatic staging needs >=80 Hz and >=5 minutes of EEG'}
    if total>max_hours*3600:
        return {'available':False,'reason':f'Sleep staging capped at {max_hours} hours to limit RAM use'}
    # Match YASA training recommendations: central EEG where possible, EOG/EMG
    # optional. Unknown channel types are not silently assumed to be EOG/EMG.
    eeg=picks[0]
    eog=next(iter(_aux_candidates(raw,'eog')),None)
    emg=next(iter(_aux_candidates(raw,'emg')),None)
    selected=[eeg]+([eog] if eog and eog!=eeg else [])+([emg] if emg and emg not in (eeg,eog) else [])
    duration=int(total//30)*30
    n=duration*target_hz
    data=np.empty((len(selected),n),dtype=np.float64)
    block_seconds=120
    for start_s in range(0,duration,block_seconds):
        end_s=min(duration,start_s+block_seconds)
        part=_read_resampled(raw,selected,start_s,end_s,target_hz)
        i=start_s*target_hz;j=end_s*target_hz
        if part.shape[1] != j-i:
            if abs(part.shape[1]-(j-i))>1:
                return {'available':False,'reason':'Staging downsampled samples did not align with epochs'}
            if part.shape[1] < j-i:
                part=np.pad(part,((0,0),(0,j-i-part.shape[1])),mode='edge')
            else:
                part=part[:,:j-i]
        data[:,i:j]=part
    types=['eeg']+(['eog'] if eog else [])+(['emg'] if emg else [])
    info=mne.create_info(selected,target_hz,ch_types=types)
    staging_raw=mne.io.RawArray(data,info,verbose='ERROR')
    # Do not prefilter / standardize: YASA handles units & internal features.
    estimator=yasa.SleepStaging(staging_raw,eeg_name=eeg,eog_name=eog,emg_name=emg)
    result=estimator.predict()
    labels=np.asarray(getattr(result,'hypno',result)).reshape(-1)
    proba=getattr(result,'proba',None)
    # YASA 0.8.x returns a Hypnogram object with .proba; older API supported
    # predict_proba(). Avoid inventing probabilities when absent.
    if proba is None and hasattr(estimator,'predict_proba'):
        proba=estimator.predict_proba()
    if proba is not None:
        prob_arr=np.asarray(proba,dtype=float)
        prob_columns=list(getattr(proba,'columns',[]))
    else:
        prob_arr=None;prob_columns=[]
    intervals=[];rows=[]
    for idx,orig in enumerate(labels):
        onset=float(idx*30)
        if onset+30>duration:break
        label=_stage_text(orig)
        if label not in SLEEP_CODES:continue
        probabilities={}
        if prob_arr is not None and idx<len(prob_arr):
            if prob_arr.ndim==2 and len(prob_columns)==prob_arr.shape[1]:
                probabilities={_stage_text(c):_safe_number(v)
                               for c,v in zip(prob_columns,prob_arr[idx])}
        confidence=max((v for v in probabilities.values() if v is not None),default=None)
        intervals.append((onset,onset+30,label))
        rows.append({'onset_s':onset,'duration_s':30.,'stage':label,
                     'model_max_probability':confidence,
                     'review_recommended':bool(confidence is not None and confidence<0.6),
                     **{f'probability_{k}':probabilities.get(k) for k in SLEEP_CODES}})
    if not intervals:
        return {'available':False,'reason':'Automatic stage model returned no valid stage labels'}
    del estimator,staging_raw,data
    return {'available':True,'intervals':intervals,'epoch_rows':rows,
            'method':'YASA pretrained LightGBM sleep staging',
            'model_source':'YASA built-in pretrained classifier',
            'yasa_version':getattr(yasa,'__version__','unknown'),
            'channels':{'eeg':eeg,'eog':eog,'emg':emg},
            'downsample_hz':target_hz,'scoring_origin':'automated_unreviewed',
            'epoch_duration_s':30,
            'n_epochs':len(intervals),
            'low_confidence_epoch_count':sum(r['review_recommended'] for r in rows),
            'mean_max_probability':_safe_number(np.mean([
                row['model_max_probability'] for row in rows
                if row['model_max_probability'] is not None])) if any(row['model_max_probability'] is not None for row in rows) else None}


def _stage_mask(intervals,start_s,stop_s,hz, length):
    mask=np.full(length,-2,dtype=np.int8)
    for a,b,label in intervals:
        if label not in SLEEP_CODES or b<=start_s or a>=stop_s:continue
        i=max(0,int(round((max(start_s,a)-start_s)*hz)))
        j=min(length,int(round((min(stop_s,b)-start_s)*hz)))
        mask[i:j]=SLEEP_CODES[label]
    return mask


def _standard_event_rows(frame, *, kind, start_s, core_start,core_stop):
    output=[]
    if frame is None:return output
    for row in frame.to_dict('records'):
        center=row.get('Peak',row.get('NegPeak',row.get('Start')))
        peak=_safe_number(center)
        if peak is None or not core_start<=start_s+peak<core_stop:continue
        stage_code=row.get('Stage',None)
        label = {v:k for k,v in SLEEP_CODES.items()}.get(stage_code, _stage_text(stage_code) if stage_code is not None else None)
        if label not in ('N2','N3'):continue
        item={'kind':kind,'channel':row.get('Channel','unknown'),
              'stage':label,'start_s':_safe_number(start_s+float(row['Start'])),
              'peak_s':_safe_number(start_s+peak),
              'end_s':_safe_number(start_s+float(row['End'])),
              'duration_s':_safe_number(row.get('Duration'))}
        if kind=='spindle':
            item.update({'amplitude_uv':_safe_number(row.get('Amplitude')),
                         'frequency_hz':_safe_number(row.get('Frequency')),
                         'rms_uv':_safe_number(row.get('RMS')),
                         'relative_power':_safe_number(row.get('RelPower'))})
        else:
            item.update({'peak_to_peak_uv':_safe_number(row.get('PTP')),
                         'negative_peak_uv':_safe_number(row.get('ValNegPeak')),
                         'slope_uv_per_s':_safe_number(row.get('Slope')),
                         'frequency_hz':_safe_number(row.get('Frequency'))})
        output.append(item)
    return output


def microstructure(raw, intervals, *, chunk_seconds=120, hz=100):
    """Detect N2/N3 spindles and slow waves using bounded-window YASA calls.

    The count is per selected channel; no cross-channel merging. A window with
    obvious gross artifacts is rejected, never silently counted as zero.
    """
    yasa=_load_yasa()
    if yasa is None:
        return {'available':False,'reason':
            'YASA event detection not installed. Run: python -m pip install -r requirements-sleep.txt'}
    eeg=_eeg_candidates(raw)
    if not eeg:
        return {'available':False,'reason':'No EEG channels suitable for detecting sleep events'}
    if raw.info['sfreq']<50:
        return {'available':False,'reason':'Sampling rate below 50 Hz; spindle band cannot be resolved reliably'}
    if not any(label in ('N2','N3') for _,_,label in intervals):
        return {'available':False,'reason':'No verified or model-estimated N2/N3 intervals for detection'}
    names=eeg[:2]
    total=raw.n_times/float(raw.info['sfreq'])
    core_seconds=max(30,min(300,int(chunk_seconds)))
    overlap_s=8
    rows={'spindle':[],'slow_wave':[]}
    skipped=[]
    for core_start in np.arange(0,total,core_seconds):
        core_end=min(total,core_start+core_seconds)
        if not any(b>core_start and a<core_end and lab in ('N2','N3') for a,b,lab in intervals):
            continue
        start_s=max(0,core_start-overlap_s)
        stop_s=min(total,core_end+overlap_s)
        samples=_read_resampled(raw,names,start_s,stop_s,hz)*1e6
        if samples.shape[1]<int(10*hz):continue
        stage_mask=_stage_mask(intervals,start_s,stop_s,hz,samples.shape[1])
        for index,name in enumerate(names):
            data=samples[index]
            if not np.isfinite(data).all() or np.std(data)<0.3 or np.max(np.abs(data))>500:
                skipped.append({'channel':name,'start_s':float(core_start),'reason':'gross_signal_artifact_or_flatline'})
                continue
            for kind,detector in [('spindle',yasa.spindles_detect),('slow_wave',yasa.sw_detect)]:
                try:
                    result=detector(data,sf=hz,ch_names=[name],hypno=stage_mask,
                                    include=(2,3),verbose=False)
                    if result is not None:
                        rows[kind].extend(_standard_event_rows(result.summary(),kind=kind,
                                    start_s=start_s,core_start=core_start,core_stop=core_end))
                except (ValueError,RuntimeError) as exc:
                    skipped.append({'channel':name,'start_s':float(core_start),
                                    'reason':f'{kind}_detector_error: {str(exc)[:160]}'})
        del samples
    nrem_h = sum(max(0,b-a) for a,b,lab in intervals if lab in ('N2','N3'))/3600
    result={'available':True,'channels':names,'yasa_version':getattr(yasa,'__version__','unknown'),
            'stage_restricted_to':['N2','N3'],'scored_n2_n3_hours':nrem_h,
            'chunk_seconds':core_seconds,'overlap_seconds':overlap_s,
            'skipped_windows':skipped,
            'warnings':['Spindle and slow-wave events are algorithmic candidate detections, not human-validated events.',
                        'Chunk-edge detection and amplitude thresholds require validation against scored PSG.']}
    for kind in rows:
        counts=dict(Counter(x['channel'] for x in rows[kind]))
        stage_counts=dict(Counter(x['stage'] for x in rows[kind]))
        durations=[x['duration_s'] for x in rows[kind] if x['duration_s'] is not None]
        amp_field='amplitude_uv' if kind=='spindle' else 'peak_to_peak_uv'
        amplitudes=[x[amp_field] for x in rows[kind] if x[amp_field] is not None]
        result[kind]={'count':len(rows[kind]),'count_by_channel':counts,
                      'count_by_stage':stage_counts,
                      'mean_duration_s':_safe_number(np.mean(durations)) if durations else None,
                      'mean_amplitude_uv':_safe_number(np.mean(amplitudes)) if amplitudes else None,
                      'events':rows[kind],
                      'density_events_per_nrem_hour_by_channel':{
                          ch:round(counts.get(ch,0)/nrem_h,4) if nrem_h>0 else None for ch in names}}
    return result


def plot_sleep_extensions(out:Path, staging:dict|None, micro:dict|None):
    """Generate plots + exact data tables for the report, not just raw JSON."""
    figs={}; tables={}
    if staging and staging.get('available'):
        rows=staging['epoch_rows']
        conf=[x['model_max_probability'] for x in rows]
        if any(x is not None for x in conf):
            fig,ax=plt.subplots(figsize=(10,3))
            ax.plot([r['onset_s']/3600 for r in rows],conf,lw=.8)
            ax.set(xlabel='Hours from EEG start',ylabel='Model maximum probability',
                   title='Automatic sleep staging: model confidence (not validation)',ylim=(0,1))
            fig.tight_layout();fig.savefig(out/'sleep_automatic_stage_confidence.png',dpi=120);plt.close(fig)
            figs['Automatic sleep stage confidence']='sleep_automatic_stage_confidence.png'
            tables['Automatic sleep stage confidence']=spec(rows,
                'All automatically predicted 30-s stages and model probabilities. Must be reviewed by an expert.',
                summary={'method':staging['method'],'channels':staging['channels'],
                         'mean_max_probability':staging['mean_max_probability']})
    if micro and micro.get('available'):
        for key,title,name in [('spindle','Sleep spindles by hour','sleep_spindle_counts.png'),
                               ('slow_wave','Sleep slow waves by hour','sleep_slow_wave_counts.png')]:
            events=micro[key]['events']
            if not events: continue
            bins=Counter(int(e['peak_s']//3600) for e in events)
            rows=[{'hour_start':int(i),'hour_end':int(i+1),'candidate_count':int(bins.get(i,0))}
                  for i in range(int(max(bins))+1)]
            fig,ax=plt.subplots(figsize=(10,3))
            ax.bar([r['hour_start'] for r in rows],[r['candidate_count'] for r in rows],width=.85)
            ax.set(xlabel='Elapsed recording hour',ylabel='Candidate count',title=title)
            fig.tight_layout();fig.savefig(out/name,dpi=120);plt.close(fig)
            figs[title]=name
            tables[title]=spec(rows,
                'Candidate event counts by hour across explicitly listed channels. These are channel events, not unique cortical events.',
                summary={'candidate_count':micro[key]['count'],
                         'channels':micro['channels'],'n2_n3_hours':micro['scored_n2_n3_hours']},
                extras={('All detected spindle events' if key=='spindle' else 'All detected slow-wave events'):events})
    return figs,tables
