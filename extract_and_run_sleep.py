#!/usr/bin/env python3
"""Create a small, real EEG excerpt from a LOCAL EEGLAB .set + .fdt and test Neuro-buddy.

No synthetic signal is generated. Designed for OpenNeuro ds005207 and other
EEGLAB recordings storing externally referenced float32 sample data.

Run inside the Neuro-buddy V2 v0.5.1 source root:
 python extract_and_run_sleep.py --minutes 20

The file picker selects a .set whose companion .fdt must exist on this computer.
Scored sleep annotations are optional. If available, place the PSGScoring TSV
and its event dictionary alongside the recording or supply --scoring / --dictionary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
from scipy.io import loadmat, savemat


def pick_mac_file(prompt: str):
    if sys.platform != 'darwin':
        raise ValueError('Please provide --set on this platform')
    cmd = ['osascript', '-e', f'POSIX path of (choose file with prompt "{prompt}")']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit('File selection cancelled')
    return Path(proc.stdout.strip())


def safe_float(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def nearest_scoring(path: Path):
    normalized = re.sub(r'\(\d+\)(?=\.set$)', '', path.name, flags=re.I)
    prefix = re.sub(r'_acq-[^_]+_eeg\.set$', '', normalized, flags=re.I)
    folder=path.parent
    candidates = [p for p in folder.glob('*Scoring_events.tsv')
                  if p.is_file() and p.name.startswith(prefix)]
    return candidates[0] if len(candidates) == 1 else None


def stage_dictionary(source: Path, explicit: Path | None):
    if explicit:
        return explicit
    # BIDS inheritance: try the recording folder and up to the dataset root.
    for parent in [source.parent, *list(source.parents)[:5]]:
        for pat in ('*psgScoring_events.json','*PSGScoring_events.json','*stages.json'):
            matches = list(parent.glob(pat))
            if matches:
                return matches[0]
    # Bundled official ds005207 dictionary. Use only for matching PSGScoring
    # file from the explicit OpenNeuro ds005207 workflow.
    if (re.search(r'sub-\d+_ses-\d+_task-sleep_acq-psg_eeg', source.name, flags=re.I)
            and any(source.parent.glob('*PSGScoring_events.tsv'))):
        official = Path(__file__).resolve().parent/'task-sleep_acq-psgScoring_events.json'
        if official.is_file():
            return official
    return None


def scoring_stage_map(dictionary: Path | None):
    if not dictionary or not dictionary.exists():
        return {}
    try:
        doc=json.loads(dictionary.read_text(encoding='utf-8-sig'))
        return {str(k):str(v).strip().upper()
                for k,v in doc.get('staging',{}).get('Levels',{}).items()}
    except (ValueError, OSError):
        return {}


def score_excerpt(rows: list[dict], start: float, stop: float, stages: dict[str,str]):
    score=0.
    for row in rows:
        val = str(row.get('staging',row.get('stage',row.get('value','')))).strip()
        name = stages.get(val,val).upper()
        if name not in ('N2','N3','STAGE N2','STAGE N3','NON-REM 2','NON-REM 3'):
            continue
        onset=safe_float(row.get('onset'))
        duration=safe_float(row.get('duration'))
        if onset is None or duration is None or duration<=0:continue
        score+=max(0,min(onset+duration,stop)-max(onset,start))
    return score


def choose_window(total: float, minutes: int, scoring: Path | None, dictionary: Path | None, start_minutes: float | None):
    duration = min(minutes*60.,total)
    if total < 60:
        raise ValueError('Recording too short for this sleep test (<60 seconds)')
    if start_minutes is not None:
        start=float(start_minutes)*60
        if start<0 or start+duration>total:
            raise ValueError(f'Chosen segment exceeds recording: start={start:.1f}s, length={duration:.1f}s, total={total:.1f}s')
        return start,duration,'explicit_start'
    if scoring is not None:
        try:
            with scoring.open('r',encoding='utf-8-sig',newline='') as fh:
                rows=list(csv.DictReader(fh,delimiter='\t'))
            mapping=scoring_stage_map(dictionary)
            # Select a window that contains a high fraction of documented N2/N3.
            # Never interpret undocumented numeric codes.
            scored=any(str(r.get('staging','')).strip() in mapping for r in rows)
            textual=any(str(r.get('stage',r.get('value',''))).upper() in ('N2','N3') for r in rows)
            if scored or textual:
                candidates=np.arange(0,max(0,total-duration)+.1,60.)
                ranking=[(score_excerpt(rows,float(x),float(x+duration),mapping),-abs(x-(total-duration)/3),float(x)) for x in candidates]
                best=max(ranking)
                if best[0]>0:
                    return best[2],duration,'scored_N2_N3_rich_window'
        except Exception as exc:
            print(f'[NOTE] Could not optimize segment against staging file: {exc}')
    # Default away from immediate setup / awake periods, but not guaranteed sleep.
    start=min(max(0.,total*.15), max(0.,total-duration))
    start=np.floor(start/30)*30
    return float(start),duration,'default_unscored_window'


def crop_event_table(src: Path, destination: Path, start:float, stop:float, sfreq:float):
    with src.open('r',encoding='utf-8-sig',newline='') as fh:
        reader=csv.DictReader(fh,delimiter='\t' if src.suffix.lower()=='.tsv' else ',')
        if not reader.fieldnames or 'onset' not in reader.fieldnames:
            raise ValueError(f'Annotation table lacks onset: {src.name}')
        fieldnames=list(reader.fieldnames)
        rows=[]
        for rec in reader:
            onset=safe_float(rec.get('onset'))
            if onset is None:continue
            dur=safe_float(rec.get('duration'))
            if dur is None: dur=0.
            if dur<0:continue
            end=onset+dur
            if dur and end<=start:continue
            if onset>=stop:continue
            new_onset=max(onset,start)-start
            new_dur=(min(end,stop)-max(onset,start)) if dur else 0.
            if dur and new_dur<=0:continue
            rec=dict(rec)
            rec['onset']=f'{new_onset:.6f}'
            if 'duration' in rec:rec['duration']=f'{new_dur:.6f}'
            if 'sample' in rec:
                rec['sample']=str(int(round(new_onset*sfreq)))
            rows.append(rec)
    with destination.open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fieldnames,delimiter='\t' if src.suffix.lower()=='.tsv' else ',',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    return len(rows)


def make_selfcontained_set(original_path:Path, destination:Path, signal_uV:np.ndarray, sfreq:float, start_sample:int):
    """Store numeric samples INSIDE the MAT v5 EEGLAB .set; no .fdt needed.

    This intentionally removes original embedded events because their latencies
    refer to the full recording. Cropped event TSVs are the timing source.
    """
    mat=loadmat(str(original_path),struct_as_record=False,squeeze_me=True)
    out={k:v for k,v in mat.items() if not k.startswith('__')}
    n_channels,n=int(signal_uV.shape[0]),int(signal_uV.shape[1])
    if int(np.asarray(mat['nbchan']).item()) != n_channels:
        raise ValueError('Channel count mismatch between EEGLAB header and extracted signal')
    out.update({
        'setname': f'{original_path.stem} real excerpt',
        'filename': destination.name,
        'filepath': str(destination.parent),
        'data': np.asarray(signal_uV,dtype=np.float32),
        'datfile': '',
        'pnts': n,
        'trials': 1,
        'nbchan': n_channels,
        'xmin': 0.0,
        'xmax': float((n-1)/sfreq),
        'times': np.arange(n,dtype=np.float32)*(1000.0/sfreq),
        'event': np.empty((0,),dtype=object),
        'urevent': np.empty((0,),dtype=object),
        'epoch': np.empty((0,),dtype=object),
        'saved': 'yes',
    })
    # Preserving original reject/stats fields may imply entire-record metrics;
    # remove inherited analytical state in the excerpt.
    out['reject']=np.empty((0,),dtype=object)
    out['stats']=np.empty((0,),dtype=object)
    destination.parent.mkdir(parents=True,exist_ok=True)
    savemat(str(destination),out,do_compression=True,long_field_names=True)
    return destination


def main():
    ap=argparse.ArgumentParser(description='Crop real .set/.fdt sleep EEG and run Neuro-buddy')
    ap.add_argument('--set',type=Path,default=None,help='Original .set; omitted opens macOS Finder')
    ap.add_argument('--scoring',type=Path,help='PSGScoring_events.tsv (recommended)')
    ap.add_argument('--dictionary',type=Path,help='Matching PSGScoring_events.json (required for numeric sleep codes)')
    ap.add_argument('--minutes',type=int,default=20,choices=(10,15,20),help='Real EEG excerpt length')
    ap.add_argument('--start-minutes',type=float,default=None,help='Crop start in source-recording minutes')
    ap.add_argument('--output',type=Path,default=Path('./real_sleep_excerpt_and_report'))
    ap.add_argument('--no-run',action='store_true',help='Only generate the self-contained real .set and sidecars')
    ap.add_argument('--sleep-staging',choices=('if_missing','always','off'),default='if_missing')
    args=ap.parse_args()
    source=(args.set if args.set else pick_mac_file('Select the ORIGINAL sleep PSG .set file')).expanduser().resolve()
    if not source.is_file() or source.suffix.lower()!='.set':
        ap.error('Please select an existing EEGLAB .set')
    scoring=args.scoring.expanduser().resolve() if args.scoring else nearest_scoring(source)
    dictionary=stage_dictionary(source,args.dictionary.expanduser().resolve() if args.dictionary else None)
    print(f'[INPUT] {source}')
    print(f'[SCORING] {scoring if scoring else "not supplied"}')
    print(f'[STAGE DICTIONARY] {dictionary if dictionary else "not supplied"}')
    import mne
    try:
        raw=mne.io.read_raw_eeglab(str(source),preload=False,verbose='ERROR')
    except FileNotFoundError as exc:
        print('\n[STOP] The real source .fdt is missing or not next to the .set.')
        print(f'[DETAIL] {exc}')
        print('Download the matching FDT onto this Mac. It does not need to be uploaded to ChatGPT.')
        return 2
    except Exception as exc:
        print(f'[STOP] Could not read the real EEG: {exc}')
        return 2
    total=raw.n_times/float(raw.info['sfreq']);sfreq=float(raw.info['sfreq'])
    start,duration,selection=choose_window(total,args.minutes,scoring,dictionary,args.start_minutes)
    start_sample=int(round(start*sfreq));stop_sample=min(raw.n_times,int(round((start+duration)*sfreq)))
    if stop_sample-start_sample < 1000:raise ValueError('Not enough samples to extract')
    print(f'[EXTRACT] {duration/60:.1f} min from {start/60:.2f} min ({selection})')
    print(f'[EXTRACT] {len(raw.ch_names)} channels at {sfreq:g} Hz; reading only {stop_sample-start_sample} samples per channel')
    # MNE returns volts. EEGLAB .set expects numerical samples in µV.
    values_uv=raw.get_data(start=start_sample,stop=stop_sample)*1e6
    out_dir=args.output.expanduser().resolve()
    eeg_dir=out_dir/'sample'
    eeg_dir.mkdir(parents=True,exist_ok=True)
    clean_stem=re.sub(r'\(\d+\)(?=\.|$)','',source.stem)
    eeg_path=eeg_dir/(clean_stem+'.set')
    make_selfcontained_set(source,eeg_path,values_uv,sfreq,start_sample)
    del values_uv
    raw.close()
    # Critical: read-back test to confirm the output is *genuinely* self-contained.
    verify=mne.io.read_raw_eeglab(str(eeg_path),preload=False,verbose='ERROR')
    assert verify.n_times==stop_sample-start_sample
    assert len(verify.ch_names)==len(raw.ch_names)
    probe=verify.get_data(picks=[0],start=0,stop=min(1000,verify.n_times))
    assert np.isfinite(probe).all()
    verify.close()
    print(f'[VERIFIED] Embedded signal is readable without any external .fdt: {eeg_path.stat().st_size/1048576:.1f} MiB')
    n_sidecars=0
    if scoring and scoring.is_file():
        dest=eeg_dir/scoring.name
        nrows=crop_event_table(scoring,dest,start,start+duration,sfreq)
        n_sidecars+=1
        print(f'[SCORING EXCERPT] {nrows} rows, {dest.name}')
    if dictionary and dictionary.is_file():
        shutil.copy2(dictionary,eeg_dir/dictionary.name)
    for candidate in source.parent.glob('*_events.tsv'):
        if candidate==scoring:continue
        if not candidate.name.startswith(clean_stem.rsplit('_eeg',1)[0].split('_acq-')[0]):continue
        dest=eeg_dir/candidate.name
        if not dest.exists():
            nrows=crop_event_table(candidate,dest,start,start+duration,sfreq)
            n_sidecars+=1
            print(f'[EVENTS EXCERPT] {nrows} rows, {dest.name}')
    manifest={'source_set':source.name,'original_source_has_external_fdt':True,
              'subject':source.name.split('_')[0],
              'segment_start_s':start,'segment_length_s':(stop_sample-start_sample)/sfreq,
              'selection_method':selection,'sampling_rate_hz':sfreq,
              'n_channels':len(raw.ch_names),'output_set':eeg_path.name,
              'scoring_file':scoring.name if scoring else None,'stage_dictionary':dictionary.name if dictionary else None,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'origin':'excerpt_from_local_companion_fdt',
              'note':'Provenance describes the local source; evaluate source authenticity independently.'}
    (out_dir/'segment_manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.no_run:
        print('[DONE] Real excerpt created. Run with --no-run omitted to produce a Neuro-buddy report.')
        return 0
    from neurobuddy_v2.pipeline import process
    print('[RUN] Analyzing the real short recording through Neuro-buddy V2...')
    reports=process(eeg_path,out_dir/'report',large_threshold_mb=0,chunk_seconds=60,sleep_staging=args.sleep_staging)
    statuses=[r['processing_status'] for r,_ in reports]
    print('[RESULT] Processing status:',statuses)
    print('[RESULT] Report:',out_dir/'report'/'report.html')
    print('[RESULT] Transfer ZIP:',out_dir/'report'/'report.zip')
    (out_dir/'run_result.json').write_text(json.dumps({'statuses':statuses,'reports':[str(archive) for _,archive in reports]},indent=2))
    return 0 if all(s in ('COMPLETE','PARTIAL') for s in statuses) else 2


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
