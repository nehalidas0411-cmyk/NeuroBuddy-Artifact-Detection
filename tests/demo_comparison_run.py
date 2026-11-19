"""Small synthetic fixture illustrating 3 baseline + 1 test. NOT human EEG."""
from pathlib import Path
import numpy as np
import mne
from neurobuddy_v2.comparative import run_athlete_workflow

root=Path('/mnt/data/neurobuddy_comparative_build/DEMO_athlete')
root.mkdir(parents=True,exist_ok=True)
files=[]
fs=128
for idx, amplitude in enumerate([8.0,9.0,10.0,15.0],1):
    t=np.arange(0,14,1/fs)
    data=np.vstack([amplitude*1e-6*np.sin(2*np.pi*10*t)+1e-6*np.sin(2*np.pi*2*t),
                    (amplitude+1)*1e-6*np.sin(2*np.pi*11*t)+1.1e-6*np.sin(2*np.pi*3*t)])
    raw=mne.io.RawArray(data,mne.create_info(['C3','C4'],fs,ch_types=['eeg','eeg']),verbose='ERROR')
    eeg=root/f'sub-001_task-rest_run-{idx}_eeg.fif'
    raw.save(eeg,overwrite=True,verbose='ERROR')
    files.append(eeg)
report=run_athlete_workflow(files[:3],files[3],root/'athlete_output',
                            team_id='demo-team',athlete_id='demo-athlete',
                            player_frames=12,large_threshold_mb=0,chunk_seconds=15,
                            sleep_staging='off')
print('result',report['processing_status'],report['comparison_summary'])
print('package',root/'athlete_output'/'athlete_report.zip')
