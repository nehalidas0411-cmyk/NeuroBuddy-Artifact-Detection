"""Real ds005207 encoding: numeric staging codes must remain strings when parsed."""
import json
from pathlib import Path
import tempfile
import mne
import numpy as np
from neurobuddy_v2.pipeline import process


def test_ds005207_scoring_numeric_codes_keep_dictionary_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        fs=100
        t=np.arange(0,180,1/fs)
        sig=np.vstack([10e-6*np.sin(2*np.pi*1*t),10e-6*np.sin(2*np.pi*12*t)])
        raw=mne.io.RawArray(sig,mne.create_info(['C3','C4'],fs,ch_types='eeg'),verbose='ERROR')
        path=root/'sub-003_ses-001_task-sleep_acq-PSG_eeg.fif'
        raw.save(path,overwrite=True,verbose='ERROR')
        (root/'sub-003_ses-001_task-sleep_acq-PSGScoring_events.tsv').write_text(
            'onset\tduration\tstaging\n0\t60\t4\n60\t60\t5\n120\t60\t2\n')
        (root/'task-sleep_acq-psgScoring_events.json').write_text(json.dumps({
            'staging':{'Levels':{'1':'Wake','2':'REM','3':'N1','4':'N2','5':'N3'}}}))
        report,_=process(path,root/'reports',large_threshold_mb=0,chunk_seconds=60,
                         sleep_staging='off',player_frames=12)[0]
        assert report['processing_status']=='COMPLETE',report['errors']
        assert report['classification']['task_family']=='SLEEP'
        assert report['specialized_metrics']['sleep']['stage_origin']=='provided_scoring'
        assert [e['semantic_label'] for e in report['events']['normalized']]==['N2','N3','REM']
        assert (root/'reports'/'sleep_hypnogram.png').exists()
