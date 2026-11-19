"""Sleep V2.5 wiring and failure-condition tests.

The CI environment might not install optional YASA. A deterministic test double
checks glue/data-contract behavior; it does not validate YASA physiology or the
accuracy of a pretrained classifier. A real YASA/PSG integration run is needed.
"""
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neurobuddy_v2.analysis import sleep
from neurobuddy_v2.pipeline import process
from neurobuddy_v2 import sleep_advanced


def fake_yasa():
    fake=types.ModuleType('yasa');fake.__version__='0.8.1-fixture'
    class FakeStaging:
        def __init__(self, raw, eeg_name, eog_name=None, emg_name=None):
            self.raw=raw
            assert eeg_name in raw.ch_names
        def predict(self):
            n=self.raw.n_times//(30*int(self.raw.info['sfreq']))
            stages=['W','N2','N3','REM']
            hypo=np.array([stages[i%4] for i in range(n)])
            proba=pd.DataFrame([{k:(0.8 if k==stages[i%4] else 0.2/3)
                                 for k in stages} for i in range(n)])
            return types.SimpleNamespace(hypno=hypo,proba=proba)
    def fake_spindles(data, sf, ch_names, hypno, include, verbose):
        assert sf==100 and tuple(include)==(2,3)
        accepted=np.where(np.isin(hypno,[2,3]))[0]
        if not accepted.size:return None
        j=int(accepted[len(accepted)//2]);center=j/sf
        class Spindle:
            def summary(self):
                return pd.DataFrame([{'Start':center-0.4,'Peak':center,'End':center+0.4,
                    'Stage':int(hypno[j]),'Channel':ch_names[0],
                    'Duration':0.8,'Amplitude':32.,'RMS':5.2,'Frequency':13.1,'RelPower':0.46}])
        return Spindle()
    def fake_slow(data, sf, ch_names, hypno, include, verbose):
        accepted=np.where(np.isin(hypno,[2,3]))[0]
        if not accepted.size:return None
        j=int(accepted[len(accepted)//2]);center=j/sf
        class Slow:
            def summary(self):
                return pd.DataFrame([{'Start':center-0.55,'NegPeak':center,'End':center+0.55,
                    'Stage':int(hypno[j]),'Channel':ch_names[0],
                    'Duration':1.1,'PTP':115.,'ValNegPeak':-85.,'Slope':120.,'Frequency':0.91}])
        return Slow()
    fake.SleepStaging=FakeStaging
    fake.spindles_detect=fake_spindles
    fake.sw_detect=fake_slow
    return fake


def make_fif(path, seconds, sf=100):
    t=np.arange(0,seconds,1/sf)
    x=np.vstack([8e-6*np.sin(2*np.pi*1*t),
                 12e-6*np.sin(2*np.pi*13*t)+8e-6*np.sin(2*np.pi*.85*t)])
    raw=mne.io.RawArray(x,mne.create_info(['C3','C4'],sf,['eeg','eeg']),verbose='ERROR')
    raw.save(path,verbose='ERROR')
    return raw


def test_unscored_sleep_automatic_staging_full_report(monkeypatch):
    monkeypatch.setitem(sys.modules,'yasa',fake_yasa())
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)
        recording=p/'sub-001_task-sleep_acq-PSG_eeg.fif'
        make_fif(recording,seconds=360)
        report,zpath=process(recording,p/'out',player_frames=8,
            large_threshold_mb=0,chunk_seconds=60,sleep_staging='if_missing')[0]
        assert report['processing_status']=='COMPLETE',report['errors']
        assert report['classification']['task_family']=='SLEEP'
        data=report['specialized_metrics']['sleep']
        assert data['stage_origin']=='automated_yasa_unreviewed'
        assert data['automatic_staging']['available'] is True
        assert data['automatic_staging']['n_epochs']==12
        assert len(data['automatic_stage_epochs'])==12
        assert data['spindle_summary']['count']>0
        assert data['slow_wave_summary']['count']>0
        assert data['spindle_events'] and data['slow_wave_events']
        names=[x['title'] for x in report['visualizations']['figures']]
        assert 'Automated sleep hypnogram (unreviewed)' in names
        assert 'Automatic sleep stage confidence' in names
        assert 'Sleep spindles by hour' in names
        assert 'Sleep slow waves by hour' in names
        html=(p/'out'/'report.html').read_text()
        assert 'Specialized pathway results' in html
        assert '<h3>Spindle Events</h3>' in html and 'Download every value as CSV' in html
        assert 'Sleep spindles by hour' in html and 'Complete graph data table' in html
        with zipfile.ZipFile(zpath) as z:
            assert z.testzip() is None
            assert 'sleep_spindle_counts_data.csv' in z.namelist()
            assert 'sleep_slow_wave_counts_data.csv' in z.namelist()
            assert 'sleep_automatic_stage_confidence_data.csv' in z.namelist()
            report_json=json.loads(z.read('report.json'))
            assert report_json['classification']['analysis_readiness']['hypnogram']


def test_scored_stages_take_precedence_and_microstructure(monkeypatch):
    monkeypatch.setitem(sys.modules,'yasa',fake_yasa())
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);path=p/'sub-001_task-sleep_acq-PSG_eeg.fif'
        make_fif(path,seconds=360)
        (p/'sub-001_task-sleep_acq-PSGScoring_events.tsv').write_text(
            'onset\tduration\tvalue\n0\t120\tN2\n120\t120\tN3\n240\t120\tREM\n')
        report,_=process(path,p/'out',player_frames=8,large_threshold_mb=0,
                          chunk_seconds=60,sleep_staging='always')[0]
        assert report['processing_status']=='COMPLETE',report['errors']
        m=report['specialized_metrics']['sleep']
        assert m['stage_origin']=='provided_scoring'
        assert m['total_sleep_time_s']==360
        assert m['automatic_staging']['available']
        assert m['automatic_vs_scored_comparison']['matched_30s_epochs']==12
        assert m['automatic_vs_scored_comparison']['cohens_kappa'] is not None
        assert m['spindle_summary']['count']>=1
        assert m['slow_wave_summary']['count']>=1
        assert (p/'out'/'sleep_hypnogram.png').exists()
        assert m['stage_transitions']==2


def test_yasa_absent_fails_gracefully(monkeypatch):
    monkeypatch.setattr(sleep_advanced,'_load_yasa',lambda:None)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);path=p/'sub-001_task-sleep_acq-PSG_eeg.fif'
        raw=make_fif(path,seconds=120)
        x=sleep(raw,raw,[],p,raw.ch_names,staging_policy='if_missing')
        assert x['metrics']['stage_origin']=='unscored'
        assert x['metrics']['automatic_staging']['available'] is False
        assert x['metrics']['sleep_event_detection']['available'] is False
        assert any('YASA' in w for w in x['warnings'])
        assert x['figures']=={}
        raw.close()


def test_auto_sleep_staging_uses_only_selected_channels(monkeypatch):
    monkeypatch.setitem(sys.modules,'yasa',fake_yasa())
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);path=p/'sub-001_task-sleep_eeg.fif'
        raw=make_fif(path,seconds=360)
        # This does not prove production memory footprint, but the staging
        # adapter only constructs 100 Hz data for the best EEG channel.
        r=sleep_advanced.auto_stage(raw)
        assert r['channels']['eeg']=='C4' or r['channels']['eeg']=='C3'
        assert r['channels']['eog'] is None and r['channels']['emg'] is None
        assert r['downsample_hz']==100
        raw.close()
