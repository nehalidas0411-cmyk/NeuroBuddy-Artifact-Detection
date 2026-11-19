"""Low-memory large-EEG tests, independent of external datasets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
import numpy as np
import mne
from scipy.io import savemat
from neurobuddy_v2.intake import discover,load_eeg
from neurobuddy_v2.large_recordings import requires_streaming
from neurobuddy_v2.pipeline import process


def test_forced_streaming_sleep_end_to_end():
    with TemporaryDirectory() as tmp:
        p=Path(tmp); sf=100
        t=np.arange(0,75,1/sf)
        sig=np.vstack([4e-6*np.sin(2*np.pi*1.5*t),3e-6*np.sin(2*np.pi*11*t)])
        raw=mne.io.RawArray(sig,mne.create_info(['C3','C4'],sf,['eeg']*2),verbose='ERROR')
        eeg=p/'sub-001_task-sleep_acq-PSG_eeg.fif'
        raw.save(eeg,verbose='ERROR')
        (p/'sub-001_task-sleep_acq-PSGScoring_events.tsv').write_text(
            'onset\tduration\tvalue\n0\t30\tW\n30\t30\tN2\n60\t15\tREM\n')
        report,archive=process(eeg,p/'out',player_frames=8,large_threshold_mb=0,chunk_seconds=25)[0]
        assert report['processing_status']=='COMPLETE'
        assert report['classification']['task_family']=='SLEEP'
        assert report['shared_metrics']['preprocessing']['memory_strategy']=='bounded_chunk_streaming'
        assert report['specialized_metrics']['sleep']['total_sleep_time_s']==45
        assert len(report['shared_metrics']['channel_statistics'])==2
        assert (p/'out'/'sleep_hypnogram.png').is_file()
        html=(p/'out'/'report.html').read_text()
        assert 'Brain Activity Player' in html and 'Complete graph data table' in html
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert {'report.html','report.json','player_data.json','sleep_hypnogram_data.csv'} <= set(z.namelist())


def test_800mb_external_eeglab_fdt_is_read_lazily():
    """Sparse >800 MiB binary verifies true EEGLAB .set/.fdt loading.

    Only short windows are read here. Complete-file streaming is covered by
    test_forced_streaming_sleep_end_to_end with a smaller recording.
    """
    with TemporaryDirectory() as tmp:
        p=Path(tmp);sf=256; ch=16;n=13_300_000
        f=p/'sub-003_task-sleep_acq-PSG_eeg.fdt'
        with f.open('wb') as handle:handle.truncate(n*ch*4)
        name=f.with_suffix('.set')
        savemat(name,{'EEG':{'setname':'sparse external data test','nbchan':ch,'trials':1,
            'pnts':n,'srate':sf,'xmin':0.,'xmax':(n-1)/sf,'data':f.name,
            'chanlocs':np.array([{'labels':f'C{i+1}'} for i in range(ch)],dtype=object),
            'event':np.array([],dtype=object),'urevent':np.array([],dtype=object)}})
        assert f.stat().st_size > 800*1024**2
        b=discover(name)[0]
        assert b.companion == f
        raw=load_eeg(b)
        assert raw.preload is False
        assert requires_streaming(raw,b)
        assert raw.get_data(picks=['C1'],start=0,stop=100).shape == (1,100)
        assert raw.get_data(picks=['C16'],start=n-100,stop=n).shape == (1,100)
        raw.close()


def test_missing_fdt_keeps_sleep_classification():
    with TemporaryDirectory() as tmp:
        p=Path(tmp);n=1000;sf=100;ch=2
        eeg=p/'sub-003_task-sleep_acq-PSG_eeg.set'
        savemat(eeg,{'EEG':{'setname':'missing fdt','nbchan':ch,'trials':1,
            'pnts':n,'srate':sf,'xmin':0.,'xmax':(n-1)/sf,
            'data':'missing_companion.fdt','chanlocs':np.array([
                {'labels':'C3'},{'labels':'C4'}],dtype=object),
            'event':np.array([],dtype=object),'urevent':np.array([],dtype=object)}})
        report,_=process(eeg,p/'out',player_frames=5)[0]
        assert report['processing_status']=='INCOMPLETE'
        assert report['classification']['task_family']=='SLEEP'
        assert not report['routing']['shared_eeg_ionm_executed']


def test_forced_streaming_oddball_erp_from_lazy_windows():
    with TemporaryDirectory() as tmp:
        p=Path(tmp);sf=100
        tt=np.arange(0,30,1/sf)
        sig=np.vstack([5e-6*np.sin(2*np.pi*8*tt),3e-6*np.sin(2*np.pi*10*tt)])
        raw=mne.io.RawArray(sig,mne.create_info(['Cz','Pz'],sf,['eeg']*2),verbose='ERROR')
        eeg=p/'sub-001_task-P300_run-1_eeg.fif'
        raw.save(eeg,verbose='ERROR')
        (p/'sub-001_task-P300_run-1_events.tsv').write_text(
            'onset\tvalue\n' + ''.join(f'{1+i*2}\t{("oddball" if i%2 else "standard")}\n' for i in range(12)))
        report,_=process(eeg,p/'out',player_frames=6,large_threshold_mb=0,chunk_seconds=15)[0]
        assert report['processing_status']=='COMPLETE'
        assert report['classification']['task_family']=='ODDBALL'
        assert 'oddball_erp.png' in [p.name for p in (p/'out').iterdir()]
        assert report['specialized_metrics']['oddball']['erp'] is not None
