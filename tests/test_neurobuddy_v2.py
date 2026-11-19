import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import mne
import pytest
from neurobuddy_v2.intake import discover,load_eeg,safe_extract_zip
from neurobuddy_v2.events import load_events
from neurobuddy_v2.routing import decide
from neurobuddy_v2.pipeline import process

SOURCE=Path('/mnt/data')


def test_real_oddball_classifies_with_semantics():
    filename=SOURCE/'sub-001_task-P300_run-1_eeg.set'
    if not filename.exists():pytest.skip('Uploaded dataset not present')
    b=discover(filename)[0];raw=load_eeg(b)
    events,warnings=load_events(b,raw)
    d=decide(b,events,warnings)
    assert d.task_family=='ODDBALL'
    assert d.analysis_readiness['erp']
    assert d.analysis_readiness['reaction_time']
    assert len(events)==863
    assert any(x.raw_value=='oddball_with_reponse' and x.semantic_label=='target' for x in events)
    assert all(x.response_time_s is None for x in events)
    raw.close()


def test_real_motor_imagery_and_duplicates():
    filename=SOURCE/'sub-001_task-motion_run-4_eeg.set'
    if not filename.exists():pytest.skip('Uploaded dataset not present')
    b=discover(filename)[0];raw=load_eeg(b)
    assert any('Duplicate identical annotation' in x for x in b.warnings)
    assert len(b.annotations)==1
    events,warnings=load_events(b,raw)
    d=decide(b,events,warnings)
    assert (d.task_family,d.subtype)==('MOTOR','imagery')
    assert len(events)==30
    assert {'left_hand','right_hand','rest'} <= {e.semantic_label for e in events}
    raw.close()


def test_general_fif_without_task_evidence():
    with tempfile.TemporaryDirectory() as t:
        t=Path(t)
        info=mne.create_info(['C3','C4'],100,['eeg','eeg'])
        eeg=mne.io.RawArray(np.random.default_rng(55).normal(0,1e-6,(2,1400)),info,verbose='ERROR')
        f=t/'sub-002_task-rest_run-1_eeg.fif';eeg.save(f,overwrite=True,verbose='ERROR')
        b=discover(f)[0];r=load_eeg(b);ev,_=load_events(b,r);d=decide(b,ev)
        assert d.task_family=='GENERAL'
        r.close()


def test_sleep_scoring_associates_psg_not_ceegrid_and_computes_hypnogram():
    from neurobuddy_v2.analysis import sleep
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)
        info=mne.create_info(['C3','C4','Cz'],100,['eeg']*3)
        data=np.random.default_rng(21).normal(0,3e-6,(3,9000))
        raw=mne.io.RawArray(data,info,verbose='ERROR')
        eeg=p/'sub-001_ses-001_task-sleep_acq-PSG_eeg.fif'
        raw.save(eeg,overwrite=True,verbose='ERROR')
        scoring=p/'sub-001_ses-001_task-sleep_acq-PSGScoring_events.tsv'
        scoring.write_text('onset\tduration\tvalue\n0\t30\tW\n30\t30\tN2\n60\t30\tREM\n')
        b=discover(eeg)[0]
        assert b.annotations == [scoring]
        events,_=load_events(b,raw);d=decide(b,events)
        assert d.task_family=='SLEEP' and d.analysis_readiness['hypnogram']
        out=p/'figs';out.mkdir();results=sleep(raw,raw,events,out,raw.ch_names)
        assert results['metrics']['total_sleep_time_s']==60
        assert results['metrics']['time_by_stage_s']['W']==30
        assert (out/'sleep_hypnogram.png').exists()


def test_conflicting_mapped_task_evidence_needs_review():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t);raw=mne.io.RawArray(np.zeros((2,200)),mne.create_info(['C3','C4'],100,['eeg']*2),verbose='ERROR')
        f=p/'sub-001_task-P300_run-1_eeg.fif';raw.save(f,overwrite=True,verbose='ERROR')
        ann=p/'sub-001_task-P300_run-1_events.tsv'
        ann.write_text('onset\tvalue\n0.1\tleft_hand\n0.8\tright_hand\n')
        b=discover(f)[0];ev,_=load_events(b,raw);decision=decide(b,ev)
        assert decision.status=='NEEDS_REVIEW'


def test_zip_traversal_rejected():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t);path=p/'bad.zip'
        with zipfile.ZipFile(path,'w') as z:z.writestr('../danger.txt','nope')
        with pytest.raises(ValueError,match='Unsafe'):safe_extract_zip(path,p/'out')


def test_complete_general_report_html_json_zip():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)
        sf=128;tm=np.arange(0,8,1/sf)
        data=np.vstack([10e-6*np.sin(2*np.pi*10*tm),6e-6*np.sin(2*np.pi*16*tm)])
        raw=mne.io.RawArray(data,mne.create_info(['C3','C4'],sf,['eeg','eeg']),verbose='ERROR')
        f=p/'sub-002_task-rest_run-1_eeg.fif';raw.save(f,overwrite=True,verbose='ERROR')
        reports=process(f,p/'report',player_frames=12)
        report,archive=reports[0]
        assert report['processing_status']=='COMPLETE'
        assert report['classification']['task_family']=='GENERAL'
        assert archive.exists()
        html=(archive.parent/'report.html').read_text()
        assert 'Brain Activity Player' in html
        assert '<script id="nb-player-data"' in html
        assert report['visualizations']['brain_activity_player']['topomap_available'] is False
        with zipfile.ZipFile(archive) as z:
            assert {'report.json','report.html','player_data.json','manifest.json'} <= set(z.namelist())
        assert (archive.parent/'report.json').exists()


def test_inherited_bids_event_dictionary_recognized():
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);(root/'dataset_description.json').write_text('{"Name":"mock","BIDSVersion":"1.9.0"}')
        eegfolder=root/'sub-001'/'ses-001'/'eeg';eegfolder.mkdir(parents=True)
        f=eegfolder/'sub-001_ses-001_task-sleep_acq-PSG_eeg.fif'
        raw=mne.io.RawArray(np.zeros((2,2000)),mne.create_info(['C3','C4'],100,['eeg']*2),verbose='ERROR')
        raw.save(f,overwrite=True,verbose='ERROR')
        (root/'task-sleep_acq-PSGScoring_events.json').write_text('{"value":{"Levels":{"W":"Wake","N2":"Sleep stage N2"}}}')
        (eegfolder/'sub-001_ses-001_task-sleep_acq-PSGScoring_events.tsv').write_text('onset\tduration\tvalue\n0\t10\tW\n10\t10\tN2\n')
        bundle=discover(f)[0]
        assert any('PSGScoring_events.json' in p.name for p in bundle.dictionaries)
        assert bundle.annotations[0].name.endswith('PSGScoring_events.tsv')


def test_zip_input_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);sf=100;tt=np.arange(0,8,1/sf)
        raw=mne.io.RawArray(np.vstack([1e-5*np.sin(2*np.pi*8*tt),1e-5*np.sin(2*np.pi*12*tt)]),
                            mne.create_info(['C3','C4'],sf,['eeg']*2),verbose='ERROR')
        f=root/'sub-123_task-rest_eeg.fif';raw.save(f,overwrite=True,verbose='ERROR')
        z=root/'uploaded.zip'
        with zipfile.ZipFile(z,'w') as dst:dst.write(f,'sub-123/eeg/'+f.name)
        reports=process(z,root/'out',player_frames=10)
        assert len(reports)==1 and reports[0][0]['processing_status']=='COMPLETE'


def test_complete_sleep_route_and_versioned_schema():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);sf=100;tt=np.arange(0,70,1/sf)
        raw=mne.io.RawArray(np.vstack([8e-6*np.sin(2*np.pi*1.5*tt), 9e-6*np.sin(2*np.pi*12*tt)]),
                            mne.create_info(['C3','C4'],sf,['eeg']*2),verbose='ERROR')
        f=p/'sub-001_task-sleep_acq-PSG_eeg.fif';raw.save(f,overwrite=True,verbose='ERROR')
        stage=p/'sub-001_task-sleep_acq-PSGScoring_events.tsv'
        stage.write_text('onset\tduration\tvalue\n0\t20\tW\n20\t20\tN2\n40\t30\tREM\n')
        results=process(f,p/'out',player_frames=10)
        report,zipfile_path=results[0]
        assert report['processing_status']=='COMPLETE'
        assert report['classification']['task_family']=='SLEEP'
        assert report['specialized_metrics']['sleep']['total_sleep_time_s']==50
        assert 'Stage band powers' in [f['title'] for f in report['visualizations']['figures']]
        assert report['events']['count']==3
        with zipfile.ZipFile(zipfile_path) as z:
            assert 'report.schema.json' in z.namelist()
            import jsonschema
            instance=json.loads(z.read('report.json'))
            schema=json.loads(z.read('report.schema.json'))
            jsonschema.validate(instance,schema)


def test_real_ds005207_numeric_stage_dictionary_schema():
    """OpenNeuro ds005207 uses 'staging' numeric codes, 0 durations, JSON Levels."""
    from neurobuddy_v2.analysis import sleep
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/'dataset_description.json').write_text('{"Name":"Surrey sleep","BIDSVersion":"1.5.0"}')
        (root/'task-sleep_acq-psgScoring_events.json').write_text(json.dumps({'staging':{
            'Levels':{'1':'Wake','2':'REM','3':'N1','4':'N2','5':'N3',
                      '6':'A','7':'Movement','8':'Artefact','9':'Unscored'}}}))
        folder=root/'sub-001'/'ses-001'/'eeg';folder.mkdir(parents=True)
        sf=100; secs=180
        data=np.random.default_rng(7).normal(0,3e-6,(2,sf*secs))
        raw=mne.io.RawArray(data,mne.create_info(['C3','C4'],sf,['eeg']*2),verbose='ERROR')
        f=folder/'sub-001_ses-001_task-sleep_acq-PSG_eeg.fif'
        raw.save(f,overwrite=True,verbose='ERROR')
        # The code 6 is an aux annotation and must not be mapped to N-stage.
        tab=folder/'sub-001_ses-001_task-sleep_acq-PSGScoring_events.tsv'
        tab.write_text('onset\tduration\tstaging\n0\t0\t6\n30\t0\t1\n60\t0\t3\n90\t0\t4\n120\t0\t9\n150\t0\t2\n')
        bundle=discover(f)[0]
        assert any('psgScoring_events.json' in x.name for x in bundle.dictionaries)
        ev,warnings=load_events(bundle,raw)
        assert len(ev)==6
        assert [e.semantic_label for e in ev]==['A','W','N1','N2','Unscored','REM']
        assert all(e.mapping_source and 'JSON' in e.mapping_source for e in ev)
        assert 'sleep_staging_numeric_codes_not_documented' not in warnings
        d=decide(bundle,ev)
        assert d.task_family=='SLEEP' and d.analysis_readiness['hypnogram']
        out=root/'report';out.mkdir()
        result=sleep(raw,raw,ev,out,raw.ch_names)
        # W=30, N1=30, N2=30, unscored gap=30; final REM 30 s
        # is inferred from consistently spaced 30-s scoring annotations.
        assert result['metrics']['total_sleep_time_s']==90
        assert result['metrics']['time_by_stage_s']['W']==30
        assert result['metrics']['time_by_stage_s']['N1']==30
        assert result['metrics']['time_by_stage_s']['N2']==30
        assert result['metrics']['scored_intervals']==4
        assert result['metrics']['final_epoch_duration_inferred_from_verified_cadence']


def test_oddball_behavior_rt_and_signal_detection_from_real_onsets():
    from neurobuddy_v2.analysis import oddball
    source=SOURCE/'sub-001_task-P300_run-1_eeg.set'
    if not source.exists():pytest.skip('Real oddball set not uploaded')
    bundle=discover(source)[0];raw=load_eeg(bundle)
    events,_=load_events(bundle,raw)
    # Timebase-based performance summary independently of original RT metadata.
    targets=sum(e.semantic_label=='target' for e in events)
    responses=sum(e.semantic_label=='response' for e in events)
    assert targets==113 and responses==113
    assert sum(e.original_response_time is not None for e in events)==113
    assert all(e.response_time_s is None for e in events)
    raw.close()


def test_every_graph_has_complete_embedded_table_and_transfer_csv():
    """Under-figure tables must stay in HTML, JSON manifest, and portable ZIP."""
    import csv
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        fs=128;tt=np.arange(0,9,1/fs)
        raw=mne.io.RawArray(np.vstack([5e-6*np.sin(2*np.pi*8*tt),
                                       8e-6*np.sin(2*np.pi*13*tt)]),
            mne.create_info(['C3','C4'],fs,['eeg','eeg']),verbose='ERROR')
        eeg=root/'sub-050_task-rest_eeg.fif'
        raw.save(eeg,overwrite=True,verbose='ERROR')
        rep,archive=process(eeg,root/'out',player_frames=6)[0]
        figures=rep['visualizations']['figures']
        assert len(figures)>=5
        html=(root/'out'/'report.html').read_text()
        assert html.count('Complete graph data table') == len(figures)
        assert 'Statistical Results tab' not in html
        for fig in figures:
            data=fig['data_table']
            assert data['rows']>0
            f=root/'out'/data['asset']
            assert f.is_file()
            with f.open(newline='') as stream:
                records=list(csv.DictReader(stream))
            assert len(records)==data['rows']
            assert list(records[0])==data['columns']
            assert data['asset'] in html
        wave=next(x for x in figures if x['title']=='Raw vs. filtered EEG waveform')
        assert wave['data_table']['rows']==len(tt)
        with zipfile.ZipFile(archive) as z:
            for fig in figures:
                assert fig['data_table']['asset'] in z.namelist()
            assert z.testzip() is None
        assert html.count('class="graph-data"')>=len(figures)



def test_specialized_metrics_display_as_tables_not_raw_json():
    from neurobuddy_v2.reporting import _specialized_markup
    import json
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        oddball={'oddball':{'event_counts':{'target':113,'standard':522},
                'erp':{'p300_difference_peak_latency_ms':343.75,
                       'target_vs_standard_trial_test':{'p_uncorrected':0.162,'cohens_d':-0.197}},
                'behavior':{'mean_reaction_time_s':0.4764}}}
        out=_specialized_markup(root,oddball)
        assert '<pre' not in out and '"erp"' not in out
        assert out.count('<table>')==3
        assert 'P300 Difference Peak Latency' in out and '343.75' in out
        assert '0.162' in out and '0.4764' in out
        motor={'motor':{'subtype':'imagery','trial_features':[
            {'cue':'left_hand','channel':'C3','band':'mu','erd_ers_pct':1.2},
            {'cue':'right_hand','channel':'C4','band':'beta','erd_ers_pct':-2.2}]}}
        out=_specialized_markup(root,motor)
        assert 'nbspecial0' in out and 'left_hand' in out and 'C3' in out
        datafile=root/'specialized_motor_trial_features_data.csv'
        assert datafile.exists() and len(datafile.read_text().splitlines())==3
        sleep={'sleep':{'time_by_stage_s':{'N2':90,'REM':30},
                        'stage_bandpower_v2':{'N2':{'delta':2.5e-12}},
                        'sleep_efficiency':None,
                        'sleep_efficiency_unavailable_reason':'No lights-out marker'}}
        out=_specialized_markup(root,sleep)
        assert 'N2' in out and '2.5e-12' in out and 'No lights-out marker' in out
        assert 'Not available' in out
