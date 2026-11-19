"""Regression coverage for the original 3-baseline + 1-test Neuro-buddy workflow."""
from pathlib import Path
import copy
import json
import tempfile
import zipfile

import pytest

from neurobuddy_v2.comparative import (build_comparison, ComparisonError,
                                      run_athlete_workflow,run_from_manifest,scalar_metrics)


def fake_report(index, value, *, fam='GENERAL', subtype=None,
                subject='001', sfreq=160, status='COMPLETE', method='Welch 4s'):
    return {'report_id':f'report-{index}', 'recording_id':f'rec-{index}',
            'pipeline_version':'0.6.0','processing_status':status,
            'input_manifest':[{'role':'signal','name':f'recording-{index}.fif','sha256':f'{index:064x}','bytes':100}],
            'classification':{'task_family':fam,'subtype':subtype},
            'recording':{'file':f'sub-{subject}_task-rest_run-{index}_eeg.fif',
                         'sample_rate_hz':sfreq},
            'shared_metrics':{'channel_statistics':{
                'C3':{'metrics_available':True,'alpha_power':value,'alpha_relative':value/10,
                      'rms':value*1e-6,'zero_crossings':40},
                'C4':{'metrics_available':True,'alpha_power':value+2,'rms':value*2e-6}},
                'preprocessing':{'psd_method':method,'raw_stats':'unfiltered EEG','filtered_quality':'1-45 Hz'}},
            'specialized_metrics':{fam.lower():{'erp':{'p300_difference_mean_uv':value,
                                                        'n_target_epochs':10,
                                                        'target_vs_standard_trial_test':{'p_uncorrected':0.04}}}}}


def test_exact_three_baselines_sample_sd_and_abnormal():
    baseline=[fake_report(i,v) for i,v in enumerate([10,11,12],1)]
    test=fake_report(4,15)
    longitudinal,comp=build_comparison(baseline,test)
    a=next(row for row in longitudinal['metrics'] if row['metric']=='shared.channel.C3.alpha_power')
    b=next(row for row in comp['metrics'] if row['metric']=='shared.channel.C3.alpha_power')
    assert (a['n'],a['mean'],a['sd']) == (3,11,1)
    assert (b['baseline_mean'],b['baseline_sd'],b['test_value'],b['z_score']) == (11,1,15,4)
    assert b['status']=='Abnormal'
    assert a['cv_percent']==pytest.approx(100/11)
    assert 'p_uncorrected' not in scalar_metrics(test)
    assert 'n_target_epochs' not in scalar_metrics(test)
    assert comp['abnormal_rule']=='abs(z) > threshold'


def test_strict_threshold_exact_three_is_within_range():
    _,comp=build_comparison([fake_report(i,v) for i,v in enumerate([10,11,12],1)],fake_report(4,14))
    row=next(r for r in comp['metrics'] if r['metric']=='shared.channel.C3.alpha_power')
    assert row['z_score']==3 and row['status']=='Within range'


def test_baseline_zero_sd_is_not_fabricated_infinite_z():
    _,comp=build_comparison([fake_report(i,10) for i in range(1,4)],fake_report(4,20))
    row=next(r for r in comp['metrics'] if r['metric']=='shared.channel.C3.alpha_power')
    assert row['baseline_sd']==0 and row['z_score'] is None
    assert row['status']=='Not comparable'


def test_duplicate_records_rejected():
    with pytest.raises(ComparisonError,match='distinct'):
        build_comparison([fake_report(1,1),fake_report(2,2),fake_report(1,3)],fake_report(4,4))
    with pytest.raises(ComparisonError,match='distinct'):
        build_comparison([fake_report(1,1),fake_report(2,2),fake_report(3,3)],fake_report(3,4))


def test_families_subtypes_and_subjects_must_match():
    base=[fake_report(i,v,fam='MOTOR',subtype='imagery') for i,v in enumerate([10,11,12],1)]
    with pytest.raises(ComparisonError,match='subtype mismatch'):
        build_comparison(base,fake_report(4,15,fam='MOTOR',subtype='execution'))
    with pytest.raises(ComparisonError,match='subtype mismatch'):
        build_comparison(base,fake_report(4,15,fam='SLEEP',subtype='scored_sleep'))
    with pytest.raises(ComparisonError,match='participant BIDS IDs differ'):
        build_comparison(base,fake_report(4,15,fam='MOTOR',subtype='imagery',subject='002'))


def test_incomplete_input_and_wrong_method_block():
    base=[fake_report(i,v) for i,v in enumerate([10,11,12],1)]
    with pytest.raises(ComparisonError,match='all inputs must complete'):
        build_comparison(base,fake_report(4,15,status='PARTIAL'))
    with pytest.raises(ComparisonError,match='sampling-rate mismatch'):
        build_comparison(base,fake_report(4,15,sfreq=256))
    with pytest.raises(ComparisonError,match='analysis-method mismatch'):
        build_comparison(base,fake_report(4,15,method='periodogram'))


def test_missing_metric_not_compared_with_only_two_baselines():
    base=[fake_report(i,v) for i,v in enumerate([10,11,12],1)]
    del base[0]['shared_metrics']['channel_statistics']['C3']['rms']
    _,comp=build_comparison(base,fake_report(4,15))
    row=next(r for r in comp['metrics'] if r['metric']=='shared.channel.C3.rms')
    assert row['status']=='Not comparable' and row['baseline_n']==2


def test_orchestration_produces_three_sections_and_transfer_zip():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        files=[]
        for i in range(1,5):
            f=root/f'sub-001_task-rest_run-{i}_eeg.fif'
            f.write_text('fixture')
            files.append(f)
        recorded=[]
        def stub_runner(input, dest, **kwargs):
            index=len(recorded)+1
            recorded.append((Path(input),dest,kwargs))
            r=fake_report(index, 10+index)
            (dest/'report.html').write_text(f'<html>Recording {index}</html>')
            (dest/'report.json').write_text(json.dumps(r))
            (dest/'report.zip').write_bytes(b'zip fixture')
            (dest/'a_figure.png').write_bytes(b'fixture')
            return [(r, dest/'report.zip')]
        result=run_athlete_workflow(files[:3],files[3],root/'out',
                team_id='team-alpha',athlete_id='athlete-beta',process_one=stub_runner,
                player_frames=6,large_threshold_mb=0)
        assert len(recorded)==4
        assert all(call[2]['large_threshold_mb']==0 for call in recorded)
        assert len(result['baseline_reports'])==3
        assert result['processing_status']=='COMPLETE'
        text=(root/'out'/'athlete_report.html').read_text()
        assert text.count('<details') == 3
        assert '<summary>BASELINE</summary>' in text
        assert '<summary>TEST EEG</summary>' in text
        assert '<summary>COMPARATIVE ANALYSIS</summary>' in text
        assert 'Abnormal' in text and 'Within range' in text
        assert 'athlete_report.json' in (root/'out').joinpath('manifest.json').read_text()
        with zipfile.ZipFile(root/'out'/'athlete_report.zip') as z:
            assert z.testzip() is None
            assert {'athlete_report.html','athlete_report.json','comparative_analysis.json',
                    'baseline/longitudinal.json','baseline/longitudinal.csv',
                    'test/report.html','baseline/01_sub-001_task-rest_run-1_eeg/report.html'} <= set(z.namelist())
            assert not any(name.endswith('.fif') for name in z.namelist())


def test_manifest_resolves_relative_paths():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for i in range(4):(root/f'{i}.fif').write_text('abc')
        manifest=root/'inputs.json'
        manifest.write_text(json.dumps({'team_id':'abc','athlete_id':'x',
            'baseline_inputs':['0.fif','1.fif','2.fif'],'test_input':'3.fif'}))
        def stub(path,out,**kwargs):
            i=int(Path(path).stem)
            report=fake_report(i+1,i+4)
            out.mkdir(parents=True,exist_ok=True)
            (out/'report.json').write_text(json.dumps(report));(out/'report.html').write_text('test')
            return [(report,out/'report.zip')]
        result=run_from_manifest(manifest,root/'analysis',process_one=stub)
        assert result['athlete_id']=='x' and result['task_family']=='GENERAL'


def test_team_manifest_separates_two_athletes():
    from neurobuddy_v2.comparative import run_team_manifest
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        athletes=[]
        for athlete in ['Ada','Ben']:
            files=[]
            for n in range(1,5):
                f=p/f'{athlete}_{n}.fif'
                f.write_text('fixture')
                files.append(f)
            athletes.append({'athlete_id':athlete,
                             'baseline_inputs':[f.name for f in files[:3]],
                             'test_input':files[3].name})
        team=p/'team.json'
        team.write_text(json.dumps({'team_id':'school-1','athletes':athletes}))
        # Stub actual EEG execution while retaining real output & ZIP packaging.
        # Team calls the production orchestrator and is covered by the four-run
        # integration test separately; use monkeypatch for deterministic fixture.
        from neurobuddy_v2 import pipeline
        seen=[]
        original=pipeline.process
        def stub(input,dest,**kwargs):
            n=(len(seen)%4)+1
            sub='001' if len(seen)<4 else '002'
            seen.append(Path(input))
            r=fake_report(n+100*(sub=='002'),n+9,subject=sub)
            dest.mkdir(parents=True,exist_ok=True)
            (dest/'report.html').write_text('report')
            (dest/'report.json').write_text(json.dumps(r))
            return [(r,dest/'report.zip')]
        pipeline.process=stub
        try:
            result=run_team_manifest(team,p/'out')
            assert result['processing_status']=='COMPLETE'
            assert len(result['athletes'])==2 and len(seen)==8
            assert (p/'out'/'team_report.zip').exists()
            with zipfile.ZipFile(p/'out'/'team_report.zip') as z:
                assert z.testzip() is None
                assert 'athletes/Ada/athlete_report.json' in z.namelist()
                assert 'athletes/Ben/athlete_report.json' in z.namelist()
            assert result['cross_athlete_averaging_performed'] is False
        finally:
            pipeline.process=original
