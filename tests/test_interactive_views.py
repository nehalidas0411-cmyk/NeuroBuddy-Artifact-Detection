"""Regression tests for additive offline visualizations."""
import json
import re
import subprocess

from neurobuddy_v2.interactive_views import heatmap_html, sleep_replay
from neurobuddy_v2.comparative import build_comparison
from test_athlete_comparative import fake_report


def _sleep_report():
    r=fake_report(1, 10, fam='SLEEP', subtype='scored_sleep')
    r['recording']['duration_s']=120
    r['specialized_metrics']['sleep']={
        'stage_origin':'provided_scoring',
        'stage_intervals':[{'start_s':0,'end_s':30,'stage':'W'},
                           {'start_s':30,'end_s':60,'stage':'N1'},
                           {'start_s':60,'end_s':120,'stage':'N2'}],
        'spindle_events':[{'onset_s':75,'channel':'C3'}],
        'slow_wave_events':[]}
    return r


def test_sleep_replay_available_and_preserves_stages():
    html=sleep_replay(_sleep_report(), sync_player=True)
    assert 'Interactive Sleep Replay' in html
    assert 'provided scoring' in html
    assert '"stage":"N2"' in html
    assert 'nb-seek' in html
    assert 'Jump to stage' in html


def test_sleep_replay_not_in_general():
    assert sleep_replay(fake_report(1, 10)) == ''


def test_sleep_replay_missing_stages_does_not_fabricate():
    r=_sleep_report()
    r['specialized_metrics']['sleep']['stage_intervals']=[]
    assert 'Unavailable' in sleep_replay(r)


def test_heatmap_matches_existing_comparison_without_recalculation():
    _, comp=build_comparison([fake_report(i,v) for i,v in enumerate((10,11,12),1)],fake_report(4,15))
    html=heatmap_html(comp)
    assert 'Baseline vs. Test Brain Heatmap' in html
    assert '"channel":"C3"' in html
    assert '"z":4.0' in html
    assert '"status":"Abnormal"' in html
    assert '"metric":"alpha_power"' in html
    assert 'baseline' in html.lower()


def test_heatmap_missing_channel_data():
    assert 'No comparable channel-level' in heatmap_html({'metrics':[]})


def test_embedded_scripts_have_valid_javascript(tmp_path):
    _, comp=build_comparison([fake_report(i,v) for i,v in enumerate((10,11,12),1)],fake_report(4,15))
    for idx, html in enumerate((heatmap_html(comp),sleep_replay(_sleep_report(),sync_player=True))):
        scripts=re.findall(r'<script>(.*?)</script>',html,re.S)
        assert len(scripts)==1
        p=tmp_path/f'visual_{idx}.js'
        p.write_text(scripts[0])
        result=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
        assert result.returncode==0, result.stderr
