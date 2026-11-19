"""Traceable data tables for each plotted figure.

A figure table contains the exact numerical series used to draw that figure,
plus scientific summary/test results where available. Every row is exported to
CSV and embedded into the self-contained HTML for paginated inspection.
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
from scipy import signal


def spec(rows, description, *, summary=None, extras=None):
    return {'rows': rows, 'description': description,
            'summary': summary or {}, 'extra_tables': extras or {}}


def shared_tables(raw, filtered, channels, eeg_stats, quality, records, tests, cfg):
    if not channels:
        return {}
    sf = float(raw.info['sfreq'])
    count = min(raw.n_times, max(2, int(cfg.display_seconds*sf)))
    times = np.arange(count)/sf
    ch = channels[0]
    orig = raw.get_data(picks=[ch], start=0, stop=count)[0]
    filt = filtered.get_data(picks=[ch], start=0, stop=count)[0]
    # Matches legacy_combined.save_graphs' plotted samples and plotted units.
    waveform = [
        {'time_s': float(t), 'channel':ch, 'raw_uv':float(r*1e6), 'filtered_uv':float(f*1e6)}
        for t,r,f in zip(times,orig,filt)
    ]
    f,p = signal.periodogram(raw.get_data(picks=[ch])[0], fs=sf)
    ff,pp = signal.periodogram(filtered.get_data(picks=[ch])[0], fs=sf)
    max_f = min(50.0, sf/2)
    # The legacy plot's axes show only frequencies <= max_f.
    mask = f <= max_f
    freq_rows = [
        {'frequency_hz':float(hz),'channel':ch,'raw_psd_v2_per_hz':float(a),
         'filtered_psd_v2_per_hz':float(b)}
        for hz,a,b in zip(f[mask],p[mask],pp[mask])
    ]
    bands = ('delta','theta','alpha','beta')
    band_rows = [{'channel':c, **{f'{band}_relative_pct':
                       100 * eeg_stats[c].get(band+'_relative')
                       if eeg_stats[c].get(band+'_relative') is not None else 0.0
                       for band in bands},
                  **{f'{band}_absolute_v2':eeg_stats[c].get(band+'_power') for band in bands}}
                 for c in channels]
    q_rows = [{'channel':c,'rms_filtered_uv':
                    float(quality[c].get('rms',0)*1e6),
               'quality_grade':quality[c]['quality'],
               'snr_difference_proxy_db':quality[c].get('snr_difference_proxy_db'),
               'flatline':quality[c].get('flatline')} for c in channels]
    qcodes={'bad':0,'poor':1,'fair':2,'good':3}
    level_rows=[{'channel':c,'quality_score_0_bad_3_good':qcodes[quality[c]['quality']],
                'quality_grade':quality[c]['quality'],
                'rms_filtered_uv':float(quality[c].get('rms',0)*1e6)} for c in channels]
    tables = {
      'Raw vs. filtered EEG waveform':spec(waveform,
          f'Every displayed waveform sample for {ch}. Values are microvolts; raw and filtered were plotted separately.',
          summary={'channel':ch,'display_seconds':float(times[-1]) if len(times) else 0.0,
                   'sample_rate_hz':sf,'raw_recording_rms_v':eeg_stats[ch].get('rms')}),
      'EEG power spectrum':spec(freq_rows,
          f'Periodogram PSD bins displayed for {ch}; both traces are in V²/Hz. Includes every visible frequency bin.',
          summary={'channel':ch,'frequency_resolution_hz':float(f[1]-f[0]) if len(f)>1 else None,
                   'raw_total_spectral_power_v2':eeg_stats[ch].get('total_spectral_power'),
                   'raw_peak_alpha_frequency_hz':eeg_stats[ch].get('peak_alpha_frequency'),
                   'raw_spectral_entropy':eeg_stats[ch].get('spectral_entropy')}),
      'Relative EEG band power':spec(band_rows,
          'Each bar corresponds to the relative power percentage shown for its channel and band. Absolute powers are also included in V².',
          summary={'channels_plotted':len(channels),'relative_scale':'0–100 percent'}),
      'Channel RMS':spec(q_rows,
          'Filtered-channel RMS values are shown in µV. Heuristic quality fields are supplementary.',
          summary={'channels_plotted':len(channels),'filter_source':'IONM filtered EEG'}),
      'Channel quality':spec(level_rows,
          'Numerical categories match bar heights: 0 bad, 1 poor, 2 fair, 3 good. Screening grades are heuristic, not diagnoses.',
          summary={'channels_plotted':len(channels),'grade_mapping':'0=bad, 1=poor, 2=fair, 3=good'})
    }
    if records:
        change_uv=np.asarray([r['mean_change']*1e6 for r in records], dtype=float)
        hist,edges=np.histogram(change_uv,bins=20)
        rows=[{'bin_start_uv':float(edges[i]),'bin_end_uv':float(edges[i+1]),
               'event_channel_records':int(hist[i])} for i in range(len(hist))]
        per_trial=[{'trial':r.get('trial'),'event_code':r.get('event_code'),
                    'channel':r.get('channel'),
                    'mean_change_uv':r.get('mean_change',0)*1e6,
                    'baseline_mean_uv':r.get('baseline_mean',0)*1e6,
                    'event_mean_uv':r.get('event_mean',0)*1e6,
                    'rms_change_uv':r.get('rms_change',0)*1e6}
                   for r in records]
        tables['Event response changes']=spec(rows,
            'Histogram bins exactly match the event-response change distribution. A supplementary table contains every event/channel measurement.',
            summary={'n_event_channel_records':len(records),
                     'mean_change_uv':float(np.mean(change_uv)),
                     'median_change_uv':float(np.median(change_uv)),
                     'standard_deviation_uv':float(np.std(change_uv))},
            extras={'Individual event/channel responses':per_trial})
    if tests:
        rows=[{'channel':r.get('channel'),'groups':', '.join(r.get('groups',[])),
               'group_trial_counts':str(r.get('group_trial_counts',{})),
               'kruskal_wallis_H':r['H_statistic'], 'p_value':r['p_value'],
               'minus_log10_p_plotted':-math.log10(max(r['p_value'],1e-300)),
               'epsilon_squared':r.get('epsilon_squared'),
               'bonferroni_alpha':r.get('bonferroni_alpha'),
               'significant_uncorrected':r.get('significant_uncorrected'),
               'significant_bonferroni':r.get('significant_bonferroni')}
              for r in tests]
        tables['Event statistical comparisons']=spec(rows,
            'Full Kruskal–Wallis comparisons underlying the −log10(p) bars, including test statistic, uncorrected p, effect size and correction threshold.',
            summary={'tests_plotted':len(tests),'uncorrected_alpha':cfg.statistical_alpha,
                     'bonferroni_alpha':tests[0].get('bonferroni_alpha')})
    return tables


def oddball_erp_table(times, target, standard, difference, erp_metrics):
    rows = [{'latency_ms':float(t*1e3),
             'target_mean_uv':float(a),'standard_mean_uv':float(b),
             'target_minus_standard_uv':float(d)}
            for t,a,b,d in zip(times,target,standard,difference)]
    summary={k:v for k,v in erp_metrics.items() if k != 'note'}
    return spec(rows, 'Every time point of the target, standard and difference-wave traces shown in the ERP plot. ERP window statistics and trial comparison appear above the data.',summary=summary)


def oddball_rt_table(reaction_times_s, behavior):
    rt_ms=np.asarray(reaction_times_s)*1000
    hist,edges=np.histogram(rt_ms,bins=min(20,max(4,len(rt_ms)//5)))
    rows=[{'bin_start_ms':float(edges[i]),'bin_end_ms':float(edges[i+1]),
           'matched_target_responses':int(hist[i])} for i in range(len(hist))]
    extras={'Individual matched reaction times':[
        {'response_index':i+1,'reaction_time_ms':float(ms)}
        for i,ms in enumerate(rt_ms)]}
    return spec(rows, 'Exact reaction-time histogram bins and counts. Individual matched response latencies are available below.',
                summary=behavior,extras=extras)


def motor_table(conditions, group, features, subtype):
    rows=[]
    for cue in conditions:
        for channel in ('C3','C4'):
            rows.append({'cue':cue,'channel':channel,
                         'mu_mean_erd_ers_pct':group[cue].get(channel+'_mu_mean_erd_ers_pct'),
                         'beta_mean_erd_ers_pct':group[cue].get(channel+'_beta_mean_erd_ers_pct'),
                         'cue_count':sum(f['cue']==cue and f['channel']==channel and f['band']=='mu'
                                         for f in features)})
    return spec(rows,'Mu-band values reproduce each plotted motor ERD/ERS bar. Beta is included as an additional calculated metric.',
                summary={'motor_subtype':subtype,'baseline_s':[-1.5,-0.25],'post_cue_s':[0.5,2.5],
                         'erd_ers_definition':'100 × (post − pre) / pre'},
                extras={'Individual motor trial features':features})


def hypnogram_table(intervals, metrics):
    rows=[{'interval':i+1,'onset_s':float(a),'end_s':float(b),
           'onset_hours':float(a/3600),'end_hours':float(b/3600),
           'stage':s,'duration_s':float(b-a)} for i,(a,b,s) in enumerate(intervals)]
    summary={k:v for k,v in metrics.items() if k in (
        'time_by_stage_s','total_sleep_time_s','stage_transitions','scored_intervals',
        'sleep_efficiency','sleep_efficiency_unavailable_reason')}
    return spec(rows,'Every scored sleep-stage interval used to render the hypnogram. Times are relative to EEG recording start.',summary=summary)


def sleep_band_table(stage_psd, channel, metrics):
    rows=[{'stage':stage, 'channel':channel,
           **{f'{band}_power_v2':values.get(band) for band in ('delta','theta','sigma','beta')}}
          for stage,values in stage_psd.items()]
    return spec(rows,'Stage-average Welch band powers displayed by the grouped bar graph; all reported powers use V².',
                summary={'channel':channel,'window_s':metrics.get('stage_psd_window_s'),
                         'samples_per_interval_max':metrics.get('stage_psd_max_samples_per_interval'),
                         'total_sleep_time_s':metrics.get('total_sleep_time_s')})
