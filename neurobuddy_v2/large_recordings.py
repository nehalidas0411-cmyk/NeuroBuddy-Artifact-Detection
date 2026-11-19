"""Bounded-memory (out-of-core) analysis for long EEGLAB / overnight PSG files.

A large .fdt is read through MNE's lazy reader in short windows. We never call
raw.load_data(), never make a full-length filtered copy, and never compute a
record-length periodogram. Estimates that differ from the legacy full-record
periodogram/median are explicitly identified in the provenance.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .legacy_combined import Settings, BANDS, band_power
from .figure_tables import spec

DEFAULT_THRESHOLD_MIB = 256
DEFAULT_CHUNK_SECONDS = 60


def requires_streaming(raw, bundle, threshold_mib=DEFAULT_THRESHOLD_MIB):
    if threshold_mib <= 0:
        return True
    n_eeg = sum(t == 'eeg' for t in raw.get_channel_types())
    expected_float_bytes = raw.n_times * n_eeg * 8
    disk_bytes = max([p.stat().st_size for p in (bundle.signal, bundle.companion) if p is not None], default=0)
    return max(expected_float_bytes, disk_bytes) >= threshold_mib * 1024 * 1024


class WindowFiltered:
    """A lazy, chunk-only filtered EEG view, not an MNE Raw object.

    Use for sleep/motor windowed features. Reject unbounded requests instead
    of quietly materializing many hours of data. SOS forward/backward filtering
    with an 8-second pad is only an approximation of whole-run filtering.
    """
    def __init__(self, raw, channels, cfg, max_window_s=120):
        self.raw = raw
        self.ch_names = list(channels)
        self.info = raw.info
        self.n_times = raw.n_times
        self.first_samp = raw.first_samp
        self.max_window_s = max_window_s
        nyq = float(raw.info['sfreq']) / 2
        self.high = min(cfg.low_pass_hz, nyq * .90)
        self.low = cfg.high_pass_hz
        if self.low >= self.high:
            raise ValueError('Sampling rate too low for the default filtering branch')
        self.sos = signal.butter(4, [self.low, self.high], fs=float(raw.info['sfreq']),
                                 btype='bandpass', output='sos')

    def get_data(self, picks=None, start=None, stop=None):
        sf = float(self.info['sfreq'])
        if start is None or stop is None:
            raise ValueError('Out-of-core filtered data requires explicit start and stop sample indexes')
        start, stop = int(start), int(stop)
        if start < 0 or stop <= start or stop > self.n_times:
            raise ValueError('Invalid filtered sample range')
        if stop - start > int(self.max_window_s * sf) + 1:
            raise MemoryError('Requested filtered window exceeds bounded-memory limit')
        context = int(8 * sf)
        a, b = max(0, start-context), min(self.n_times, stop+context)
        names = self.ch_names if picks is None else ([picks] if isinstance(picks,str) else list(picks))
        data = self.raw.get_data(picks=names, start=a, stop=b)
        if data.shape[1] < 25:
            return data[:,start-a:stop-a]
        filtered = signal.sosfiltfilt(self.sos, data, axis=1)
        return filtered[:,start-a:stop-a]


class Accumulator:
    """Streaming raw/signal moments plus a fixed-grid averaged Welch PSD."""
    def __init__(self, channels, fs, *, n_fft):
        self.channels = list(channels)
        self.fs = fs
        self.n = 0
        self.s = np.zeros(len(channels),dtype=float)   # microvolts, stable moments
        self.s2 = self.s.copy(); self.s3 = self.s.copy(); self.s4 = self.s.copy()
        self.low = np.full(len(channels),np.inf)
        self.high = np.full(len(channels),-np.inf)
        self.abspeak = self.s.copy()
        self.zc = np.zeros(len(channels),dtype=np.int64)
        self.diff_sq = self.s.copy()
        self.diff_n = 0
        self.diff2_sq = self.s.copy()
        self.diff2_n = 0
        self.prev = None
        self.prev2 = None
        self.nonfinite = np.zeros(len(channels),dtype=np.int64)
        self.n_fft = n_fft
        self.psd_sum = np.zeros((len(channels),n_fft//2+1),dtype=float)
        self.psd_windows = 0
        self.reservoir = [[] for _ in channels]

    def add(self, data, *, start, stride, accumulate_psd=True):
        if data.size == 0: return
        uv = np.asarray(data,dtype=float)*1e6
        finite = np.isfinite(uv)
        self.nonfinite += np.sum(~finite,axis=1)
        # Preserve non-finite status for reporting. Replace NaNs only for
        # bounded moment arithmetic, not with scientifically claimed samples.
        valid = np.where(finite,uv,0.0)
        self.n += valid.shape[1]
        self.s += np.sum(valid,axis=1)
        self.s2 += np.sum(valid*valid,axis=1)
        self.s3 += np.sum(valid*valid*valid,axis=1)
        self.s4 += np.sum((valid*valid)**2,axis=1)
        self.low = np.minimum(self.low,np.min(valid,axis=1))
        self.high = np.maximum(self.high,np.max(valid,axis=1))
        self.abspeak = np.maximum(self.abspeak,np.max(np.abs(valid),axis=1))
        if self.prev is not None:
            self.zc += np.not_equal(np.signbit(self.prev),np.signbit(valid[:,0])).astype(int)
            self.diff_sq += np.square(valid[:,0]-self.prev)
            self.diff_n += 1
            if self.prev2 is not None:
                self.diff2_sq += np.square(valid[:,0]-2*self.prev+self.prev2)
                self.diff2_n += 1
        self.zc += np.sum(np.signbit(valid[:,1:]) != np.signbit(valid[:,:-1]),axis=1)
        delta=np.diff(valid,axis=1)
        self.diff_sq += np.sum(delta*delta,axis=1)
        self.diff_n += valid.shape[1]-1
        if delta.shape[1]>1:
            self.diff2_sq += np.sum(np.square(np.diff(valid, n=2, axis=1)),axis=1)
            self.diff2_n += delta.shape[1]-1
        self.prev2 = valid[:,-2].copy() if valid.shape[1]>=2 else self.prev
        self.prev = valid[:,-1].copy()
        # Deterministic, bounded subsample only for approximate quantiles.
        selected = np.arange(valid.shape[1],dtype=int)
        selected = selected[(selected + start) % stride == 0]
        if len(selected):
            sample=valid[:,selected]
            for i in range(len(self.channels)):
                self.reservoir[i].extend(sample[i].tolist())
        if accumulate_psd and valid.shape[1]>=self.n_fft:
            f,p = signal.welch(np.asarray(data,dtype=float), fs=self.fs,
                  nperseg=self.n_fft, noverlap=self.n_fft//2,
                  axis=1,scaling='density')
            # scipy.welch outputs mean per chunk; weighting by segment count
            n_segments=1+(valid.shape[1]-self.n_fft)//(self.n_fft-self.n_fft//2)
            self.psd_sum += p*n_segments
            self.psd_windows += n_segments

    def result(self):
        n=max(1,self.n)
        mean=self.s/n
        variance=np.maximum(self.s2/n-mean**2,0)
        std=np.sqrt(variance)
        fourth=(self.s4/n-4*mean*self.s3/n+6*mean**2*self.s2/n-3*mean**4)
        third=(self.s3/n-3*mean*self.s2/n+2*mean**3)
        skew=np.divide(third,std**3,out=np.full(len(mean),np.nan),where=std>0)
        kurt=np.divide(fourth,variance**2,out=np.full(len(mean),np.nan),where=variance>0)-3
        return mean,std,variance,skew,kurt

    def psd(self):
        return self.psd_sum / max(1,self.psd_windows)


def _num(v):
    x=float(v)
    return x if np.isfinite(x) else None


def analyze_large(raw, out: Path, *, chunk_seconds=DEFAULT_CHUNK_SECONDS):
    """Calculate bounded-memory shared statistics and five companion figures."""
    from .legacy_combined import Settings
    cfg=Settings()
    fs=float(raw.info['sfreq'])
    chans=[c for c,t in zip(raw.ch_names,raw.get_channel_types()) if t=='eeg']
    filtered=WindowFiltered(raw,chans,cfg,max_window_s=max(120,chunk_seconds))
    # Keep a single float64 EEG chunk below ~32 MiB even if the operator
    # chooses a very long time window or the recording has 128+ channels.
    max_samples_by_memory=max(128,int((32*1024*1024)//(8*len(chans))))
    samples_per_chunk=min(max(int(12*fs),int(chunk_seconds*fs)),max_samples_by_memory)
    if samples_per_chunk < int(12*fs):
        raise MemoryError('Channel count / sampling rate exceeds the 32 MiB per-chunk safety budget')
    # Avoid high FFT memory even on high-sample-rate recordings.
    nfft=max(8,min(int(4*fs),samples_per_chunk))
    stride=max(1,raw.n_times//20000)
    a=Accumulator(chans,fs,n_fft=nfft)
    af=Accumulator(chans,fs,n_fft=nfft)
    num_chunks=(raw.n_times+samples_per_chunk-1)//samples_per_chunk
    for ix,start in enumerate(range(0,raw.n_times,samples_per_chunk)):
        stop=min(raw.n_times,start+samples_per_chunk)
        chunk=raw.get_data(picks=chans,start=start,stop=stop)
        filt=filtered.get_data(picks=chans,start=start,stop=stop)
        a.add(chunk,start=start,stride=stride)
        af.add(filt,start=start,stride=stride)
        if (ix+1)%max(1,num_chunks//10)==0 or ix+1==num_chunks:
            print(f'[V2 large EEG] Processed {ix+1}/{num_chunks} chunks',flush=True)
    mean,sd,var,skew,kurt=a.result()
    fmean,fsd,fvar,_,_=af.result()
    frequencies=np.fft.rfftfreq(nfft,d=1/fs)
    spectra=a.psd()
    filtered_spectra=af.psd()
    # Count global >5 SD and near-rail values without storing the signal.
    high_count=np.zeros(len(chans),dtype=np.int64)
    rail_count=np.zeros(len(chans),dtype=np.int64)
    transient=np.zeros(len(chans),dtype=np.int64)
    stimulation=np.zeros(len(chans),dtype=np.int64)
    transient_first=[[] for _ in chans]
    for ix,start in enumerate(range(0,raw.n_times,samples_per_chunk)):
        stop=min(raw.n_times,start+samples_per_chunk)
        chunk=raw.get_data(picks=chans,start=start,stop=stop)*1e6
        fil=filtered.get_data(picks=chans,start=start,stop=stop)*1e6
        high_count += np.sum(np.abs(chunk-mean[:,None])>5*sd[:,None],axis=1)
        rail_count += np.sum((chunk==a.low[:,None])|(chunk==a.high[:,None]),axis=1)
        z=np.divide(np.abs(fil-fmean[:,None]),fsd[:,None],out=np.zeros_like(fil),where=fsd[:,None]>0)
        tr=z>=cfg.transient_z
        transient+=np.sum(tr,axis=1)
        stimulation+=np.sum(z>=cfg.stimulation_z,axis=1)
        for i in range(len(chans)):
            remaining=50-len(transient_first[i])
            if remaining>0:transient_first[i].extend(((np.flatnonzero(tr[i])[:remaining]+start)/fs).tolist())
        if (ix+1)%max(1,num_chunks//10)==0 or ix+1==num_chunks:
            print(f'[V2 large EEG] QC pass {ix+1}/{num_chunks} chunks',flush=True)
    q,arts,stats,outliers={},{},{},{}
    grade_codes={'bad':0,'poor':1,'fair':2,'good':3}
    band_names=tuple(BANDS)
    for i,ch in enumerate(chans):
        p=spectra[i];total=float(np.sum(p)*fs/nfft)
        powers={k:band_power(frequencies,p,*ab) for k,ab in BANDS.items()}
        btot=sum(x for x in powers.values() if np.isfinite(x))
        ratios={}
        for name,numer,denom in (('theta_delta_ratio','theta','delta'),('alpha_theta_ratio','alpha','theta'),
                                 ('beta_alpha_ratio','beta','alpha'),('beta_theta_ratio','beta','theta')):
            ratios[name]=_num(powers[numer]/powers[denom]) if powers[denom]>0 else None
        pnorm=p/np.sum(p) if np.sum(p)>0 else np.zeros_like(p)
        positive=pnorm[pnorm>0]
        edge=float(frequencies[min(len(frequencies)-1,np.searchsorted(np.cumsum(p),.95*np.sum(p)))]) if np.sum(p)>0 else None
        alphamask=(frequencies>=8)&(frequencies<13)
        peakalpha=float(frequencies[alphamask][np.argmax(p[alphamask])]) if np.any(alphamask) else None
        sample=np.asarray(a.reservoir[i],dtype=float)
        median=float(np.median(sample)) if len(sample) else float('nan')
        mad=float(np.median(np.abs(sample-median))) if len(sample) else float('nan')
        stats[ch]={'metrics_available':not bool(a.nonfinite[i]),'finite_fraction':1-a.nonfinite[i]/a.n,
          'mean':mean[i]*1e-6,'standard_deviation':sd[i]*1e-6,'variance':var[i]*1e-12,
          'rms':np.sqrt(a.s2[i]/a.n)*1e-6,'peak_to_peak':(a.high[i]-a.low[i])*1e-6,
          'min':a.low[i]*1e-6,'max':a.high[i]*1e-6,'median':_num(median*1e-6),
          'mad':_num(mad*1e-6),'skewness':_num(skew[i]),'kurtosis':_num(kurt[i]),
          'zero_crossings':int(a.zc[i]),'zero_crossing_rate':a.zc[i]/a.n,
          'crest_factor':_num(a.abspeak[i]/np.sqrt(a.s2[i]/a.n)) if a.s2[i]>0 else None,
          'high_amplitude_fraction':high_count[i]/a.n,'total_spectral_power':total,
          'dominant_frequency':float(frequencies[np.argmax(p)]),'peak_alpha_frequency':peakalpha,
          'spectral_entropy':float(-np.sum(positive*np.log2(positive))), 'spectral_edge':edge,
          'total_band_power':btot,'estimation_method':'streamed_chunk_welch',
          'median_mad_method':'systematic_bounded_subsample',
          **{name+'_power':_num(power) for name,power in powers.items()},
          **{name+'_relative':_num(power/btot) if btot else None for name,power in powers.items()},**ratios}
        r=float(np.sqrt(af.s2[i]/af.n))*1e-6
        clip=float(rail_count[i]/a.n)
        proxy=float(10*np.log10(af.s2[i]/af.diff_sq[i])) if af.diff_sq[i]>0 and af.s2[i]>0 else None
        flat=(af.high[i]-af.low[i])*1e-6 <= cfg.flatline_ptp_volts
        clipping=clip>=cfg.clipping_edge_fraction
        gr='bad' if flat or clipping or (proxy is not None and proxy<0) else 'poor' if proxy is not None and proxy<5 else 'fair' if proxy is not None and proxy<10 else 'good'
        if a.nonfinite[i] or af.nonfinite[i]:
            gr='bad'
        q[ch]={'quality':gr,'finite_fraction':1-af.nonfinite[i]/af.n,'mean':fmean[i]*1e-6,'rms':r,
            'standard_deviation':fsd[i]*1e-6,'variance':fvar[i]*1e-12,
            'peak_to_peak':(af.high[i]-af.low[i])*1e-6,
            'minimum':af.low[i]*1e-6,'maximum':af.high[i]*1e-6,
            'snr_difference_proxy_db':proxy,'flatline':bool(flat),
            'possible_clipping':bool(clipping),'clipping_edge_fraction':clip,
            'flags':(['flatline'] if flat else [])+(['possible_clipping'] if clipping else [])}
        mains=cfg.mains_frequency_hz
        target=(frequencies>=mains-1)&(frequencies<=mains+1)
        near=(frequencies>=mains-5)&(frequencies<=mains+5)&(~target)
        if fs/2 < mains+5 or not target.any() or not near.any():
            line={'assessable':False,'reason':'Mains frequency outside reliable spectral range'}
        else:
            background=float(np.median(p[near])); prominence=float(np.max(p[target])/background) if background>0 else None
            line={'assessable':True,'prominence_ratio':prominence,
                  'detected':bool(prominence is not None and prominence>=cfg.line_noise_prominence),
                  'mains_frequency_hz':mains}
        flags=(['line_noise'] if line.get('detected') else [])+(['transient'] if transient[i] else [])
        if stimulation[i]:flags.append('stimulation_pattern_not_cause')
        if flat:flags.append('electrode_problem')
        arts[ch]={'artifact_flags':flags,'artifact_present':bool(flags),
          'transient_count':int(transient[i]),'stimulation_threshold_count':int(stimulation[i]),
          'transient_first_times_s':transient_first[i], 'line_noise':line,
          'electrode_problem':'flatline' if flat else None}
    for name in chans:
        outliers[name]={'outlier':False,'note':'Full-record pairwise outlier audit not run in streaming mode'}
    # Create figures from bounded 10-second windows and accumulated PSD tables.
    ch=chans[0];n=min(raw.n_times,int(cfg.display_seconds*fs))
    t=np.arange(n)/fs
    wave=raw.get_data(picks=[ch],start=0,stop=n)[0]*1e6
    filt_wave=filtered.get_data(picks=[ch],start=0,stop=n)[0]*1e6
    figs={};tables={}
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    axes[0].plot(t,wave,lw=.8);axes[0].set(title=f'Original EEG: {ch}',ylabel='Amplitude (µV)')
    axes[1].plot(t,filt_wave,lw=.8);axes[1].set(title=f'Window-filtered EEG: {ch}',ylabel='Amplitude (µV)',xlabel='Time (seconds)')
    fig.tight_layout();fig.savefig(out/'waveform_raw_vs_filtered.png',dpi=125);plt.close(fig)
    figs['Raw vs. filtered EEG waveform']='waveform_raw_vs_filtered.png'
    tables['Raw vs. filtered EEG waveform']=spec(
        [{'time_s':float(tt),'channel':ch,'raw_uv':float(r),'filtered_uv':float(f)} for tt,r,f in zip(t,wave,filt_wave)],
        'Complete 10-second waveform window; filtered version uses chunk-local SOS filtering.',
        summary={'sample_rate_hz':fs,'window_duration_s':n/fs})
    plotmask=frequencies<=min(50,fs/2)
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(frequencies[plotmask],spectra[0][plotmask],label='Original')
    ax.plot(frequencies[plotmask],filtered_spectra[0][plotmask],label='Filtered')
    ax.set(xlabel='Frequency (Hz)',ylabel='PSD (V²/Hz)',title=f'Chunk-Welch power spectrum: {ch}')
    ax.legend();fig.tight_layout();fig.savefig(out/'power_spectrum.png',dpi=125);plt.close(fig)
    figs['EEG power spectrum']='power_spectrum.png'
    tables['EEG power spectrum']=spec(
        [{'frequency_hz':float(f),'channel':ch,'raw_psd_v2_per_hz':float(p),
          'filtered_psd_v2_per_hz':float(qp)}
          for f,p,qp in zip(frequencies[plotmask],spectra[0][plotmask],filtered_spectra[0][plotmask])],
        'Every displayed Welch spectral bin; unlike the legacy full-length periodogram, this is an estimate across chunks.',
        summary={'frequency_resolution_hz':fs/nfft,'method':'segment-weighted Welch across bounded chunks'})
    names=np.arange(len(chans));width=.20
    fig,ax=plt.subplots(figsize=(max(11,min(20,len(chans)*.38)),5))
    rows=[]
    for i,band in enumerate(band_names):
        ax.bar(names+(i-1.5)*width,[100*(stats[c][band+'_relative'] or 0) for c in chans],width,label=band.title())
    ax.set_xticks(names,labels=chans,rotation=90)
    ax.set(ylabel='Relative power (%)',title='Relative EEG band power by channel');ax.legend()
    fig.tight_layout();fig.savefig(out/'relative_band_power.png',dpi=125);plt.close(fig)
    figs['Relative EEG band power']='relative_band_power.png'
    for c in chans:
        rows.append({'channel':c,**{band+'_relative_pct':100*(stats[c][band+'_relative'] or 0) for band in band_names},
                     **{band+'_absolute_v2':stats[c][band+'_power'] for band in band_names}})
    tables['Relative EEG band power']=spec(rows,'Displayed relative band-power values, including absolute power in V².')
    fig,ax=plt.subplots(figsize=(max(10,min(20,len(chans)*.38)),4))
    ax.bar(chans,[q[c]['rms']*1e6 for c in chans]);ax.tick_params(axis='x',rotation=90)
    ax.set(title='Filtered channel RMS',ylabel='RMS (µV)')
    fig.tight_layout();fig.savefig(out/'channel_rms.png',dpi=125);plt.close(fig)
    figs['Channel RMS']='channel_rms.png'
    tables['Channel RMS']=spec([{'channel':c,'rms_filtered_uv':q[c]['rms']*1e6,'quality_grade':q[c]['quality']} for c in chans],
                               'Filtered channel RMS and screening grades.')
    fig,ax=plt.subplots(figsize=(max(10,min(20,len(chans)*.38)),4))
    ax.bar(chans,[grade_codes[q[c]['quality']] for c in chans]);ax.set_yticks([0,1,2,3],labels=['Bad','Poor','Fair','Good']);ax.tick_params(axis='x',rotation=90)
    ax.set(title='IONM-style channel quality (heuristic)')
    fig.tight_layout();fig.savefig(out/'channel_quality.png',dpi=125);plt.close(fig)
    figs['Channel quality']='channel_quality.png'
    tables['Channel quality']=spec([{'channel':c,'quality_score_0_bad_3_good':grade_codes[q[c]['quality']],
                                   'quality_grade':q[c]['quality'],'rms_filtered_uv':q[c]['rms']*1e6} for c in chans],
                                 'Heuristic screening grades and exact bar heights.')
    # Use the streamed *filtered* moments for the general Hjorth metrics.
    hjorth={}
    for i,ch in enumerate(chans):
        mobility=np.sqrt((af.diff_sq[i]/max(1,af.diff_n))/fvar[i]) if fvar[i]>0 else None
        complexity=(np.sqrt((af.diff2_sq[i]/max(1,af.diff2_n))/(af.diff_sq[i]/max(1,af.diff_n)))/mobility
                    if mobility and af.diff_sq[i]>0 and af.diff2_sq[i]>0 else None)
        hjorth[ch]={'hjorth_activity_v2':fvar[i]*1e-12,
                    'hjorth_mobility':_num(mobility) if mobility is not None else None,
                    'hjorth_complexity':_num(complexity) if complexity is not None else None}
    filtered.hjorth=hjorth
    warnings=[
      *(['Non-finite EEG values detected: channel statistics and PSD estimates may be unreliable.']
        if np.any(a.nonfinite) or np.any(af.nonfinite) else []),
      'Large-data out-of-core mode: legacy full-record periodogram replaced with chunk-wise Welch PSD.',
      'Large-data median/MAD are approximate estimates from a bounded deterministic subsample.',
      'Filtering uses an 8-second context and Butterworth SOS per chunk, not the legacy full-record FIR.',
      'Global IONM event-locked statistics and cross-channel outlier screening are not run in out-of-core mode.',
      'Screening grades and SNR proxy are heuristic; no clinical diagnostic interpretation.',
      'Artifact flags indicate possible issues and do not mean artifacts were removed.',
    ]
    return {'channel_statistics':stats,'ionm_quality':q,'ionm_artifacts':arts,
      'ionm_outliers':outliers,
      'ionm_event_audit':{'source':'streaming_mode','counts':{},'marker_pairs':{},'trial_statistics':{},'tests':[],
                         'screening':{'status':'unavailable','reason':'Global IONM event epoching disabled for memory safety'}},
      'preprocessing':{'raw_stats':'lazy_chunked_EEG',
          'filtered_quality':f'{cfg.high_pass_hz}-{filtered.high:.2f} Hz chunked Butterworth SOS',
          'artifact_removal_performed':False,'line_noise_assessed_on':'unfiltered_chunk_welch_PSD',
          'memory_strategy':'bounded_chunk_streaming', 'chunk_seconds':samples_per_chunk/fs,
          'maximum_raw_chunk_bytes':int(samples_per_chunk*len(chans)*8),
          'psd_method':'segment_weighted_welch', 'median_mad':'bounded_subsample',
          'legacy_ionm_event_comparisons':'not_run_in_streaming_mode'},
      'warnings':warnings,'figures':figs,'figure_tables':tables,
      '_filtered':filtered,'_eeg_channels':chans}
