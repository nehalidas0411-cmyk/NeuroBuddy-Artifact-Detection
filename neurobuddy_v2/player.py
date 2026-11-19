"""Transferable derived-only 2D EEG scalp / channel playback payload."""
from __future__ import annotations
import numpy as np
from scipy import signal
import mne

BANDS = {'delta':(.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30)}


def make_payload(raw, events, *, max_frames=1500, n_channels=48):
    eegs=[ch for ch,kind in zip(raw.ch_names,raw.get_channel_types()) if kind=='eeg']
    # Select scalp positions from provided montage only. No synthetic sensor positions.
    montage=raw.get_montage()
    channel_positions={}
    if montage:
        channel_positions = montage.get_positions().get('ch_pos',{})
    spatial=[]
    for ch in eegs:
        p=channel_positions.get(ch)
        if p is not None and np.all(np.isfinite(p)) and np.linalg.norm(p)>0:
            spatial.append((ch,np.array(p,dtype=float)))
    spatial=spatial[:n_channels]
    coords=np.array([p[:2] for _,p in spatial]) if spatial else np.empty((0,2))
    valid_map=bool(len(spatial)>=8 and np.linalg.matrix_rank(coords-coords.mean(axis=0)) == 2
                   and np.ptp(coords[:,0])>0.03 and np.ptp(coords[:,1])>0.03)
    if valid_map:
        channel_names=[ch for ch,_ in spatial]
        center=np.mean(coords,axis=0)
        coords=coords-center
        radius=np.max(np.sqrt(np.sum(coords**2,axis=1)))
        xy=(coords/radius * .87).round(5).tolist()
    else:
        channel_names=eegs[:min(8,len(eegs))]
        xy=[]
    sf=float(raw.info['sfreq']);dur=float(raw.n_times/sf)
    frames=max(2,min(max_frames,int(np.ceil(dur*2))))
    times=np.linspace(0,max(0,dur-.001),frames)
    channels=len(channel_names)
    modes={'voltage':[],'delta':[],'theta':[],'alpha':[],'beta':[]}
    # Spectral frames summarize centered 2s windows; voltage is 100-ms median.
    for t in times:
        center=int(round(t*sf));a=max(0,center-int(sf));b=min(raw.n_times,center+int(sf))
        if b-a<max(8,int(.3*sf)):
            a=max(0,b-max(8,int(.3*sf)))
        data=raw.get_data(picks=channel_names,start=a,stop=b)
        ci=max(0,min(data.shape[1]-1,center-a))
        lo=max(0,ci-int(.05*sf));hi=min(data.shape[1],ci+int(.05*sf)+1)
        volts=np.median(data[:,lo:hi],axis=1)*1e6
        modes['voltage'].append(np.around(volts,3).tolist())
        if data.shape[1]<3:
            for k in BANDS:modes[k].append([None]*channels)
            continue
        f,psd=signal.welch(data,fs=sf,axis=1,nperseg=min(data.shape[1],int(2*sf)))
        step=f[1]-f[0]
        total=np.sum(psd[:,(f>=.5)&(f<=min(45,sf/2))],axis=1)*step
        for name,(start,end) in BANDS.items():
            mask=(f>=start)&(f<end)
            values = (np.sum(psd[:,mask],axis=1)*step/np.maximum(total,1e-24)*100) if np.any(mask) else np.zeros(channels)
            modes[name].append(np.around(values,3).tolist())
    ranges={}
    for name,arr in modes.items():
        vals=np.array(arr,dtype=float)
        finite=vals[np.isfinite(vals)]
        if len(finite)==0: ranges[name]=[0,1]
        elif name=='voltage':
            peak=max(1.,float(np.percentile(np.abs(finite),97)))
            ranges[name]=[-round(peak,3),round(peak,3)]
        else:
            ranges[name]=[0,round(max(1.,float(np.percentile(finite,98))),2)]
    timeline=[{'t':float(round(e.onset_s,3)),'label':e.semantic_label,'original':e.raw_value,
               'source':e.source} for e in events if e.alignment=='verified_within_recording' and e.onset_s>=0]
    return {'schema_version':'1.0.0','duration_s':dur,'sampled_points':frames,
      'time_s':times.round(3).tolist(), 'channels':channel_names,'geometry_xy':xy,
      'topomap_available':valid_map,
      'mode_units':{'voltage':'µV','delta':'% relative bandpower','theta':'% relative bandpower',
                    'alpha':'% relative bandpower','beta':'% relative bandpower'},
      'ranges':ranges,'modes':modes,'events':timeline,
      'notes':['Derived EEG sensor values, not 3D source localization or literal brain activation.',
               'Power values use 2-second Welch windows and are % of 0.5–45 Hz PSD power.',
               'Scalp geometry is based only on the provided montage.',
               'Playback is sampled in time and does not embed continuous raw EEG.']}


PLAYER_HTML = r'''
<section class="nb-player" aria-label="Interactive EEG activity player">
  <header><h2>Brain Activity Player</h2><p class="muted">Recorded scalp EEG. Colors display sensor values or interpolated scalp distribution, not internal brain source activity.</p></header>
  <div class="nb-player-grid">
    <div class="nb-map-wrap"><canvas id="nb-map" width="420" height="420" aria-label="EEG scalp activity map"></canvas><p id="nb-channel-note" class="muted"></p></div>
    <div class="nb-controls">
      <label for="nb-mode">View mode</label><select id="nb-mode"><option value="voltage">Voltage (µV)</option><option value="delta">Delta (0.5-4 Hz)</option><option value="theta">Theta (4-8 Hz)</option><option value="alpha">Alpha (8-13 Hz)</option><option value="beta">Beta (13-30 Hz)</option></select>
      <p><strong id="nb-clock">00:00</strong> / <span id="nb-total"></span></p>
      <input type="range" id="nb-seek" min="0" value="0" step="1" aria-label="Seek EEG recording"/>
      <div class="nb-buttons"><button type="button" id="nb-prev">◀ Event</button><button type="button" id="nb-play">▶ Play</button><button type="button" id="nb-next">Event ▶</button><label>Speed <select id="nb-speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="5">5×</option></select></label></div>
      <div class="nb-metadata"><strong id="nb-legend"></strong><p id="nb-current-channel" class="muted"></p></div>
      <p class="muted">Select electrode: <select id="nb-electrode"></select></p>
      <canvas id="nb-trace" width="580" height="135" aria-label="Selected channel EEG feature trace"></canvas>
      <p id="nb-event" class="muted"></p>
    </div>
  </div>
</section>
<script id="nb-player-data" type="application/json">__PLAYER_JSON__</script>
<script>
(()=>{'use strict';
const json = document.getElementById('nb-player-data');if(!json)return;
const d=JSON.parse(json.textContent), $=x=>document.getElementById('nb-'+x);
const map=$('map'),ctx=map.getContext('2d'),trace=$('trace'),tc=trace.getContext('2d');
let fi=0, mode='voltage', chosen=0,playing=false,last=null,frameTime=0,speed=1;
const n=d.time_s.length, W=132, center=66, rad=56;const scale=map.width/W;
let pts=[];if(d.topomap_available)pts=d.geometry_xy.map(([x,y])=>[(x+1)*rad+10,(1-y)*rad+10]);
const weights=[];if(d.topomap_available)for(let y=0;y<W;y++)for(let x=0;x<W;x++){
 const dist=(x-center)**2+(y-center)**2;if(dist>rad**2){weights.push(null);continue;}
 const list=pts.map(([a,b],i)=>[i,(x-a)**2+(y-b)**2]).sort((a,b)=>a[1]-b[1]).slice(0,5);
 const inv=list.map(z=>1/(z[1]+.25)),s=inv.reduce((a,b)=>a+b,0);weights.push([list.map(z=>z[0]),inv.map(z=>z/s)]);
}
function pct(v){return Math.max(0,Math.min(1,v))}
function color(v){const [low,high]=d.ranges[mode];if(mode==='voltage'){
 const a=pct((v-low)/Math.max(.000001,high-low));return [Math.round(26+229*a),Math.round(70+130*(1-Math.abs(2*a-1))),Math.round(221-200*a)];}
 const a=pct(v/Math.max(1,high));return [Math.round(16+235*a),Math.round(49+125*a),Math.round(111-75*a)];}
function time(t){let h=Math.floor(t/3600),m=Math.floor(t/60)%60,s=Math.floor(t)%60;
 return (h>0?String(h).padStart(2,'0')+':':'')+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');}
const sel=$('electrode');d.channels.forEach((ch,i)=>{let o=document.createElement('option');o.value=i;o.textContent=ch;sel.appendChild(o)});
sel.addEventListener('change',e=>{chosen=+e.target.value;draw()});
const off=document.createElement('canvas');off.width=W;off.height=W;const oc=off.getContext('2d');
function renderHead(frame){ctx.clearRect(0,0,map.width,map.height);
 if(!d.topomap_available){ctx.fillStyle='#b7cce0';ctx.font='17px sans-serif';ctx.fillText('Channel-level view',95,55);
 d.channels.forEach((ch,i)=>{let y=78+i*26;if(y<map.height-4){ctx.fillStyle='#b7cce0';ctx.fillText(ch,34,y);ctx.fillStyle='rgb('+color(frame[i]).join(',')+')';ctx.fillRect(120,y-15,250,16)}});return;}
 let pixels=oc.createImageData(W,W);for(let z=0;z<weights.length;z++){const a=weights[z],o=z*4;if(!a){pixels.data[o+3]=0;continue;}
 let v=0;for(let q=0;q<a[0].length;q++)v+=(frame[a[0][q]]||0)*a[1][q];let rgb=color(v);pixels.data[o]=rgb[0];pixels.data[o+1]=rgb[1];pixels.data[o+2]=rgb[2];pixels.data[o+3]=255;}
 oc.putImageData(pixels,0,0);ctx.drawImage(off,0,0,map.width,map.height);
 ctx.strokeStyle='#24455f';ctx.lineWidth=2;ctx.beginPath();ctx.arc(center*scale,center*scale,rad*scale,0,Math.PI*2);ctx.stroke();
 ctx.beginPath();ctx.moveTo(202,35);ctx.lineTo(210,15);ctx.lineTo(218,35);ctx.stroke();
 pts.forEach(([x,y],i)=>{ctx.beginPath();ctx.fillStyle=i===chosen?'#ffffff':'#152b40';ctx.arc(x*scale,y*scale,i===chosen?4:2,0,Math.PI*2);ctx.fill()});
}
function drawTrace(){const ctx=tc,w=trace.width,h=trace.height;
 ctx.clearRect(0,0,w,h);const values=d.modes[mode].map(v=>v[chosen]);const finite=values.filter(Number.isFinite);
 if(!finite.length)return;const lo=Math.min(...finite),hi=Math.max(...finite);ctx.strokeStyle='#45aecb';ctx.lineWidth=1.5;ctx.beginPath();
 for(let i=0;i<n;i++){let x=5+(w-10)*i/(n-1),y=h-8-(h-16)*pct((values[i]-lo)/(hi-lo||1));if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
 ctx.strokeStyle='#f59e0b';ctx.beginPath();let x=5+(w-10)*fi/(n-1);ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
function draw(){const data=d.modes[mode][fi];renderHead(data);drawTrace();$('clock').textContent=time(d.time_s[fi]);$('seek').value=fi;
 $('legend').textContent=mode.toUpperCase()+' · '+d.mode_units[mode]+' · scale '+d.ranges[mode][0]+' to '+d.ranges[mode][1];
 $('current-channel').textContent=d.channels[chosen]+': '+(Number.isFinite(data[chosen])?data[chosen].toFixed(2):'unavailable')+' '+d.mode_units[mode];
 let nearby=d.events.filter(e=>Math.abs(e.t-d.time_s[fi]) <= Math.max(1, (d.duration_s/n)*.7));$('event').textContent=nearby.length?'Near event: '+nearby.slice(0,4).map(e=>e.label).join(', '):'No nearby annotated event';}
function set(i){fi=Math.max(0,Math.min(n-1,Math.round(i)));frameTime=d.time_s[fi];draw()}
function tick(t){if(!playing)return;if(last!=null){frameTime+=(t-last)/1000*speed;if(frameTime>=d.duration_s){playing=false;$('play').textContent='▶ Play';return;}while(fi+1<n&&d.time_s[fi+1]<=frameTime)fi++;draw();}last=t;requestAnimationFrame(tick)}
$('seek').max=n-1;$('total').textContent=time(d.duration_s);$('seek').addEventListener('input',e=>{set(+e.target.value);last=null});
$('mode').addEventListener('change',e=>{mode=e.target.value;draw()});$('speed').addEventListener('change',e=>{speed=+e.target.value});
$('play').addEventListener('click',()=>{playing=!playing;last=null;if(fi==n-1)set(0);$('play').textContent=playing?'Ⅱ Pause':'▶ Play';if(playing)requestAnimationFrame(tick)});
function jump(sign){let t=d.time_s[fi],arr=d.events.map(e=>e.t).filter(v=>sign>0?v>t+.1:v<t-.1);if(arr.length){let to=sign>0?Math.min(...arr):Math.max(...arr);let i=0;while(i<n-1&&d.time_s[i]<to)i++;set(i)}}
$('prev').addEventListener('click',()=>jump(-1));$('next').addEventListener('click',()=>jump(1));
map.addEventListener('click',e=>{if(!d.topomap_available)return;let box=map.getBoundingClientRect();let x=(e.clientX-box.left)/box.width*W,y=(e.clientY-box.top)/box.height*W;let best=pts.map(([a,b],i)=>[i,(a-x)**2+(b-y)**2]).sort((a,b)=>a[1]-b[1])[0];if(best&&best[1]<250){chosen=best[0];sel.value=chosen;draw()}});
set(0);})();
</script>
'''
