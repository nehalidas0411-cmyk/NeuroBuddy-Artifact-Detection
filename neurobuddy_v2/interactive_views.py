"""Additive, offline-capable athlete visualizations. No EEG statistics are recomputed."""
from __future__ import annotations
import html
import json
import math
import re


def _json(obj):
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')


def sleep_replay(report: dict, element_id: str = 'nb-sleep-replay', *, sync_player: bool = False) -> str:
    """Render an independent, self-contained sleep-stage replay for a single EEG report."""
    if report.get('classification', {}).get('task_family') != 'SLEEP':
        return ''
    sleep = report.get('specialized_metrics', {}).get('sleep', {})
    intervals = sleep.get('stage_intervals') or []
    if not intervals:
        return '<section><h2>Interactive Sleep Replay</h2><p>Unavailable: no verified or estimated sleep-stage intervals.</p></section>'
    safe = re.sub('[^a-zA-Z0-9_-]', '_', element_id)
    origin = sleep.get('stage_origin', 'unscored')
    payload = {'intervals': intervals, 'origin': origin, 'duration_s': report.get('recording', {}).get('duration_s'),
               'spindles': [{'onset_s': e.get('onset_s', e.get('start_s')), 'channel': e.get('channel')}
                            for e in sleep.get('spindle_events', []) if isinstance(e, dict)],
               'slow_waves': [{'onset_s': e.get('onset_s', e.get('start_s')), 'channel': e.get('channel')}
                              for e in sleep.get('slow_wave_events', []) if isinstance(e, dict)]}
    return f'''<section id="{safe}" class="nb-sleep-replay" style="background:#102c3e;border:1px solid #34586c;border-radius:12px;padding:18px;margin:20px 0;color:#e4eff6">
<h2>Interactive Sleep Replay</h2>
<p style="color:#a9c4d2">{html.escape(origin.replace('_',' '))}. Playback shows scored or estimated sleep stages, not reconstructed raw EEG. Estimated stages are unreviewed.</p>
<canvas data-role="plot" width="1050" height="240" style="width:100%;height:auto;cursor:crosshair;background:#091d2c;border-radius:8px" aria-label="Interactive sleep hypnogram"></canvas>
<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:12px">
<button type="button" data-role="play">▶ Play</button>
<label>Speed <select data-role="speed"><option value="1">1×</option><option value="10">10×</option><option value="60" selected>60×</option><option value="300">300×</option></select></label>
<label>Jump to stage <select data-role="stage"><option value="">Select stage</option><option>W</option><option>N1</option><option>N2</option><option>N3</option><option>REM</option></select></label>
<strong data-role="clock">00:00:00</strong><strong data-role="current">Stage: —</strong>
</div>
<input type="range" data-role="seek" min="0" max="1000" step="1" value="0" style="width:100%;margin:14px 0" aria-label="Seek sleep recording">
<p data-role="detail" style="color:#a9c4d2"></p>
<script type="application/json" data-role="data">{_json(payload)}</script>
<script>(function(){{'use strict';
const root=document.getElementById({_json(safe)}), get=k=>root.querySelector('[data-role="'+k+'"]');
const d=JSON.parse(get('data').textContent), cv=get('plot'),ctx=cv.getContext('2d');
const rows=d.intervals.map(r=>({{start:Number(r.start_s),end:Number(r.end_s),stage:r.stage}})).filter(r=>Number.isFinite(r.start)&&Number.isFinite(r.end)&&r.end>r.start).sort((a,b)=>a.start-b.start);
if(!rows.length)return;
const duration=Math.max(Number(d.duration_s)||0,...rows.map(r=>r.end)), colors={{W:'#a4b6c3',N1:'#76c7eb',N2:'#378bd2',N3:'#294b9b',REM:'#f27b9d'}}, ys={{W:35,REM:70,N1:105,N2:140,N3:175}};
let time=0, playing=false,last=null,speed=60;
function fmt(t){{const s=Math.floor(t);return [Math.floor(s/3600),Math.floor(s/60)%60,s%60].map(x=>String(x).padStart(2,'0')).join(':')}}
function stageAt(t){{return rows.find(r=>r.start<=t&&t<r.end)}}
function x(t){{return 56+(cv.width-72)*t/duration}}
function draw(){{ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle='#091d2c';ctx.fillRect(0,0,cv.width,cv.height);
ctx.font='14px system-ui';Object.entries(ys).forEach(([s,y])=>{{ctx.fillStyle='#a9c4d2';ctx.fillText(s,9,y+5);ctx.strokeStyle='#23445b';ctx.beginPath();ctx.moveTo(55,y);ctx.lineTo(cv.width-16,y);ctx.stroke()}});
rows.forEach(r=>{{ctx.strokeStyle=colors[r.stage]||'#ddd';ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(x(r.start),ys[r.stage]);ctx.lineTo(x(r.end),ys[r.stage]);ctx.stroke()}});
ctx.lineWidth=1;ctx.fillStyle='#a9c4d2';for(let i=0;i<=6;i++){{const t=duration*i/6;ctx.fillText(fmt(t),Math.min(cv.width-90,x(t)-23),224)}}
ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(time),15);ctx.lineTo(x(time),200);ctx.stroke();
get('clock').textContent=fmt(time)+' / '+fmt(duration);const r=stageAt(time);get('current').textContent='Stage: '+(r?r.stage:'Unscored');get('seek').value=Math.round(1000*time/duration);
const count=(arr)=>arr.filter(e=>Number.isFinite(Number(e.onset_s))&&Math.abs(Number(e.onset_s)-time)<=15).length;
get('detail').textContent=(r?'Current stage interval: '+fmt(r.start)+' to '+fmt(r.end):'Unscored interval')+' · Spindle candidates within ±15 s: '+count(d.spindles)+' · Slow-wave candidates within ±15 s: '+count(d.slow_waves);
}}
function seek(t){{time=Math.max(0,Math.min(duration,t));draw();{'''if(sync_player){const p=document.getElementById('nb-seek');const data=document.getElementById('nb-player-data');if(p&&data){try{const obj=JSON.parse(data.textContent),times=obj.time_s;let lo=0,hi=times.length-1;while(lo<hi){const m=(lo+hi)>>1;if(times[m]<time)lo=m+1;else hi=m;}p.value=lo;p.dispatchEvent(new Event('input',{bubbles:true}));}catch(e){}}}''' if sync_player else ''}}}
function tick(t){{if(!playing)return;if(last!==null)time+=(t-last)/1000*speed;last=t;if(time>=duration){{time=duration;playing=false;get('play').textContent='▶ Play'}}seek(time);if(playing)requestAnimationFrame(tick)}}
get('play').addEventListener('click',()=>{{playing=!playing;last=null;if(playing&&time>=duration)time=0;get('play').textContent=playing?'Ⅱ Pause':'▶ Play';if(playing)requestAnimationFrame(tick)}});
get('speed').addEventListener('change',e=>speed=Number(e.target.value));get('seek').addEventListener('input',e=>{{seek(duration*Number(e.target.value)/1000);last=null}});
get('stage').addEventListener('change',e=>{{const r=rows.find(r=>r.stage===e.target.value&&r.start>time+.01)||rows.find(r=>r.stage===e.target.value);if(r)seek(r.start)}});
cv.addEventListener('click',e=>{{const box=cv.getBoundingClientRect();const px=(e.clientX-box.left)/box.width*cv.width;seek((px-56)/(cv.width-72)*duration);last=null}});draw();
}})();</script></section>'''


def heatmap_html(comparison: dict, element_id: str = 'nb-compare-heatmap') -> str:
    """Interactive channel × statistic heatmap; derived solely from existing comparison rows."""
    rows=[]
    for r in comparison.get('metrics', []):
        name=r.get('metric', '')
        match=re.match(r'^shared\.(?:channel|quality)\.([^.]+)\.(.+)$', name)
        if not match: continue
        channel, metric=match.groups()
        rows.append({'channel':channel,'metric':metric,'path':name,'z':r.get('z_score'),
                     'status':r.get('status'),'mean':r.get('baseline_mean'), 'sd':r.get('baseline_sd'),
                     'test':r.get('test_value'),'unit':r.get('unit'),'reason':r.get('reason')})
    if not rows:
        return '<section><h2>Baseline vs. Test Brain Heatmap</h2><p>No comparable channel-level EEG metrics available.</p></section>'
    safe=re.sub('[^a-zA-Z0-9_-]','_',element_id)
    return f'''<section id="{safe}" style="background:#102c3e;border:1px solid #34586c;border-radius:12px;padding:18px;margin:18px 0">
<h2>Baseline vs. Test Brain Heatmap</h2>
<p class="muted">Each cell is one electrode/channel and EEG statistic. Blue = below baseline, red = above baseline, gray = unavailable. Darker color means larger |z|. Abnormal when |z| &gt; {_json(comparison.get('z_threshold',3))}; color alone is not a diagnosis.</p>
<label>Metric group <select data-role="group"><option value="">All metrics</option></select></label>
<div style="overflow:auto;max-width:100%"><canvas data-role="plot" width="1050" height="440" style="max-width:none;cursor:crosshair" aria-label="Interactive baseline versus test channel metric heatmap"></canvas></div>
<div data-role="detail" style="margin-top:12px;padding:12px;border:1px solid #34586c;border-radius:8px">Click a heatmap cell for baseline mean, SD, test value, z-score and status.</div>
<script type="application/json" data-role="data">{_json(rows)}</script>
<script>(function(){{'use strict';const root=document.getElementById({_json(safe)}), get=k=>root.querySelector('[data-role="'+k+'"]');
const rows=JSON.parse(get('data').textContent),cv=get('plot'),ctx=cv.getContext('2d'),group=get('group');
const metrics=[...new Set(rows.map(r=>r.metric))].sort(),channels=[...new Set(rows.map(r=>r.channel))].sort();
const groups=[...new Set(metrics.map(m=>m.split('.')[0]))].sort();groups.forEach(g=>{{const opt=document.createElement('option');opt.value=g;opt.textContent=g;group.appendChild(opt)}});
let shown=[],x0=130,y0=130,cw=50,ch=30;
function color(z){{if(z===null||!Number.isFinite(z))return '#536675';const p=Math.min(1,Math.abs(z)/5);return z<0?`rgb(${{Math.round(45-12*p)}},${{Math.round(115-55*p)}},${{Math.round(175+65*p)}})`:`rgb(${{Math.round(90+155*p)}},${{Math.round(120-65*p)}},${{Math.round(115-65*p)}})`}}
function fmt(v){{return typeof v==='number'&&Number.isFinite(v)?v.toPrecision(6):'N/A'}}
function draw(){{shown=metrics.filter(m=>!group.value||m.split('.')[0]===group.value);const W=Math.max(550,x0+shown.length*cw+25),H=Math.max(280,y0+channels.length*ch+45);cv.width=W;cv.height=H;
ctx.fillStyle='#091d2c';ctx.fillRect(0,0,W,H);ctx.font='12px system-ui';ctx.textBaseline='middle';
shown.forEach((m,i)=>{{ctx.save();ctx.translate(x0+i*cw+cw/2,y0-8);ctx.rotate(-Math.PI/3);ctx.fillStyle='#bed4e0';ctx.fillText(m,0,0);ctx.restore()}});
channels.forEach((c,j)=>{{ctx.fillStyle='#bed4e0';ctx.fillText(c,10,y0+j*ch+ch/2);shown.forEach((m,i)=>{{const r=rows.find(r=>r.channel===c&&r.metric===m);ctx.fillStyle=r?color(r.z):'#364956';ctx.fillRect(x0+i*cw+1,y0+j*ch+1,cw-2,ch-2);if(r&&r.status==='Abnormal'){{ctx.strokeStyle='#f8e5a3';ctx.lineWidth=2;ctx.strokeRect(x0+i*cw+3,y0+j*ch+3,cw-6,ch-6)}}}})}});
ctx.fillStyle='#a9c4d2';ctx.fillText('Outlined = ABNORMAL  ·  gray = not comparable / missing',x0,H-16);
}}
cv.addEventListener('click',e=>{{const rect=cv.getBoundingClientRect(),px=(e.clientX-rect.left)/rect.width*cv.width,py=(e.clientY-rect.top)/rect.height*cv.height;const i=Math.floor((px-x0)/cw),j=Math.floor((py-y0)/ch);if(i<0||j<0||i>=shown.length||j>=channels.length)return;const r=rows.find(r=>r.channel===channels[j]&&r.metric===shown[i]);const d=get('detail');d.replaceChildren();if(!r){{d.textContent='Metric unavailable for this channel';return}};
const title=document.createElement('strong');title.textContent=r.channel+' · '+r.metric;d.appendChild(title);
const table=document.createElement('table');[['Baseline mean',r.mean],['Baseline SD',r.sd],['Test value',r.test],['Z-score',r.z],['Status',r.status],['Unit',r.unit],['Explanation',r.reason]].forEach(([k,v])=>{{const tr=document.createElement('tr'),th=document.createElement('th'),td=document.createElement('td');th.textContent=k;td.textContent=typeof v==='number'?fmt(v):String(v??'N/A');tr.append(th,td);table.appendChild(tr)}});d.appendChild(table);
}});group.addEventListener('change',draw);draw();}})();</script></section>'''
