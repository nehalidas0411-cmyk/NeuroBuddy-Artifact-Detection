"""Schema-checked JSON, self-contained HTML, derived player payload and portable ZIP."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import base64
import hashlib
import html
import json
import csv
import re
import zipfile
import shutil

from . import __version__
from .legacy_combined import safe_json
from .player import PLAYER_HTML
from .interactive_views import sleep_replay

STYLES = '''
:root{font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#e4eff6;background:#081c2a}
*{box-sizing:border-box}body{margin:0}main{max-width:1250px;margin:auto;padding:28px 20px}h1{font-size:30px;line-height:1.2}h2{font-size:20px}h3{font-size:16px}
section{background:#102c3e;border:1px solid #24465a;border-radius:13px;padding:22px;margin:22px 0}.muted{color:#a7c0cc}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.stat{padding:14px;background:#17394c;border-radius:9px}
small,code{color:#aac4d1}code{word-break:break-word}a{color:#79d2ef}img{width:100%;max-width:960px;border-radius:6px}
.scroll{max-height:500px;overflow:auto}
.graph-data{margin-top:18px;padding:15px;border:1px solid #34586c;border-radius:8px;background:#0b2333}.graph-data h3{margin:0 0 9px}.graph-data .scroll{max-height:380px}.graph-data table{width:max-content;min-width:100%}.graph-page{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:10px 0}.graph-page button{padding:6px 12px;border:1px solid #547b91;border-radius:6px;background:#173b50;color:#e4eff6;cursor:pointer}.graph-page button:disabled{opacity:.45;cursor:default}.graph-summary{margin:8px 0 14px}.graph-summary table{width:max-content;min-width:100%;white-space:normal}.graph-download{font-size:13px;margin:8px 0 12px}
.specialized-family{margin-top:18px}.specialized-group{margin-top:16px;padding:14px;border:1px solid #34586c;border-radius:8px;background:#0b2333}
.specialized-group h3{margin:0 0 10px}.specialized-group table{white-space:normal;min-width:100%}.specialized-group .scroll{max-height:500px}
.specialized-group td:first-child{overflow-wrap:anywhere;max-width:650px}.specialized-group td:nth-child(2){font-variant-numeric:tabular-nums}
.specialized-group td{vertical-align:top}.specialized-group th:nth-child(2){min-width:120px}table{border-collapse:collapse;font-size:12px;min-width:100%;white-space:nowrap}td,th{text-align:left;border-bottom:1px solid #305267;padding:8px}th{position:sticky;top:0;background:#15384b}
.nb-player-grid{display:grid;grid-template-columns:minmax(250px,1fr) minmax(320px,1.25fr);gap:30px;align-items:start}
.nb-map-wrap{display:flex;flex-direction:column;align-items:center}#nb-map{width:100%;max-width:420px;aspect-ratio:1}#nb-trace{width:100%;height:135px}
.nb-controls{display:flex;flex-direction:column;gap:10px}.nb-controls input[type=range]{width:100%;accent-color:#35bee2}
.nb-controls select,.nb-controls button{background:#1a4960;color:#e1f1f6;border:1px solid #47758c;border-radius:6px;padding:8px}
.nb-buttons{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.nb-metadata{background:#14354a;border-radius:8px;padding:10px 15px}
@media(max-width:740px){.nb-player-grid{grid-template-columns:1fr}main{padding:18px 12px}}
'''


def _esc(v):
    return html.escape(str(v), quote=True)


def _table(items, fields=None, limit=200):
    if not items: return '<p class="muted">No results available.</p>'
    fields = fields or list(items[0])
    rows=''.join('<tr>'+''.join(f'<td>{_esc(item.get(k,""))}</td>' for k in fields)+'</tr>' for item in items[:limit])
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{_esc(k)}</th>' for k in fields)+'</tr></thead><tbody>'+rows+'</tbody></table></div>'



GRAPH_PAGE_SIZE = 75


def _columns(rows):
    """Stable union across rows, so no variable or statistical field is silently lost."""
    names=[]
    for r in rows:
        for k in r:
            if k not in names: names.append(k)
    return names


def _csv_value(v):
    if v is None: return ''
    if isinstance(v,(dict,list,tuple)): return json.dumps(safe_json(v),ensure_ascii=False,allow_nan=False)
    if isinstance(v,bool): return str(v).lower()
    if isinstance(v,float): return format(v,'.17g')
    return str(v)


def _write_table_csv(out, filename, rows, columns):
    with (out/filename).open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(columns)
        for r in rows:w.writerow([_csv_value(r.get(c)) for c in columns])


def _safe_name(text):
    return re.sub(r'[^a-z0-9]+','_',text.lower()).strip('_')[:55]


def _summary_rows(items, prefix=''):
    rows=[]
    for k,v in items.items():
        path=f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v,dict):
            rows.extend(_summary_rows(v,path))
        else:
            rows.append({'statistic':path,'value':_display_value(v)})
    return rows


def _display_value(v):
    if isinstance(v,(dict,list,tuple)): return json.dumps(v,ensure_ascii=False)
    if v is None: return 'N/A'
    if isinstance(v,bool): return 'Yes' if v else 'No'
    if isinstance(v,float): return f'{v:.12g}'
    return str(v)


def _paged_graph_table(table_id, title, rows, description, csv_name, label):
    fields=_columns(rows)
    if not fields:
        return '<p class="muted">No numerical data available for this graph.</p>'
    initial=''.join('<tr>'+''.join(f'<td>{_esc(_display_value(r.get(c)))}</td>' for c in fields)+'</tr>' for r in rows[:GRAPH_PAGE_SIZE])
    # All plotted rows stay embedded in the offline HTML. Client-side paging
    # avoids constructing thousands of table DOM nodes at once.
    matrix=[[r.get(c) for c in fields] for r in rows]
    inline=json.dumps(safe_json(matrix),separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('<', chr(92)+'u003c')
    head=''.join(f'<th scope="col">{_esc(c)}</th>' for c in fields)
    n=len(rows)
    count=f'Showing 1–{min(n,GRAPH_PAGE_SIZE):,} of {n:,} rows' if n else '0 rows'
    return f"""<div class="graph-data" id="{_esc(table_id)}" data-page="0" data-total="{n}">
    <h3>{_esc(title)}</h3>
    <p class="muted">{_esc(description)}</p>
    <p class="graph-download"><a href="{_esc(csv_name)}" download="{_esc(csv_name)}">Download every value as CSV</a> · {n:,} rows · {len(fields)} columns · also embedded for offline viewing</p>
    <div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{initial}</tbody></table></div>
    <div class="graph-page"><button type="button" onclick="nbGraphPage('{table_id}',-1)" disabled>Previous</button>
    <span class="graph-count">{count}</span>
    <button type="button" onclick="nbGraphPage('{table_id}',1)" {'disabled' if n<=GRAPH_PAGE_SIZE else ''}>Next</button></div>
    <script type="application/json" id="{table_id}-rows">{inline}</script></div>"""


def _graph_markup(out, title, filename, data, graph_index):
    p=out/filename
    if not p.is_file(): return ''
    encoded=base64.b64encode(p.read_bytes()).decode('ascii')
    if not data:
        # Every chart must have a visible explanation even if old releases do
        # not supply numerical graph backing yet.
        return f'<section><h2>{_esc(title)}</h2><img src="data:image/png;base64,{encoded}" alt="{_esc(title)}"><p class="muted">Graph data are unavailable in this report.</p></section>'
    rows=data.get('rows',[])
    summary=data.get('summary') or {}
    section=f'<section><h2>{_esc(title)}</h2><img src="data:image/png;base64,{encoded}" alt="{_esc(title)}">'
    if summary:
        summary_rows=_summary_rows(summary)
        section+='<div class="graph-data graph-summary"><h3>Summary statistics and test results</h3>'+_table(summary_rows,limit=None)+'</div>'
    slug=Path(filename).stem
    primary_csv=f'{slug}_data.csv'
    section+=_paged_graph_table(f'nbgraph{graph_index}', 'Complete graph data table',rows,
                                data.get('description','Underlying values of the graph.'),primary_csv,title)
    for j,(extra_title, extra_rows) in enumerate(data.get('extra_tables',{}).items()):
        fname=f'{slug}_{_safe_name(extra_title)}_data.csv'
        section+=_paged_graph_table(f'nbgraph{graph_index}x{j}',extra_title,extra_rows,
                                    'Supplementary individual measurements that support this graph.',fname,title)
    return section+'</section>'



def _pretty_metric(name):
    """Human-readable presentation only; original metric keys stay in report.json."""
    parts=str(name).replace('-', '_').split('_')
    names={'erp':'ERP','erd':'ERD','ers':'ERS','eeg':'EEG','ionm':'IONM',
           'psd':'PSD','rms':'RMS','n200':'N200','p300':'P300','rt':'RT',
           'uv':'µV','hz':'Hz','v2':'V²','s':'s','ms':'ms',
           'c3':'C3','c4':'C4','cz':'Cz','pz':'Pz','d':'d',
           'loglinear':'log-linear','pct':'%','pre':'pre','post':'post'}
    return ' '.join(names.get(w.lower(),w.capitalize()) for w in parts)


def _metric_unit(name):
    key=str(name).lower()
    if key.endswith('_pct'):return '%'
    if key.endswith('_uv'):return 'µV'
    if key.endswith('_ms'):return 'ms'
    if key.endswith('_hz'):return 'Hz'
    if key.endswith('_v2'):return 'V²'
    if key.endswith('_s'):return 's'
    if key.endswith('_relative') or key.endswith('sleep_efficiency'):return 'fraction'
    return ''


def _special_value(value):
    if isinstance(value,(list,tuple)):
        if not value:return 'None supplied'
        return ' to '.join(_special_value(v) for v in value)
    if value is None:return 'Not available'
    return _display_value(value)


def _special_rows(mapping, prefix=''):
    """Flatten nested statistical tests to one readable row per numerical field."""
    result=[]
    for key,value in mapping.items():
        label=' / '.join(x for x in (prefix, _pretty_metric(key)) if x)
        if isinstance(value,dict):
            result.extend(_special_rows(value,label))
        elif isinstance(value,list) and value and isinstance(value[0],dict):
            # Trial-level records are separately rendered as a full table.
            continue
        else:
            result.append({'Statistic':label,
                           'Value':_special_value(value),
                           'Unit':_metric_unit(key)})
    return result


def _specialized_table(rows):
    return _table(rows, fields=['Statistic','Value','Unit'], limit=None)


def _specialized_markup(out:Path,metrics:dict):
    """Render numerical specialized metrics like the existing report tables.

    The JSON contract and statistics remain unchanged. Array-of-dict trial data
    are paginated for readability, embedded for offline use, and saved to CSV.
    """
    parts=['<section id="specialized-results"><h2>Specialized pathway results</h2>',
           '<p class="muted">Calculated results presented as tables. All original metric keys and full precision values remain available in report.json.</p>']
    if not metrics:
        parts.append('<p class="muted">No specialized results available for this recording.</p></section>')
        return ''.join(parts)
    table_index=0
    for family,data in metrics.items():
        parts.append(f'<div class="specialized-family"><h3>{_esc(_pretty_metric(family))} analysis</h3>')
        if not isinstance(data,dict):
            parts.append(_specialized_table([{'Statistic':_pretty_metric(family),'Value':_special_value(data),'Unit':''}]))
            parts.append('</div>')
            continue
        scalars={k:v for k,v in data.items() if not isinstance(v,dict) and not (isinstance(v,list) and v and isinstance(v[0],dict))}
        if scalars:
            parts.append('<div class="specialized-group"><h3>Summary and processing details</h3>')
            parts.append(_specialized_table(_special_rows(scalars)))
            parts.append('</div>')
        for group,values in data.items():
            if isinstance(values,dict):
                parts.append(f'<div class="specialized-group"><h3>{_esc(_pretty_metric(group))}</h3>')
                if values:
                    parts.append(_specialized_table(_special_rows(values)))
                else:
                    parts.append('<p class="muted">No measurements available.</p>')
                parts.append('</div>')
            elif isinstance(values,list) and values and all(isinstance(row,dict) for row in values):
                tab_id=f'nbspecial{table_index}'
                name=f'specialized_{_safe_name(family)}_{_safe_name(group)}_data.csv'
                cols=_columns(values)
                _write_table_csv(out,name,values,cols)
                parts.append(_paged_graph_table(tab_id, _pretty_metric(group), values,
                    'All individual measurements from this specialized analysis.',name,group))
                table_index+=1
        parts.append('</div>')
    parts.append('</section>')
    return ''.join(parts)


GRAPH_PAGING_JS = r"""
<script>
const nbGraphCache = {};
function nbGraphPage(id, change) {
  const root = document.getElementById(id);
  if (!root) return;
  if (!nbGraphCache[id]) {
    const payload = document.getElementById(id+'-rows');
    try { nbGraphCache[id] = JSON.parse(payload.textContent); }
    catch(e) { console.error('Cannot load graph table', e); return; }
  }
  const values = nbGraphCache[id];
  const pageSize = 75;
  const maxPage = Math.max(0, Math.ceil(values.length/pageSize)-1);
  const page = Math.min(maxPage,Math.max(0,Number(root.dataset.page)+change));
  const start=page*pageSize, end=Math.min(values.length,start+pageSize);
  const tbody=root.querySelector('tbody');
  tbody.replaceChildren();
  for (const record of values.slice(start,end)) {
    const tr=document.createElement('tr');
    for (const cell of record) {
      const td=document.createElement('td');
      if (cell===null || cell===undefined) td.textContent='N/A';
      else if (typeof cell==='number') td.textContent=cell.toLocaleString('en-US', {maximumSignificantDigits:12, useGrouping:false});
      else if (typeof cell==='object') td.textContent=JSON.stringify(cell);
      else td.textContent=String(cell);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  root.dataset.page=page;
  root.querySelector('.graph-count').textContent=`Showing ${start+1}–${end} of ${values.length.toLocaleString()} rows`;
  const buttons=root.querySelectorAll('.graph-page button');
  buttons[0].disabled=page===0;
  buttons[1].disabled=page>=maxPage;
}
</script>
"""

def save_report(out: Path, report: dict, player: dict | None, figures: dict[str,str], figure_tables: dict | None = None):
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(__file__).with_name('report.schema.json'),out/'report.schema.json')
    figure_tables=figure_tables or {}
    # The machine-readable report points to complete raw plotted values as
    # individual CSV assets inside the same portable ZIP.
    for item in report.get('visualizations',{}).get('figures',[]):
        title=item['title']; filename=item['file']
        data=figure_tables.get(title)
        if not data:
            continue
        rows=data.get('rows',[])
        table_filename=Path(filename).stem+'_data.csv'
        fields=_columns(rows)
        _write_table_csv(out,table_filename,rows,fields)
        extras=[]
        for extra_title,extra_rows in data.get('extra_tables',{}).items():
            fn=Path(filename).stem+'_'+_safe_name(extra_title)+'_data.csv'
            cols=_columns(extra_rows)
            _write_table_csv(out,fn,extra_rows,cols)
            extras.append({'title':extra_title,'asset':fn,'rows':len(extra_rows),'columns':cols})
        item['data_table']={'asset':table_filename,'rows':len(rows),
                            'columns':fields,'summary':data.get('summary',{}),
                            'supplementary_tables':extras}
    clean=safe_json(report)
    report_file=out/'report.json'
    report_file.write_text(json.dumps(clean,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    if player is not None:
        (out/'player_data.json').write_text(json.dumps(safe_json(player),separators=(',',':'),allow_nan=False,ensure_ascii=False),encoding='utf8')
    # Keep the existing summary sections, but expose every original metric
    # rather than selecting only six columns.
    metric_rows=[{'channel':name, **stats}
                  for name,stats in (clean.get('shared_metrics') or {}).get('channel_statistics',{}).items()]
    qual_rows=[{'channel':name, **q}
               for name,q in (clean.get('quality') or {}).get('ionm_quality',{}).items()]
    summary=clean.get('recording',{})
    desc=clean.get('classification',{})
    images=[_graph_markup(out,title,path,figure_tables.get(title),i)
            for i,(title,path) in enumerate(figures.items())]
    player_html=PLAYER_HTML.replace('__PLAYER_JSON__',json.dumps(safe_json(player),separators=(',',':'),ensure_ascii=False).replace('</','<\\/')) if player else '<section><h2>Brain Activity Player</h2><p>Unavailable because valid EEG geometry or signal data could not be loaded.</p></section>'
    warnings=''.join('<li>'+_esc(v)+'</li>' for v in clean.get('warnings',[]))
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neuro-buddy V2 EEG Report</title><style>{STYLES}</style></head><body><main>
<header><h1>Neuro-buddy V2 EEG Analysis</h1><p class="muted">Research and engineering output only. No clinical diagnosis or validated athlete readiness score.</p></header>
<section><h2>Recording and classification</h2><div class="summary">
<div class="stat"><small>File</small><div>{_esc(summary.get('file','N/A'))}</div></div>
<div class="stat"><small>Selected pipeline</small><div>{_esc(desc.get('task_family','N/A'))}</div></div>
<div class="stat"><small>Status</small><div>{_esc(clean.get('processing_status','N/A'))}</div></div>
<div class="stat"><small>Duration</small><div>{_esc(summary.get('duration_s','N/A'))} s</div></div></div>
<h3>Decision evidence</h3><p>{_esc(', '.join(desc.get('evidence',[])))}</p></section>
{player_html}
{sleep_replay(clean, sync_player=True)}
{_specialized_markup(out,clean.get('specialized_metrics',{}))}
<section><h2>Shared EEG spectral statistics</h2><p class="muted">Raw-signal PSD bandpower integrated in V². Spectral estimator: {_esc((clean.get('shared_metrics') or {}).get('preprocessing',{}).get('psd_method','legacy full-record periodogram'))}. Interpret band ratios with preprocessing and montage context.</p>{_table(metric_rows)}</section>
<section><h2>IONM-style signal-quality screening</h2><p class="muted">Heuristic grading, not validated artifact removal or medical IONM interpretation.</p>{_table(qual_rows)}</section>
{''.join(images)}
{GRAPH_PAGING_JS}
<section><h2>Warnings and limitations</h2><ul>{warnings}</ul><p class="muted">Portable JSON and derived visualization payload are supplied alongside this HTML in the transfer ZIP.</p></section>
<footer class="muted">Generated by Neuro-buddy {__version__}. Sensor-level EEG visualization, not source localization.</footer>
</main></body></html>'''
    (out/'report.html').write_text(document,encoding='utf8')
    assets={}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != 'manifest.json' and p.name != 'report.zip':
            assets[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    manifest={'schema_version':'1.0.0','report_id':clean['report_id'],'package_format':'neurobuddy-portable-v2',
              'html_entrypoint':'report.html','json_entrypoint':'report.json',
              'player_entrypoint':'player_data.json' if player is not None else None,
              'assets':assets,'includes_original_eeg':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    with zipfile.ZipFile(out/'report.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name != 'report.zip':z.write(p,p.name)
    return out/'report.zip'


def schema_check(report):
    required=('schema_version','pipeline_version','report_id','processing_status','input_manifest',
              'classification','routing','recording','quality','shared_metrics','specialized_metrics',
              'visualizations','events','warnings','errors','provenance')
    missing=[f for f in required if f not in report]
    if missing:raise ValueError('Incomplete report schema: '+', '.join(missing))
    # Ensure no numpy objects or non-JSON floats escape in downstream transport.
    normalized=safe_json(report)
    json.dumps(normalized,allow_nan=False)
    import jsonschema
    schema=json.loads(Path(__file__).with_name('report.schema.json').read_text())
    jsonschema.validate(instance=normalized,schema=schema)
    return True
