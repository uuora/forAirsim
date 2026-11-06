"""Offline interactive scientific plots of recorded trajectory samples."""
import hashlib
import html
import json
from pathlib import Path


ASSETS=Path(__file__).with_name('assets')
FILES=('trajectory_view.html','trajectory_view.css','trajectory_view.js','plotly-gl3d-3.1.0.min.js')
COLORS={'bezier':'#007c91','hermite':'#b6417a','bspline':'#b46b05'}
LABELS={'bezier':'Bézier','hermite':'Hermite','bspline':'B样条'}


def campaign_view(folder,environment_id=None,label=None):
    folder=Path(folder).resolve()
    if not (folder/'config_frozen.json').is_file() or not (folder/'summary.json').is_file():return None
    cfg=json.loads((folder/'config_frozen.json').read_text(encoding='utf-8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    env=summary.get('environment',{})
    series=[];inputs=[folder/'config_frozen.json',folder/'summary.json']
    for result in summary['results']:
        if not result.get('run_dir'):continue
        path=Path(result['run_dir'])/'samples.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        rows=[r for r in rows if r.get('phase')=='TRACKING']
        if not rows:continue
        inputs.append(path);method=result['method']
        def height(p):return [p[0],p[1],-p[2]]
        series.append(dict(method=method,label=LABELS[method],color=COLORS[method],
            t=[r['t'] for r in rows],measured=[height(r['position']) for r in rows],
            reference=[height(r['reference']['p']) for r in rows]))
    if not series:return None
    points=[[p[0],p[1],-p[2]] for p in cfg['comparison']['task']['points_ned']]
    cloud=points+[p for s in series for kind in ('measured','reference') for p in s[kind]]
    limits=[]
    for axis in range(3):
        low=min(p[axis] for p in cloud);high=max(p[axis] for p in cloud)
        center=(low+high)/2;span=max(1.,high-low)*1.15
        limits.append([center-span/2,center+span/2])
    return dict(id=environment_id or env.get('id','baseline'),label=label or env.get('label','轨迹对比'),
        points=points,series=series,limits=limits,
        planar=max(p[2] for p in points)-min(p[2] for p in points)<.1,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})


def write_view(folder,views,title='三维轨迹对比'):
    views=[v for v in views if v is not None]
    if not views:return None
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(dict(views=views),ensure_ascii=False,allow_nan=False,separators=(',',':'))
    payload=payload.replace('<','\\u003c')
    template=(ASSETS/'trajectory_view.html').read_text(encoding='utf-8')
    page=template.replace('__TITLE__',html.escape(title)).replace('__STYLE__',(ASSETS/'trajectory_view.css').read_text(encoding='utf-8'))
    page=page.replace('__PLOTLY__',(ASSETS/'plotly-gl3d-3.1.0.min.js').read_text(encoding='utf-8').replace('</script','<\\/script'))
    page=page.replace('__DATA__',payload).replace('__SCRIPT__',(ASSETS/'trajectory_view.js').read_text(encoding='utf-8'))
    target=folder/'trajectory_view.html';target.write_text(page,encoding='utf-8')
    snapshot=folder/'plot_source/trajectory_view';snapshot.mkdir(parents=True,exist_ok=True)
    sources={}
    for source in [Path(__file__)]+[ASSETS/name for name in FILES]:
        copied=snapshot/source.name;copied.write_bytes(source.read_bytes())
        sources[source.name]=hashlib.sha256(copied.read_bytes()).hexdigest()
    inputs={name:sha for view in views for name,sha in view['inputs'].items()}
    provenance=dict(inputs=inputs,sources=sources,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        samples={v['id']:{s['method']:len(s['t']) for s in v['series']} for v in views},
        scope='Recorded TRACKING samples only; no smoothing, offsets, new flight, AirSim RPC or camera calls')
    (folder/'trajectory_view_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    return target
