"""Explicit E0-E4 campaign, preserving every requested flight and raw record."""
from datetime import datetime
import csv
import html
import io
import json
import math
import os
from pathlib import Path
import threading

from .campaign import run_campaign, safe_cleanup
from .environments import CATALOG, ROOT, load_catalog, config_for, load_environment_config
from .planners import METHODS
from .runtime import digest, filesystem_path, load_config, write_json


def run_suite(output=None, launch=False, stop_event=None, campaign_runner=None):
    from .runtime import run

    stop_event=stop_event or threading.Event()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder=Path(output or ROOT/'reports'/('geometric_environment_suite_'+stamp)).resolve()
    folder.mkdir(parents=True,exist_ok=False)
    entries=load_catalog()['environments']
    requests=[]
    for index,entry in enumerate(entries):
        load_environment_config(entry['id'])
        cfg=load_config(config_for(entry['id']))
        frozen=folder/('E'+str(index)+'_config.json');write_json(frozen,cfg)
        requests.append(dict(environment_id=entry['id'],label=entry['label'],config=str(frozen),
            config_sha256=digest(frozen),original_config=str(config_for(entry['id'])),
            campaign_dir=str(ROOT/'reports'/('geometric_three_planners_'+stamp+'_E'+str(index))),status='NOT_RUN'))
    paths=list((ROOT/'python/geometric_validation').glob('*.py'))+[ROOT/'python/run_geometric_environment_suite.py',
          ROOT/'python/run_geometric_planner_comparison.py']
    sources={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for source in paths:
        destination=filesystem_path(folder/'source'/source.relative_to(ROOT))
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.read_bytes())
    design=dict(catalog_sha256=digest(CATALOG),environments=requests,sources=sources,methods=list(METHODS),
        requested_flights=len(entries)*len(METHODS),no_automatic_retries=True,
        policy='Independent fixed tasks, no speed escalation; continue after control failure only if cleanup verified; stop after unsafe/unknown cleanup or user stop')
    write_json(folder/'design_frozen.json',design)
    stopped=False
    def flight(*args,**kwargs):
        if stop_event.is_set():raise InterruptedError('Suite stopped before next flight')
        result=run(*args,**kwargs,stop_event=stop_event)
        error=str(result['report'].get('error',''))
        if 'USER_STOP' in error or 'KeyboardInterrupt' in error:stop_event.set()
        return result
    for request in requests:
        if stopped or stop_event.is_set():
            request['reason']='Previous unsafe/unknown cleanup, changed inputs, or user stop'
        else:
            try:
                if digest(CATALOG)!=design['catalog_sha256'] or any(digest(ROOT/p)!=value for p,value in sources.items()):
                    raise RuntimeError('Source or catalog changed during suite')
                if any(digest(r['config'])!=r['config_sha256'] for r in requests):
                    raise RuntimeError('Frozen suite input changed')
                request['status']='RUNNING';write_json(folder/'suite.json',dict(design=design,requests=requests))
                print('Starting '+request['environment_id'],flush=True)
                _,rows=(campaign_runner or run_campaign)(request['config'],request['campaign_dir'],METHODS,launch,
                                                        runner=flight if campaign_runner is None else None)
                request.update(status='RECORDED',results=rows)
                stopped=not all(r.get('safe_cleanup') is True for r in rows)
                print(json.dumps(dict(environment=request['environment_id'],accepted=sum(r.get('accepted') is True for r in rows),
                    recorded=len(rows),safe_to_continue=not stopped)),flush=True)
            except BaseException as exc:
                request.update(status='FAILED',error=repr(exc));stopped=True
        write_json(folder/'suite.json',dict(design=design,requests=requests))
    return folder


def _write_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:return
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=fieldnames or list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=stream.getvalue().encode('utf-8-sig');path=filesystem_path(path)
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('Existing export differs: '+str(path))
    else:
        with path.open('xb') as target:target.write(payload)


def _states_export(directory):
    from .audit import rotation, angle
    from .core import matrix_to_euler

    samples=filesystem_path(directory/'samples.jsonl')
    rows=[json.loads(s) for s in samples.read_text(encoding='utf-8').splitlines() if s.strip()]
    output=[]
    for r in rows:
        row={k:r.get(k) for k in ('timestamp_ns','phase','t')}
        for key,labels in (('position','xyz'),('velocity','xyz'),('acceleration','xyz'),('angular_velocity','xyz'),('quaternion_wxyz','wxyz')):
            for i,label in enumerate(labels):row[key+'_'+label]=r.get(key,[None]*len(labels))[i]
        R=rotation(r['quaternion_wxyz'])
        for label,value in zip(('roll','pitch','yaw'),matrix_to_euler(R)):row['actual_'+label+'_deg']=math.degrees(value)
        reference=r.get('reference',{})
        for key in ('p','v','a','jerk','snap'):
            for i,label in enumerate('xyz'):row['reference_'+key+'_'+label]=reference.get(key,[None]*3)[i]
        row['reference_yaw_rad']=reference.get('yaw')
        cmd=r.get('command',{})
        for key in ('roll','pitch','yaw','throttle'):row['command_'+key]=cmd.get(key)
        row['attitude_reference_error_deg']=math.degrees(r['attitude_reference_error_rad']) if 'attitude_reference_error_rad' in r else None
        row['attitude_command_error_deg']=math.degrees(angle(R,cmd['R_des'])) if 'R_des' in cmd else None
        row['command_sent']=r.get('command_sent');row['failure']=r.get('failure')
        output.append(row)
    _write_csv(directory/'states_export.csv',output)
    return dict(samples=len(rows),actual_flight=any(r.get('phase')!='LANDING' and r['position'][2]<-1. for r in rows),
                released=any(r.get('phase')=='TRACKING' for r in rows))


def collect_suite(folder):
    from .audit import audit
    from .comparison import render

    folder=Path(folder).resolve();suite=json.loads((folder/'suite.json').read_text(encoding='utf-8'))
    rows=[];passages=[];checks=[];campaigns=[];evidence={}
    fields=('control_constraints_passed','data_audit_passed','waypoint_passage_passed')
    for request in suite['requests']:
        campaign_path=Path(request['campaign_dir']);campaign_file=campaign_path/'campaign.json'
        campaign=json.loads(campaign_file.read_text(encoding='utf-8')) if campaign_file.is_file() else {'results':[]}
        by_method={r['method']:r for r in campaign['results']}
        cfg=json.loads(Path(request['config']).read_text(encoding='utf-8'))
        fingerprints=[]
        for method in METHODS:
            item=by_method.get(method,{});directory=Path(item['run_dir']) if item.get('run_dir') else None
            report=item.get('report',{});audit_result={};export={};consistent=False
            if directory:
                initial=directory/'audit_initial.json'
                if not initial.exists() and (directory/'audit.json').exists():
                    original=json.loads((directory/'audit.json').read_text(encoding='utf-8'))
                    write_json(initial,original)
                audit_result=audit(directory)
                consistent=all(audit_result.get(k)==report.get(k) for k in fields)
                checks.append(dict(environment=request['environment_id'],method=method,
                    report_verdicts_match=consistent,independent_audit=audit_result))
                export=_states_export(directory)
                manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
                fingerprints.append(manifest.get('comparison_fingerprint'))
                for name in ('samples.jsonl','states_export.csv','metrics_samples.csv','manifest.json','config.json',
                             'report.json','audit_initial.json','audit.json','events.json','cleanup.json'):
                    path=directory/name
                    if path.is_file():evidence[str(path)]=digest(path)
            metrics=audit_result.get('metrics',report.get('metrics',{}))
            def get(metric,key):return metrics.get(metric,{}).get(key)
            def degrees(value):return math.degrees(value) if value is not None else None
            receipts=audit_result.get('waypoint_passages',report.get('waypoint_passages',[]))
            for p in receipts:passages.append(dict(environment=request['environment_id'],method=method,**p))
            row=dict(environment=request['environment_id'],label=request['label'],method=method,
                status=item.get('status','NOT_RUN'),actual_flight=export.get('actual_flight',False),
                normal_completion=report.get('normal_completion'),control_passed=audit_result.get('control_constraints_passed'),
                passage_passed=audit_result.get('waypoint_passage_passed'),audit_passed=audit_result.get('data_audit_passed'),
                safe_cleanup=safe_cleanup(report),report_accepted=report.get('accepted'),
                accepted=bool(report.get('accepted') is True and report.get('normal_completion') is True
                    and safe_cleanup(report) and consistent and all(audit_result.get(k) is True for k in fields)),
                actual_completion_s=report.get('actual_completion_time_s'),
                reference_duration_s=sum(cfg['cases'][method+'_pass']['segment_durations']),
                offline_planning_wall_s=report.get('planning_wall_s'),reference_decode_wall_s=report.get('reference_decode_wall_s'),
                position_rmse_m=get('position_error_m','rmse'),position_peak_m=get('position_error_m','peak'),
                position_exceedance_s=get('position_error_m','exceedance_s'),
                attitude_reference_rmse_deg=degrees(get('attitude_reference_error_rad','rmse')),
                attitude_reference_peak_deg=degrees(get('attitude_reference_error_rad','peak')),
                attitude_reference_exceedance_s=get('attitude_reference_error_rad','exceedance_s'),
                attitude_command_peak_deg=degrees(get('attitude_command_error_rad','peak')),
                actual_speed_peak_mps=get('actual_speed_mps','peak'),actual_tilt_peak_deg=degrees(get('actual_tilt_rad','peak')),
                max_state_interval_s=metrics.get('max_state_interval_s'),tracking_samples=metrics.get('samples'),
                passed_waypoints=sum(p['passed'] for p in receipts),expected_waypoints=len(cfg['comparison']['task']['points_ned'])-2,
                command_saturation_s=metrics.get('command_saturation_s'),run_dir=str(directory) if directory else None,
                error=report.get('error',item.get('error',request.get('error'))))
            rows.append(row)
        same=bool(len(fingerprints)==len(METHODS) and None not in fingerprints and len(set(fingerprints))==1)
        campaigns.append(dict(environment=request['environment_id'],same_conditions=same,
                              path=str(campaign_path),status=request['status']))
        if (campaign_path/'summary.json').is_file():render(campaign_path)
    _write_csv(folder/'all_flights.csv',rows)
    _write_csv(folder/'all_waypoint_passages.csv',passages,fieldnames=['environment','method','waypoint_index',
        'passed','reference_time_s','actual_time_s','distance_m','actual_speed_mps','nearest_distance_m','nearest_speed_mps'])
    write_json(folder/'independent_audits.json',checks)
    summary=dict(requested=len(rows),actual_flights=sum(r['actual_flight'] for r in rows),
        normal_completion=sum(r['normal_completion'] is True for r in rows),
        control_passed=sum(r['control_passed'] is True for r in rows),passage_passed=sum(r['passage_passed'] is True for r in rows),
        audit_passed=sum(r['audit_passed'] is True for r in rows),safe_cleanup=sum(r['safe_cleanup'] for r in rows),
        accepted=sum(r['accepted'] is True for r in rows),not_run=sum(r['status']=='NOT_RUN' for r in rows),
        reports_match_reaudit=all(c['report_verdicts_match'] for c in checks),environments=campaigns,results=rows,
        scope='One fixed attempt per environment and planner; no retries, cross-environment ranking or reliability claim')
    write_json(folder/'summary.json',summary)
    write_json(folder/'data_dictionary.json',dict(
        coordinates='position/velocity/reference in world NED; quaternion wxyz body FRD to NED; angular velocity body FRD',
        units='SI; actual Euler and error fields ending _deg in degrees; command angles and reference_yaw_rad in radians',
        timing='actual completion from synchronous release to completed terminal hold; preparation/landing excluded',
        planning='offline plan() wall, cold/warm import differences uncontrolled; reference decoding separate; no AirSim optimizer calls',
        blank_cells='Unrecorded or not applicable; never interpreted as zero',
        passage='First measured sample meeting ordered radius/speed/time-window gates; nearest-distance sample separately recorded',
        attitude='Main gate compares actual SO(3) to nominal synchronous R_reference, separately from feedback R_des',
        scope='SimpleFlight state estimates; no real sensor, motor saturation or camera evidence'))
    render_suite_page(folder,summary)
    evidence.update({str(folder/p):digest(folder/p) for p in ('all_flights.csv','all_waypoint_passages.csv','summary.json','independent_audits.json','index.html')})
    write_json(folder/'provenance.json',dict(evidence=evidence,source_design_sha256=digest(folder/'design_frozen.json')))
    return summary


def render_suite_page(folder,summary=None):
    """Refresh the suite display from existing records, without auditing or flight."""
    folder=Path(folder).resolve()
    if summary is None:summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    suite=json.loads((folder/'suite.json').read_text(encoding='utf-8'))
    labels={r['environment_id']:r['label'] for r in suite['requests']}
    rows=summary['results'];campaigns=summary['environments']
    headings=['环境','方案','正常完成','位置/姿态','过点','审计','安全收尾','整体接受','实际时间(s)','位置峰值(m)','姿态峰值(°)']
    keys=['label','method','normal_completion','control_passed','passage_passed','audit_passed','safe_cleanup','accepted',
          'actual_completion_s','position_peak_m','attitude_reference_peak_deg']
    def value(x):
        if x is True:return '通过'
        if x is False:return '未通过'
        if x is None:return '无记录'
        if isinstance(x,float):return f'{x:.4f}'
        return html.escape(str(x))
    table=''.join('<tr>'+''.join('<td>'+value(r.get(k))+'</td>' for k in keys)+'</tr>' for r in rows)
    tabs=[];panels=[];inputs=[folder/'summary.json',folder/'suite.json']
    for index,c in enumerate(campaigns):
        label=html.escape(labels.get(c['environment'],c['environment']));panel='environment-'+str(index)
        selected='true' if index==0 else 'false';tabindex='0' if index==0 else '-1'
        tabs.append('<button type="button" role="tab" id="tab-'+str(index)+'" aria-controls="'+panel+
                    '" aria-selected="'+selected+'" tabindex="'+tabindex+'">'+label+'</button>')
        path=Path(c['path'])
        def url(name):return html.escape(os.path.relpath(path/name,folder).replace('\\','/'),quote=True)
        if (path/'trajectory_3d.png').is_file():
            metric=url('trajectory_3d_metric.png') if (path/'trajectory_3d_metric.png').is_file() else url('trajectory_3d.png')
            content='<img class="trajectory" src="'+url('trajectory_3d.png')+'" data-display-src="'+url('trajectory_3d.png')+'" data-metric-src="'+metric+'" width="1600" height="960" alt="'+label+'三方案三维轨迹对比">'
            inputs.append(path/'trajectory_3d.png')
            if (path/'trajectory_3d_metric.png').is_file():inputs.append(path/'trajectory_3d_metric.png')
            content+='<p><a class="spatial-svg" href="'+url('trajectory_3d.svg')+'" data-display-href="'+url('trajectory_3d.svg')+'" data-metric-href="'+url('trajectory_3d_metric.svg')+'">三维图SVG</a> · <a class="spatial-png" href="'+url('trajectory_3d.png')+'" data-display-href="'+url('trajectory_3d.png')+'" data-metric-href="'+metric+'">三维图PNG</a> · <a href="'+url('index.html')+'">本环境完整对比</a></p>'
        else:content='<p>该环境暂无三维轨迹记录。</p>'
        panels.append('<section role="tabpanel" id="'+panel+'" aria-labelledby="tab-'+str(index)+'"'+
                      ('' if index==0 else ' hidden')+'><h2>'+label+' · 三维轨迹对比</h2>'+content+'</section>')
    from .trajectory_view import campaign_view, write_view
    interactive=write_view(folder,[campaign_view(c['path'],c['environment'],labels.get(c['environment'])) for c in campaigns])
    if interactive:inputs.extend([interactive,folder/'trajectory_view_provenance.json'])
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>E0–E4 实测数据</title>
<style>body{font:14px sans-serif;margin:0;line-height:1.6;color:#20252b;background:#fff}
main{max-width:1400px;margin:auto;padding:20px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:14px 0 0}
p{margin:8px 0}a{color:#00788a}nav{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid #d5dce3;margin-top:16px}
button{font:inherit;cursor:pointer;color:#4a525b;background:#fff;border:0;border-bottom:3px solid transparent;padding:9px 12px}
button[aria-selected=true]{color:#006b7b;border-bottom-color:#007c91;background:#edf7f8}
button:focus-visible{outline:2px solid #007c91;outline-offset:-2px}[hidden]{display:none!important}
.view-bar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:10px}
.view-modes{display:inline-flex;border:1px solid #c6d2d8;border-radius:4px;overflow:hidden}
.view-modes button{border:0;padding:5px 12px}.view-modes button[aria-pressed=true]{background:#007c91;color:#fff}
#scale-note{font-size:12px;color:#59636e}
.trajectory-frame{width:100%;height:720px;border:0;display:block}.static-exports{margin-top:10px}
.static-exports summary{cursor:pointer;color:#526372;padding:6px 0}
section img{display:block;width:100%;max-width:1100px;max-height:65vh;height:auto;object-fit:contain;margin:0 auto}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:1100px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #d5dce3}
th{white-space:nowrap}@media(max-width:600px){main{padding:12px}h1{font-size:22px}button{padding:8px 9px}}</style></head><body><main>
<h1>E0–E4 实测数据</h1><p>实际飞行 '''+str(summary['actual_flights'])+' / '+str(summary['requested'])+' · 整体接受 '+str(summary['accepted'])+''' · 每环境三种方案各一次</p>
'''+('<iframe class="trajectory-frame" src="trajectory_view.html" title="五环境三方案三维轨迹对比"></iframe><details class="static-exports"><summary>静态图与导出</summary>' if interactive else '')+'''
<nav role="tablist" aria-label="实验环境">'''+''.join(tabs)+'''</nav>
<div class="view-bar"><div class="view-modes" role="group" aria-label="三维比例">
<button type="button" data-view-mode="display" aria-pressed="true">展示比例</button>
<button type="button" data-view-mode="metric" aria-pressed="false">等米制</button></div>
<span id="scale-note">展示轴比例 X:Y:H = 1.55:1.10:0.80</span></div>'''+''.join(panels)+('</details>' if interactive else '')+'''
<h2>全部实测指标</h2><div class="table-scroll"><table><tr>'''+''.join('<th>'+h+'</th>' for h in headings)+'</tr>'+table+'''</table></div>
<p><a href="all_flights.csv">全部指标CSV</a> · <a href="all_waypoint_passages.csv">逐点判据CSV</a> ·
<a href="data_dictionary.json">数据字典</a> · <a href="summary.json">完整JSON</a> · <a href="independent_audits.json">独立重算</a></p>
<p>位置误差≤0.5m，SO(3)参考姿态误差≤10°；中途实测过点，终点静止保持。不同环境分别比较；Bézier/Hermite为等价换基。本批不证明全局最快、算法优越性或可靠性。</p>
</main><script>
window.addEventListener('message',event=>{const frame=document.querySelector('.trajectory-frame');if(frame&&event.source===frame.contentWindow&&event.data&&event.data.type==='trajectory-height')frame.style.height=Math.max(500,Math.min(1800,event.data.height))+'px';});
const tabs=Array.from(document.querySelectorAll('[role=tab]'));
const panels=Array.from(document.querySelectorAll('[role=tabpanel]'));
document.querySelectorAll('[data-view-mode]').forEach(button=>{
  button.addEventListener('click',()=>{
    const metric=button.dataset.viewMode==='metric';
    document.querySelectorAll('[data-view-mode]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    document.querySelectorAll('img.trajectory').forEach(img=>{img.src=metric?img.dataset.metricSrc:img.dataset.displaySrc;});
    document.querySelectorAll('a.spatial-png,a.spatial-svg').forEach(link=>{link.href=metric?link.dataset.metricHref:link.dataset.displayHref;});
    document.getElementById('scale-note').textContent=metric?'各轴 1 米等长':'展示轴比例 X:Y:H = 1.55:1.10:0.80';
  });
});
function select(tab){
  tabs.forEach(item=>{const active=item===tab;item.setAttribute('aria-selected',String(active));item.tabIndex=active?0:-1;});
  panels.forEach(panel=>{panel.hidden=panel.id!==tab.getAttribute('aria-controls');});
}
tabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>select(tab));
  tab.addEventListener('keydown',event=>{
    let next=index;
    if(event.key==='ArrowRight')next=(index+1)%tabs.length;
    else if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;
    else if(event.key==='Home')next=0;
    else if(event.key==='End')next=tabs.length-1;
    else return;
    event.preventDefault();select(tabs[next]);tabs[next].focus();
  });
});
</script></body></html>'''
    (folder/'index.html').write_text(page,encoding='utf-8')
    source=folder/'page_source/environment_suite.py';source.parent.mkdir(exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    write_json(folder/'page_provenance.json',dict(inputs={str(p):digest(p) for p in inputs},
        source_sha256=digest(source),outputs={'index.html':digest(folder/'index.html')},
        scope='Existing suite display only; no audit, AirSim connection or flight'))
    return folder/'index.html'
