"""Read-only intake and comparison of selected, completed geometric records."""
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import shutil

from .planners import METHODS, common_fingerprint
from .runtime import ROOT, digest, filesystem_path, write_json
from .trajectory_view import LABELS, campaign_view, write_view


def read_json(path):
    return json.loads(filesystem_path(path).read_text(encoding='utf-8-sig'))


def inspect_record(directory):
    directory=Path(directory).resolve()
    record=dict(path=str(directory),id=directory.name,environment='',environment_id='',method='',
        time='',fingerprint='',eligible=False,reason='',normal_completion=None,
        tracking_passed=None,passage_passed=None,audit_passed=None,safe_cleanup=None)
    try:
        cfg=read_json(directory/'config.json');manifest=read_json(directory/'manifest.json')
        environment=cfg.get('comparison',{}).get('environment',{})
        record.update(environment=environment.get('label','E0 · 基准三维过点'),environment_id=environment.get('id','baseline'))
        case=manifest['case'];spec=cfg['cases'][case];method=spec.get('method','')
        record['method']=method
        if manifest.get('created_utc_ns'):
            record['time']=datetime.fromtimestamp(manifest['created_utc_ns']/1e9,
                timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
        report=read_json(directory/'report.json')
        cleanup=report.get('cleanup',{})
        record.update(normal_completion=report.get('normal_completion'),tracking_passed=report.get('control_constraints_passed'),
            passage_passed=report.get('waypoint_passage_passed'),audit_passed=report.get('data_audit_passed'),
            safe_cleanup=all(cleanup.get(k) is True for k in ('landed','disarmed','api_control_released')))
        if cfg.get('protocol')!='geometric_planner_pass_v1' or method not in METHODS:
            raise ValueError('仅支持三方案连续过点协议')
        if manifest.get('protocol')!=cfg['protocol'] or digest(directory/'config.json')!=manifest.get('config_sha256'):
            raise ValueError('配置与原始清单不一致')
        fingerprint=hashlib.sha256(json.dumps(dict(common=common_fingerprint(cfg),sources=manifest['sources'],
            runtime=manifest['local_preflight']['files'],settings=manifest['live']['loaded_settings']),sort_keys=True).encode()).hexdigest()
        if fingerprint!=manifest.get('comparison_fingerprint'):
            raise ValueError('运行条件指纹缺失或不一致')
        for name in ('samples.jsonl','events.json','cleanup.json'):
            if not filesystem_path(directory/name).is_file():raise ValueError('记录缺少 '+name)
        if not isinstance(report.get('normal_completion'),bool):raise ValueError('记录尚未结束')
        record.update(fingerprint=fingerprint,eligible=True)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,OverflowError) as exc:
        record['reason']=str(exc)
    return record


def discover_records(root=ROOT, extra=()):
    root=Path(root)
    paths=set(Path(p).resolve() for p in extra)
    paths.update(p.resolve() for p in (root/'logs').glob('geometric_*') if p.is_dir())
    paths.update(p.resolve() for p in (root/'reports').glob('geometric_three_planners_*/flights/geometric_*') if p.is_dir())
    return sorted((inspect_record(p) for p in paths),key=lambda r:(r['time'],r['id']),reverse=True)


def group_records(records):
    if not records:raise ValueError('尚未选择记录')
    groups=defaultdict(list);seen=set()
    for record in records:
        if not record['eligible']:raise ValueError(record['id']+'：'+record['reason'])
        if record['path'] in seen:raise ValueError('重复选择同一记录')
        seen.add(record['path']);groups[record['fingerprint']].append(record)
    for items in groups.values():
        methods=[r['method'] for r in items]
        if len(methods)!=len(set(methods)):
            raise ValueError(items[0]['environment']+'：同一条件组每种方案请选择一条记录')
        items.sort(key=lambda r:METHODS.index(r['method']))
    return dict(groups)


def _copy_evidence(record, destination, inputs):
    original=Path(record['path']);manifest=read_json(original/'manifest.json')
    names=['config.json','manifest.json','samples.jsonl','events.json','report.json','cleanup.json']
    names.extend(name for name in ('metrics_samples.csv','audit.json') if filesystem_path(original/name).is_file())
    for source in manifest['sources']:
        path=PurePosixPath(source.replace('\\','/'))
        if path.is_absolute() or '..' in path.parts or ':' in str(path):raise ValueError('源码快照路径无效')
        names.append('source/'+path.as_posix())
    for name in names:
        source=filesystem_path(original/name)
        data=source.read_bytes();sha=hashlib.sha256(data).hexdigest()
        if name.startswith('source/'):
            key=next(k for k in manifest['sources'] if k.replace('\\','/')==name[7:])
            if sha!=manifest['sources'][key]:raise ValueError('源码快照哈希不一致：'+name)
        inputs[str(original/name)]=sha
        target=filesystem_path(destination/('audit_original.json' if name=='audit.json' else name))
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)


def _result_row(record, copied, cfg):
    from .audit import audit
    report=read_json(copied/'report.json');recomputed=audit(copied);cleanup=read_json(copied/'cleanup.json')
    fields=('control_constraints_passed','data_audit_passed','waypoint_passage_passed')
    consistent=all(recomputed.get(k)==report.get(k) for k in fields) and cleanup==report.get('cleanup')
    safe=all(cleanup.get(k) is True for k in ('landed','disarmed','api_control_released'))
    metrics=recomputed.get('metrics',{})
    angle=metrics.get('attitude_reference_error_rad',{}).get('peak')
    issues=list(recomputed.get('issues',[]))
    if not consistent:issues.append('独立重算与原始报告或收尾回执不一致')
    return dict(method=record['method'],status='RECORDED',normal_completion=report['normal_completion'],
        tracking_passed=recomputed.get('control_constraints_passed'),passage_passed=recomputed.get('waypoint_passage_passed'),
        audit_passed=recomputed.get('data_audit_passed') is True and consistent,safe_cleanup=safe,
        accepted=report.get('accepted') is True and report['normal_completion'] is True and safe and consistent
            and all(recomputed.get(k) is True for k in fields),
        actual_time_s=report.get('actual_completion_time_s'),
        reference_time_s=sum(cfg['cases'][record['method']+'_pass']['segment_durations']),
        planning_wall_s=report.get('planning_wall_s'),
        position_peak_m=metrics.get('position_error_m',{}).get('peak'),
        attitude_peak_deg=angle*180/3.141592653589793 if angle is not None else None,
        run_dir=str(copied),original_run_dir=record['path'],error=report.get('error'),audit_issues=issues)


def generate_comparison(paths, output=None):
    records=[inspect_record(p) for p in paths];groups=group_records(records)
    folder=Path(output or ROOT/'reports'/('geometric_selected_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if any(folder==Path(r['path']) or folder.is_relative_to(Path(r['path'])) for r in records):
        raise ValueError('输出目录不能位于原始记录内')
    folder.mkdir(parents=True,exist_ok=False)
    inputs={};views=[];group_results=[]
    write_json(folder/'selection.json',dict(records=records,scope='Selected original records; new derived output only'))
    try:
        from .comparison import render
        for number,(fingerprint,items) in enumerate(groups.items(),1):
            child=folder/('group_'+str(number));child.mkdir()
            cfg=deepcopy(read_json(Path(items[0]['path'])/'config.json'));cfg['cases']={}
            for item in items:
                original=Path(item['path']);source_cfg=read_json(original/'config.json')
                spec=source_cfg['cases'][read_json(original/'manifest.json')['case']]
                cfg['cases'][item['method']+'_pass']=deepcopy(spec)
            write_json(child/'config_frozen.json',cfg)
            rows=[]
            for item in items:
                destination=child/'records'/item['method'];_copy_evidence(item,destination,inputs)
                current=inspect_record(destination)
                if not current['eligible'] or current['fingerprint']!=fingerprint:
                    raise ValueError('读取期间原始条件发生变化：'+item['id'])
                copied_config=read_json(destination/'config.json')
                copied_case=read_json(destination/'manifest.json')['case']
                if copied_config['cases'][copied_case]!=cfg['cases'][item['method']+'_pass']:
                    raise ValueError('读取期间冻结参考发生变化：'+item['id'])
                rows.append(_result_row(item,destination,cfg))
            write_json(child/'summary.json',dict(title='所选记录 · 实测对比',results=rows,comparable_conditions=True,
                environment=cfg['comparison'].get('environment',dict(id='baseline',label='E0 · 基准三维过点')),
                selection=True,condition_fingerprint=fingerprint))
            write_json(child/'design_frozen.json',dict(condition_fingerprint=fingerprint,original_records=items))
            with (child/'comparison.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            render(child)
            label=items[0]['environment']+' · 条件 '+fingerprint[:8]
            views.append(campaign_view(child,'group_'+str(number),label))
            group_results.append(dict(label=label,page=child.relative_to(folder).as_posix()+'/index.html',
                fingerprint=fingerprint,results=rows))
        for source,sha in inputs.items():
            if digest(source)!=sha:raise ValueError('生成期间原始记录发生变化：'+source)
        interactive=write_view(folder,views,title='所选记录 · 三维轨迹对比')
        write_json(folder/'summary.json',dict(groups=group_results,selected_records=len(records),new_flights=0))
        _write_index(folder,group_results,interactive)
        shutil.copy2(__file__,folder/'record_comparison_source.py')
        write_json(folder/'provenance.json',dict(inputs=inputs,original_inputs_unchanged=True,
            source_sha256=digest(__file__),outputs={p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!='provenance.json'},
            scope='Original records copied and independently audited only in derived folders; no AirSim RPC or new flight'))
        return folder/'index.html'
    except Exception as exc:
        write_json(folder/'generation_error.json',dict(error=str(exc),completed=False))
        raise


def _write_index(folder,groups,interactive):
    def value(x):
        if x is True:return '通过'
        if x is False:return '未通过'
        if x is None:return '无记录'
        if isinstance(x,float):return f'{x:.4f}'
        return html.escape(str(x))
    headings=['方案','完成','位置/姿态','过点','审计','收尾','整体接受','实际时间(s)','位置峰值(m)','姿态峰值(°)']
    keys=['method','normal_completion','tracking_passed','passage_passed','audit_passed','safe_cleanup','accepted',
        'actual_time_s','position_peak_m','attitude_peak_deg']
    sections=[]
    for group in groups:
        rows=''.join('<tr>'+''.join('<td>'+value(LABELS.get(r[k],r[k]) if k=='method' else r.get(k))+'</td>' for k in keys)+'</tr>' for r in group['results'])
        sections.append('<section><h2>'+html.escape(group['label'])+'</h2><p><a href="'+group['page']+'">本组完整曲线与原始记录</a></p>'
            '<div class="table-scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in headings)+'</tr>'+rows+'</table></div></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>所选记录 · 实测对比</title><style>body{margin:0;font:14px "Segoe UI","Microsoft YaHei",sans-serif;color:#26333d;line-height:1.6}
main{max-width:1400px;margin:auto;padding:20px}h1{font-size:24px;margin:0}h2{font-size:18px}a{color:#00788a}
.trajectory-frame{border:0;width:100%;height:720px}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:900px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #d5dce3}section{margin:20px 0}@media(max-width:620px){main{padding:12px}}</style>
<main><h1>所选记录 · 实测对比</h1><p>'''+str(sum(len(g['results']) for g in groups))+' 条已有记录 · '+str(len(groups))+''' 个独立条件组 · 本次新增飞行 0</p>
'''+('<iframe class="trajectory-frame" src="trajectory_view.html" title="所选记录三维轨迹对比"></iframe>' if interactive else '<p>所选记录没有跟踪采样。</p>')+''.join(sections)+'''
<p>不同任务、源码或运行条件分别展示，不跨组排名；未选择的方案不计为失败。Bézier/Hermite为等价换基，本页不证明全局最快、算法优越性或可靠性。</p>
<p><a href="selection.json">记录清单</a> · <a href="summary.json">完整汇总</a> · <a href="provenance.json">来源哈希</a></p></main>
<script>window.addEventListener('message',event=>{const f=document.querySelector('.trajectory-frame');if(f&&event.source===f.contentWindow&&event.data&&event.data.type==='trajectory-height')f.style.height=Math.max(500,Math.min(1800,event.data.height))+'px';});</script></html>'''
    (folder/'index.html').write_text(page,encoding='utf-8')
