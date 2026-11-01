"""Read-only workflow audit, paired raw tables and scientific figures."""
import argparse,csv,hashlib,html,json
from pathlib import Path
import numpy as np
from analyze_online_study import record
from audit_online_control import audit_run as online_audit
from dual_target_control.curve_audit import audit_run as fixed_audit
from dual_target_control.curve_results import STYLE,table,TITLES,render_selection
from dual_target_control.reproduction_paths import workspace_record_path

ARMS=['O','A','P','AG','PG','AGL','PGL','RGL']
COLORS=dict(zip(ARMS,['#667786','#3475bd','#c77c25','#8a65b5','#299a8b','#82623e','#ce5175','#345b64']))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csvsave(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k],(list,dict))))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def fmt(x):return '未达/无有效值' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x)

def analyze(study,output=None):
    study=Path(study).resolve();design=read(study/'design_frozen.json');campaign=read(study/'campaign.json')
    if hashlib.sha256((study/'design_frozen.json').read_bytes()).hexdigest()!=campaign['design_sha256']:raise ValueError('冻结设计被修改')
    if campaign['status']=='RUNNING':raise ValueError('运行结束后再分析，避免额外负载污染飞行')
    output=Path(output).resolve() if output else study/'analysis';output.mkdir(parents=True,exist_ok=False)
    tasks={t['id']:t for t in design['tasks']};fixed=[];online=[];issues=[];audits=[];inputs={}
    trial_by_id={t['task_id']:t for t in campaign['trials']}
    for task_id in tasks:
        trial=trial_by_id.get(task_id,dict(task_id=task_id,status='NOT_RUN',closed_safely=False,run_dir=None))
        task=tasks[trial['task_id']];folder=workspace_record_path(trial['run_dir']) if trial.get('run_dir') else None
        base=dict(task_id=task['id'],method=task['method'],family=task['family'],run_dir=str(folder or ''),dispatch_status=trial['status'],closed_safely=trial['closed_safely'])
        if folder is None or not (folder/'summary.json').exists():
            issues.append(dict(**base,error=trial.get('error','缺少原始日志/未运行')))
            append_missing(base,task,fixed,online);continue
        summary=read(folder/'summary.json')
        try:
            audit=online_audit(folder) if task['family']=='online' else fixed_audit(folder)
            audits.append(dict(task_id=task['id'],**audit))
        except Exception as ex:
            audit={'status':'AUDIT_ERROR','error':repr(ex)};audits.append(dict(task_id=task['id'],**audit));issues.append(dict(task_id=task['id'],error=repr(ex)))
        if task['family']=='fixed':
            m=summary.get('metrics') or {};planning=summary.get('planning_evidence') or {};manifest=read(folder/'live_manifest.json')
            row=dict(**base,policy=task['policy'],outcome=summary['outcome'],control_passed=m.get('control_error_gates_passed',False),
                audit_passed=audit.get('status')=='PASS',condition_fingerprint=manifest['condition_fingerprint'],
                task_success=summary['outcome']=='COMPLETED' and m.get('control_error_gates_passed') is True and audit.get('status')=='PASS',
                planning_wall_s=planning.get('parent_total_planning_wall_s'),full_proxy_calls=(planning.get('counts') or {}).get('total_objective_calls'),
                peak_actual_speed_mps=audit.get('peak_actual_speed_mps'),tracking_host_wall_s=summary.get('tracking_host_wall_s'),error=summary.get('error'))
            row.update({k:m.get(k) for k in ('reference_duration_s','position_rmse_m','max_position_error_m','terminal_position_error_m','terminal_velocity_error_mps','quality_term','max_observed_state_gap_s')})
            row['waypoint_results']=m.get('waypoint_results',[]);fixed.append(row)
        else:
            try:item=record(folder,task['arm'],task['config']['online']['research']['label'])
            except Exception as ex:
                issues.append(dict(**base,error='在线指标解析失败: '+repr(ex)));append_missing(base,task,fixed,online);continue
            row=item['row'];row.update(base,scene_seed=task['scene_seed'],optimizer_seed=task['optimizer_seed'],audit_passed=audit.get('audit_passed',False))
            row['task_success']=row['outcome'] in ('DURATION_COMPLETE','STREAM_COMPLETE') and row['tracking_pass'] and row['audit_passed']
            row['condition_fingerprint']=item['manifest']['condition_fingerprint'];row['study_control_fingerprint']=item['manifest']['study_control_fingerprint']
            online.append(row)
        for filename in ('summary.json','live_manifest.json','trace.json','states.csv','preparation.json','online_manifest.json','online_metrics.json','online_states.jsonl','online_events.jsonl','online_worker_evaluations.jsonl','online_reference_windows.json'):
            p=folder/filename
            if p.exists():inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Algorithm treatment is the ONLY field removed within each paired block.
    for scene in sorted(set(r['scene_seed'] for r in online)):
        block=[r for r in online if r['scene_seed']==scene and r.get('study_control_fingerprint')];conditions=[]
        if not block:continue
        for r in block:
            manifest=read(Path(r['run_dir'])/'online_manifest.json');c=manifest['conditions'];c['online'].pop('research',None);conditions.append(c)
        if any(c!=conditions[0] for c in conditions[1:]) or len({r['study_control_fingerprint'] for r in block})!=1:
            raise ValueError(f'场景{scene}共同条件不一致')
    for policy in ('stop','pass'):
        if len({r['condition_fingerprint'] for r in fixed if r['policy']==policy and r.get('condition_fingerprint')})>1:raise ValueError('固定路线同规则条件不一致')
    paired=[]
    for scene in sorted(set(r['scene_seed'] for r in online)):
        b={r['arm']:r for r in online if r['scene_seed']==scene}
        for candidate,control in [('PGL','P'),('PGL','PG'),('PGL','AGL'),('PGL','RGL'),('AGL','A')]:
            if candidate not in b or control not in b:continue
            c,a=b[candidate],b[control]
            paired.append(dict(scene_seed=scene,candidate=candidate,control=control,both_pass=c['task_success'] and a['task_success'],
                delta_targets=difference(c,a,'targets'),delta_rmse_m=difference(c,a,'rmse_m'),delta_proxy_calls=difference(c,a,'full_proxy_calls'),
                delta_work_wall_s=difference(c,a,'work_wall_s'),
                delta_time_to_4_s=c['time_to_4_s']-a['time_to_4_s'] if c['time_to_4_s'] is not None and a['time_to_4_s'] is not None else None))
    csvsave(output/'fixed_metrics.csv',fixed);csvsave(output/'online_metrics.csv',online);csvsave(output/'paired_differences.csv',paired)
    save(output/'results.json',dict(fixed=fixed,online=online,paired_differences=paired,audits=audits,issues=issues,not_run=campaign['not_run'],
        interpretation='Raw development replication. Three paired scenario/optimizer blocks. No significance or general superiority claim. A failed trial remains in denominators.'))
    for family,folders in [('fixed',[r['run_dir'] for r in fixed if r.get('condition_fingerprint')])]:
        if folders:
            try:render_selection(folders,output/family)
            except Exception as ex:issues.append(dict(stage='fixed_detail_view',error=repr(ex)))
    figures=plots(fixed,online,output)
    fixedtable=table(['规则','方法','执行','控制＋审计','参考时长/s','跟踪墙钟/s','RMSE/m','峰值/m','完整规划/s','完整模型调用'],
        [[r['policy'],TITLES.get(r['method'],r['method']),r['outcome'],r['task_success'],*[fmt(r[k]) for k in ('reference_duration_s','tracking_host_wall_s','position_rmse_m','max_position_error_m','planning_wall_s','full_proxy_calls')]] for r in fixed])
    onlinetable=table(['场景','组','任务通过','40s过点数','第4点/s','RMSE/m','峰值/m','完整代理调用','工作墙钟/s','P95收件/s','推迟次数'],
        [[r['scene_seed'],r['arm'],r['task_success'],r['targets'],*[fmt(r[k]) for k in ('time_to_4_s','rmse_m','peak_m','full_proxy_calls','work_wall_s','latency_p95_s','postponed_windows')]] for r in online])
    labels={r['arm']:r['label'] for r in online}
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整控制实验流程与结果</title>'+STYLE+'<main><h1>完整控制实验：真实运行与结果</h1><p>原03地图、无障碍物、统一反馈控制器、实时物理。模型用于候选评价，AirSim搜索调用为0。本页由原始日志重算生成；AI协助编程与分析。</p>'
    page+='<section><h2>怎样判断结果</h2><p>固定路线须正常完成、全部控制门槛和独立审计通过。在线须完成40秒、同步跟踪通过和独立审计通过；再比较通过目标数及到第k点时间。运行完成、审计一致、控制通过是三个不同结论。</p><p>峰值误差≤0.5m、采样间隔≤0.1s；固定路线终端位置≤0.25m、速度≤0.5m/s；中间点位置≤0.5m，停稳速度≤0.3m/s、连续通过速度≥0.5m/s。在线命中要求0.5m范围与机头误差≤20°。RMSE无独立通过阈值。</p><p>固定路线优化80次搜索＋1次验证，为等候选开发设置；在线每个搜索任务0.8秒，种群8，验证计入。启动、累计搜索工作、飞行时长分别报告。三场景每组一次，尚不足以证明普遍优势或统计显著。</p></section>'
    page+='<section><h2>在线计时的证据边界</h2><p>0.8秒从请求准备完成后开始。父进程的请求准备、fallback与运动学检查，以及结果收件后的复查／解码尚未分段计时。本页在线累计工作墙钟来自工作进程，不是完整规划总开销，不能据此宣称端到端计算提速。时序保护可能在坏间隔状态写入前触发，因此即使已保存样本的跟踪检查通过，FAILED仍判任务失败。</p></section>'
    page+='<section><h2>固定路线12次</h2>'+fixedtable+'<a href="fixed/index.html">轨迹、误差、航点和指令详细对比</a> · <a href="fixed_metrics.csv">完整CSV</a></section>'
    page+='<section><h2>在线24次</h2><p>'+html.escape('；'.join(k+'='+labels[k] for k in ARMS if k in labels))+'</p>'+onlinetable+'<a href="online_metrics.csv">完整CSV</a> · <a href="paired_differences.csv">同场景配对差</a></section>'
    for name,title in figures:page+=f'<section><h2>{title}</h2><img src="{name}.png"></section>'
    page+='<section><h2>原始数据与单次图</h2><ul>'
    for r in [*fixed,*online]:
        if not r['run_dir']:continue
        folder=Path(r['run_dir']);target=folder/('online_result.html' if r['family']=='online' else 'summary.json')
        page+=f'<li>{html.escape(r["task_id"])}：<a href="{target.as_uri()}">单次图/摘要</a> · <a href="{folder.as_uri()}">原始目录</a></li>'
    page+='</ul><a href="results.json">审计、原始指标与问题清单</a></section><section><h2>失败与未运行</h2><pre>'+html.escape(json.dumps(dict(issues=issues,not_run=campaign['not_run']),ensure_ascii=False,indent=2))+'</pre></section></main></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    results=read(output/'results.json');results['issues']=issues;save(output/'results.json',results)
    inputs[str(study/'campaign.json')]=hashlib.sha256((study/'campaign.json').read_bytes()).hexdigest()
    helpers=[Path(__file__),Path(__file__).with_name('analyze_online_study.py'),Path(__file__).with_name('audit_online_control.py')]
    helpers += [Path(__file__).with_name('dual_target_control')/n for n in ('curve_audit.py','curve_results.py','curve_plots.py')]
    save(output/'provenance.json',dict(input_sha256=inputs,design_sha256=campaign['design_sha256'],analysis_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}))
    (output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    return output


def difference(a,b,key):return a[key]-b[key] if a.get(key) is not None and b.get(key) is not None else None


def append_missing(base,task,fixed,online):
    row=dict(**base,outcome=base['dispatch_status'],task_success=False,audit_passed=False)
    if task['family']=='fixed':
        row.update(policy=task['policy'],control_passed=False,condition_fingerprint=None,waypoint_results=[])
        row.update({k:None for k in ('reference_duration_s','tracking_host_wall_s','position_rmse_m','max_position_error_m','planning_wall_s','full_proxy_calls','terminal_position_error_m','terminal_velocity_error_mps','quality_term','max_observed_state_gap_s')});fixed.append(row)
    else:
        row.update(arm=task['arm'],label=task['config']['online']['research']['label'],scene_seed=task['scene_seed'],optimizer_seed=task['optimizer_seed'],tracking_pass=False,condition_fingerprint=None,study_control_fingerprint=None)
        row.update({k:None for k in ('targets','time_to_3_s','time_to_4_s','rmse_m','peak_m','full_proxy_calls','work_wall_s','latency_p95_s','postponed_windows','duration_s','proxy_calls_per_target')});online.append(row)


def plots(fixed,online,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    figures=[]
    def savefig(fig,name,title):
        fig.tight_layout()
        for ext in ('png','svg','pdf'):fig.savefig(output/(name+'.'+ext),dpi=170)
        plt.close(fig);figures.append((name,title))
    if fixed:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for policy,ax in zip(('stop','pass'),axes):
            for i,r in enumerate(r for r in fixed if r['policy']==policy):
                if r['position_rmse_m'] is None:continue
                short=r['method'].replace('bezier','B').replace('hermite','H').replace('_ordinary','-O').replace('_abom_improved_candidate','-P').replace('_abom','-A')
                is_bezier=r['method'].startswith('bezier')
                color='#667786' if r['method'].endswith('ordinary') else '#ce5175' if r['method'].endswith('candidate') else '#3475bd'
                ax.scatter(r['planning_wall_s'],r['position_rmse_m'],color=color,marker=('o' if is_bezier else '^') if r['task_success'] else 'x',s=65)
                ax.annotate(short,(r['planning_wall_s'],r['position_rmse_m']),xytext=(12,13 if is_bezier else -19),textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.6,color=color))
            ax.set(title={'stop':'逐点停稳','pass':'连续过点'}[policy]+'：每方法一次',xlabel='完整模型规划墙钟 / s',ylabel='同步RMSE / m')
            ax.margins(x=.12,y=.15)
        savefig(fig,'fixed_tradeoff','固定路线：计算成本与跟踪精度')
    online=[r for r in online if r['targets'] is not None]
    if online:
        fig,axes=plt.subplots(1,3,figsize=(15,4.7))
        for scene,ax in zip(sorted(set(r['scene_seed'] for r in online)),axes):
            for r in [r for r in online if r['scene_seed']==scene]:
                ev=[json.loads(x) for x in (Path(r['run_dir'])/'online_events.jsonl').read_text(encoding='utf-8').splitlines() if x]
                ts=[e['reference_time_s'] for e in ev if e['stage']=='TARGET_REACHED']
                ax.step([0,*ts,r['duration_s']],[0,*range(1,len(ts)+1),len(ts)],where='post',color=COLORS[r['arm']],label=r['arm']+(' ×' if not r['task_success'] else ''))
                if not r['task_success']:ax.scatter(r['duration_s'],len(ts),marker='X',s=65,color=COLORS[r['arm']],zorder=6)
            ax.set(title=f'场景{scene}（×提前失败）',xlabel='飞行时间 / s',ylabel='实际通过点数',xlim=(0,40));ax.legend(fontsize=8,ncol=4)
        savefig(fig,'online_progress','三个配对场景的任务进度')
        fig,axes=plt.subplots(1,3,figsize=(15,5));jitter={1401:-.17,1402:0,1403:.17}
        for i,arm in enumerate(ARMS):
            for r in [r for r in online if r['arm']==arm]:
                for ax,key in zip(axes,('targets','rmse_m','full_proxy_calls')):
                    ax.scatter(i+jitter[r['scene_seed']],r[key],color=COLORS[arm],marker='o' if r['task_success'] else 'x',s=42)
        for ax,label in zip(axes,('40秒通过点数','同步RMSE / m','完整代理调用（含验证）')):ax.set(xticks=range(len(ARMS)),xticklabels=ARMS,ylabel=label)
        savefig(fig,'online_raw_points','每个点代表整次飞行：保留场景差异与失败')
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for i,arm in enumerate(ARMS):
            for r in [r for r in online if r['arm']==arm]:
                for ax,key in zip(axes,('proxy_calls_per_target','work_wall_s','latency_p95_s')):
                    if r[key] is not None:ax.scatter(i+jitter[r['scene_seed']],r[key],color=COLORS[arm],marker='o' if r['task_success'] else 'x',s=42)
        for ax,label in zip(axes,('完整评价 / 已通过目标','累计工作墙钟 / s（与飞行重叠）','规划收件P95 / s')):ax.set(xticks=range(len(ARMS)),xticklabels=ARMS,ylabel=label)
        axes[2].axhline(.8,color='#ba3648',ls='--');savefig(fig,'online_resources','效率与实时性，不能把调用次数当CPU或内存')
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for r in online:
            marker='o' if r['task_success'] else 'x'
            axes[0].scatter(r['postponed_windows'],r['targets'],color=COLORS[r['arm']],marker=marker,s=40);axes[0].annotate(r['arm'],(r['postponed_windows'],r['targets']),fontsize=8)
            if r['time_to_4_s'] is not None:axes[1].scatter(r['time_to_4_s'],r['rmse_m'],color=COLORS[r['arm']],marker=marker,s=40);axes[1].annotate(r['arm'],(r['time_to_4_s'],r['rmse_m']),fontsize=8)
        axes[0].set(xlabel='已采用方案推迟当前目标时刻的次数',ylabel='通过目标数',title='机制账目；不作因果延误估计')
        axes[1].set(xlabel='到第4点时间 / s（未达不填）',ylabel='同步RMSE / m',title='速度—精度联合展示')
        savefig(fig,'online_mechanism','时间约束的作用与代价')
    save(output/'figure_index.json',[dict(name=n,title=t,formats=['png','svg','pdf']) for n,t in figures]);return figures


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();print(analyze(a.study,a.output))
