"""Read-only study analysis; source outcomes and failures are never rewritten."""
import argparse,csv,hashlib,html,json,math
from pathlib import Path
import numpy as np
from dual_target_control.online_results import read_lines
from dual_target_control.online_curve import ReferenceTimeline
from audit_online_control import audit_run

COLORS=['#687786','#3475bd','#c77c25','#8a65b5','#299a8b','#ce5175']


def record(folder,arm,label):
    folder=Path(folder);m=json.loads((folder/'online_metrics.json').read_text(encoding='utf-8'));manifest=json.loads((folder/'online_manifest.json').read_text(encoding='utf-8'))
    states=read_lines(folder/'online_states.jsonl');events=read_lines(folder/'online_events.jsonl');journal=read_lines(folder/'online_worker_evaluations.jsonl')
    windows=json.loads((folder/'online_reference_windows.json').read_text(encoding='utf-8'));hits=[e for e in events if e['stage']=='TARGET_REACHED'];active=set(s['window_id'] for s in states);shifts=[]
    timeline=ReferenceTimeline(windows[0]['request'],windows[0]['parameters']) if windows else None
    for w in windows[1:]:
        _,prior=timeline.at(w['start_s']);before=prior['start_s']+float(prior['_curve'].ends[0]);after=w['start_s']+float(w['parameters'][3])
        if w['source']=='optimizer' and w['id'] in active:
            shifts.append(dict(window_id=w['id'],target_index=w['request'].get('target_index'),old_arrival_s=before,new_arrival_s=after,delta_s=after-before))
        timeline.add(w['start_s'],w['request'],w['parameters'],w['source'],now=w['start_s']-.1)
    results=[e for e in events if e['stage']=='SEARCH_RESULT'];work=sum(e['reply']['result']['wall_s'] for e in results)
    latency=m['planning_reply_latencies_s'];times=[e['reference_time_s'] for e in hits];calls=m['counts']['full_surrogate_search_calls']+m['counts']['selected_validation_calls']
    row=dict(arm=arm,label=label,run_dir=str(folder),outcome=m['outcome'],tracking_pass=m['tracking_checks_passed'],duration_s=m['executed_duration_s'],targets=m['targets_reached'],
        time_to_1_s=times[0] if len(times)>0 else None,time_to_2_s=times[1] if len(times)>1 else None,time_to_3_s=times[2] if len(times)>2 else None,time_to_4_s=times[3] if len(times)>3 else None,
        rmse_m=m['position_rmse_m'],peak_m=m['max_position_error_m'],max_state_gap_s=m['max_state_gap_s'],peak_speed_mps=max((float(np.linalg.norm(s['velocity_zup_mps'])) for s in states),default=None),
        full_proxy_calls=calls,proxy_calls_per_target=calls/len(hits) if hits else None,work_wall_s=work,startup_s=sum(m.get('worker_startup_s',[])),
        latency_p50_s=float(np.median(latency)) if latency else None,latency_p95_s=float(np.percentile(latency,95)) if latency else None,latency_max_s=max(latency) if latency else None,
        dispatched=m['counts']['dispatched_jobs'],scheduled=m['counts']['scheduled_candidates'],executed=m['counts']['optimizer_windows_executed'],deadline_rejections=m['counts']['deadline_rejections'],
        skipped_windows=sum(e['stage']=='SEARCH_SKIPPED_PROGRESS' for e in events),postponed_windows=sum(s['delta_s']>1e-7 for s in shifts),
        positive_planned_delay_s=sum(max(0.,s['delta_s']) for s in shifts),net_planned_delay_s=sum(s['delta_s'] for s in shifts),reference_time_changes=shifts,
        phase_times=m['accumulated_job_timings_s'],speed_limit_fraction=m['speed_command_clipped_fraction'],slew_limit_fraction=m['command_slew_clipped_fraction'])
    return dict(row=row,states=states,events=events,manifest=manifest,hits=times)


def analyze(study):
    study=Path(study).resolve();campaign=json.loads((study/'campaign.json').read_text(encoding='utf-8'));design=json.loads((study/'design_frozen.json').read_text(encoding='utf-8'))
    if hashlib.sha256((study/'design_frozen.json').read_bytes()).hexdigest()!=campaign['design_sha256']:raise ValueError('冻结方案哈希变化')
    output=study/'analysis';output.mkdir(exist_ok=False);data=[];issues=[];audits=[]
    for trial in campaign['trials']:
        folder=trial.get('run_dir');arm=trial['arm']
        if folder and (Path(folder)/'online_metrics.json').exists():
            item=record(folder,arm,design['arms'][arm]['label']);data.append(item)
            try:audits.append(audit_run(folder))
            except Exception as ex:issues.append(dict(arm=arm,error='audit: '+str(ex)))
        else:issues.append(dict(arm=arm,status=trial['status'],error=trial.get('error'),run_dir=folder))
    order={key:i for i,key in enumerate(['O','A','P','AG','PG','PGL'])};data.sort(key=lambda d:order[d['row']['arm']])
    controls=set(d['manifest'].get('study_control_fingerprint') for d in data)
    if len(controls)!=1 or None in controls:raise ValueError('物理/控制/预算/源码条件不一致，禁止画成同组比较')
    # Independently inspect actual condition differences, not only a stored hash.
    frozen=[]
    for item in data:
        conditions=json.loads(json.dumps(item['manifest']['conditions']));conditions['online'].pop('research',None);frozen.append(conditions)
    if any(c!=frozen[0] for c in frozen[1:]):raise ValueError('删除预先声明的算法处理字段后仍存在条件差异')
    rows=[d['row'] for d in data];fields=[k for k in rows[0] if k not in ('reference_time_changes','phase_times')]
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    result=dict(design_sha256=campaign['design_sha256'],study_control_fingerprint=next(iter(controls)),records=rows,issues=issues,not_run=campaign['not_run'],audits=audits,
        scope='One scenario, one run per mechanism. No confidence intervals/significance. All failures retained. Planned-time shift is a reference ledger, not causal flight delay. Work is accumulated worker wall time, not CPU time.')
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    figures=plot(data,output,design['duration_s'])
    headings=['组','方法','结果','跟踪通过','目标数','到第3点/s','到第4点/s','RMSE/m','完整代理调用','实际采用','计划推迟次数']
    values=[[r['arm'],r['label'],r['outcome'],r['tracking_pass'],r['targets'],r['time_to_3_s'],r['time_to_4_s'],r['rmse_m'],r['full_proxy_calls'],r['executed'],r['postponed_windows']] for r in rows]
    from dual_target_control.curve_results import STYLE,table
    links=''.join(f'<li>{r["arm"]}：<a href="{Path(r["run_dir"]).joinpath("online_result.html").as_uri()}">单次轨迹/误差/原始数据</a></li>' for r in rows)
    body='<!doctype html><meta charset="utf-8">'+STYLE+'<main><h1>在线轨迹搜索：进度约束与局部扰动消融</h1><p>种子'+str(design['scenario_seed'])+'；每组25秒；优化组每任务0.8秒，含初始化和验证；种群8、同一网络容量。O/AG等简称见表。全为Hermite，隔离搜索机制；不是原Bézier/Hermite六组。</p><section>'+table(headings,values)+'</section><section><p>未到达的目标时刻为空，不能当25秒完成。红色/失败不删除。这里只是一组新场景开发验证，不能证明普遍优势。普通曲线也加载公共工作进程，启动开销单列；CPU占用和内存峰值本轮未采集。</p><p>图中的计划推迟是每次承诺参考到达时刻变化，不是实际多耗时的因果估计。不同滚动状态的代理分数不直接汇总比较。</p><a href="metrics.csv">完整指标CSV</a> · <a href="results.json">审计与完整JSON</a><ul>'+links+'</ul></section>'
    for name,title in figures:body+=f'<section><h2>{title}</h2><img src="{name}.png"></section>'
    body+='<section><h2>未运行或审计缺口</h2><pre>'+html.escape(json.dumps(dict(issues=issues,not_run=campaign['not_run']),ensure_ascii=False,indent=2))+'</pre></section></main>'
    (output/'index.html').write_text(body,encoding='utf-8');(output/'figure_index.json').write_text(json.dumps([dict(title=t,png=n+'.png',svg=n+'.svg',pdf=n+'.pdf') for n,t in figures],ensure_ascii=False),encoding='utf-8')
    sources={str(study/'design_frozen.json'):campaign['design_sha256'],str(study/'campaign.json'):hashlib.sha256((study/'campaign.json').read_bytes()).hexdigest()}
    for d in data:
        for name in ('online_manifest.json','online_metrics.json','online_states.jsonl','online_events.jsonl','online_worker_evaluations.jsonl','online_reference_windows.json'):
            p=Path(d['row']['run_dir'])/name
            if p.exists():sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps(dict(input_sha256=sources,declared_treatment_fields=design['treatment_fields'],analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    (output/'analysis_source.py').write_bytes(Path(__file__).read_bytes());return output


def plot(data,output,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'figure.facecolor':'white'})
    figures=[]
    def save(fig,name,title):
        fig.tight_layout()
        for fmt in ('png','svg','pdf'):fig.savefig(output/(name+'.'+fmt),dpi=170)
        plt.close(fig);figures.append((name,title))
    fig,ax=plt.subplots(figsize=(11,5.5))
    for d,c in zip(data,COLORS):
        r=d['row'];times=d['hits'];ax.step([0.,*times,r['duration_s']],[0,*range(1,len(times)+1),len(times)],where='post',color=c,label=r['arm']+' '+r['label'])
    ax.set(xlim=(0,horizon),xlabel='飞行时间 / s',ylabel='已通过目标数',title='实际任务进度（停止后不延长曲线）');ax.legend(fontsize=9,ncol=2);save(fig,'progress','目标累计曲线与到达时间')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for d,c in zip(data,COLORS):
        r=d['row'];x=r['time_to_3_s'];y=r['rmse_m']
        if x is not None and y is not None:axes[0].scatter(x,y,color=c,s=80);axes[0].annotate(r['arm'],(x,y),xytext=(5,4),textcoords='offset points')
        axes[1].scatter(r['full_proxy_calls'],r['targets'],color=c,s=80);axes[1].annotate(r['arm'],(r['full_proxy_calls'],r['targets']),xytext=(5,4),textcoords='offset points')
    axes[0].set(xlabel='到达第3点的时间 / s（未达到者不填虚值）',ylabel='同步位置RMSE / m',title='速度—精度取舍');axes[1].set(xlabel='完整代理调用（含验证）',ylabel='已通过目标数',title='任务推进与计算开销');save(fig,'tradeoff','速度、精度与资源的联合展示')
    fig,axes=plt.subplots(1,2,figsize=(12,5));x=np.arange(len(data));bottom=np.zeros(len(data))
    parts=[('initialization_s','网络初始化'),('evaluation_s','候选评价'),('neural_generation_s','网络生成'),('neural_update_s','网络更新'),('validation_s','最终验证')]
    for key,label in parts:
        v=np.array([d['row']['phase_times'].get(key,0.) for d in data]);axes[0].bar(x,v,bottom=bottom,label=label);bottom+=v
    other=np.maximum(0,np.array([d['row']['work_wall_s'] for d in data])-bottom);axes[0].bar(x,other,bottom=bottom,label='其他搜索工作')
    axes[0].set(xticks=x,xticklabels=[d['row']['arm'] for d in data],ylabel='累计工作进程墙钟 / s',title='工作时间拆分（与飞行重叠）');axes[0].legend(fontsize=8)
    axes[1].bar(x,[d['row']['startup_s'] for d in data],color=COLORS[:len(data)]);axes[1].set(xticks=x,xticklabels=[d['row']['arm'] for d in data],ylabel='启动/导入/预热 / s',title='启动成本单列，不隐去');save(fig,'resources','完整资源开销')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for d,c in zip(data,COLORS):
        r=d['row'];shifts=r['reference_time_changes'];axes[0].plot(range(1,len(shifts)+1),[s['delta_s'] for s in shifts],'.-',color=c,label=r['arm'])
        t=[s['reference_elapsed_s'] for s in d['states']];err=[s['position_error_m'] for s in d['states']];axes[1].plot(t,err,color=c,label=r['arm'])
    axes[0].axhline(0,color='black',lw=.8);axes[0].set(xlabel='已执行优化窗口序号',ylabel='新的绝对到达时刻 − 原时刻 / s',title='正值代表计划被推迟（非实际延误）');axes[0].legend()
    axes[1].axhline(.5,color='#b32f39',ls='--',label='门槛');axes[1].set(xlabel='飞行时间 / s',ylabel='位置误差 / m',title='同步原始误差');axes[1].legend();save(fig,'mechanism','机制是否起效与跟踪约束')
    fig,axes=plt.subplots(1,2,figsize=(12,5));width=.24
    for j,(key,label) in enumerate([('dispatched','已提交'),('scheduled','安排切换'),('executed','实际执行')]):axes[0].bar(x+(j-1)*width,[d['row'][key] for d in data],width,label=label)
    axes[0].set(xticks=x,xticklabels=[d['row']['arm'] for d in data],title='任务流转（不能把安排当执行）');axes[0].legend()
    for d,c in zip(data,COLORS):
        lat=[e['host_monotonic_s']-next(e0['host_monotonic_s'] for e0 in d['events'] if e0['stage']=='SEARCH_DISPATCHED' and e0['request']['job_id']==e['reply']['job_id']) for e in d['events'] if e['stage']=='SEARCH_RESULT' and e['disposition']!='FLIGHT_ENDED']
        if lat:axes[1].plot(sorted(lat),np.arange(1,len(lat)+1)/len(lat),'.-',label=d['row']['arm'],color=c)
    axes[1].axvline(.8,color='#b32f39',ls='--');axes[1].set(xlabel='提交至完整收件 / s',ylabel='经验累计比例',title='规划响应延迟（同次飞行内样本）');axes[1].legend();save(fig,'pipeline','候选采用链与实时预算')
    return figures


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',type=Path);args=parser.parse_args();print(analyze(args.study))
