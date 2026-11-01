"""Read-only protocol16 report; recomputes reference from the window ledger."""
import csv,hashlib,html,json
from pathlib import Path
import numpy as np
from .online_curve import decode,ReferenceTimeline


def read_lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def build_report(folder,plots=True):
    folder=Path(folder);rows=read_lines(folder/'online_states.jsonl');events=read_lines(folder/'online_events.jsonl')
    windows=json.loads((folder/'online_reference_windows.json').read_text(encoding='utf-8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'));manifest=json.loads((folder/'online_manifest.json').read_text(encoding='utf-8'))
    curves={w['id']:decode(w['request'],w['parameters']) for w in windows};lookup={w['id']:w for w in windows}
    t=np.array([r['reference_elapsed_s'] for r in rows]);p=np.array([r['position_zup_m'] for r in rows]).reshape(-1,3)
    q=np.array([curves[r['window_id']].sample(min(max(0.,r['reference_elapsed_s']-lookup[r['window_id']]['start_s']),curves[r['window_id']].duration_s)) for r in rows]).reshape(-1,3)
    err=np.linalg.norm(p-q,axis=1);duration=float(t[-1]-t[0]) if len(t)>1 else 0.
    rmse=float(np.sqrt(np.trapz(err*err,t)/duration)) if duration>0 else None
    results=[e for e in events if e['stage']=='SEARCH_RESULT'];dispatch=[e for e in events if e['stage']=='SEARCH_DISPATCHED'];hits=[e for e in events if e['stage']=='TARGET_REACHED']
    active_ids=set(r['window_id'] for r in rows)
    source_ok=all((folder/'source_snapshot'/Path(path).name).is_file() and hashlib.sha256((folder/'source_snapshot'/Path(path).name).read_bytes()).hexdigest()==digest for path,digest in manifest['source_sha256'].items())
    reference_error=max((float(np.linalg.norm(np.asarray(r['reference_position_zup_m'])-expected)) for r,expected in zip(rows,q)),default=0.)
    continuity_ok=True;reference_id_ok=True;derivative_error=0.
    if windows:
        reconstructed=ReferenceTimeline(windows[0]['request'],windows[0]['parameters'])
        try:
            for w in windows[1:]:reconstructed.add(w['start_s'],w['request'],w['parameters'],w['source'],now=w['start_s']-.1)
        except ValueError:continuity_ok=False
        for r in rows:
            expected,active=reconstructed.at(r['reference_elapsed_s']);reference_id_ok=reference_id_ok and active['id']==r['window_id']
            for d,key in ((1,'reference_velocity_zup_mps'),(2,'reference_acceleration_zup_mps2')):
                derivative_error=max(derivative_error,float(np.linalg.norm(expected[d]-r[key])))
    gaps=np.diff(t);max_gap=float(gaps.max()) if len(gaps) else None
    checks=dict(synchronized_reference_reconstruction=reference_error<1e-7,monotonic_state_time=bool(len(t)>1 and np.all(gaps>0)),
        state_gap=bool(max_gap is not None and max_gap<=.1),sampled_peak_error=bool(len(err) and err.max()<=.5),source_snapshot_intact=source_ok)
    checks['source_unchanged_during_run']=summary.get('sources_unchanged',summary.get('online_metrics',{}).get('tracking_checks',{}).get('source_unchanged')) is True
    checks.update(C2_handoff_reconstruction=continuity_ok,executed_reference_window_ids=reference_id_ok,reference_derivatives=derivative_error<1e-7)
    counts=dict(dispatched_jobs=len(dispatch),completed_jobs=len(results),scheduled_candidates=sum(e['stage']=='CANDIDATE_SCHEDULED' for e in events),
        optimizer_windows_executed=sum(w['source']=='optimizer' and w['id'] in active_ids for w in windows),
        deadline_rejections=sum(e['disposition']=='WALL_DEADLINE' for e in results),stale_rejections=sum(e['disposition']=='STALE_REFERENCE_OR_TARGET' for e in results),
        aborted_jobs=sum(e['stage']=='SEARCH_ABORTED' for e in events),full_surrogate_search_calls=sum(e['reply']['result']['counts']['search_calls'] for e in results),
        selected_validation_calls=sum(e['reply']['result']['counts']['validation_calls'] for e in results),airsim_search_calls=0)
    journal=read_lines(folder/'online_worker_evaluations.jsonl')
    counts['resource_count_scope']='All completed worker-journal calls' if journal else 'Completed result replies only; aborted/uncollected jobs may have additional calls'
    if journal:
        counts.update(full_surrogate_search_calls=sum(e['kind']=='EVALUATION_COMPLETE' for e in journal),selected_validation_calls=sum(e['kind']=='VALIDATION_COMPLETE' for e in journal),
            unfinished_evaluations=sum(e['kind'] in ('EVALUATION_STARTED','VALIDATION_STARTED') for e in journal)-sum(e['kind'] in ('EVALUATION_COMPLETE','VALIDATION_COMPLETE') for e in journal))
    completed_by_id={e['reply']['job_id']:e for e in results}
    latencies=[e['host_monotonic_s']-d['host_monotonic_s'] for d in dispatch if (e:=completed_by_id.get(d['request']['job_id'])) is not None and e['disposition']!='FLIGHT_ENDED']
    metrics=dict(protocol_revision=16,outcome=summary['outcome'],error=summary.get('error'),method=manifest['method'],target_stream_sha256=manifest['target_stream_sha256'],
        executed_duration_s=duration,targets_reached=len(hits),targets_per_minute=len(hits)*60/duration if duration else None,
        position_rmse_m=rmse,max_position_error_m=float(err.max()) if len(err) else None,max_state_gap_s=max_gap,
        speed_command_clipped_fraction=float(np.mean([r['command']['speed_clipped'] for r in rows])) if rows else None,
        command_slew_clipped_fraction=float(np.mean([r['command']['slew_clipped'] for r in rows])) if rows else None,
        tracking_checks=checks,tracking_checks_passed=all(checks.values()),counts=counts,planning_reply_latencies_s=latencies,
        worker_startup_s=[e['startup_wall_s'] for e in events if e['stage']=='WORKER_READY'],
        accumulated_job_timings_s={key:sum(e['reply']['result']['timings'].get(key,0.) for e in results) for key in ('initialization_s','evaluation_s','neural_generation_s','neural_update_s','validation_s')},
        reference_reconstruction_max_error_m=reference_error,
        scope='Two-target constrained stream; executed stitched reference RMSE; no terminal-stop score, no proxy-to-AirSim ranking claim; timed completion differs from tracking pass')
    (folder/'online_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'online_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['指标','数值','含义'])
        for key,description in DICTIONARY.items():writer.writerow([key,metrics.get(key),description])
    with (folder/'online_states.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['t_s','x_m','y_m','h_m','qx_m','qy_m','qh_m','vx_mps','vy_mps','vh_mps','cx_mps','cy_mps','ch_mps','error_m','window_id','target_index'])
        for row,ref,error in zip(rows,q,err):writer.writerow([row['reference_elapsed_s'],*row['position_zup_m'],*ref,*row['velocity_zup_mps'],*row['command']['limited'],error,row['window_id'],row['target_index']])
    if plots and len(rows)>1:_plots(folder,rows,t,p,q,err,events)
    images=''.join(f'<img src="{name}.png" style="width:100%"/>' for name in ('online_trajectory','online_tracking') if (folder/(name+'.png')).exists())
    table=''.join(f'<tr><td>{html.escape(key)}</td><td>{html.escape(str(metrics.get(key)))}</td><td>{html.escape(description)}</td></tr>' for key,description in DICTIONARY.items())
    result_rows=''.join(f'<tr><td>{e["reply"]["job_id"]}</td><td>{html.escape(e["disposition"])}</td><td>{e["reply"]["result"]["wall_s"]:.3f}</td><td>{e["reply"]["result"]["counts"]["search_calls"]}</td></tr>' for e in results)
    page=f'''<!doctype html><meta charset="utf-8"><title>连续飞行·后台规划</title><style>body{{font:15px 'Microsoft YaHei',sans-serif;background:#eef3f8;color:#203248;max-width:1100px;margin:24px auto}}section{{background:white;border-radius:12px;padding:22px;margin:18px 0}}td,th{{padding:10px;border-bottom:1px solid #d9e2ef;text-align:left}}table{{border-collapse:collapse;width:100%}}</style><h1>连续飞行 · 后台规划 / 协议16</h1><p>{html.escape(manifest['method'])} · {html.escape(summary['outcome'])}</p><section><b>证据边界</b><p>原03地图、起降与观察器；时序曲线反馈跟踪器。搜索采用简化滞后质点模型，AirSim搜索调用为0。0.5米峰值门槛保持；定时结束不等于控制通过。Bézier/Hermite共用同一物理曲线族，不能作为换基创新证据。单次飞行不证明ABOM获益。</p><a href="online_metrics.json">指标JSON</a> · <a href="online_states.csv">状态CSV</a> · <a href="online_events.jsonl">事件/搜索账本</a> · <a href="online_reference_windows.json">参考切换账本</a></section><section><table><tr><th>指标</th><th>数值</th><th>含义</th></tr>{table}</table><pre>{html.escape(json.dumps(counts,ensure_ascii=False,indent=2))}</pre></section><section>{images}</section><section><h2>每次后台规划</h2><table><tr><th>任务</th><th>采用/拒绝原因</th><th>子进程耗时/秒</th><th>代理候选数</th></tr>{result_rows}</table><p>迟到、目标已变、验证失败或解未变化均保持既有参考；计划切换数与实际执行的优化窗口数分别统计。启动耗时、排队至收件延迟与训练耗时都保留；它们与飞行重叠，不能简单相加当作任务耗时。</p></section>'''
    (folder/'online_result.html').write_text(page,encoding='utf-8')
    return metrics


DICTIONARY=dict(executed_duration_s='飞行参考时钟覆盖的时长，准备/起降另存',targets_reached='通过半径0.5米且机头误差20度的目标数，不是物理碰撞数',
    targets_per_minute='单位执行时间通过目标数，须同条件重复对比',position_rmse_m='同一仿真时刻实际位置与当前已执行参考之差的时间加权均方根，单位米',
    max_position_error_m='所有采样时刻的位置误差峰值；门槛0.5米',max_state_gap_s='连续实际状态时间戳的最大间隔；超过0.1秒失败',
    speed_command_clipped_fraction='速度指令限幅的采样比例，不能称电机饱和',command_slew_clipped_fraction='速度指令变化率被限制的采样比例',
    tracking_checks_passed='参考重算、时间、采样峰值、源码检查均通过；与定时飞完分开',worker_startup_s='独立规划进程导入网络及预热时间，未隐去',
    planning_reply_latencies_s='提交到控制线程收到完整结果的墙钟延迟，包含队列和通信',outcome='定时完成、用户中止或失败均保留',error='失败原因；空值不能单独证明控制通过')


def _plots(folder,rows,t,p,q,err,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f4f7fb','axes.grid':True,'grid.alpha':.18})
    fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(111,projection='3d');ax.plot(*q.T,color='#16a377',label='已执行参考');ax.plot(*p.T,color='#d026ac',label='AirSim实际');ax.scatter(*p[0],marker='s',s=55,label='起点')
    hits=[e for e in events if e['stage']=='TARGET_REACHED']
    for i,e in enumerate(hits):ax.scatter(*e['position'],color='#df9633',marker='D');ax.text(*e['position'],f'W{i+1}')
    extent=np.maximum(np.ptp(np.vstack([p,q]),axis=0),1.);ax.set_box_aspect(extent)
    ax.set(xlabel='X / m',ylabel='Y / m',zlabel='高度 / m',title='连续飞行：已执行参考与实测轨迹（等米比例）');ax.legend();fig.tight_layout();fig.savefig(folder/'online_trajectory.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True);axes[0].plot(t,err,color='#6858bd');axes[0].axhline(.5,color='#cc4b55',ls='--',label='0.5米门槛');axes[0].legend();axes[0].set_ylabel('位置误差 / m')
    axes[1].plot(t,[np.linalg.norm(r['reference_velocity_zup_mps']) for r in rows],label='参考');axes[1].plot(t,[np.linalg.norm(r['velocity_zup_mps']) for r in rows],label='实际');axes[1].plot(t,[np.linalg.norm(r['command']['limited']) for r in rows],label='速度指令',alpha=.7);axes[1].legend(ncol=3);axes[1].set_ylabel('速度 / m/s')
    axes[2].plot(t,[r['window_id'] for r in rows],drawstyle='steps-post',label='已执行参考窗口');axes[2].set(ylabel='参考窗口编号',xlabel='仿真参考时间 / s')
    for e in events:
        if e['stage']=='SEARCH_DISPATCHED':axes[2].axvline(e['reference_time_s'],color='#239eb0',alpha=.3)
    axes[2].legend();fig.tight_layout();fig.savefig(folder/'online_tracking.png',dpi=170);plt.close(fig)


def render_comparison(directories,output):
    """Keep all failures; split exact fingerprints, no cross-group ranking."""
    from .curve_results import STYLE,TITLES,table,condition_differences
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=False);groups={};first=None;differences=[]
    for directory in directories:
        folder=Path(directory);manifest=json.loads((folder/'online_manifest.json').read_text(encoding='utf-8'))
        metrics=json.loads((folder/'online_metrics.json').read_text(encoding='utf-8'))
        groups.setdefault(manifest['condition_fingerprint'],[]).append((folder,manifest,metrics))
        if first and first['condition_fingerprint']!=manifest['condition_fingerprint']:differences+=condition_differences(first,manifest)
        first=first or manifest
    links=[]
    for i,(fingerprint,records) in enumerate(groups.items(),1):
        leaf=output/f'online_{i}';leaf.mkdir();source=leaf/'plot_source';source.mkdir();(source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        columns=['run','method','outcome','tracking_pass','targets','duration_s','RMSE_m','peak_m','executed_optimizer_windows','deadline_rejections','planning_reply_p95_s']
        rows=[];inputs={}
        for folder,manifest,m in records:
            for name in ('online_manifest.json','online_metrics.json','online_states.jsonl','online_events.jsonl','online_reference_windows.json','summary.json'):
                path=folder/name
                if path.exists():inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            latencies=m['planning_reply_latencies_s']
            rows.append([folder.name,TITLES.get(manifest['method'],manifest['method']),m['outcome'],m['tracking_checks_passed'],m['targets_reached'],m['executed_duration_s'],m['position_rmse_m'],m['max_position_error_m'],m['counts']['optimizer_windows_executed'],m['counts']['deadline_rejections'],float(np.percentile(latencies,95)) if latencies else None])
        with (leaf/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:writer=csv.writer(stream);writer.writerow(columns);writer.writerows(rows)
        plt.rcParams['font.family']='Microsoft YaHei';fig,axes=plt.subplots(1,3,figsize=(14,5))
        labels=[f'{i+1}. {TITLES.get(manifest["method"],manifest["method"])}' for i,(_,manifest,_) in enumerate(records)]
        colors=['#198a86' if m['tracking_checks_passed'] and m['outcome'] in ('DURATION_COMPLETE','STREAM_COMPLETE') else '#c75568' for _,_,m in records]
        for ax,key,title in zip(axes,('targets_reached','position_rmse_m','max_position_error_m'),('目标通过数（同时间预算）','同步位置 RMSE / m','峰值位置误差 / m')):
            values=[m.get(key) for _,_,m in records];positions=np.arange(len(records));valid=[(j,v) for j,v in enumerate(values) if v is not None]
            ax.bar([j for j,v in valid],[v for j,v in valid],color=[colors[j] for j,v in valid],width=.6);ax.set_title(title);ax.set_xticks(positions,labels,rotation=32,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
            for j,v in valid:ax.text(j,v,f'{v:.3g}',ha='center',va='bottom')
        axes[2].axhline(.5,color='#c75568',ls='--');fig.tight_layout()
        for suffix in ('png','svg','pdf'):fig.savefig(leaf/f'comparison.{suffix}',dpi=170)
        plt.close(fig)
        (leaf/'figure_index.json').write_text(json.dumps([dict(title='连续在线任务：每次原始结果（红色保留失败）',png='comparison.png',svg='comparison.svg',pdf='comparison.pdf')],ensure_ascii=False),encoding='utf-8')
        provenance=dict(protocol_revision=16,condition_fingerprint=fingerprint,input_sha256=inputs,plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Raw per-run values only, failures included; no significance or ABOM superiority inference')
        (leaf/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
        raw_links=''.join(f'<li><a href="{html.escape((folder/"online_result.html").as_uri())}">{html.escape(folder.name)}</a></li>' for folder,_,_ in records)
        (leaf/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+'<main><h1>同条件连续在线对比</h1><p>指纹 '+fingerprint+'。只比较同一目标流、时长、控制器与源码的记录；全部失败保留。单次演示不证明算法优势。</p><section>'+table(columns,rows)+'</section><section><img src="comparison.png"></section><section><h2>逐次轨迹、数据与原因</h2><ul>'+raw_links+'</ul></section></main>',encoding='utf-8')
        links.append(f'<li><a href="online_{i}/index.html">条件组{i} · {len(records)}次飞行</a></li>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+'<main><h1>所选连续在线实验 / 协议16</h1><p>条件不同分开展示，不产生跨条件总排名。红色为运行/跟踪未通过，仍纳入原始结果表。</p><section><ul>'+''.join(links)+'</ul></section><section><h2>具体条件差异</h2><pre>'+html.escape('\n'.join(dict.fromkeys(differences)) or '无')+'</pre></section></main>',encoding='utf-8')
    return output/'index.html'
