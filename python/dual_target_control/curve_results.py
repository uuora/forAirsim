"""Read-only Chinese dashboards, with strict comparisons within one protocol."""
import csv,hashlib,html,json
from pathlib import Path
from .metric_explanations import glossary_section,write_glossary

TITLES=dict(bezier_ordinary='Bézier 未优化参考',hermite_ordinary='Hermite 未优化参考',bezier_abom='Bézier + 基础ABOM',
    hermite_abom='Hermite + 基础ABOM',bezier_abom_improved_candidate='Bézier + 扰动改进候选',hermite_abom_improved_candidate='Hermite + 扰动改进候选')
STYLE='''<style>body{font:15px system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#edf2f5;color:#203348}main{max-width:1200px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:21px}section{background:white;border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 3px 16px #18364b0b}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #e0e8ee}img{width:100%;height:auto}small{color:#597081}.scroll{overflow:auto}a{color:#147b96}code{overflow-wrap:anywhere}.pass{color:#16745c}.fail{color:#ba423f}</style>'''

def table(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def condition_differences(first,other):
    result=[]
    def compare(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b)):compare(a.get(k),b.get(k),path+'.'+k)
        elif a!=b:result.append(path+': '+str(a)[:100]+' → '+str(b)[:100])
    compare(first['conditions'],other['conditions'],'条件');compare(first['source_sha256'],other['source_sha256'],'源码')
    return result


def render_results(directories,output):
    directories=[Path(p).resolve() for p in directories];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if not directories:raise ValueError('至少选择一条已有记录')
    source_paths=[Path(__file__),Path(__file__).with_name('curve_plots.py'),Path(__file__).with_name('metric_explanations.py'),Path(__file__).with_name('curve_audit.py')]
    source_bytes={p.name:p.read_bytes() for p in source_paths};source_dir=output/'plot_source';source_dir.mkdir(exist_ok=True)
    for name,value in source_bytes.items():(source_dir/name).write_bytes(value)
    records=[];inputs={};audits=[];first=None
    for folder in directories:
        manifest=json.loads((folder/'live_manifest.json').read_text(encoding='utf-8'))
        if first and first['condition_fingerprint']!=manifest['condition_fingerprint']:
            raise ValueError('实验条件不同，拒绝混比：\n'+'\n'.join(condition_differences(first,manifest)[:10]))
        first=first or manifest
        from .curve_audit import audit_run
        audits.append(audit_run(folder))
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary.get('sources_unchanged') is False:raise ValueError('运行期间源码变化；保留原始记录，不进入排名')
        trace=json.loads((folder/'trace.json').read_text(encoding='utf-8'));records.append((folder,summary,trace,manifest))
        for name in ('live_manifest.json','summary.json','trace.json','states.csv'):
            path=folder/name;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    revision=first['protocol_revision'];route=first['plan'].get('route',{})
    rows=[];waypoint_rows=[];csv_rows=[]
    short=dict(bezier_ordinary='B0',hermite_ordinary='H0',bezier_abom='B+ABOM',hermite_abom='H+ABOM',bezier_abom_improved_candidate='B+Pair',hermite_abom_improved_candidate='H+Pair')
    def fmt(value):return '—' if value is None else f'{value:.4f}'
    for index,(folder,summary,trace,manifest) in enumerate(records):
        states=trace['states'];commands=trace['commands'];method=summary['method'];label=short.get(method,method)+' #'+str(index+1)
        metrics=summary.get('metrics') or {};planning=summary.get('planning_evidence') or {}
        passed=metrics.get('control_error_gates_passed');outcome=summary['outcome']
        result='通过' if passed is True else '未通过' if metrics else '无有效分数'
        rows.append([TITLES.get(method,method)+' #'+str(index+1),{'COMPLETED':'执行完成','FAILED':'执行失败'}.get(outcome,outcome),result,
            *[fmt(metrics.get(k)) for k in ('reference_duration_s','position_rmse_m','max_position_error_m','terminal_position_error_m','terminal_velocity_error_mps')]])
        for point in metrics.get('waypoint_results',[]):
            waypoint_rows.append([label,point['index']+1,{'stop':'中间点停稳','pass':'中间点连续通过','terminal_stop':'末点停稳'}.get(point.get('rule'),point.get('status')),
                fmt(point.get('reference_time_s')),fmt(point.get('position_error_m')),fmt(point.get('actual_speed_mps')),'通过' if point['passed'] else '未通过'])
        row=dict(method=method,method_name=TITLES.get(method,method),policy=route.get('policy','single'),outcome=outcome,control_passed=passed,task_success=bool(outcome=='COMPLETED' and passed),
            historical_planning_wall_s=(planning.get('timings') or {}).get('search_plus_selected_validation_wall_s') if revision!=15 else None,
            model_search_validation_wall_s=(planning.get('timings') or {}).get('search_plus_selected_validation_wall_s') if revision==15 else None,
            planning_wall_s=planning.get('parent_total_planning_wall_s') if revision==15 else (planning.get('timings') or {}).get('search_plus_selected_validation_wall_s'),
            full_model_calls=(planning.get('counts') or {}).get('total_objective_calls'),
            neural_generation_s=(planning.get('timings') or {}).get('neural_generation_s'),neural_update_s=(planning.get('timings') or {}).get('neural_update_s'),
            airsim_search_calls=summary.get('airsim_search_calls'),airsim_reference_execution_calls=summary.get('airsim_reference_execution_calls'),
            tracking_host_wall_s=summary.get('tracking_host_wall_s'),data_path=str(folder))
        row.update({k:metrics.get(k) for k in ('reference_duration_s','position_rmse_m','max_position_error_m','terminal_position_error_m','terminal_velocity_error_mps','speed_clipped_time_fraction','command_slew_clipped_time_fraction','quality_term')})
        csv_rows.append(row)
    from .curve_plots import render_plots
    figures=render_plots(records,output)
    title=route.get('policy_name','单目标')+' · 控制实验数据'
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title>'+STYLE+'<main><h1>'+title+'</h1>'
    text+=f'<p>协议{revision} · 原03实时框架 · 无障碍物 · 状态反馈，无视觉输入</p><section><a href="#definitions">数据怎么读、单位和判据</a> · <a href="指标中文.csv">中文指标CSV</a> · <a href="comparison.csv">程序读取CSV</a> · <a href="audit.json">独立重算回执</a>'
    text+=table(['方法/重复','执行状态','控制判据','参考时长/秒','RMSE/米','峰值/米','终端位置/米','终端速度/米每秒'],rows)
    text+='<p>峰值≤0.50米、终端位置≤0.25米、终端速度≤0.50米/秒，并满足全部过点规则，才算控制通过。执行完成与控制通过分别记录。相同轨迹的曲线会重叠。</p></section>'
    if route:
        text+='<section><h2>共同轨迹定义</h2><p>每段由端点位置、速度、加速度约束的五次基线，加保持端状态的六次中点位移扰动构成。未优化参考的扰动为0，因此实际次数不超过5；扰动族最高6次。Bézier/Hermite使用相同物理参数域。</p>'
        text+=f"<p>{len(route['waypoints'])}个航点联合搜索{4*len(route['waypoints'])}个变量：每段中点位移dx/dy/dz及基础时长T。位移范围为±(0.8,1.4,0.8)米；基础T为3至7秒，执行时长除以共同倍率{route['time_scale']}。底层控制器固定。</p></section>"
        text+='<section><h2>每个航点是否按要求通过？</h2>'+table(['方法','航点','规则','核验时刻/秒','位置偏差/米','实际速度/米每秒','判定'],waypoint_rows)+'</section>'
    for image,heading in figures:text+=f'<section><h2>{heading}</h2><p><a href="{image}.png">PNG高清图</a> · <a href="{image}.svg">SVG矢量图</a> · <a href="{image}.pdf">PDF矢量图</a></p><img src="{image}.png" alt="{heading}" loading="lazy"></section>'
    text+='<section><h2>图例、计时与原始日志</h2><p>B/H分别为Bézier/Hermite，0为未优化参考，Pair为扰动改进候选。虚线为参考/指令，实线为实际。两种曲线基可表示相同物理轨迹，换基不构成性能创新。</p>'
    text+=table(['方法','本次完整规划/秒' if revision==15 else '历史模型规划/秒','完整模型调用/次','本次跟踪墙钟/秒','数据目录'],[[r['method_name'],fmt(r['planning_wall_s']),r['full_model_calls'],fmt(r['tracking_host_wall_s']),r['data_path']] for r in csv_rows])
    text+=table(['方法','网络生成/秒','网络更新/秒','AirSim搜索调用/次','冻结参考执行/次'],[[r['method_name'],fmt(r['neural_generation_s']),fmt(r['neural_update_s']),r['airsim_search_calls'],r['airsim_reference_execution_calls']] for r in csv_rows])
    text+=('<p>协议15：到准备点后，在悬停期间重新进行所选方法的模型规划；规划墙钟含进程启动/初始化/搜索/训练/验证。AirSim只执行最终参考，AirSim搜索调用为0。准备/悬停与参考执行分开记录。</p>' if revision==15 else '<p>本次AirSim执行冻结参考；搜索在简化模型中完成。历史规划耗时与本次飞行时间分开统计。重复执行不是重新搜索。</p>')
    if revision==13:text+='<p>协议13执行时统一重定时，重定时后没有重新搜索；不能据此证明搜索整体提效。</p>'
    else:text+='<p>协议14/15的时间倍率已纳入模型搜索，逐点停稳与连续过点分别比较。当前为固定候选数的开发核验，尚未完成固定墙钟、多种子留出测试。</p>'
    text+='<small>图来自保存的日志。审计通过只说明记录一致，不等于控制门槛通过或真机验证。单次开发飞行不证明算法优越性。AI辅助开发。</small></section>'+glossary_section()+'</main></html>'
    (output/'index.html').write_text(text,encoding='utf-8')
    fields=list(csv_rows[0])
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(csv_rows)
    from .metric_explanations import METRICS
    names={key:label+' /'+unit for key,label,unit,meaning,rule in METRICS}
    names.update(method='方法代码',method_name='方法名称',policy='实验组',outcome='执行状态',control_passed='控制判据通过',task_success='任务成功',data_path='原始数据目录',
        neural_generation_s='网络候选生成 /秒',neural_update_s='网络更新 /秒',airsim_search_calls='AirSim搜索调用 /次',airsim_reference_execution_calls='冻结参考AirSim执行 /次')
    with (output/'指标中文.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow([names.get(k,k) for k in fields]);w.writerows([[row.get(k) for k in fields] for row in csv_rows])
    write_glossary(output/'字段说明')
    (output/'audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(p.read_bytes()!=source_bytes[p.name] for p in source_paths):raise ValueError('绘图期间源码变化，请重新生成；原始记录保留')
    (output/'provenance.json').write_text(json.dumps(dict(input_sha256=inputs,condition_fingerprint=first['condition_fingerprint'],source_sha256=hashlib.sha256(source_bytes['curve_results.py']).hexdigest(),plot_source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in source_bytes.items()}),indent=2),encoding='utf-8')
    return output/'index.html'


def render_campaign(campaign_path,output=None):
    """Keep every slot; partition differing conditions, never rank across them."""
    path=Path(campaign_path).resolve();data=json.loads(path.read_text(encoding='utf-8'));output=Path(output or path.parent/'data_view').resolve();output.mkdir(parents=True,exist_ok=True)
    groups={};rows=[];issues=[]
    for i,slot in enumerate(data['slots']):
        folder=Path(slot['run_dir']) if slot.get('run_dir') else None;metrics=slot.get('control_metrics') or {}
        rows.append([i+1,{'stop':'逐点停稳','pass':'连续过点'}.get(slot['policy'],slot['policy']),TITLES.get(slot['method'],slot['method']),slot['repeat'],slot['status'],
            '通过' if metrics.get('control_error_gates_passed') is True else '未通过' if metrics else '无有效分数',str(folder or '未产生运行日志')])
        if folder and (folder/'live_manifest.json').is_file() and (folder/'summary.json').is_file():
            manifest=json.loads((folder/'live_manifest.json').read_text(encoding='utf-8'));key=(slot['policy'],manifest['condition_fingerprint'])
            groups.setdefault(key,[]).append(folder)
        elif slot['status']=='COMPLETED':issues.append(f'第{i+1}项标为完成但缺少独立运行日志；不能计为已验证飞行。')
    links=[];first_by_policy={}
    for i,((policy,fingerprint),folders) in enumerate(groups.items(),1):
        target=output/(policy+'_'+str(i));manifest=json.loads((folders[0]/'live_manifest.json').read_text(encoding='utf-8'))
        if policy in first_by_policy:issues.extend(condition_differences(first_by_policy[policy],manifest))
        first_by_policy.setdefault(policy,manifest)
        try:
            page=render_results(folders,target);links.append('<li><a href="'+page.relative_to(output).as_posix()+'">'+('逐点停稳' if policy=='stop' else '连续过点')+f'：{len(folders)}条记录</a>（独立条件组{i}）</li>')
        except Exception as exc:issues.append(f'条件组{i}不能生成有效对比：{exc}')
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两组多点批量数据</title>'+STYLE+'<main><h1>两组多点实验 · 批量数据</h1><p>批量状态：'+html.escape(data['phase'])+'。下列两组任务分别比较，不合并排名。</p><section><ul>'+''.join(links)+'</ul><a href="字段说明/index.html">数据怎么读</a></section>'
    text+='<section><h2>全部任务，包括失败及未运行</h2>'+table(['序号','实验组','方法','重复','执行状态','控制判据','原始数据目录'],rows)+'</section>'
    if issues:text+='<section><h2>条件差异或证据缺口</h2><p>'+'</p><p>'.join(html.escape(v) for v in issues)+'</p></section>'
    text+='<section><p>相同冻结方案的执行重复不是独立优化训练；小规模飞行只核验系统和数据接口。来源：<code>'+html.escape(str(path))+'</code></p></section></main></html>'
    (output/'index.html').write_text(text,encoding='utf-8');write_glossary(output/'字段说明')
    (output/'provenance.json').write_text(json.dumps(dict(campaign=str(path),campaign_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),issues=issues),ensure_ascii=False,indent=2),encoding='utf-8')
    return output/'index.html'


def render_selection(directories,output):
    """Group an arbitrary user selection by protocol, rule and fingerprint."""
    directories=list(dict.fromkeys(str(Path(p).resolve()) for p in directories));output=Path(output).resolve()
    if not directories:raise ValueError('没有选择飞行记录')
    if any((Path(p)/'online_manifest.json').exists() for p in directories):
        if not all((Path(p)/'online_manifest.json').exists() for p in directories):raise ValueError('协议16连续在线任务不能与13/14/15的固定路线混比；请分开勾选。')
        from .online_results import render_comparison
        return render_comparison(directories,output)
    groups={};first=None;differences=[]
    for folder in directories:
        manifest=json.loads((Path(folder)/'live_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('protocol_revision') not in (13,14,15):raise ValueError('该入口目前只支持实时协议13/14/15，旧协议保留单独查看')
        key=(manifest['plan'].get('route',{}).get('policy','single'),manifest['protocol_revision'],manifest['condition_fingerprint'])
        groups.setdefault(key,[]).append(folder)
        if first and first['condition_fingerprint']!=manifest['condition_fingerprint']:differences.extend(condition_differences(first,manifest))
        first=first or manifest
    if len(groups)==1:return render_results(directories,output)
    output.mkdir(parents=True,exist_ok=True);links=[];issues=[]
    for i,((policy,revision,fingerprint),folders) in enumerate(groups.items(),1):
        label={'stop':'逐点停稳','pass':'连续过点','single':'单目标'}.get(policy,policy)+f' · 协议{revision} · {len(folders)}条记录'
        try:
            page=render_results(folders,output/(policy+'_'+str(i)));links.append('<li><a href="'+page.relative_to(output).as_posix()+'">'+html.escape(label)+'</a></li>')
        except Exception as exc:issues.append(label+'：'+str(exc))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>所选记录分组对比</title>'+STYLE+'<main><h1>所选记录 · 按条件分组</h1><p>共选择'+str(len(directories))+'条记录；不同协议、规则、参数或源码分开展示，不产生跨条件总排名。</p><section><ul>'+''.join(links)+'</ul></section>'
    if differences:page+='<section><h2>具体条件差异</h2><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+html.escape('\n'.join(dict.fromkeys(differences)))+'</pre></section>'
    if issues:page+='<section><h2>未生成有效对比的记录组</h2><p>'+'</p><p>'.join(html.escape(s) for s in issues)+'</p></section>'
    page+='<section><h2>完整选择清单</h2>'+table(['原始目录'],[[p] for p in directories])+'</section></main></html>'
    (output/'index.html').write_text(page,encoding='utf-8');(output/'selection.json').write_text(json.dumps(dict(directories=directories,groups=len(groups),issues=issues,differences=list(dict.fromkeys(differences))),ensure_ascii=False,indent=2),encoding='utf-8')
    return output/'index.html'
