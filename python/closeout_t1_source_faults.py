"""Write bounded source-fault development results into the existing chapters."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import re
from analyze_t1_identity_flight import digest

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def closeout(analysis,figures,folder):
    analysis,figures,folder=[Path(p).resolve() for p in (analysis,figures,folder)]
    doc=ROOT/'docs/research-plan-20261004/任务一源证据扰动验证结果.md'
    if doc.exists() or (folder/'receipt.json').exists():raise FileExistsError('SOURCE_FAULT_CLOSEOUT_ALREADY_EXISTS')
    data=json.loads(analysis.read_text(encoding='utf-8'));runs=data['runs']
    if not runs:raise ValueError('EXECUTED_SOURCE_FAULT_RUNS_REQUIRED')
    fpath=figures/'artifact_receipt.json';f=json.loads(fpath.read_text(encoding='utf-8'))
    if (f['input_analysis']['sha256']!=digest(analysis) or f['font']['glyph_coverage']!='PASS'
            or any(x['text_bounds_status']!='PASS' for x in f['layout_checks'])
            or f['visual_review']['status']!='PASS_RENDERED_PNG_REVIEW'
            or len(f['visual_review']['reviewed_pngs'])!=len(runs)):raise ValueError('CHINESE_FIGURE_EVIDENCE_NOT_VERIFIED')
    for row in f['artifacts']:
        if digest(Path(row['path']))!=row['sha256']:raise ValueError('FIGURE_ARTIFACT_CHANGED')
    before=json.loads((folder/'before.json').read_text(encoding='utf-8'))
    for row in before['files']:
        if digest(Path(row['path']))!=row['sha256']:raise ValueError('CANONICAL_FILE_CHANGED_DURING_EXPERIMENT')
    declared=json.loads((folder/'predeclared_plans.json').read_text(encoding='utf-8'))
    if digest(Path(declared['protocol_path']))!=declared['protocol_sha256']:
        raise ValueError('PREDECLARED_SOURCE_FAULT_PROTOCOL_CHANGED')
    planned={row['session_id']:row for row in declared['plans']}
    if {r['session_id'] for r in runs}!=set(planned) or len(runs)!=len(planned):
        raise ValueError('ALL_FIVE_PREDECLARED_SOURCE_FAULT_SESSIONS_REQUIRED')
    for r in runs:
        row=planned[r['session_id']]
        if digest(Path(row['plan_path']))!=row['sha256'] or r['source_perturbation']['condition']!=row['condition']:
            raise ValueError('SOURCE_FAULT_PLAN_CHANGED')
    for name in ('第二章_问题建模与仿真平台.md','第三章_单机感知轨迹优化.md'):
        if '### AI协助说明' not in (ROOT/'docs/thesis'/name).read_text(encoding='utf-8'):
            raise ValueError('CHAPTER_AI_FOOTER_MISSING')
    tests_path=folder/'final_tests.stderr.log';text=tests_path.read_text(encoding='utf-8')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',text)
    if not match or int(match[1])<407 or not text.rstrip().endswith('OK'):raise ValueError('SOURCE_FAULT_REGRESSION_NOT_PASSED')
    idle_path=folder/'final_idle_snapshot.json';idle=json.loads(idle_path.read_text(encoding='utf-8-sig'))
    if idle['engine_process_count']!=0 or idle['rpc_connection_count']!=0:raise ValueError('FINAL_IDLE_SNAPSHOT_NOT_VERIFIED')
    reports=[]
    for r in runs:
        if not r['base_evidence']['fresh_ground_closure_verified'] or not r['base_evidence']['engine_closed_after_safe_completion']:
            raise ValueError('FRESH_GROUND_CLOSURE_REQUIRED_BEFORE_CLOSEOUT')
        raw=json.loads((Path(r['run'])/'flight/report.json').read_text(encoding='utf-8'));reports.append(raw)
        for row in raw['configuration']['source_inventory']+raw['configuration']['legacy_source_inventory']:
            if digest(Path(row['path']))!=row['sha256'] or digest(Path(row['snapshot_path']))!=row['sha256']:
                raise ValueError('RUNTIME_OR_LEGACY_SOURCE_CHANGED')
    names={
        'baseline':'零输出门控','delivery_delay':'回复延迟交付','occlusion':'遮挡与返回','glint':'小白斑干扰'}
    table=['| 条件/会话 | 保存/计划主帧 | 正确/无输出/其他 | 净XY位移（毫米） | 扰动独立判定 | 运动尺度/零漂移 | 原完整完成晚到 |',
           '|---|---:|---:|---:|---|---|---:|']
    for r in runs:
        b=r['base_evidence'];c=b['measured_identity_counts'];other=sum(n for k,n in c.items() if k not in ('CORRECT','NO_OUTPUT'))
        label=names[r['source_perturbation']['condition']]+('独立重复' if '_rep' in r['session_id'] else '')
        verdict='通过' if r['independent_fault_qualification'].startswith('PASS') else r['independent_fault_qualification']
        scale='通过' if r['endpoint_scale_status'].startswith('PASS') else r['endpoint_scale_status']
        table.append(f"| {label} | {b['measured_frame_count']}/{r['plan_check']['expected_main_slots']} | {c.get('CORRECT',0)}/{c.get('NO_OUTPUT',0)}/{other} | {r['endpoint']['net_xy_displacement_m']*1000:.3f} | {verdict} | {scale} | {r['timing_summary']['historical_full']['counts'].get('LATE',0)} |")
    table='\n'.join(table);nmain=sum(r['base_evidence']['measured_frame_count'] for r in runs)
    nplanned=sum(r['plan_check']['expected_main_slots'] for r in runs);nacq=sum(len(p['acquisition_frames']) for p in reports)
    allpassed=all(r['independent_fault_qualification'].startswith('PASS') for r in runs)
    motion=[r for r in runs if r['mode']=='SHORT_XY'];scale_failed=[r['session_id'] for r in motion if not r['endpoint_scale_status'].startswith('PASS')]
    service_count=sum(r['reply_delivery_audit']['actual_service_wall_distribution']['count'] for r in runs)
    slow_count=sum(r['reply_delivery_audit']['natural_service_over_two_s_count'] for r in runs)
    holds=[h for r in runs for h in r['reply_delivery_audit']['holds']]
    late=sum(r['timing_summary']['historical_full']['counts'].get('LATE',0) for r in runs)
    processing_late=sum(r['timing_summary']['frame_processing']['counts'].get('LATE',0) for r in runs)
    identity_totals={}
    late_limiters={}
    for r in runs:
        for key,value in r['base_evidence']['measured_identity_counts'].items():
            identity_totals[key]=identity_totals.get(key,0)+value
        for row in r['frame_timing_rows']:
            if row['recorded_timing']['historical_full']['status']=='LATE':
                key=row['full_limiter'];late_limiters[key]=late_limiters.get(key,0)+1
    evidence=f'[逐帧、媒体、源链与控制独立分析]({analysis.as_posix()})'
    overview=(f"本轮实际启动{sum(r['new_flights'] for r in runs)}个AirSim飞行会话，保存主帧{nmain}/{nplanned}个，"
        f"初始化帧另计{nacq}个。主帧独立评分为正确{identity_totals.get('CORRECT',0)}帧、无输出{identity_totals.get('NO_OUTPUT',0)}帧、"
        f"其他{sum(n for key,n in identity_totals.items() if key not in ('CORRECT','NO_OUTPUT'))}帧。"
        f"原完整完成晚到{late}/{nmain}，帧处理晚到{processing_late}/{nmain}。"
        "每个会话分别完成新进程地面复核和自有UE关闭；帧具有时间关联，独立实验单位是会话。")
    delivery=(f"真实后端完成回执共{service_count}个，其中自然耗时超过2秒的回执{slow_count}个。"
        f"人工延迟交付另计{len(holds)}次，预先声明的等待为3秒与9秒，均作用于真实VLX返回结果。"
        "原图像来源、原回复时刻和结果内容保持，人工等待时刻单列；自然服务耗时和人工等待分别评价。")
    details=[];recovery_metrics=[]
    for r,raw in zip(runs,reports):
        cond=r['source_perturbation']['condition'];b=r['source_fault_behavior']
        if cond=='delivery_delay':
            details.append(f"{r['session_id']}："+'；'.join(f"{h['declared_hold_s']:g}秒等待对应终结状态{h['terminal_status']}、原因{h['terminal_reason']}" for h in r['reply_delivery_audit']['holds'])+'。')
        elif cond=='occlusion':
            if b['first_correct_return_frame'] is None:
                details.append(f"遮挡保存{b['hidden_frames']}帧；本会话没有建立返回后的正确输出。")
            else:
                transitions=[x for x in raw['phase_transitions'] if x['phase_id']=='return']
                if len(transitions)!=1 or transitions[0]['scene_fault']['occluder_active']:
                    raise ValueError('ACTUAL_OCCLUSION_CLEAR_RECEIPT_REQUIRED')
                cleared=transitions[0]['completed_monotonic_s'];fid=b['first_correct_return_frame']
                clock=next(x for x in r['frame_timing_rows'] if x['frame_id']==fid)['recorded_timing']
                delays={key:clock[key]['monotonic_s']-cleared for key in ('capture','gate','frame_processing','historical_full')}
                if any(value<0 for value in delays.values()) or not delays['capture']<=delays['gate']<=delays['frame_processing']<=delays['historical_full']:
                    raise ValueError('OCCLUSION_RECOVERY_CLOCKS_NOT_ORDERED')
                recovery_metrics.append(dict(session_id=r['session_id'],first_correct_return_frame=fid,
                    actual_occluder_clear_completed_monotonic_s=cleared,after_actual_clear_s=delays,
                    nominal_return_capture_delay_s=b['return_first_correct_relative_s'],clock_scope='saved host monotonic clocks'))
                details.append(f"遮挡保存{b['hidden_frames']}帧，遮挡拒绝违例{b['hidden_refusal_failures']}个。"
                    f"以解除遮挡调用实际完成时刻为起点，首个正确帧的图像回读间隔为{delays['capture']:.3f}秒，"
                    f"目标决策完成间隔为{delays['gate']:.3f}秒，帧处理完成间隔为{delays['frame_processing']:.3f}秒，"
                    f"包含停稳的完整完成间隔为{delays['historical_full']:.3f}秒。首次正确帧按保存样本识别，采样间隔为0.4秒。")
        elif cond=='glint':
            white=[x['distractor_white_pixels'] for x in r['source_fault_frame_rows'] if x['phase_id']=='glint']
            details.append(f"小白斑阶段保留{len(white)}个主帧，干扰气球分割区域内的白像素范围为{min(white) if white else 0}—{max(white) if white else 0}。"
                "该条件测试局部白斑外观，不建立物理光照或完整相同属性干扰的结论。")
    corrected_rows=[dict(session_id=r['session_id'],rows=r['review_handoff'].get(
        'legacy_audit_rows_replaced_by_explicit_saved_chain_proof',[])) for r in runs
        if r['review_handoff'].get('legacy_audit_rows_replaced_by_explicit_saved_chain_proof')]
    if corrected_rows:
        details.append("原交接审计要求待确认复核编号等于当前HTTP请求编号。延迟试次中，已接纳的复核仍在当前帧确认时，"
            "下一条HTTP请求已经提交，两个编号不同。初始失败报告保留；新版离线审计从已保存的已接纳回复出发，"
            "逐个实际当前帧复算唯一候选链接、原源时效和确认次数，不使用未来帧，不改飞行代码或任何判据。"
            f"[复算后再重复的门控记录]({(folder/'delivery_bound_review_pre_repeat_gate.json').as_posix()})。")
    issues=[dict(session_id=r['session_id'],qualification=r['independent_fault_qualification'],
        behavior_issues=r['source_fault_behavior']['issues'],scene_issues=r['physical_scene_audit']['issues'],delivery_issues=r['reply_delivery_audit']['issues'])
        for r in runs if not r['independent_fault_qualification'].startswith('PASS')]
    scope=(f"扰动行为独立判定{'全部通过' if allpassed else '存在未通过条件，具体问题见分析日志'}。"
        f"运动试次仍按原20毫米净位移标准单独评价，未达标会话数为{len(scale_failed)}；"
        "拒绝旧证据和恢复正确输出不能替代运动尺度验证。控制参数沿用V7P3，原0.4秒完整完成截止点保持。"
        "这些结果属于少量受控开发会话，正式四方法效能、完整几何、YOLO/VLX消融和正式独立重复仍待补。")
    timing_analysis=(f"完整完成晚到帧中，{late_limiters.get('CONTROL_SETTLEMENT',0)}帧的最后完成环节是控制停稳，"
        f"另有{late_limiters.get('UNKNOWN',0)}帧没有确定最后完成环节，保留为未知。"
        "源证据拒绝、采样停稳和0.4秒完整完成分别对应不同约束。"
        "当前执行频繁进入采样停稳；后续较大尺度执行需要检验这一控制节奏对位移、观测连续性和完成时间的影响。")
    steps=(("下一步修复已记录的未通过项，再" if not allpassed or scale_failed else "下一步")+
        "单独冻结50/100毫米运动尺度及完整几何执行；"
        "再以统一接口、场景、种子、回合判据开展固定跟随、距离规划、可见性规划、身份复核联合规划及YOLO/VLX消融。")
    figures_md=[]
    for i,r in enumerate(runs,1):
        if r['source_perturbation']['condition']=='baseline':continue
        figures_md.append(f"![{names[r['source_perturbation']['condition']]}：身份、位移、完成时钟与回复耗时]({(figures/f'source_fault_{i}_zh.png').as_posix()})")
    doc.write_text(f"# 任务一源证据扰动验证结果\n\n2026-10-06。单机单目标，保留YOLO＋VLX。\n\n"
        f"## 1. 执行结果\n\n{overview} {evidence}\n\n{table}\n\n"
        f"## 2. 证据时间与行为分析\n\n{delivery} {evidence}\n\n"+'\n\n'.join(details)+f"\n\n{scope}\n\n"
        f"{timing_analysis} {evidence}\n\n"
        f"[原始失败及尺度未达标列表]({analysis.as_posix()})；源码、原判据和全部帧记录保留。\n\n"
        "## 3. 中文图表\n\n"+'\n\n'.join(figures_md)+f"\n\n[字体、布局、图点映射与PNG视觉核查记录]({fpath.as_posix()})。PDF同源导出，未单独渲染复查。\n\n"
        f"## 4. 软件回归与后续工作\n\n{match[1]}项软件回归通过。[测试日志]({tests_path.as_posix()})。"
        f"最终进程快照中UE与41451端口连接均为0。[快照]({idle_path.as_posix()})。\n\n{steps}\n\n"
        "AI协助说明：AI辅助实现、独立复算与文字整理；数值来自真实AirSim记录，人工故障显式标注，研究解释由作者审定。\n",encoding='utf-8')
    chapter2=ROOT/'docs/thesis/第二章_问题建模与仿真平台.md';chapter3=ROOT/'docs/thesis/第三章_单机感知轨迹优化.md'
    additions={chapter2:f"### 2026-10-06 源证据扰动与时钟分离\n\n{delivery} {evidence}\n\n"
        "后台等待由唯一复核线程执行，不占用前台采图或独立控制线程。源RGB、真实回复时刻与后来交付时刻分别保留，"
        "过期证据的拒绝按原链期限检查。遮挡物和干扰材料只用于场景执行，真实分割只用于独立评价。\n\n",
        chapter3:f"### 2026-10-06 延迟交付、遮挡返回与小白斑干扰\n\n{overview} {evidence}\n\n"
        f"表3-9 源证据扰动开发试次；尺度和扰动行为分别判断。\n\n{table}\n\n{delivery}\n\n"+
        '\n\n'.join(details)+f"\n\n{scope}\n\n{timing_analysis} {evidence}\n\n{steps}\n\n[完整结果及中文图表]({doc.as_posix()})\n\n"}
    for p,addition in additions.items():
        old=p.read_text(encoding='utf-8');marker='### AI协助说明'
        if marker not in old:raise ValueError('CHAPTER_AI_FOOTER_MISSING')
        p.write_text(old.replace(marker,addition+marker,1),encoding='utf-8')
    note=f"\n\n### 2026-10-06 源证据扰动验证\n\n{overview} {evidence}\n\n{scope}\n\n{steps}\n\n[本轮完整记录]({doc.as_posix()})\n"
    other_docs=[ROOT/'README.md']+[ROOT/'docs/research-plan-20261004'/n for n in ('硕士论文章节任务与图表安排.md','任务一论文内容与完成标准.md')]
    for p in other_docs:p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
    tp=ROOT/'configs/thesis_simulation_tasks.json';pp=ROOT/'configs/t1_completion_protocol_v1.json'
    t=json.loads(tp.read_text(encoding='utf-8'));p=json.loads(pp.read_text(encoding='utf-8'));old_tasks=json.loads(json.dumps(t['tasks'][1:]));old_protocol=json.loads(json.dumps(p))
    status='SOURCE_FAULT_DEVELOPMENT_MEASURED_FORMAL_T1_PENDING' if allpassed else 'SOURCE_FAULT_DEVELOPMENT_NEEDS_REMEDIATION_FORMAL_T1_PENDING'
    summary=dict(date='2026-10-06',status=status,analysis_path=str(analysis),analysis_sha256=digest(analysis),report_path=str(doc),
        figure_receipt_path=str(fpath),actual_flights=sum(r['new_flights'] for r in runs),all_main_frames=nmain,all_planned_main_slots=nplanned,
        all_acquisition_frames=nacq,all_source_fault_conditions_passed=allpassed,failed_conditions=issues,scale_failed_sessions=scale_failed,
        all_main_identity_counts=identity_totals,late_full_completion_limiters=late_limiters,
        original_failed_audit_preserved_and_bound_pending_review_recomputed=corrected_rows,
        actual_occlusion_recovery_clocks=recovery_metrics,
        natural_service_reply_count=service_count,natural_service_over_two_s_count=slow_count,controlled_delivery_holds_count=len(holds),
        complete_late_frames=late,frame_processing_late_frames=processing_late,regression_tests=int(match[1]),
        formal_efficacy_results=False,whole_thesis_complete=False)
    next_gate=('source_fault_remediation_then_50_100mm_full_geometry_and_formal_methods' if not allpassed or scale_failed
        else '50_100mm_motion_scale_full_geometry_then_formal_methods')
    t.update(updated='2026-10-06',active_stage='task_1_'+next_gate)
    t['tasks'][0].update(status=status,next_gate=next_gate,source_fault_development_20261006=summary)
    p['latest_source_fault_development']=summary;save(tp,t);save(pp,p)
    assert t['tasks'][1:]==old_tasks
    assert {k:v for k,v in p.items() if k!='latest_source_fault_development'}==old_protocol
    documents=[doc,chapter2,chapter3,tp,pp]+other_docs
    receipt=dict(kind='T1_SOURCE_FAULT_DEVELOPMENT_CLOSEOUT',created_utc=datetime.now(timezone.utc).isoformat(),
        status='PASS_BOUNDED_EVIDENCE_CLOSEOUT',summary=summary,T2_T3_unchanged=True,old_formal_protocol_keys_unchanged=True,
        runtime_and_legacy_sources_verified=True,figure_visual_review=f['visual_review'],regression_tests=int(match[1]),final_idle_snapshot=idle,
        documents=[dict(path=str(p),sha256=digest(p)) for p in documents],input_artifacts=[dict(path=str(p),sha256=digest(p)) for p in
            (analysis,fpath,tests_path,folder/'final_tests.stdout.log',idle_path,folder/'before.json',folder/'predeclared_plans.json',
             folder/'zero_pre_motion_gate.json',folder/'delivery_bound_review_pre_repeat_gate.json',
             folder/'all_conditions_pre_repeat_gate.json',folder/'regression_plan.json',folder/'final_idle_attempt_1.json')],
        closeout_source_sha256=digest(Path(__file__)),formal_efficacy_results=False,whole_thesis_complete=False)
    save(folder/'receipt.json',receipt);print(status,nmain,nplanned,int(match[1]));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);p.add_argument('--folder',type=Path,required=True)
    a=p.parse_args();closeout(a.analysis,a.figures,a.folder)
