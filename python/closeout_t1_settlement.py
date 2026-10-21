"""Update the two chapter drafts from verified local development evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from analyze_t1_identity_flight import digest

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/research-plan-20261004/任务一严格停稳与清零预算验证结果.md'


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def closeout(analysis, figures, folder):
    analysis,figures,folder=map(lambda p:Path(p).resolve(),(analysis,figures,folder))
    if DOC.exists() or (folder/'receipt.json').exists():raise FileExistsError('CLOSEOUT_ALREADY_EXISTS')
    data=json.loads(analysis.read_text(encoding='utf-8')); runs=data['runs']
    primary=[r for r in runs if r['architecture'].endswith('V7P3')]
    motion=[r for r in primary if r['mode']=='SHORT_XY']
    if len(primary)!=3 or len(motion)!=2:raise ValueError('ONE_ZERO_AND_TWO_INDEPENDENT_MOTION_RUNS_REQUIRED')
    for r in primary:
        if (r['base_evidence']['independent_integration_qualification']!='PASS_CONTROLLED_DEVELOPMENT_SCOPE'
                or r['feedback_stop_audit']['status']!='PASS_INDEPENDENT_STRICT_FEEDBACK_STOP_AUDIT'
                or r['zero_hold_audit']['status']!='PASS_ACTUAL_STATIONARY_ZERO_HOLD_AUDIT'
                or r['measured_grid_check']['status']!='PASS_COMPLETE_SAMPLED_VISIBLE_GRID'
                or r['actual_clear_admission_audit']['status']!='PASS_DECLARED_ACTUAL_CLEAR_RESERVE_AUDIT'
                or not r['endpoint_scale_status'].startswith('PASS')):
            raise ValueError('LATEST_SOURCE_DEVELOPMENT_GATE_FAILED')
    f=json.loads((figures/'artifact_receipt.json').read_text(encoding='utf-8'))
    if f['input_analysis']['sha256']!=digest(analysis) or f['font']['glyph_coverage']!='PASS' or any(
            x['text_bounds_status']!='PASS' for x in f['layout_checks']):raise ValueError('FIGURE_INPUT_FONT_OR_LAYOUT_FAILED')
    if (f['visual_review']['status']!='PASS_RENDERED_PNG_REVIEW'
            or len(f['visual_review']['reviewed_pngs'])!=len(runs)):
        raise ValueError('ALL_RENDERED_PNGS_REQUIRE_RECORDED_VISUAL_REVIEW')
    for artifact in f['artifacts']:
        if digest(Path(artifact['path']))!=artifact['sha256']:
            raise ValueError('FIGURE_ARTIFACT_CHANGED')
    tests_path=folder/'final_tests.stderr.log'
    tests_text=tests_path.read_text(encoding='utf-8')
    tests_match=re.search(r'Ran (\d+) tests in ([\d.]+)s',tests_text)
    if not tests_match or int(tests_match[1])!=392 or not tests_text.rstrip().endswith('OK'):
        raise ValueError('FINAL_SOFTWARE_REGRESSION_NOT_PASSED')
    regression=dict(status='PASS_SOFTWARE_REGRESSION',tests=392,wall_s=float(tests_match[2]),
        path=str(tests_path),sha256=digest(tests_path))
    idle_path=folder/'final_idle_snapshot.json'
    idle=json.loads(idle_path.read_text(encoding='utf-8-sig'))
    if (idle['status']!='PASS_IDLE_AT_OBSERVATION' or idle['engine_process_count']!=0
            or idle['rpc_connection_count']!=0):raise ValueError('FINAL_IDLE_SNAPSHOT_FAILED')
    before=json.loads((folder/'before.json').read_text(encoding='utf-8'))
    before_map={Path(r['path']).resolve():r['text'] for r in before['files']}
    for path,prior_text in before_map.items():
        if path.read_bytes().decode('utf-8')!=prior_text:
            raise ValueError('CANONICAL_DOCUMENT_CHANGED_SINCE_START')
    task_path=ROOT/'configs/thesis_simulation_tasks.json'; protocol_path=ROOT/'configs/t1_completion_protocol_v1.json'
    old_task=json.loads(before_map[task_path]); old_protocol=json.loads(before_map[protocol_path])
    task=json.loads(task_path.read_text(encoding='utf-8')); protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if task['tasks'][1:]!=old_task['tasks'][1:] or protocol!=old_protocol:
        raise ValueError('UNEXPECTED_PRIOR_TASK_OR_PROTOCOL_CHANGE')
    source_plans=[json.loads((Path(r['run'])/'flight/report.json').read_text(encoding='utf-8'))['configuration'] for r in primary]
    if len({json.dumps(p['source_sha256'],sort_keys=True) for p in source_plans})!=1:raise ValueError('PRIMARY_SOURCE_VERSIONS_DIFFER')
    for plan in source_plans:
        for source in plan['source_inventory']+plan['legacy_source_inventory']:
            if digest(Path(source['path']))!=source['sha256'] or digest(Path(source['snapshot_path']))!=source['sha256']:
                raise ValueError('RUNTIME_OR_LEGACY_SOURCE_CHANGED')
        for field,expected in (
                ('weights','92b0dc0f4e17ff14e9c1e110b9cad9ef249274136ee9a28d99b7885dfb396844'),
                ('training_receipt','f98db6fb0f916bd00ea74ee36596fb222d662f3a913d36610d9e96aa7c4b22a5')):
            if plan[field+'_sha256']!=expected or digest(Path(plan[field+'_path']))!=expected:
                raise ValueError('PRIMARY_MODEL_OR_TRAINING_RECEIPT_CHANGED')
    chapter_paths=[ROOT/'docs/thesis'/n for n in ('第二章_问题建模与仿真平台.md','第三章_单机感知轨迹优化.md')]
    if any('### AI协助说明' not in p.read_text(encoding='utf-8') for p in chapter_paths):
        raise ValueError('CHAPTER_AI_FOOTER_MISSING')
    new=[r for r in runs if not r['architecture'].endswith('V6P1')]
    all_raw=[json.loads((Path(r['run'])/'flight/report.json').read_text(encoding='utf-8')) for r in new]
    if any(not r['base_evidence']['engine_closed_after_safe_completion'] or not r['base_evidence']['fresh_ground_closure_verified'] for r in new):
        raise ValueError('NEW_FLIGHT_GROUND_CLOSURE_UNVERIFIED')
    labels={'t1_settlement_zero_v7_20261006':'V7零输出：阶段切换失锁',
        't1_settlement_zero_v7p1_20261006':'V7P1零输出',
        't1_settlement_visible20_v7p1_20261006':'V7P1运动20秒：指令记录顺序失败',
        't1_settlement_zero_v7p2_20261006':'V7P2零输出',
        't1_settlement_visible20_v7p2_20261006':'V7P2运动20秒：尺度未达到',
        't1_settlement_visible60_v7p2_20261006':'V7P2运动60秒：实际清零越界',
        't1_settlement_zero_v7p3_20261006':'V7P3零输出',
        't1_settlement_visible60_v7p3_20261006':'V7P3运动60秒',
        't1_settlement_visible60_v7p3_rep1_20261006':'V7P3运动60秒独立重复'}
    lines=['| 开发试次 | 保存/计划 | 正确/无输出/错选 | 非零目标派发 | 净XY位移（毫米） | 独立集成/严格停稳 | 原整帧晚到 |',
        '|---|---:|---:|---:|---:|---|---:|']
    for r in new:
        b=r['base_evidence']; ids=b['measured_identity_counts']; plan=r['plan_check']; late=r['timing_summary']['historical_full']['counts'].get('LATE',0)
        integration='通过' if b['independent_integration_qualification']=='PASS_CONTROLLED_DEVELOPMENT_SCOPE' else '失败'
        stopping='通过' if r['feedback_stop_audit']['status']=='PASS_INDEPENDENT_STRICT_FEEDBACK_STOP_AUDIT' else '失败'
        lines.append(f"| {labels[r['session_id']]} | {b['measured_frame_count']}/{plan.get('expected_main_slots',b['measured_frame_count'])} | {ids.get('CORRECT',0)}/{ids.get('NO_OUTPUT',0)}/{ids.get('WRONG_DISTRACTOR',0)} | {b['nonzero_target_command_count']} | {r['endpoint']['net_xy_displacement_m']*1000:.3f} | {integration}/{stopping} | {late} |")
    table='\n'.join(lines)
    nframes=sum(r['base_evidence']['measured_frame_count'] for r in motion)
    correct=sum(r['base_evidence']['measured_identity_counts'].get('CORRECT',0) for r in motion)
    no_output=sum(r['base_evidence']['measured_identity_counts'].get('NO_OUTPUT',0) for r in motion)
    nonzero=sum(r['base_evidence']['nonzero_target_command_count'] for r in motion)
    net='、'.join(f"{r['endpoint']['net_xy_displacement_m']*1000:.3f}" for r in motion)
    processing_late=sum(r['timing_summary']['frame_processing']['counts'].get('LATE',0) for r in motion)
    full_late=sum(r['timing_summary']['historical_full']['counts'].get('LATE',0) for r in motion)
    slow=sum(1 for r in motion for t in r['base_evidence']['sanitized_transport_trace'] if t.get('backend_completed_monotonic_s') is not None
        and t['backend_completed_monotonic_s']-t['backend_started_monotonic_s']>2.)
    transport_count=sum(r['base_evidence']['backend_wall_distribution']['count'] for r in motion)
    primary_summary=(f"两次V7P3独立UE会话均在冻结的60秒条件下达到20毫米门槛，净位移分别为{net}毫米。"
        f"共保存{nframes}/{nframes}个主测量帧，独立评分{correct}个正确、{no_output}个无输出、0个错选；"
        f"实际非零目标派发{nonzero}次，严格停稳、实际清零时效及派发预算均经独立核查。")
    sample_scope=("这里的300帧来自两个独立启动的受控开发会话，帧之间有时间关联，不能当作300个独立实验。"
        "独立会话重复支持该开发条件的可重复执行；正式随机种子和多场景效能估计仍待开展。")
    evidence=f'[逐帧、指令及自身状态独立分析]({analysis.as_posix()})'
    fig_index=next(i+1 for i,r in enumerate(runs) if r['session_id']=='t1_settlement_visible60_v7p3_20261006')
    figure=figures/f'settlement_{fig_index}_zh.png'
    method=("反馈制动以清零时无人机自身XY位置为参考，按自身速度及位置误差生成期望速度，再施加边界、速度和按真实前一指令时间计算的变化率限制。"
        "反馈期望速度上限0.015米/秒，实际指令仍受0.03米/秒全局上限约束。采样停稳要求XY速度不超过0.5毫米/秒、"
        "连续稳定至少0.10秒、清零后观察至少0.30秒、至少三个不同仿真状态时刻，且自身参考位置误差不超过2毫米。"
        "目标指令和清零后的自身状态反馈分别记账，反馈不续期目标图像或VLX源证据。")
    timing=(f"帧处理和控制停稳分别评价。原完整完成时间继续取两者最大值，仍与原0.4秒截止点比较。"
        f"最新两次60秒运动共有{processing_late}/{nframes}个帧处理晚到，原完整完成晚到{full_late}/{nframes}；"
        "分开计时不能将这些晚到改标为及时。控制停稳另与从声明租约结束起2秒的预先声明时限比较。")
    budget=("目标派发为实际清零预留0.12秒，但保留原观测截止点。令原观测截止点为当前RGB读回时间加0.5秒与复核截止点中的较早值，"
        "再扣除原0.02秒新鲜度余量；新版本将声明租约结束限制在这个原截止点之前0.12秒。剩余预算不足0.01秒时撤销该目标动作。"
        "这一策略缩短或拒绝动作，实际清零是否及时仍由日志核查。")
    failures=("V7等待停稳后再进入主段导致图像链间隔超限，20帧中6帧无输出；V7P1保持连续采图后修复了这个切换。"
        "V7P1运动中的148个诊断样本记录了查询后派发的反馈指令，均未进入最终稳定时窗，原分析失败保留；V7P2修正记录顺序。"
        "V7P2的20秒运动净位移7.550毫米，未达到20毫米标准。其60秒试次虽达到21.483毫米，"
        "一次实际清零超过原观测截止点，独立集成仍判失败；该版本已准备的重复未执行。"
        "V7P3增加派发前清零预算。首轮V7P3审计将两对同时间、同状态读回匹配到较早日志，初始失败报告保留；"
        "新审计依据冻结V1P4控制器的查询前指令字段及实际事件顺序找到对应读回，没有改变状态值、时间或停稳阈值。")
    failures+=("表中的独立集成通过仅表示其受控证据检查通过；V7另设的持续身份切换门控失败已保留，"
        "其6个无输出帧未被删除，也未据此启动运动试次。")
    interpretation=("严格停稳改变了短动作后的执行安排，端点也在严格采样停稳后记录。60秒条件根据早期20秒试次选择，"
        "因此这些结果支持本阶段受控闭环和运动尺度核验，不能把20秒未达标记录改判为通过，"
        "也不能用不同版本的净位移大小推断感知轨迹优劣。帧间连续身份、连续物理静止和硬实时均未由采样检查建立。")
    next_steps=("下一步在同一执行接口上补真实慢VLX回复、白条高光干扰和遮挡返回，继续保留所有失锁、无输出与晚到；"
        "随后单独冻结50/100毫米尺度及完整几何执行协议，统一固定跟随、距离规划、可见性规划和身份复核联合规划，"
        "开展YOLO检测后端及VLX开关消融。正式场景、参数、种子、全启动回合指标和独立重复另行冻结。"
        "任务一正式四方法效能、A3完整几何及A7—A10仍待补，六章结构与任务二、三安排保持原计划。")
    report_text=(f"# 任务一严格停稳与清零预算验证结果\n\n2026-10-06。单机、指定白色横条红气球，保留YOLO＋VLX链路。\n\n"
        f"## 1. 最新结果\n\n{primary_summary} {evidence}\n\n{sample_scope}\n\n{table}\n\n"
        f"本轮共启动{sum(r.get('new_flights',0) for r in new)}次新飞行，各次均完成落地、解除控制、暂停、新进程地面复核和自有UE关闭。"
        f"失败记录与原运行器通过状态分别保留。{evidence}\n\n## 2. 控制与时钟修正\n\n{method}\n\n{budget}\n\n{timing} {evidence}\n\n"
        f"## 3. 失败及复算\n\n{failures}\n\n{interpretation}\n\n"
        f"## 4. 中文图表\n\n![60秒试次的自身状态位移、速度、计时和目标派发]({figure.as_posix()})\n\n"
        f"全部图件保存PNG/PDF、图点映射、来源SHA-256、字体覆盖和布局检查。"
        "已逐张查看10幅PNG的中文、单位和布局；PDF保留同源导出，未单独渲染复查。"
        f"[图表核查记录]({(figures/'artifact_receipt.json').as_posix()})\n\n"
        f"## 5. 软件回归与收尾\n\n392项软件回归全部通过。"
        f"[测试日志]({tests_path.as_posix()})。"
        f"最终进程快照中UE进程与41451端口连接均为0。"
        f"[进程快照]({idle_path.as_posix()})。这个快照与各飞行的新进程落地复核分别保存。\n\n"
        f"## 6. 后续工作\n\n{next_steps}\n\n"
        f"最新两次运动保存{transport_count}个实际后端完成回执，其中超过2秒的回执{slow}个；未覆盖的真实延迟条件仍单独补测。{evidence}\n\n"
        "AI协助说明：AI辅助代码、独立复算和文字组织；所有数值来自实际AirSim记录，研究解释由作者审定。开发会话不混算正式方法成功率。\n")
    DOC.write_text(report_text,encoding='utf-8')
    additions={ROOT/'docs/thesis/第二章_问题建模与仿真平台.md':
        f"### 2026-10-06 严格采样停稳与实际清零预算\n\n{method}\n\n{budget}\n\n{timing} {evidence}\n\n{primary_summary} {evidence}\n\n",
        ROOT/'docs/thesis/第三章_单机感知轨迹优化.md':
        f"### 2026-10-06 严格停稳后的尺度验证与失败分析\n\n{primary_summary} {evidence}\n\n"
        f"{sample_scope}\n\n表3-8 新版开发试次及保留的失败；完整完成的原0.4秒截止点不变。\n\n{table}\n\n"
        f"{failures} {evidence}\n\n{timing}\n\n"
        f"![图3-9 严格停稳与清零预算修复后的60秒运动，中文标注]({figure.as_posix()})\n\n"
        f"{interpretation}\n\n{next_steps}\n\n"}
    for path,addition in additions.items():
        text=path.read_text(encoding='utf-8'); marker='### AI协助说明'
        if marker not in text:raise ValueError('CHAPTER_AI_FOOTER_MISSING')
        path.write_text(text.replace(marker,addition+marker,1),encoding='utf-8')
    note=f"\n\n### 2026-10-06 严格停稳与清零预算验证\n\n{primary_summary} {evidence}\n\n{timing}\n\n{next_steps}\n\n[本轮完整记录]({DOC.as_posix()})\n"
    for name in ('硕士论文章节任务与图表安排.md','任务一论文内容与完成标准.md'):
        path=ROOT/'docs/research-plan-20261004'/name; path.write_text(path.read_text(encoding='utf-8')+note,encoding='utf-8')
    readme=ROOT/'README.md';readme.write_text(readme.read_text(encoding='utf-8')+note,encoding='utf-8')
    summary=dict(date='2026-10-06',status='STRICT_STOP_AND_CLEAR_BUDGET_DEVELOPMENT_VALIDATED_FORMAL_T1_PENDING',
        latest_runtime='V7P3',latest_controller='V1P4',analysis=str(analysis),analysis_sha256=digest(analysis),
        figure_receipt=str(figures/'artifact_receipt.json'),report=str(DOC),latest_motion_main_frames=nframes,
        latest_motion_correct_frames=correct,latest_motion_no_output_frames=no_output,
        latest_motion_nonzero_commands=nonzero,latest_motion_net_xy_m=[r['endpoint']['net_xy_displacement_m'] for r in motion],
        frame_processing_late_count=processing_late,historical_full_late_count=full_late,
        same_source_independent_motion_runs=2,new_flights_this_turn=sum(r.get('new_flights',0) for r in new),
        software_regression=regression,final_idle_snapshot=idle,
        primary_source_sha256=source_plans[0]['source_sha256'],formal_efficacy_results=False,whole_thesis_complete=False,
        unused_prepared_repeat='configs/t1_settlement_visible60_v7p2_rep1_20261006; not executed after actual-clear failure')
    task['updated']='2026-10-06';task['active_stage']='task_1_slow_reply_attribute_occlusion_and_larger_scale_then_frozen_formal_comparisons'
    task['tasks'][0]['status']=summary['status'];task['tasks'][0]['next_gate']='slow_reply_attribute_occlusion_then_50_100mm_full_geometry_and_formal_four_methods'
    task['tasks'][0]['strict_settlement_development_20261006']=summary
    protocol['latest_strict_settlement_development']=summary
    save(task_path,task);save(protocol_path,protocol)
    if task['tasks'][1:]!=old_task['tasks'][1:]:raise ValueError('T2_T3_CHANGED')
    if {k:v for k,v in protocol.items() if k!='latest_strict_settlement_development'}!=old_protocol:raise ValueError('OLD_FORMAL_PROTOCOL_KEYS_CHANGED')
    files=list(additions)+[DOC,readme,task_path,protocol_path]+[ROOT/'docs/research-plan-20261004'/n for n in
        ('硕士论文章节任务与图表安排.md','任务一论文内容与完成标准.md','任务一严格停稳与异步完成验证方案.md')]
    receipt=dict(kind='T1_STRICT_SETTLEMENT_DEVELOPMENT_CLOSEOUT',status='PASS_BOUNDED_DEVELOPMENT_CLOSEOUT',
        created_utc=datetime.now(timezone.utc).isoformat(),summary=summary,T2_T3_unchanged=True,
        old_formal_protocol_keys_unchanged=True,all_primary_runtime_and_legacy_sources_verified=True,
        primary_weights_sha256=source_plans[0]['weights_sha256'],primary_training_receipt_sha256=source_plans[0]['training_receipt_sha256'],
        software_regression=regression,final_idle_snapshot=idle,figure_visual_review=f['visual_review'],
        all_new_main_frames=sum(len(r['frames']) for r in all_raw),all_new_acquisition_frames=sum(len(r['acquisition_frames']) for r in all_raw),
        documents=[dict(path=str(p),sha256=digest(p)) for p in files],
        input_artifacts=[dict(path=str(p),sha256=digest(p)) for p in
            (analysis,figures/'artifact_receipt.json',folder/'before.json',tests_path,folder/'final_tests.stdout.log',idle_path)],
        closeout_source_sha256=digest(Path(__file__)),formal_efficacy_results=False,whole_thesis_complete=False)
    save(folder/'receipt.json',receipt)
    print('Updated chapters, evidence report and task state; formal comparisons remain pending.')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);p.add_argument('--folder',type=Path,required=True)
    args=p.parse_args();closeout(args.analysis,args.figures,args.folder)
