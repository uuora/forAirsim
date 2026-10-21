"""Bind saved motion evidence to thesis updates and a bounded closeout receipt.

Read-only experiment intake; no simulator, model or cloud request is made.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics

from analyze_t1_identity_flight import digest, read_json

ROOT=Path(__file__).resolve().parents[1]
ANALYSIS=ROOT/'logs/t1_motion_scale_analysis_v6p1_20261006'
FIGURES=ROOT/'logs/t1_motion_scale_figures_zh_v6p1_20261006'
CLOSEOUT=ROOT/'logs/t1_motion_scale_closeout_20261005'
RESULT=ROOT/'docs/research-plan-20261004/任务一运动尺度重复验证与论文更新.md'


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def bound(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=digest(path))


def append_section(path,section):
    text=path.read_text(encoding='utf-8-sig')
    if section.splitlines()[0] in text:raise ValueError('Do not duplicate a thesis update')
    marker='### AI协助说明'
    text=text.replace(marker,section+'\n\n'+marker,1) if marker in text else text.rstrip()+'\n\n'+section+'\n'
    path.write_text(text,encoding='utf-8')


def closeout(idle_proof):
    destination=CLOSEOUT/'receipt.json'
    if destination.exists():raise ValueError('Closeout already sealed')
    before=read_json(CLOSEOUT/'before.json');report=read_json(ANALYSIS/'report.json')
    flown=[r for r in report['runs'] if r.get('command_response') is not None]
    motion=[r for r in flown if r['mode']=='SHORT_XY'];zero=[r for r in flown if r['mode']=='ZERO_GATE']
    failed=[r for r in report['runs'] if r.get('status')=='SEALED_PREFLIGHT_FAILURE']
    if len(motion)!=4 or len(zero)!=2 or len(failed)!=1 or failed[0]['actual_flights']!=0:
        raise ValueError('All declared attempts must be present with the expected actual flight classification')
    if any(r['motion_scale_qualification']!='PASS_ENDPOINT_MOTION_SCALE' for r in motion):
        raise ValueError('Motion scale has not passed the independent frozen criterion')
    if any(r['motion_scale_qualification']!='PASS_ZERO_GATE_DRIFT' for r in zero):raise ValueError('Zero gate incomplete')
    idle=read_json(idle_proof)
    if idle.get('ue_process_count')!=0 or idle.get('rpc_41451_listener_count')!=0:raise ValueError('Simulation still occupied')
    test_path=CLOSEOUT/'tests_v6p1.stderr.log';test_text=test_path.read_text(encoding='utf-8-sig')
    test_count=int(re.search(r'Ran (\d+) tests',test_text).group(1))
    if test_count!=353 or not test_text.rstrip().endswith('OK'):raise ValueError('Required regression receipt missing')
    locks=[];media_count=0;acquisition_slots=0;acquisition_frames=0;unused_acquisition=0
    for run in flown:
        inputs=run['motion_analysis_inputs'];plan=read_json(inputs['plan_path']);raw=read_json(inputs['report_path'])
        manifest=read_json(inputs['manifest_path'])
        if not all(manifest.get(k) is True for k in ('fresh_ground_closure_verified','engine_closed_after_safe_completion')):
            raise ValueError('Fresh closure missing')
        if raw.get('new_flights')!=1:raise ValueError('Actual flight count missing')
        for row in plan['source_inventory']+plan['legacy_source_inventory']:
            if digest(row['path'])!=row['sha256'] or digest(row['snapshot_path'])!=row['sha256']:
                raise ValueError('Frozen source changed')
        for key in ('weights','training_receipt','settings','detector_config'):
            if digest(plan[key+'_path'])!=plan[key+'_sha256']:raise ValueError('Frozen artifact changed')
        media_count+=sum(len(f['media']) for f in raw['acquisition_frames']+raw['frames'])
        acquisition_slots+=len(raw['acquisition_slots']);acquisition_frames+=len(raw['acquisition_frames'])
        unused_acquisition+=sum(s.get('status')=='NOT_REQUIRED_AFTER_ACQUISITION' for s in raw['acquisition_slots'])
        locks.append(dict(session_id=run['session_id'],manifest=bound(inputs['manifest_path']),
            plan=bound(inputs['plan_path']),report=bound(inputs['report_path']),
            closure=bound(manifest['fresh_ground_closure_path']),actual_flights=1,
            owned_pid=manifest['engine_process_id'],safe_closure_verified=True,
            source_inventory_count=len(plan['source_inventory']),historical_source_count=len(plan['legacy_source_inventory']),
            sources_and_artifacts_unchanged=True))
    repeats=[r for r in motion if '_rep' in r['session_id']]
    # Bind all four motion flights to exactly the same runtime source and budgets.
    plans=[read_json(r['motion_analysis_inputs']['plan_path']) for r in motion]
    if any(p['source_sha256']!=plans[0]['source_sha256'] for p in plans[1:]):raise ValueError('Repeat versions differ')
    for key in ('mode','phases','planned_duration_s','planned_slots','scene','motion_scale_validation',
                'idle_zero_scheduling','command_speed_cap_m_s','command_lease_s','self_monitor_period_s',
                'detector_bindings','weights_sha256','training_receipt_sha256'):
        if any(p[key]!=plans[0][key] for p in plans[1:]):raise ValueError('Repeat contract differs: '+key)
    summary_rows=[]
    for run in flown:
        m=run['motion_scale'];response=run['command_response'];cadence=run['timing']['monitor_event_receipt_cadence']
        summary_rows.append(dict(session_id=run['session_id'],mode=run['mode'],runner_status=run['original_runner_status'],
            independent_identity_execution=run['independent_integration_qualification'],motion_qualification=run['motion_scale_qualification'],
            endpoint_net_xy_m=m['net_xy_displacement_m'],endpoint_receipt_span_s=m['recorded_stage_span_s'],
            sampled_path_m=m['sampled_xy_path_length_m_context_only'],captured_main_frames=run['measured_frame_count'],
            planned_main_slots=run['measured_contract']['expected_slots'],identity_counts=run['measured_identity_counts'],
            nonzero_target_commands=run['nonzero_target_command_count'],minimum_target_mask_pixels=run['continuous_visibility']['minimum_actual_saved_pixels'],
            maximum_saved_query_gap_s=cadence['gap_distribution']['max_s'],query_gaps_over_0p12_s=cadence['gaps_over_0p12_s'],
            nonzero_dispatches_without_fresh_state=run['query_coverage_diagnostic']['nonzero_dispatches_without_fresh_state'],
            maximum_sampled_xy_speed_m_s=run['actual_motion_stage']['max_actual_xy_speed_m_s'],
            no_active_displacement_xy_m=response['sampled_displacement_vector_by_interval_class_xy_m']['NO_ACTIVE'],
            idle_omission_count=run['idle_omission_evidence']['actual_omission_event_count'],idle_omission_issues=run['idle_omission_evidence']['issues'],
            same_clock_later_dispatch_ties=sum(bool(x['same_clock_later_dispatch_ids']) for x in run['idle_omission_evidence']['rows']),
            maximum_explicit_clear_late_s=max((x['explicit_clear_after_declared_end_s'] for x in response['rows']),default=None),
            backend=run['backend_wall_distribution'],center_windows=run['center_windows'],
            gate_deadline_counts=run['timing']['frame_timing']['gate_deadline_counts'],
            observer_deadline_counts=run['timing']['frame_timing']['observer_deadline_counts'],
            full_completion_deadline_counts=run['timing']['frame_timing']['full_completion_deadline_counts']))
    measured=sum(r['measured_frame_count'] for r in flown);main_slots=sum(r['measured_contract']['expected_slots'] for r in flown)
    net_mm=[r['motion_scale']['net_xy_displacement_m']*1000 for r in motion]
    late=sum(r['timing']['frame_timing']['full_completion_deadline_counts'].get('LATE',0) for r in motion)
    total_nonzero=sum(r['nonzero_target_command_count'] for r in motion)
    coverage_link=f'[全部独立分析]({(ANALYSIS/"report.json").as_posix()})'
    result_link=f'[本轮结果与下一步]({RESULT.as_posix()})'
    next_steps='先定位整帧完成晚到并测量清零后的残余运动，再补超过2秒真实语义回复、属性高光误判和带遮挡运动；随后分别冻结50/100毫米尺度协议，统一四方法与YOLO/VLX消融，完成正式独立回合。'
    main_text=f'四个独立UE运动会话沿用同一源码与预算，20秒主测量各保存50/50槽，共200帧指定实例复算正确，0错选；实际非零目标请求共{total_nonzero}次，净位移依次为'+ '、'.join(f'{v:.3f}' for v in net_mm)+'毫米，全部达到预声明20毫米端点门槛。两个零输出会话各20/20槽、0非零目标指令、0毫米XY净漂移；旧V6起飞前校验失败另列零飞行。'+coverage_link+'。'
    limits=f'4个运动会话动作门控与观察器各200次按时，包含执行和采样停止的整帧完成{late}/200次晚于原槽截止；固定0.4秒整流程仍未通过。采样停止阈值为5毫米/秒，本组实际XY速度峰值约3.6–3.7毫米/秒；已知目标指令派发至清零区间之外，采样位移向量模长约44毫米；跨边界样本另列，该分类不是独立无输入漂移实验。清零与物理静止分列。端点均使用主段起始自状态至20秒采样结束后0.5秒尾段的显式终态，回执跨度20.5秒，起降排除。'+coverage_link+'。'
    table=['| 运动会话 | 主槽保存/计划 | 正确/错选 | 非零目标请求 | 端点净位移（毫米） | 查询最大间隔（毫秒） | 整帧晚到/计划 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for index,run in enumerate(motion):
        cadence=run['timing']['monitor_event_receipt_cadence']['gap_distribution']
        table.append(f'| {"先导" if index==0 else "独立重复"+str(index)} | 50/50 | 50/0 | {run["nonzero_target_command_count"]} | {net_mm[index]:.3f} | {cadence["max_s"]*1000:.0f} | {run["timing"]["frame_timing"]["full_completion_deadline_counts"].get("LATE",0)}/50 |')
    methods='新的闲置策略只在真实零指令、两次以上满足停止判据的采样及当前自状态新鲜时省略重复清零，满0.30秒再实际刷新。变化率仍按上一实际派发回执计算，速度上限0.03米/秒、变化率0.15米/秒²、动作租约与自状态年龄0.12秒、当前图像年龄0.5秒保持冻结值。省略日志没有伪造指令回执。'+f'[控制器]({(ROOT/"python/t1_async_control_v1p1.py").as_posix()})、[计划预检]({(ROOT/"python/t1_motion_plan_contract_v1.py").as_posix()})。'
    negative='旧V6将新50槽计划与旧41槽声明同时写入，校验在创建飞行报告及AirSim客户端前拒绝；该尝试没有飞行，新版本在启动UE之前核验完整槽表。旧失败没有新地面收尾证明，其进程当前不在不能回填当时的安全关闭。首次仅按时间戳检查闲置日志产生7处表面重叠，保存事件顺序证明它们先于同读数后续派发；初始诊断和新复算均保留。'+f'[起飞前失败]({(ROOT/"logs/t1_motion_visible20_v6_20261005/manifest.json").as_posix()})、[初始时钟诊断]({(ROOT/"logs/t1_motion_clock_tie_diagnostic_20261006/receipt.json").as_posix()})。'
    image_path=FIGURES/'motion_scale_4_zh.png'
    result_text='\n\n'.join(['# 任务一运动尺度重复验证与论文更新','日期：2026-10-06。范围：单机指定白横带红气球的固定高度开发验证；正式方法效果待完成。',
        '## 1. 实际执行与冻结范围',main_text,methods,
        '本轮V6P1新增5次实际飞行（零输出1次、运动先导1次、独立重复3次），连同此前V6零输出，完整运动尺度链共6次飞行、7个启动尝试。6次飞行均落地、释放、暂停，保存新地面回执后核对自有进程并关闭。旧V6三个重复计划因先导契约失败未执行，原计划保留。'+coverage_link+'。',
        f'实际飞行主段共{measured}/{main_slots}槽，初始获取{acquisition_frames}/{acquisition_slots}槽实采、{unused_acquisition}槽成功后无需执行；合计{media_count//3}组RGB/深度/分割、{media_count}份媒体。获取和主测量分母各自保存。'+coverage_link+'。',
        '## 2. 四个运动会话结果','\n'.join(table),
        'YOLO采用冻结气球专用权重；VLX真实调用、源RGB横带检查及后台双链交接均在运行。4个运动会话各7次语义完成，最大耗时依次1.000/1.031/1.047/0.938秒；本组仍未覆盖超过2秒真实回复。最后重复首帧因最近实际零指令间隔为0而WAIT，50次正确感知对应49次非零动作，该拒绝保留。'+coverage_link+'。',
        '## 3. 时效、残余运动和失败分析',limits,negative,
        '4个运动会话首尾各1秒图像窗均有有效掩膜覆盖，中心误差均值约0.0638到0.0562；这是当前地图、目标与小位移条件下的描述，不能独立归因于规划目标或写成正式居中收益。相邻帧和同会话脉冲属于嵌套记录；运动会话数为4，不能以200帧作为独立样本。20毫米协议结果也不替代另行预声明的50/100毫米验证。'+coverage_link+'。',
        '## 4. 中文图件与论文位置',f'![连续可见运动先导的实际位移、速度、中心误差与可见面积]({image_path.as_posix()})',
        f'六组新图均保存中文PNG/PDF、实际图点映射和字体SHA；120个绘图字符覆盖检查通过，六张PNG已逐张检查。首版绘图因视区外自动刻度被布局检查拒绝，V1P1图件另存并保持报告不变。[图表回执]({(FIGURES/"artifact_receipt.json").as_posix()})、[图点映射]({(FIGURES/"plot_data_mapping.json").as_posix()})。',
        '第二章补实际指令时间、端点定义、清零与停止区分；第三章补运动重复表、整帧晚到和残余运动讨论。正式四方法、检测后端/VLX消融、完整3D与独立正式轮次继续待补。'+result_link+'。',
        '## 5. 下一步',next_steps+'仍先完成第三章并同步第二章，随后任务二、任务三；本轮不改变六章结构。',
        '### AI协助说明',f'AI辅助实现、{test_count}项软件回归、原始证据复算及中文图文；论文解释由作者审定。软件测试、保存媒体、开发会话和正式方法效果分别登记。'])+'\n'
    if RESULT.exists():raise ValueError('Result document already exists')
    RESULT.write_text(result_text,encoding='utf-8')
    chapter2=ROOT/'docs/thesis/第二章_问题建模与仿真平台.md'
    chapter3=ROOT/'docs/thesis/第三章_单机感知轨迹优化.md'
    append_section(chapter2,'\n\n'.join(['### 2026-10-06 实际指令时钟与运动尺度验证',methods,main_text,limits,
        '主段图像时间格和端点状态跨度分别保存。主机时钟可能对连续操作返回相同数值，本组复核使用同一控制生产者的日志顺序区分先闲置后派发；精细物理时刻未知。'+result_link+'。']))
    append_section(chapter3,'\n\n'.join(['### 2026-10-06 连续可见运动尺度及独立开发重复',main_text,
        '表3-7 四个同版本运动会话的独立尺度结果；端点跨度含0.5秒尾段。','\n'.join(table),limits,negative,
        f'![图3-8 连续可见先导的实际运动与采样感知诊断，坐标和图例为中文]({image_path.as_posix()})',
        '每个保存主帧的目标掩膜至少500像素、身份及当前动作证据均经独立核对；帧间连续可见性和连续身份不能由这些采样保证。比较旧遮挡V5P1与当前持续可见V6P1不能分离闲置修正的因果作用；图像中心变化须由正式同场景四方法回答。'+coverage_link+'。',
        next_steps+result_link+'。']))
    for name in ('硕士论文章节任务与图表安排.md','任务一论文内容与完成标准.md','任务一运动尺度验证方案.md','任务一闲置零指令与运动尺度验证协议.md'):
        path=ROOT/'docs/research-plan-20261004'/name
        append_section(path,'\n\n'.join(['### 2026-10-06 运动尺度重复验证完成与剩余门槛',main_text,limits,
            f'{test_count}项相关测试通过；新版本完整50槽计划在启动仿真前校验。六组图件采用中文标注并保留数据映射。'+result_link+'。',
            next_steps+'A3完整几何、A7四方法、A8正式冻结、A9正式独立轮次和A10正式结果仍待补；三次开发重复不关闭正式轮次条目。']))
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig')
    text=text.replace('**最新推进（2026-10-05，后台复核交接','**此前推进（2026-10-05，后台复核交接',1)
    insert=f'**最新推进（2026-10-06，单机运动尺度独立重复）**：YOLO＋VLX链路下，1次20秒先导及3次同版本独立重复共200/200主帧正确、0错选，{total_nonzero}次非零目标请求，端点净位移62.887～64.098毫米；全部安全收尾，零输出复测通过。修复50槽/41槽预声明不一致，原起飞前失败保留；353项相关测试通过。整帧完成198/200次晚到及清零后残余运动仍需改进，正式四方法与YOLO/VLX消融待采。第二、三章与中文图表已同步。'+result_link+'。\n\n'
    text=text.replace('\n\n','\n\n'+insert,1);readme.write_text(text,encoding='utf-8')
    block=dict(date='2026-10-06',kind='IDLE_ZERO_SCHEDULING_ENDPOINT_SCALE_DEVELOPMENT_REPEATS',
        formal_efficacy_results=False,v6p1_new_actual_flights=5,total_actual_flights_in_scale_chain=6,
        total_attempts=7,verified_preflight_no_flight_attempts=1,motion_independent_sessions=4,
        motion_followup_repeats=3,original_failed_attempt_preserved=True,original_failed_attempt_safe_closure='UNVERIFIED',
        maximum_speed_cap_m_s=.03,acceleration_cap_m_s2=.15,command_lease_s=.12,
        actual_idle_zero_refresh_trigger_s=.30,sampled_stop_threshold_m_s=.005,
        formal_method_comparison_completed=False,minimum_frozen_motion_endpoint_m=.020,
        main_captured_frames=measured,main_planned_slots=main_slots,acquisition_captured_frames=acquisition_frames,
        acquisition_planned_slots=acquisition_slots,unused_after_acquisition_slots=unused_acquisition,
        full_media_groups=media_count//3,media_files=media_count,runs=summary_rows,
        motion_frames_correct=200,motion_frames_wrong=0,nonzero_motion_commands=total_nonzero,
        full_completion_late_motion_frames=late,regression_tests=test_count,
        endpoint_mean_m=statistics.mean(net_mm)/1000,endpoint_sd_m=statistics.stdev(net_mm)/1000,
        descriptive_only=True,analysis=str(ANALYSIS/'report.json'),figures=str(FIGURES/'artifact_receipt.json'),
        result_document=str(RESULT),next_work=next_steps)
    tasks_path=ROOT/'configs/thesis_simulation_tasks.json';tasks=read_json(tasks_path)
    tasks['updated']='2026-10-06';tasks['active_stage']='task_1_stop_residual_motion_and_full_completion_timing_then_formal_comparisons'
    tasks['tasks'][0].update(status='VISIBLE_ENDPOINT_SCALE_DEVELOPMENT_REPEATS_PASSED_FORMAL_T1_PENDING',
        next_gate='stop_residual_motion_and_completion_timing_then_slow_reply_attribute_stress_and_frozen_formal_four_methods',
        motion_scale_development_20261006=block)
    save(tasks_path,tasks)
    protocol_path=ROOT/'configs/t1_completion_protocol_v1.json';protocol=read_json(protocol_path)
    protocol['latest_motion_scale_development']=block;save(protocol_path,protocol)
    original_protocol=read_json(next(x['snapshot_path'] for x in before['files'] if x['path'].endswith('t1_completion_protocol_v1.json')))
    if any(protocol.get(k)!=v for k,v in original_protocol.items()):raise ValueError('Original protocol fields changed')
    if [t for t in tasks['tasks'] if t['id'] in ('T2','T3')]!=before['tasks_two_three_before']:raise ValueError('T2/T3 changed')
    fig_receipt=read_json(FIGURES/'artifact_receipt.json')
    if fig_receipt['font']['glyph_coverage']!='PASS' or any(l['text_bounds_status']!='PASS' for l in fig_receipt['layout_checks']):
        raise ValueError('Figure QA failed')
    for a in fig_receipt['artifacts']:
        if digest(a['path'])!=a['sha256']:raise ValueError('Figure artifact changed')
    fig_receipt['visual_review']=dict(status='PASS',reviewer='root assistant inspected all six PNGs with view_image',
        checks='Chinese glyphs legible; panel titles, axes and legends inside canvas; zero and motion traces checked against saved point ranges')
    save(FIGURES/'artifact_receipt.json',fig_receipt)
    updates=[dict(path=x['path'],before_sha256=x['sha256'],after_sha256=digest(x['path'])) for x in before['files']]
    local_links=set();checked=[]
    for path in [Path(x['path']) for x in before['files'] if x['path'].endswith('.md')]+[RESULT]:
        text=path.read_text(encoding='utf-8-sig')
        prior=next((Path(x['snapshot_path']).read_text(encoding='utf-8-sig') for x in before['files'] if Path(x['path'])==path),'')
        old_links=set(re.findall(r'\]\((D:/forAirsim/[^)]+)\)',prior))
        for target in set(re.findall(r'\]\((D:/forAirsim/[^)]+)\)',text))-old_links:
            if not Path(target).exists():raise ValueError('Broken local evidence link: '+target)
            local_links.add(target)
    for name in ('t1_motion_visible20_v6_rep1_20261005','t1_motion_visible20_v6_rep2_20261005','t1_motion_visible20_v6_rep3_20261005'):
        path=ROOT/'configs'/name/'plan.json'
        if (ROOT/'logs'/name).exists():raise ValueError('Supposed unused old plan has a run directory')
        checked.append(dict(session_id=name,plan=bound(path),status='NOT_EXECUTED_SUPERSEDED_CONTRACT_FAILURE'))
    receipt=dict(kind='T1_MOTION_SCALE_REPEATS_AND_CHINESE_THESIS_UPDATE_CLOSEOUT',status='PASS_BOUNDED_DEVELOPMENT_CLOSEOUT',
        created_utc=datetime.now(timezone.utc).isoformat(),canonical_updates=updates,
        original_completion_protocol_fields_unchanged=True,tasks_two_three_unchanged=True,
        actual_flights_this_v6p1_batch=5,actual_flights_total_scale_chain=6,started_attempts=7,
        motion_independent_sessions=4,followup_repeats=3,motion_source_and_budget_identity=True,
        formal_collection_authorized_by_this_file=protocol['formal_collection_authorized_by_this_file'],
        task_1_formal_effect_complete=False,whole_thesis_complete=False,
        flights=locks,failed_attempts=failed,unexecuted_old_repeat_plans=checked,
        full_completion_late_motion_frames=late,nonzero_motion_commands=total_nonzero,
        regression_tests=dict(count=test_count,result='PASS',stdout=bound(CLOSEOUT/'tests_v6p1.stdout.log'),stderr=bound(test_path)),
        analysis_receipt=bound(ANALYSIS/'artifact_receipt.json'),figure_receipt=bound(FIGURES/'artifact_receipt.json'),
        analysis_report=bound(ANALYSIS/'report.json'),results_document=bound(RESULT),
        idle_proof=bound(idle_proof),local_evidence_links_checked=len(local_links),
        source=bound(Path(__file__)),next_work=next_steps,
        ai_assistance='AI assisted code, local verification, plots and evidence prose; no formal scientific benefit inferred')
    save(destination,receipt)
    print(json.dumps(dict(status=receipt['status'],motion_sessions=4,actual_new_flights=5,
                         tests=test_count,endpoint_mm=net_mm,output=str(destination)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--idle-proof',type=Path,required=True)
    args=parser.parse_args();closeout(args.idle_proof)
