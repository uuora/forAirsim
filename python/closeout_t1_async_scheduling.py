"""Update the task ledger from sealed scheduling evidence; preserve all failures."""
from __future__ import annotations
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/t1_async_closeout_20261005'
DOC='docs/research-plan-20261004/任务一异步调度与语义时效开发结果.md'
TIMING='logs/t1_async_timing_final_20261005/report.json'
IDENTITY='logs/t1_async_identity_analysis_20261005/report.json'
PROFILE='logs/t1_vlx_latency_diagnostic_20261005/report.json'
NEXT='vlx_completion_polling_and_initial_anchor_feasibility_then_nonzero_async_retest'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def closeout():
    if OUT.exists():raise ValueError('CLOSEOUT_DESTINATION_MUST_BE_NEW')
    timing=read(ROOT/TIMING);identity=read(ROOT/IDENTITY);profile=read(ROOT/PROFILE)
    if timing['status']!='PASS' or len(timing['runs'])!=5:raise ValueError('TIMING_INPUTS_UNVERIFIED')
    checks=[]
    for folder in (Path(TIMING).parent,Path(IDENTITY).parent,Path(PROFILE).parent):
        for item in read(ROOT/folder/'artifact_receipt.json')['artifacts']:
            if digest(item['path'])!=item['sha256']:raise ValueError('ANALYSIS_ARTIFACT_CHANGED')
            checks.append(dict(path=item['path'],sha256=item['sha256']))
    runs=identity['runs'][1:]
    if len(runs)!=4 or sum(r['planned_slots'] for r in runs)!=122 or sum(r['captured_frames'] for r in runs)!=103:
        raise ValueError('ALL_ATTEMPT_ACCOUNTING_CHANGED')
    if any(r['status']!='FAIL' or not r['observed_airborne_from_execution'] or
           not r['engine_closed_after_safe_completion'] or not r['fresh_ground_closure_verified'] or
           not all(r['safe_end'].values()) for r in runs):raise ValueError('ATTEMPT_OR_SAFE_CLOSURE_CHANGED')
    if any(r['command_counts']['target_commands_nonzero'] for r in runs):raise ValueError('UNEXPECTED_NEW_TARGET_MOTION')
    # Historical source locks checked directly, without changing their files.
    locks=read(ROOT/'configs/trajectory_timed_pilot_v6p2/center.json')['flight_source_sha256']
    from prepare_t1_identity_flight import LEGACY_NAMES
    for name in LEGACY_NAMES:
        if digest(ROOT/'python'/name)!=locks[name]:raise ValueError('LEGACY_SOURCE_CHANGED')
    files=['configs/thesis_simulation_tasks.json','configs/t1_completion_protocol_v1.json','README.md',
           'docs/research-plan-20261004/硕士论文章节任务与图表安排.md',
           'docs/research-plan-20261004/任务一论文内容与完成标准.md']
    OUT.mkdir();before={}
    for name in files:
        source=ROOT/name;backup=OUT/'before'/name;backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,backup);before[name]=digest(source)
    task=read(ROOT/files[0]);other_tasks=copy.deepcopy(task['tasks'][1:])
    block=dict(status='OWN_STATE_ZERO_HOLD_TIMING_RECORDED_NO_NEW_TARGET_IDENTITY_OUTPUT',
        date='2026-10-05',timing_analysis=TIMING,identity_analysis=IDENTITY,
        component_latency_diagnostic=PROFILE,development_report=DOC,
        observed_airborne_attempts_all=4,retained_failed_attempts_all=4,
        planned_slots_all=122,captured_rgb_frames_all=103,complete_paired_media_groups=102,
        no_output_frames=102,unknown_output_frames=1,nonzero_target_commands=0,
        known_flight_semantic_submissions=14,additional_latency_diagnostic_submissions=3,
        corrected_runs_max_between_SELF_STATE_receipts_s=[.078,.094,.078],
        nominal_monitor_period_s=.05,measured_monitor_not_hard_real_time=True,
        blocked_interval_self_state_counts=[15,14],blocked_interval_max_gap_s=[.063,.063],
        blocking_under_active_nonzero_target_lease_verified=False,
        all_attempts_grounded_released_paused_owned_closed=True,
        tests_passed=141,formal_method_effect_complete=False,whole_thesis_complete=False,next_gate=NEXT,
        source_scope='Original plans and failed media gaps preserved; historical comparison is not matched efficacy')
    task.update(updated='2026-10-05',active_stage='task_1_semantic_deadline_and_initial_anchor_feasibility')
    task['tasks'][0].update(status='ASYNC_ZERO_HOLD_TIMING_RECORDED_IDENTITY_OUTPUT_RETEST_PENDING_FORMAL_T1_PENDING',
        next_gate=NEXT,async_control_development_20261005=block)
    if task['tasks'][1:]!=other_tasks:raise ValueError('TASK_TWO_OR_THREE_CHANGED')
    save(ROOT/files[0],task)
    protocol=read(ROOT/files[1]);protocol['latest_async_control_development']=block
    protocol['latest_development_review_date']='2026-10-05'
    protocol['perception_factors'].update(
        main_planner_comparison_detector='YOLO_BALLOON_SPECIALIST_FROZEN_DEVELOPMENT_FORMAL_COMPARISON_PENDING',
        yolo_current_evidence='Controlled balloon training and held-out development testing recorded; current frozen specialist checkpoint, formal trajectory detector factor pending',
        yolo_development_binding=dict(runtime='configs/t1_balloon_detector_runtime_cuda_v1.json',
            weights_sha256='92b0dc0f4e17ff14e9c1e110b9cad9ef249274136ee9a28d99b7885dfb396844'))
    save(ROOT/files[1],protocol)
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    text=text.replace('**最新复核（2026-10-05，机载身份—深度—短XY闭环）**','**此前复核（2026-10-05，机载身份—深度—短XY闭环）**',1)
    new='**最新推进（2026-10-05，感知与自身控制解耦）**：新增独立控制线程并保存4次实际飞行，全部安全关闭，整体联合验证均FAIL。修正传输配置后三次零保持试次的保存自状态间隔最大78/94/78 ms；两处0.65 s阻塞期间仍保存15/14个状态读回。新批次没有正确身份输出或非零目标指令，不能替代此前厘米级闭环。另3次VLX组件请求均在约3.06–3.09 s读超时；141项相关测试通过。下一步定位语义时效与初始身份获取，再复验非零异步闭环，之后开展四方法和消融。[本轮记录与失败](D:/forAirsim/'+DOC+')。\n\n'
    text=text.replace('# AirSim dual-drone balloon simulation\n\n','# AirSim dual-drone balloon simulation\n\n'+new,1)
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    addition='\n\n### 2026-10-05 异步控制测量与当前语义时效入口\n\n新增4次飞行全部保留，均安全关闭，联合闭环均判FAIL。独立控制在零保持及两处0.65秒阻塞期间持续取得状态回执；新批次没有有效指定身份及非零目标指令。完整数据、失败、媒体缺失与时钟口径见[异步调度开发结果](D:/forAirsim/'+DOC+')。\n\n接下来先定位VLX完成、轮询、源年龄与初始确认条件，再验证非零运动下的异步租约清除，最后推进四方法与YOLO/VLX消融及正式独立重复。已有历史身份闭环仍按原版本保留，任务一正式方法效果及任务二、三均未提前标为完成。\n'
    for name in files[3:]:
        path=ROOT/name;path.write_text(path.read_text(encoding='utf-8')+addition,encoding='utf-8')
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    receipt=dict(kind='T1_ASYNC_SCHEDULING_CLOSEOUT',status='RECORDED_FAILURES_AND_MEASURED_SCOPE',
        created_utc=datetime.now(timezone.utc).isoformat(),evidence=checks,summary=block,
        historical_twelve_sources_unchanged=True,tasks_two_and_three_semantically_unchanged=True,
        files=[dict(path=str(ROOT/name),before_sha256=before[name],after_sha256=digest(ROOT/name)) for name in files],
        source_sha256=digest(__file__),formal_method_effect_complete=False,whole_thesis_complete=False)
    save(OUT/'receipt.json',receipt)
    print(receipt['status'],len(checks),'sealed artifacts',len(files),'updated indexes')

if __name__=='__main__':closeout()
