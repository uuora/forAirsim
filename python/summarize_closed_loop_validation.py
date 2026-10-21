"""Postflight closure evidence and figures, using original simulator records only.

No aircraft commands, scene target queries, or saved-frame freshness promotion.
Run after audit_closed_loop_retest.py has produced its independent decision.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fixed_map_perception import ROOT, read, save, sha, utc
from analyze_step3_development import lines, csv_save


def summarize(root):
    root = Path(root).resolve()
    decision = read(root / 'CLOSED_LOOP_DECISION.json')
    trial = root / 'online_trial'
    result = read(trial / 'result.json')
    events = lines(trial / 'session_events.jsonl')
    task = read(trial / 'fixed_version/task.json')
    version = read(trial / 'declaration.json')['config']['schema']
    states = [r for p in sorted(trial.glob('flight_*/timestamped.jsonl'))
              for r in lines(p) if r['kind'] == 'state']
    if not states:
        raise ValueError('NO_MEASURED_FLIGHT_STATES')
    states.sort(key=lambda r: r['host_monotonic_s'])
    start = states[0]['host_monotonic_s']
    goal = None
    phase = 'observation preparation'
    updates = sorted([e for e in events if e['kind'] in
                      ['PERCEPTION_START_ACCEPTED', 'RETURN_PLAN']],
                     key=lambda e: e['host_monotonic_s'])
    cursor = 0
    rows = []
    for s in states:
        while cursor < len(updates) and updates[cursor]['host_monotonic_s'] <= s['host_monotonic_s']:
            e = updates[cursor]
            if e['kind'] == 'PERCEPTION_START_ACCEPTED':
                goal = e['task']['goal_xyz_m']
                phase = 'sensor guidance'
            else:
                phase = 'return'
            cursor += 1
        p = s['position_xyz_m']
        rows.append(dict(host_elapsed_s=s['host_monotonic_s']-start,
                         sim_timestamp_ns=s['state_timestamp_ns'], stage=s['stage'], phase=phase,
                         x_m=p[0], y_m=p[1], z_m=p[2],
                         speed_m_s=float(np.linalg.norm(s['velocity_xyz_m_s'])),
                         static_envelope_clearance_m=s['envelope_clearance_calculated_m'],
                         sensor_goal_error_m=float(np.linalg.norm(np.array(p)-goal)) if goal is not None and phase == 'sensor guidance' else None,
                         DroneB_landed=s['DroneB_landed'], DroneB_api_control=s['DroneB_api_control']))
    csv_save(root / 'audit/measured_time_series.csv', rows)
    csv_save(root / 'audit/closure_event_timeline.csv',
             [dict(host_elapsed_s=e['host_monotonic_s']-start, kind=e['kind'],
                   stage=e.get('stage'), reason=e.get('reason'),
                   plan_version=e.get('plan_version')) for e in events
              if e['kind'] in ['PERCEPTION_START_ACCEPTED', 'PERCEPTION_OLD_ACTION_CANCELLED',
                               'PERCEPTION_RECOVERY_HEIGHT_REQUIRED', 'PERCEPTION_RECOVERY_HEIGHT_RESTORED',
                               'PERCEPTION_PLAN_REJECTED', 'LANGUAGE_SENSOR_APPROACH_HOVER_PASSED',
                               'HOVER_VERIFIED', 'RETURN_LANDED_RELEASED']])
    proofs = []
    for e in events:
        if e['kind'] != 'HOVER_VERIFIED' or e['stage'] != 'ARRIVED':
            continue
        plans = [u for u in updates if u['kind'] == 'PERCEPTION_START_ACCEPTED' and u['host_monotonic_s'] <= e['host_monotonic_s']]
        if not plans:
            continue
        plan = plans[-1]
        samples = [s for s in states if s['stage'] == 'ARRIVAL_HOVER'
                   and e['hold_start_timestamp_ns'] <= s['state_timestamp_ns']
                   and s['host_monotonic_s'] <= e['host_monotonic_s']]
        if not samples:
            continue
        positions = np.array([s['position_xyz_m'] for s in samples])
        errors = np.linalg.norm(positions-np.array(plan['task']['goal_xyz_m']), axis=1)
        speeds = [float(np.linalg.norm(s['velocity_xyz_m_s'])) for s in samples]
        span = (samples[-1]['state_timestamp_ns']-samples[0]['state_timestamp_ns'])/1e9
        drift = float(np.max(np.linalg.norm(positions-positions[0], axis=1)))
        proofs.append(dict(plan_version=plan['plan_version'], original_event=e,
                           sample_count=len(samples), sampled_span_sim_s=span,
                           max_goal_error_m=float(max(errors)), max_speed_m_s=max(speeds),
                           max_displacement_from_first_sample_m=drift,
                           measured_criteria_passed=bool(span >= task['arrival_dwell_s']
                               and max(errors) <= task['arrival_error_m']
                               and max(speeds) <= task['arrival_speed_m_s']
                               and drift <= task['hover_drift_max_m'])))
    # Preserve the gate decision and publish supplemental measured checks separately.
    manifest = read(root / 'source_before_trial/manifest.json')
    hashes = dict(source_unchanged=all(sha(e['source']) == e['sha256'] for e in manifest['files']),
                  snapshot_verified=all(sha(e['snapshot']) == e['sha256'] for e in manifest['files']),
                  maps_unchanged=all(sha(e['path']) == e['sha256'] for e in manifest['maps']),
                  weights_unchanged=sha(manifest['weights']['path']) == manifest['weights']['sha256'])
    b_ok = all(s['DroneB_landed'] and not s['DroneB_api_control'] for s in states)
    closure = dict(utc=utc(),evidence_type='one prospective development simulator trial',
                   basic_gate_passed=decision['basic_closed_loop_passed'],
                   measured_sensor_hover_proofs=proofs, DroneB_ground_invariant_passed=b_ok,
                   hashes=hashes, final_ground=result['final_ground'],
                   sensor_hover_sample_check_passed=any(p['measured_criteria_passed'] for p in proofs),
                   controlled_occlusion_recovery='NOT_RUN', position_and_distractor_trials='NOT_RUN',
                   formal_method_comparison='NOT_RUN',
                   ai_disclosure='Codex executed existing simulation scripts and generated this audit and plots. Values come from saved AirSim state and onboard RGB/depth records; no synthetic flight samples.')
    save(root / 'VALIDATION_CLOSURE.json',closure)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    colors={'observation preparation':'#64748b','sensor guidance':'#087e8b','return':'#d17632'}
    for name,color in colors.items():
        selected=[r for r in rows if r['phase']==name]
        if selected:
            axes[0,0].scatter([r['x_m'] for r in selected],[r['y_m'] for r in selected],s=4,color=color,label=name,rasterized=True)
    for p in sorted(trial.glob('flight_*/perception_plan_*.json')):
        path=np.array(read(p)['path'])
        axes[0,0].plot(path[:,0],path[:,1],'--',lw=1,color='#087e8b',alpha=.65)
    axes[0,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Measured trajectory and sensor plans')
    axes[0,0].set_aspect('equal',adjustable='datalim');axes[0,0].legend(fontsize=8)
    t=[r['host_elapsed_s'] for r in rows]
    axes[0,1].scatter(t,[r['speed_m_s'] for r in rows],s=3,color='#087e8b',rasterized=True)
    axes[0,1].axhline(task['max_speed_measured_m_s'],ls='--',color='#b44',lw=1,label='Measured speed limit')
    axes[0,1].set(xlabel='Host elapsed time (s)',ylabel='Measured speed (m/s)',title='Actual speed');axes[0,1].legend(fontsize=8)
    active=[r for r in rows if r['stage'] in ['APPROACH','RETURN']]
    axes[1,0].scatter([r['host_elapsed_s'] for r in active],[r['static_envelope_clearance_m'] for r in active],s=3,color='#64748b',rasterized=True)
    axes[1,0].axhline(task['clearance_stop_m'],ls='--',color='#b44',lw=1,label='Stop threshold')
    axes[1,0].set(xlabel='Host elapsed time (s)',ylabel='Calculated static clearance (m)',title='Approach / return clearance');axes[1,0].legend(fontsize=8)
    guided=[r for r in rows if r['sensor_goal_error_m'] is not None]
    axes[1,1].scatter([r['host_elapsed_s'] for r in guided],[r['sensor_goal_error_m'] for r in guided],s=4,color='#087e8b',rasterized=True)
    axes[1,1].axhline(task['arrival_error_m'],ls='--',color='#b44',lw=1,label='Arrival error threshold')
    axes[1,1].set(xlabel='Host elapsed time (s)',ylabel='Calculated sensor goal error (m)',title='Sensor approach, including recovery stops');axes[1,1].legend(fontsize=8)
    fig.suptitle(version+' development validation — '+('basic gate PASS' if decision['basic_closed_loop_passed'] else 'basic gate FAILED'))
    figs=root/'figures';figs.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(figs/('closed_loop_measured.'+ext),dpi=220)
    plt.close(fig)
    metrics=read(root/'audit/flight_metrics.json')[0]
    text='# '+version+' 在线闭环实验记录\n\n'
    text+='基础验收：'+('通过' if decision['basic_closed_loop_passed'] else '未通过')+'。任务结果：'+str(decision['mission_outcome'])+'。\n\n'
    text+=f"本轮实际处理 {decision['online_processed_frames']} 帧，原处理时刻有效表面定位 {decision['valid_surface_frames_at_original_processing']} 帧；新鲜计划复核 {decision['planning_fresh_revalidation_count']} 次，实际感知执行启动 {decision['execution_start_count']} 次，来源绑定指令 {decision['source_bound_actual_commands']} 条。\n\n"
    text+=f"感知接近悬停事件 {decision['stable_approach_hover_events']} 次，自然证据失效停止 {decision['loss_stop_events']} 次，恢复重新规划 {decision['recovery_plan_count']} 次。自然失效不是预设的真实遮挡扰动；后者仍未运行。\n\n"
    text+=f"实际最大速度 {metrics['max_speed_m_s']:.3f} m/s，连续运动最大状态间隔 {metrics['max_continuous_motion_state_gap_s']:.3f} s，运动静态包络最小净空 {metrics['minimum_approach_return_envelope_clearance_m']:.3f} m。起降平台接触单独保留，不计作运动净空。完整指标见 audit/flight_metrics.json。\n\n"
    text+='单独复算的悬停判据与样本见 VALIDATION_CLOSURE.json；所有源码、地图、权重哈希检查：'+str(all(hashes.values()))+'。两机最终落地且控制释放：'+str(all(s['landed'] and not s['api_control'] for s in result['final_ground'].values()))+'。\n\n'
    text+='原始数据 online_trial/；状态与指令 CSV、拒识、所有帧索引及来源绑定 audit/；飞前快照 source_before_trial/；交付快照 delivery/；图表 figures/。图中所有点来自真实仿真状态，虚线为计划路线，定位是可见表面点而非目标中心。\n\n'
    text+='范围：仅一次前瞻性开发仿真试次，不能据此声称正式算法收益。位置变化、相似干扰、受控遮挡及四方法对照尚未验证。AI 使用说明：Codex 执行既有仿真流程、计算指标并生成本记录与图表；未合成或补写飞行样本。\n'
    (root/'SUMMARY.md').write_text(text,encoding='utf-8')
    print(json.dumps(closure,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();summarize(a.root)
