"""Aggregate a frozen same-scene batch from postflight receipts only."""
import argparse
import csv
import json
from pathlib import Path

from fixed_map_perception import read, save, sha, utc


def summarize(root):
    root = Path(root).resolve()
    protocol = read(root / 'PROTOCOL.json')
    rows = []
    baseline_sources = None
    for index in range(1, protocol['additional_trials'] + 1):
        trial = root / f'trial_{index:02d}'
        if not (trial / 'CLOSED_LOOP_DECISION.json').exists():
            rows.append(dict(trial=index, status='NOT_RUN_OR_NOT_AUDITED'))
            continue
        decision = read(trial / 'CLOSED_LOOP_DECISION.json')
        closure = read(trial / 'VALIDATION_CLOSURE.json')
        driver_result = read(trial / 'online_trial/result.json')
        declaration = read(trial / 'RETEST_DECLARATION.json')
        manifest = read(trial / 'source_before_trial/manifest.json')
        sources = {r['source']: r['sha256'] for r in manifest['files']}
        if baseline_sources is None:
            baseline_sources = sources
        same_sources = sources == baseline_sources
        config_hash_ok = sources.get(str(Path(protocol['config']).resolve())) == protocol['config_sha256']
        scene_ok = declaration['config']['runtime_scene_version'] == protocol['scene']['scene_version']
        scene_ok = scene_ok and declaration['config']['runtime_engine_identity'] == protocol['scene']['engine_identity']
        ground_ok = all(s['landed'] and not s['api_control'] for s in closure['final_ground'].values())
        hashes_ok = all(closure['hashes'].values())
        proofs = [p for p in closure['measured_sensor_hover_proofs'] if p['measured_criteria_passed']]
        passed = (decision['basic_closed_loop_passed'] and closure['sensor_hover_sample_check_passed']
                  and closure['DroneB_ground_invariant_passed'] and ground_ok and hashes_ok
                  and same_sources and config_hash_ok and scene_ok and decision['issue'] is None
                  and driver_result.get('result_record_provenance','ORIGINAL_DRIVER_OUTPUT') in ['ORIGINAL_DRIVER_OUTPUT','ORIGINAL_DRIVER_OUTPUT_V18'])
        metrics = read(trial / 'audit/flight_metrics.json')
        row = dict(trial=index, status='PASS' if passed else 'FAIL',
                   flight_start_requested=driver_result.get('flight_start_requested',True),
                   mission_outcome=decision['mission_outcome'], issue=decision['issue'],
                   terminal_result_provenance=driver_result.get('result_record_provenance','ORIGINAL_DRIVER_OUTPUT'),
                   processed_frames=decision['online_processed_frames'],
                   valid_surface_frames=decision['valid_surface_frames_at_original_processing'],
                   source_bound_commands=decision['source_bound_actual_commands'],
                   fresh_plans=decision['planning_fresh_revalidation_count'],
                   natural_stops=decision['loss_stop_events'], recovery_plans=decision['recovery_plan_count'],
                   safety_gate=decision['safety_checks_passed'], ground_released=ground_ok,
                   source_hashes_ok=hashes_ok, same_sources=same_sources,
                   config_hash_ok=config_hash_ok, scene_epoch_ok=scene_ok,
                   max_speed_m_s=max((m['max_speed_m_s'] for m in metrics),default=None),
                   min_motion_clearance_m=min((m['minimum_approach_return_envelope_clearance_m'] for m in metrics),default=None),
                   max_motion_state_gap_s=max((m['max_continuous_motion_state_gap_s'] for m in metrics),default=None),
                   hover_duration_s=proofs[-1]['sampled_span_sim_s'] if proofs else None,
                   hover_error_m=proofs[-1]['max_goal_error_m'] if proofs else None,
                   hover_speed_m_s=proofs[-1]['max_speed_m_s'] if proofs else None,
                   receipt=str(trial / 'VALIDATION_CLOSURE.json'))
        rows.append(row)
    completed = [r for r in rows if r['status'] in ['PASS', 'FAIL']]
    successes = sum(r['status'] == 'PASS' for r in rows)
    final_ground = read(root / 'FINAL_GROUND_RECEIPT.json')
    current_ground_ok = all(s['landed'] and not s['api_control'] and s['fresh']
                           for s in final_ground['vehicles'].values())
    complete = len(completed) == protocol['additional_trials']
    flights=sum(r['flight_start_requested'] is True for r in completed)
    result = dict(utc=utc(),protocol_sha256=sha(root / 'PROTOCOL.json'),
                  planned=protocol['additional_trials'],completed=len(completed),passed=successes,
                  failed=sum(r['status'] == 'FAIL' for r in rows),
                  actual_takeoff_attempts=flights,prelaunch_rejections=len(completed)-flights,
                  repeatability_gate_passed=complete and successes == protocol['additional_trials'] and current_ground_ok,
                  final_ground_fresh_released=current_ground_ok, trials=rows,
                  historical_success_excluded_from_new_batch=True,
                  position_generalization='NOT_RUN', controlled_occlusion='NOT_RUN',
                  method_comparison='NOT_RUN', gui_flight='NOT_RUN',
                  scope=f"{len(completed)} completed attempts of {protocol['additional_trials']} prospectively planned, {flights} requested takeoffs; descriptive counts only, no population success rate or algorithm benefit claim.")
    save(root / 'BATCH_DECISION.json', result)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (root / 'TRIAL_METRICS.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    text = '# '+protocol.get('version','v17')+' 同场景重复验证\n\n'
    verdict = 'PASS' if result['repeatability_gate_passed'] else ('FAIL' if complete else 'INCOMPLETE')
    text += f"本轮完成 {len(completed)} 次试验尝试，其中请求起飞 {flights} 次、飞前拒绝 {len(completed)-flights} 次。完整闭环通过 {successes} 次，失败/拒绝 {result['failed']} 次。三次全部通过门槛：{verdict}。此前 v17 成功试次不计入本轮分母。\n\n"
    text += '|试次|结果|任务结果|有效表面帧|实际感知指令|新鲜计划|自然停止|悬停时长/s|最大悬停误差/m|\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        duration = '-' if row.get('hover_duration_s') is None else f"{row['hover_duration_s']:.3f}"
        error = '-' if row.get('hover_error_m') is None else f"{row['hover_error_m']:.3f}"
        text += f"|{row['trial']}|{row['status']}|{row.get('mission_outcome','-')}|{row.get('valid_surface_frames','-')}|{row.get('source_bound_commands','-')}|{row.get('fresh_plans','-')}|{row.get('natural_stops','-')}|{duration}|{error}|\n"
    text += '\n所有成功与失败均保留。配置、源码和场景引擎生命周期逐次绑定；控制器不使用 Actor 真值，不重训、不修改控制参数、不传送或重置无人机。每轮详细复算、图表和 RGB/深度/轨迹位于 trial_XX/。\n\n'
    for row in completed:
        if row['terminal_result_provenance'] not in ['ORIGINAL_DRIVER_OUTPUT','ORIGINAL_DRIVER_OUTPUT_V18']:
            text += f"试次 {row['trial']} 存在收尾异常 {row['issue']}，原驱动未写出最终结果。其 result.json 是显式标注的失败终态摘要，依据原状态记录、原异常及独立新鲜地面 RPC 补充；不是原驱动终态，更不构成成功证据。参见该试次 driver_stderr.txt、POSTFAILURE_GROUND_RECEIPT.json。\n\n"
    text += '这是同一固定随机布局下的工程重复验证，不能解释为随机位置泛化、受控遮挡恢复或方法对比。自然证据过期导致的停止另行计数，不能替代预设遮挡试验。桌面界面仍需接入当前已验证版本并单独验收。\n\n'
    text += f"最终新鲜 RPC 核验：两机落地且控制释放={current_ground_ok}。原安全门槛保持：实测速度≤0.8m/s、连续运动状态间隔≤0.75s、静态包络运动净空≥0.75m、到达误差≤0.2m且速度≤0.12m/s持续至少3模拟秒。\n\n"
    text += 'AI 使用说明：Codex 执行仿真、整理原始记录并生成本报告；所有飞行样本来自 AirSim，未合成成功记录。\n'
    (root / 'VALIDATION_REPORT.md').write_text(text, encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'trials'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    summarize(parser.parse_args().root)
