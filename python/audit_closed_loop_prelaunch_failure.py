"""Preserve a genuine prelaunch refusal without inventing flight evidence."""
import argparse
import shutil
from pathlib import Path

from fixed_map_perception import read, save, sha, utc


def audit(root):
    root=Path(root).resolve();trial=root/'online_trial'
    result=read(trial/'result.json')
    if result.get('flight_start_requested') is not False or not result.get('issue'):
        raise ValueError('REQUIRES_ORIGINAL_FAILED_PRELAUNCH_RESULT')
    if result.get('terminal_phase')!='FINAL' or result.get('result_record_provenance')!='ORIGINAL_DRIVER_OUTPUT_V18':
        raise ValueError('REQUIRES_ORIGINAL_FINAL_DRIVER_RECORD')
    if list(trial.glob('flight_*')) or (trial/'session_events.jsonl').exists():
        raise ValueError('UNEXPECTED_EXECUTION_EVIDENCE_REQUIRES_FULL_FLIGHT_AUDIT')
    receipt=read(trial/'final_ground_receipt.json');vehicles=receipt['vehicles']
    ground=receipt.get('passed') and set(vehicles)=={'DroneA','DroneB'} and all(v.get('landed') is True and v.get('api_control') is False and v.get('fresh') is True for v in vehicles.values())
    manifest=read(root/'source_before_trial/manifest.json')
    hashes=dict(source_unchanged=all(sha(r['source'])==r['sha256'] for r in manifest['files']),
                snapshot_verified=all(sha(r['snapshot'])==r['sha256'] for r in manifest['files']),
                maps_unchanged=all(sha(r['path'])==r['sha256'] for r in manifest['maps']),
                weights_unchanged=sha(manifest['weights']['path'])==manifest['weights']['sha256'])
    output=root/'audit';output.mkdir(exist_ok=False)
    save(output/'flight_metrics.json',[])
    save(output/'PRELAUNCH_AUDIT.json',dict(utc=utc(),flight_start_requested=False,issue=result['issue'],ground_fresh_released=bool(ground),hashes=hashes,scope='Original ground refusal only; no measured flight safety or mission success claim'))
    delivery=root/'delivery';delivery.mkdir(exist_ok=False)
    sources=[]
    for row in manifest['files']:
        target=delivery/Path(row['source']).name;shutil.copyfile(row['source'],target)
        sources.append(dict(row,current_sha256=sha(target),unchanged_since_frozen_prelaunch=sha(target)==row['sha256']))
    shutil.copyfile(__file__,delivery/Path(__file__).name)
    save(delivery/'manifest.json',dict(utc=utc(),files=sources,auditor_sha256=sha(__file__),no_flight=True))
    decision=dict(utc=utc(),basic_closed_loop_passed=False,flight_start_requested=False,online_processed_frames=0,raw_frame_directories=0,candidate_frames=0,valid_surface_frames_at_original_processing=0,recorded_valid_certificate_failures=[],planning_fresh_revalidation_count=0,execution_start_count=0,source_bound_actual_commands=0,stable_approach_hover_events=0,loss_stop_events=0,recovery_plan_count=0,issue=result['issue'],mission_outcome='PRELAUNCH_REJECTED',final_ground=vehicles,safety_checks_passed=None,safety_status='NOT_RUN_NO_TAKEOFF',no_formal_algorithm_benefit_claim=True)
    save(root/'CLOSED_LOOP_DECISION.json',decision)
    save(root/'VALIDATION_CLOSURE.json',dict(utc=utc(),evidence_type='Original prelaunch refusal; no flight occurred',basic_gate_passed=False,measured_sensor_hover_proofs=[],DroneB_ground_invariant_passed=bool(ground),hashes=hashes,final_ground=vehicles,sensor_hover_sample_check_passed=False,flight_start_requested=False,controlled_occlusion_recovery='NOT_RUN',position_and_distractor_trials='NOT_RUN',formal_method_comparison='NOT_RUN'))
    text=f"# 飞前拒绝记录\n\n原驱动拒绝原因：`{result['issue']}`。未加载感知模型，未请求起飞，未产生飞行状态或感知运动指令。\n\n两机新鲜地面与控制释放核验：{bool(ground)}。冻结源码、地图、权重检查：{all(hashes.values())}。飞行安全、悬停与闭环结果均为 NOT_RUN；本次不能计为成功试次。\n\n原始记录位于 online_trial/result.json、online_trial/online_prelaunch_memory.json 与 online_trial/failure.json。AI 使用说明：Codex 仅核对原始拒绝与地面记录，未补写飞行样本。\n"
    (root/'ACCEPTANCE.md').write_text(text,encoding='utf-8')
    print(dict(issue=result['issue'],flight_start_requested=False,ground_fresh_released=bool(ground),hashes=hashes))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');audit(parser.parse_args().root)
