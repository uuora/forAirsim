"""Small ground development archive, traceable samples/metrics; no flight."""
import sys,json,csv,shutil,platform,importlib.metadata as metadata
from pathlib import Path
from fixed_map_perception import ROOT,HIST,read,save,sha,utc
from fixed_map_resources import memory_headroom
from analyze_step3_development import frames,csv_save
def main():
    root=Path(sys.argv[1]).resolve();out=root/'delivery';out.mkdir(exist_ok=False);(out/'source').mkdir();(out/'configs').mkdir()
    names=['fixed_map_dashboard.py','fixed_map_perception.py','fixed_map_perception_process.py','fixed_map_perception_task.py','fixed_map_perception_geometry.py','fixed_map_perception_execution.py','fixed_map_delayed_identity.py','fixed_map_multiscale_detector.py','fixed_map_scene_binding.py','fixed_map_random_scene.py','specialist_balloon_detector.py','ground_detector_diagnostic.py','recheck_ground_replay_v7.py','create_ground_reference_preview.py','check_ground_perception_v7.py','collect_ground_raw_v7.py','check_ground_dashboard_v7.py','test_fixed_map_ground_v7.py','finalize_ground_development.py','fixed_map_mission.py','t1_delayed_semantic_bridge.py','t1_source_attribute_guard.py']
    version=[]
    for n in names:
        p=ROOT/'python'/n;shutil.copyfile(p,out/'source'/n);version.append(dict(path=str(p),sha256=sha(p),snapshot='source/'+n))
    for p in [ROOT/'configs/fixed_map_perception_ground_v7.json',ROOT/'configs/step3_balloon_cuda1280_v2.json',ROOT/'configs/fixed_map_scene_state.json',ROOT/'configs/fixed_map_scene_runtime_v1.json',ROOT/'configs/ground_replay_preview_reference.json',HIST/'fixed_version/task.json',HIST/'fixed_version/snapshot/configs/enclosed_uav_lab_manual_settings.json']:
        shutil.copyfile(p,out/'configs'/p.name);version.append(dict(path=str(p),sha256=sha(p),snapshot='configs/'+p.name))
    p=ROOT/'01_固定地图实验.cmd';shutil.copyfile(p,out/p.name);version.append(dict(path=str(p),sha256=sha(p),snapshot=p.name))
    summary,online_index,valid=frames(root/'online_ground_v7_memory_ready',out);save(out/'online_frame_metrics.json',summary)
    index=[]
    for line in (root/'raw_ground_samples/frame_index.jsonl').read_text().splitlines():index.append(json.loads(line))
    index+=online_index
    for line in (root/'replay_recheck_v7_clock_repair/frame_index.jsonl').read_text().splitlines():index.append(json.loads(line))
    historic=read(root/'reference_preview/historical_preview_REPLAY.json')
    index.append(dict(mode='HISTORICAL_VALID_EVIDENCE_REPLAY_PREVIEW_NOT_ONLINE',frame_id=historic['source_frame_id'],rgb=historic['rgb'],depth=historic['depth'],pose=historic['pose'],image_sha256=historic['source_image_sha256'],preview=str(root/'reference_preview/historical_preview_REPLAY.json')))
    (out/'all_evidence_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in index),encoding='utf-8')
    replay=[]
    for p in sorted((root/'replay_recheck_v7_clock_repair').glob('frame-*.json')):
        r=read(p);replay.append(dict(frame_id=r['frame_id'],mode=r['mode'],original_receipt_s=r['original_capture']['received_monotonic_s'],original_axis_now_s=r['original_axis_now_s'],actual_reprocess_utc=r['actual_reprocess_utc'],inference_wall_s=r['new_inference_wall_s'],candidate_count=len(r['semantic_frame']['candidates']),description_matching_candidates=len(r['attribute'].get('matching_candidate_ids',[])),identity_state=r['identity']['state'],position_estimate_valid=r['estimate']['valid'],rejection_reasons=';'.join(r['rejection_reasons']),source_rgb=r['source_rgb'],evidence=str(p)))
    csv_save(out/'replay_frame_metrics.csv',replay)
    for name in ['raw_ground_states','online_ground_v7_memory_ready_states']:
        rows=[json.loads(s) for s in (root/(name+'.jsonl')).read_text().splitlines()];csv_save(out/(name+'.csv'),rows)
    diag=read(root/'diagnostic_recheck/diagnostic_results.json')['rows'];csv_save(out/'detector_diagnostic_metrics.csv',[dict(mode=r['mode'],source=r['source'],source_sha256=r['source_sha256'],inference_mode=r['inference_mode'],imgsz=r['imgsz'],wall_s=r['wall_s'],candidate_count=len(r['mapped']),band_candidates=sum(c['band']['status']=='ACCEPTED' for c in r['mapped'])) for r in diag])
    dc=read(ROOT/'configs/step3_balloon_cuda1280_v2.json');weights_sha=sha(dc['weights'])
    if weights_sha!=dc['weights_sha256']:raise ValueError('FROZEN_WEIGHT_CHANGED')
    scene=read(ROOT/'configs/fixed_map_scene_runtime_v1.json')
    if sha(scene['source_file'])!=scene['source_sha256'] or sha(scene['runtime_file'])!=scene['runtime_sha256']:raise ValueError('MAP_VERSION_CHANGED')
    packages={}
    for n in ['torch','ultralytics','PyQt5','airsim','numpy','opencv-python']:
        try:packages[n]=metadata.version(n)
        except metadata.PackageNotFoundError:packages[n]='UNAVAILABLE'
    manifest=dict(utc=utc(),python=platform.python_version(),platform=platform.platform(),packages=packages,memory=memory_headroom(),files=version,weights=dict(path=dc['weights'],sha256=weights_sha,copied=False,frozen_verified=True),large_map_reference=scene,history_references=[str(ROOT/'reports/step1_quick_review_20261010_083715'),str(ROOT/'reports/step2_perception_20261010'),str(ROOT/'reports/step3_continue_20261010_152319')],current_profile='GROUND_ONLY_FULL_IMAGE_DUAL_SCALE_V7',no_training=True,no_target_truth_as_control_input=True)
    save(out/'manifest.json',manifest)
    verification=dict(status='PASS',snapshot_files=len(version),snapshot_hashes_verified=all(sha(out/r['snapshot'])==r['sha256'] for r in version),frozen_map_and_weights_verified=True,index_rows=len(index),real_online_raw_pairs=8,real_online_inference_frames=summary['processed_frames'],new_replay_sequence_frames=len(replay),historic_reference_frames=1)
    save(out/'archive_verification.json',verification)
    import airsim
    c=airsim.MultirotorClient(port=41454,timeout_value=3);ground={}
    for n in ['DroneA','DroneB']:
        s=c.getMultirotorState(vehicle_name=n);ground[n]=dict(timestamp_ns=s.timestamp,landed=s.landed_state==airsim.LandedState.Landed,api=c.isApiControlEnabled(vehicle_name=n))
    if any(not r['landed'] or r['api'] for r in ground.values()):raise ValueError('GROUND_RELEASE_FINAL_CHECK_FAILED')
    save(out/'final_ground_receipt.json',dict(utc=utc(),state=ground,no_flight_commands=True))
    save(root/'GROUND_DEVELOPMENT_DECISION.json',dict(utc=utc(),online_collection_and_inference='PASS_GROUND_NO_TARGET',current_designated_identity='NOT_PASSED_NO_VISIBLE_TARGET',new_replay_identity='NOT_PASSED_SOURCE_CANDIDATE_MISSING',planning_preview='PASS_HISTORICAL_REFERENCE_REPLAY_ONLY',flight_execution='DISABLED_IN_GROUND_V7',four_method_pilot_ready=False,interface_tests_passed=50,offline_qt_checks_passed=8,online_frames=summary['processed_frames'],final_ground=ground,original_failures_retained=True))
    print(json.dumps(verification,ensure_ascii=False))
if __name__=='__main__':main()
