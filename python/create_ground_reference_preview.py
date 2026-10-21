"""Verify an existing historical positive and build an explicitly offline preview."""
import sys,copy,time,json
from pathlib import Path
from fixed_map_perception import ROOT,HIST,read,save,sha,utc
from fixed_map_perception_geometry import make_replay_preview,perception_safety_parameters
from fixed_map_perception_execution import static_geometry_v2,TrackingMarginGeometry
from fixed_map_perception_task import candidate_approach_guard
from t1_delayed_semantic_bridge import _frame_from_dict,verify_delayed_selection
def main():
    out=Path(sys.argv[1]).resolve()/'reference_preview';out.mkdir(exist_ok=False)
    source=ROOT/'reports/step3_development_20261010_101819/online_designated_positive_v4.json'
    data=read(source);row=data['report'];original=row['processed_monotonic_s'];cfg=read(ROOT/'reports/step3_development_20261010_101819/base_trial_v4_config.json')
    certificate=verify_delayed_selection(_frame_from_dict(row['semantic_frame']),row['temporal_identity'],cfg['target_description'],original)
    estimate=copy.deepcopy(row['position_estimate']);estimate['mode']='replay'
    parts,_=static_geometry_v2(read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives']);geometry=TrackingMarginGeometry(parts,.75,.15);safety=perception_safety_parameters(read(HIST/'fixed_version/task.json'))
    start=data['actual_self']['position_xyz_m'];start[2]=3.1;begin=time.perf_counter();preview=make_replay_preview(estimate,start,geometry,safety,cfg,original);preview['sensor_guard']=candidate_approach_guard(preview['path'],row,cfg)
    result=dict(mode='HISTORICAL_VALID_EVIDENCE_REPLAY_PREVIEW_NOT_ONLINE',utc=utc(),source=str(source),source_sha256=sha(source),source_detector_version='v4 CUDA1280; NOT new dual-scale v7',source_frame_id=row['frame_id'],source_image_sha256=row['capture']['image_sha256'],rgb=row['capture']['rgb_path'],depth=row['capture']['depth_path'],pose=str(Path(row['capture']['rgb_path']).parent/'capture.json'),overlay=row.get('overlay_path',row['capture']['rgb_path']),source_capture_utc=row['capture']['received_utc'],original_processing_monotonic_s=original,actual_preview_monotonic_s=time.monotonic(),original_time_certificate=certificate,estimate=estimate,recorded_start_xyz_m=start,preview=preview,preview_wall_s=time.perf_counter()-begin,flight_authorized=False,current_randomized_target_position_inferred=False)
    path=out/'historical_preview_REPLAY.json';save(path,result)
    save(ROOT/'configs/ground_replay_preview_reference.json',dict(path=str(path),sha256=sha(path),mode=result['mode']))
    print(json.dumps(dict(status=preview['status'],mode=result['mode'],path_vertices=len(preview['path']),wall_s=result['preview_wall_s'],flight_authorized=False)))
if __name__=='__main__':main()
