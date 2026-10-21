"""YOLO-only reprocessing of saved real frames; isolated from online commands."""
import copy,sys,time,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from fixed_map_perception import ROOT,HIST,read,save,sha
from fixed_map_perception_geometry import surface_from_box,make_replay_preview,perception_safety_parameters
from fixed_map_perception_execution import static_geometry_v2,TrackingMarginGeometry
from fixed_map_yolo_identity import YoloImageBackend,YoloAssociation,AssociationConfig

def main(root):
    out=Path(root)/('replay_yolo_v8'+(sys.argv[2] if len(sys.argv)>2 else ''));out.mkdir(exist_ok=False);cfg=read(ROOT/'configs/fixed_map_perception_yolo_v8.json');dc=read(ROOT/cfg['detector_config'])
    from train_t1_balloon_yolo import configure_offline_runtime
    configure_offline_runtime()
    import torch
    torch.set_num_threads(1);torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.5,0)
    from specialist_balloon_detector import SpecialistBalloonDetector
    from fixed_map_multiscale_detector import MultiScaleBalloonDetector
    if sha(dc['weights'])!=dc['weights_sha256']:raise ValueError('WEIGHT_HASH_CHANGED')
    adapter=YoloImageBackend(MultiScaleBalloonDetector(SpecialistBalloonDetector(dc['weights'],confidence=dc['confidence'],imgsz=dc['imgsz'],device=dc['device'],nms_iou=dc['nms_iou'])))
    source=ROOT/'reports/step3_continue_20261010_152319/online_base_v6/frames';adapter.warmup(source/'frame-000250/rgb.png')
    first=read(source/'frame-000250/capture.json');tracker=YoloAssociation(first['session_id'],first['target_generation'],cfg['target_description'],AssociationConfig())
    safety=perception_safety_parameters(read(HIST/'fixed_version/task.json'));g=TrackingMarginGeometry(static_geometry_v2(read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives'])[0],.75,.15);rows=[]
    for seq in range(250,267):
        folder=source/f'frame-{seq:06d}';meta=read(folder/'capture.json');begin=time.perf_counter();frame=adapter.detect_frame(folder/'rgb.png',session_id=meta['session_id'],target_generation=meta['target_generation'],frame_id=meta['frame_id'],frame_sequence=seq,readback_receipt_monotonic_s=meta['received_monotonic_s'],expected_image_sha256=meta['image_sha256']);duration=time.perf_counter()-begin;now=meta['received_monotonic_s']+duration;track=tracker.update(frame,now=now);positions=[]
        for c in frame.candidates:
            try:p=surface_from_box(c.bbox_xyxy_pixels,np.load(folder/'depth_ray_range_m.npy'),meta,cfg,now)
            except ValueError as ex:p=dict(valid=False,reason=str(ex))
            positions.append(dict(candidate_id=c.candidate_id,position=p))
        p=next((p['position'] for p in positions if p['candidate_id']==track['selected_candidate_id']),None)
        estimate={k:meta[k] for k in ['session_id','target_generation','description_sha256','frame_id','image_sha256','received_monotonic_s']};estimate.update(mode='replay',valid=track['state']=='TRACKING' and p is not None and p['valid'],perception_backend='yolo_only_v8',target_state=track['state'],semantic_anchor=track['semantic_anchor'],semantic_review_deadline_monotonic_s=track['semantic_review_deadline_monotonic_s'],meaning='visible_surface_point_not_center')
        preview=None
        if estimate['valid']:
            estimate.update(surface=p,surface_world_xyz_m=p['surface_world_xyz_m'])
            try:
                preview=make_replay_preview(estimate,meta['start_world_xyz_m'],g,safety,cfg,now)
                preview['task']['source_kind']='recorded_RGB_depth_YOLO_attributes_OFFLINE_v8'
            except ValueError as ex:preview=dict(status='REJECTED',reason=str(ex),execution_allowed=False)
        row=dict(mode='REPLAY_NOT_ONLINE_NOT_EXECUTABLE',source_capture=meta,semantic_frame=frame.to_dict(),temporal_identity=track,candidate_positions=positions,position_estimate=estimate,planning_preview=preview,inference_wall_s=duration,external_review_calls=0,flight_commands=0);save(out/(frame.frame_id+'.json'),row);rows.append(row)
    save(out/'summary.json',dict(mode='REPLAY_NOT_ONLINE_NOT_EXECUTABLE',frames=len(rows),candidate_frames=sum(bool(r['semantic_frame']['candidates']) for r in rows),valid_surface_frames=sum(r['position_estimate']['valid'] for r in rows),valid_preview_frames=sum(bool(r['planning_preview']) and r['planning_preview']['status']=='REPLAY_STATIC_PREVIEW_VALID' for r in rows),external_review_calls=0,flight_commands=0,inference_max_s=max(r['inference_wall_s'] for r in rows)))
    print(json.dumps(read(out/'summary.json')))

if __name__=='__main__':main(sys.argv[1])
