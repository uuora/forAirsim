"""One dual-scale recheck on a saved real sequence and saved VLX reply.

New inference/identity on the original time axis is a counterfactual replay,
never the original online result, and never fresh online evidence.
"""
import sys,time,json,copy,hashlib
from pathlib import Path
from dataclasses import replace
from concurrent.futures import Future
from collections import Counter
import cv2,numpy as np
from fixed_map_perception import ROOT,HIST,read,save,sha,utc
from t1_semantic_perception import YoloVlxBackend,SemanticReply
from t1_delayed_semantic_bridge import DelayedSemanticPolicy,DelayedBridgeConfig,verify_delayed_selection
from fixed_map_delayed_identity import ReviewHandoffTracker
from t1_source_attribute_guard import verify_source_attribute
from fixed_map_perception_execution import current_attribute_audit,static_geometry_v2,TrackingMarginGeometry
from fixed_map_perception_geometry import surface_from_box,make_replay_preview,perception_safety_parameters

class ManualReplySlot:
    def submit(self,operation,request):
        self.request=request;self.future=Future();self.future.set_running_or_notify_cancel();return self.future

def main():
    out=Path(sys.argv[1]).resolve()/('replay_recheck_v7'+(sys.argv[2] if len(sys.argv)>2 else ''));out.mkdir(exist_ok=False)
    original=ROOT/'reports/step3_continue_20261010_152319/online_base_v6'
    events=[json.loads(s) for s in (original/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    saved=next(r for r in events if r['kind']=='VLX_REPLY' and r['binding']['frame_id']=='frame-000250');binding=saved['binding']
    cfg=read(ROOT/'configs/fixed_map_perception_ground_v7.json');dc=read(ROOT/cfg['detector_config'])
    save(out/'declaration.json',dict(utc=utc(),mode='REPLAY_REPROCESSING_NOT_ORIGINAL_ONLINE_NOT_EXECUTABLE',first_frame=250,last_frame=266,semantic_reply_source=str(original/'events.jsonl'),semantic_reply_binding=binding,semantic_reply_completed_original_s=saved['completed_monotonic_s'],source_timestamps_not_rebased=True,original_axis='stored receipt + newly measured detection/attribute processing latency; explicitly counterfactual timing',network_calls=0,flight_commands=0,repair_version='full_image_dual_scale_v7',no_truth=True,maximum_rechecks=1))
    from train_t1_balloon_yolo import configure_offline_runtime
    configure_offline_runtime()
    import torch
    torch.set_num_threads(1);torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.5,0)
    from specialist_balloon_detector import SpecialistBalloonDetector
    from fixed_map_multiscale_detector import MultiScaleBalloonDetector
    if sha(dc['weights'])!=dc['weights_sha256']:raise ValueError('FROZEN_WEIGHT_HASH_MISMATCH')
    detector=MultiScaleBalloonDetector(SpecialistBalloonDetector(dc['weights'],confidence=dc['confidence'],imgsz=1280,device='0',nms_iou=dc['nms_iou']))
    clock=[0.];client=type('NoNetwork',(),dict(request_mode='image_text',proposals_transmitted=False))()
    adapter=YoloVlxBackend(detector=detector,vlx_client=client,clock=lambda:clock[0]);adapter.warmup(original/'frames/frame-000250/rgb.png')
    slot=ManualReplySlot();policy=DelayedSemanticPolicy(lambda r:None,session_id=binding['session_id'],target_generation=binding['target_generation'],description=cfg['target_description'],config=DelayedBridgeConfig(response_timeout_s=6.),executor=slot);policy._serial=int(binding['request_id'].rsplit('-',1)[1])-1
    tracker=ReviewHandoffTracker(binding['session_id'],binding['target_generation'],cfg['target_description'])
    parts,_=static_geometry_v2(read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives']);geometry=TrackingMarginGeometry(parts,.75,.15);safety=perception_safety_parameters(read(HIST/'fixed_version/task.json'))
    records=[];previews=[];index=[];source_audit=None
    for seq in range(250,267):
        source=original/'frames'/f'frame-{seq:06d}';meta=read(source/'capture.json');receipt=meta['received_monotonic_s'];clock[0]=receipt
        begin=time.perf_counter();frame=adapter.detect_frame(source/'rgb.png',session_id=binding['session_id'],target_generation=binding['target_generation'],frame_id=meta['frame_id'],frame_sequence=seq,readback_receipt_monotonic_s=receipt,expected_image_sha256=meta['image_sha256']);inference=time.perf_counter()-begin
        frame=replace(frame,candidates_ready_monotonic_s=receipt+inference);now=receipt+inference+.005
        if seq==250:
            submitted=policy.request(frame,now=binding['submitted_monotonic_s'])
            if submitted['status']!='SUBMITTED':raise ValueError('REPLAY_SOURCE_REQUEST_REJECTED_'+submitted['reason'])
            if slot.request.binding()!=binding:raise ValueError('SAVED_REPLY_BINDING_NOT_REPRODUCED')
            reply=SemanticReply(binding,saved['completed_monotonic_s'],saved['result']);source_audit=verify_source_attribute(slot.request,reply);save(out/'saved_reply_source_attribute_recheck.json',source_audit)
            now=max(now,binding['submitted_monotonic_s'])
        if not slot.future.done() and now>=saved['completed_monotonic_s']:
            if source_audit['status']=='ACCEPTED':slot.future.set_result(reply)
            else:slot.future.set_exception(RuntimeError('SOURCE_RGB_ATTRIBUTE_NOT_VERIFIED_'+source_audit['reason']))
        semantic=policy.poll(frame,now=now);tracking=tracker.update(frame,semantic if semantic['status']=='SEMANTIC_LOCAL_ASSOCIATION_VALID' else None,now)
        attr=current_attribute_audit(frame,cfg['target_description'],tracking.get('selected_candidate_id'))
        positions=[];depth=np.load(source/'depth_ray_range_m.npy')
        for c in frame.candidates:
            try:position=surface_from_box(c.bbox_xyxy_pixels,depth,meta,cfg,now)
            except ValueError as ex:position=dict(valid=False,reason=str(ex))
            positions.append(dict(candidate_id=c.candidate_id,position=position))
        selected=next((c for c in positions if c['candidate_id']==tracking.get('selected_candidate_id')),None)
        estimate={k:meta[k] for k in ['session_id','target_generation','description_sha256','frame_id','image_sha256','received_monotonic_s']}
        estimate.update(mode='replay',valid=False,target_state=tracking['state'],semantic_review_deadline_monotonic_s=tracking.get('semantic_review_deadline_monotonic_s'),meaning='visible_surface_point_not_center',flight_authorized=False)
        if tracking['state']=='TRACKING' and attr['status']=='ACCEPTED' and selected and selected['position']['valid']:
            verify_delayed_selection(frame,tracking,cfg['target_description'],now)
            estimate.update(valid=True,surface_world_xyz_m=selected['position']['surface_world_xyz_m'],surface=selected['position'],semantic_anchor=tracking['semantic_anchor'])
        plan=None
        if estimate['valid']:
            start=meta['start_world_xyz_m'];start[2]=3.1
            try:
                plan=make_replay_preview(estimate,start,geometry,safety,cfg,now)
                from fixed_map_perception_task import candidate_approach_guard
                plan['sensor_guard']=candidate_approach_guard(plan['path'],dict(candidate_positions=positions),cfg);previews.append(dict(frame_id=frame.frame_id,plan=plan))
            except ValueError as ex:plan=dict(mode='REPLAY_NOT_EXECUTABLE',status='REJECTED',reason=str(ex),flight_authorized=False)
        reasons=[]
        if not estimate['valid']:reasons=[attr['reason'],semantic['reason'],tracking['reason']]
        row=dict(mode='REPLAY_REPROCESSING_NOT_ONLINE_NOT_EXECUTABLE',frame_id=frame.frame_id,actual_reprocess_utc=utc(),original_axis_now_s=now,original_capture=meta,source_rgb=str(source/'rgb.png'),source_depth=str(source/'depth_ray_range_m.npy'),source_pose=str(source/'capture.json'),semantic_frame=frame.to_dict(),new_inference_wall_s=inference,candidate_positions=positions,semantic_decision=semantic,identity=tracking,attribute=attr,estimate=estimate,planning_preview=plan,rejection_reasons=reasons)
        save(out/(frame.frame_id+'.json'),row);records.append(row);index.append(dict(frame_id=frame.frame_id,reprocess=str(out/(frame.frame_id+'.json')),rgb=row['source_rgb'],depth=row['source_depth'],pose=row['source_pose'],image_sha256=meta['image_sha256'],depth_sha256=meta['depth_sha256'],mode='REPLAY_REPROCESSING'))
        if seq==266:
            canvas=cv2.imread(str(source/'rgb.png'))
            for c in frame.candidates:
                x0,y0,x1,y1=map(round,c.bbox_xyxy_pixels);cv2.rectangle(canvas,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.imwrite(str(out/'last_overlay_REPLAY.png'),canvas)
    policy.close();tracker.close()
    (out/'frame_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index),encoding='utf-8')
    summary=dict(utc=utc(),mode='REPLAY_REPROCESSING_NOT_ONLINE_NOT_EXECUTABLE',frames=len(records),candidate_frames=sum(bool(r['semantic_frame']['candidates']) for r in records),confirmed_surface_frames=sum(r['estimate']['valid'] for r in records),replay_valid_plans=len(previews),inference_max_s=max(r['new_inference_wall_s'] for r in records),source_attribute_status=source_audit['status'],rejections=dict(Counter(x for r in records for x in r['rejection_reasons'])),last_row=str(out/'frame-000266.json'),network_calls=0,flight_commands=0,current_random_scene_online_positive=False)
    save(out/'summary.json',summary)
    if previews:save(out/'selected_preview_REPLAY.json',dict(mode='REPLAY_REPROCESSING_NOT_ONLINE_NOT_EXECUTABLE',selected=previews[-1],source_evidence=str(out/(previews[-1]['frame_id']+'.json')),overlay=str(out/'last_overlay_REPLAY.png'),flight_authorized=False))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
