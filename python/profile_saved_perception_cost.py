"""Offline timing of existing raw frames; never supplies current evidence."""
import sys,time,statistics
from pathlib import Path
from dataclasses import replace
import numpy as np
from fixed_map_perception import read,save
from t1_delayed_semantic_bridge import _frame_from_dict
from fixed_map_perception_execution import current_attribute_audit
from fixed_map_perception_geometry import surface_from_box

root=Path(sys.argv[1]).resolve();cfg=read(root/'online_trial/online/version_manifest.json')['config'];rows=[];profiles=[]
for path in sorted((root/'online_trial/online/frames').glob('*/perception.json')):
    r=read(path);m=r['capture'];frame=r['semantic_frame'];candidate_ready=frame['candidates_ready_monotonic_s']
    rows.append(dict(frame_id=r['frame_id'],candidate_count=len(frame['candidates']),recorded_valid=r['valid'],association_state=r['temporal_identity']['state'],pre_inference_s=candidate_ready-m['received_monotonic_s']-r['inference_wall_s'],inference_s=r['inference_wall_s'],post_detector_to_decision_s=r['processed_monotonic_s']-candidate_ready,total_receipt_to_decision_s=r['processed_monotonic_s']-m['received_monotonic_s']))
    if r['temporal_identity']['state']!='TRACKING' or len(profiles)>=12:continue
    f=replace(_frame_from_dict(frame),encoded_image=Path(m['rgb_path']).read_bytes())
    started=time.perf_counter();a=current_attribute_audit(f,cfg['target_description'],None);first=time.perf_counter()-started
    selected=r['temporal_identity']['selected_candidate_id'];started=time.perf_counter();b=current_attribute_audit(f,cfg['target_description'],selected);second=time.perf_counter()-started
    depth=np.load(m['depth_path']);started=time.perf_counter();depth_reasons=[]
    for c in f.candidates:
        try:surface_from_box(c.bbox_xyxy_pixels,depth,m,cfg,m['received_monotonic_s']+.1)
        except ValueError as ex:depth_reasons.append(str(ex))
    profiles.append(dict(frame_id=f.frame_id,attribute_first_s=first,attribute_second_s=second,all_candidate_depth_s=time.perf_counter()-started,depth_reasons=depth_reasons,attribute_first_matches=a.get('matching_candidate_ids'),attribute_second_status=b['status']))
from analyze_step3_development import csv_save
csv_save(root/'audit/recorded_processing_stage_times.csv',rows)
save(root/'OFFLINE_PROCESSING_PROFILE.json',dict(mode='OFFLINE_SAVED_IMAGE_TIMING_NO_ONLINE_FRESHNESS_CLAIM',profiles=profiles,stage_medians={k:statistics.median(r[k] for r in rows) for k in ['pre_inference_s','inference_s','post_detector_to_decision_s','total_receipt_to_decision_s']},current_attribute_mean_s=statistics.mean(p['attribute_first_s']+p['attribute_second_s'] for p in profiles),depth_mean_s=statistics.mean(p['all_candidate_depth_s'] for p in profiles)))
print('Offline saved-frame processing profile saved')
