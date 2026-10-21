"""Explicit historical replay; no RPC, truth, online estimate or motion input.

Closed extraction keeps scene actor transforms, segmentation and evaluation
fields outside the replay estimator. Historical certificates retain their
original description, policy and clocks. An optional current VLX transport
probe on one archived image never turns that old image into current evidence.
"""
import argparse, copy, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
from fixed_map_perception import ROOT, read, save, sha, utc
from fixed_map_perception_geometry import surface_from_box
from t1_semantic_perception import Candidate, SemanticFrame
from t1_delayed_semantic_bridge import verify_delayed_selection

def replay(source, output, probe_vlx=False):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=read(source);cfg=read(ROOT/'configs/fixed_map_perception_v1.json')
    manifest=dict(mode='historical_replay',source_report=str(source),source_report_sha256=sha(source),source_started_utc=report['started_utc'],source_finished_utc=report['finished_utc'],source_description=report['configuration']['description'],current_description=cfg['target_description'],descriptions_equal=report['configuration']['description']==cfg['target_description'],historical_weights_sha256=report['configuration']['weights_sha256'],current_weights_sha256=read(ROOT/cfg['detector_config'])['weights_sha256'],original_virtual_camera_relocated=report['virtual_camera_relocated'],scene='historical controlled T1 scene; NOT current enclosed fixed map',truth_fields_consumed=False,segmentation_opened=False,flight_commands_issued=False,clock_rebased=False,certificate_policy='original delayed-chain 8s; does not change online 2s policy',started_utc=utc())
    save(output/'version_manifest.json',manifest);indices=[]
    for row in report['frames'][:3]:
        # Only declared media RGB/depth and observation/self-clock metadata.
        media={k:copy.deepcopy(row['media'][k]) for k in ['rgb','depth']}
        for m in media.values():
            if sha(m['path'])!=m['sha256']:raise ValueError('REPLAY_MEDIA_HASH_MISMATCH')
        obs=row['observer_result'];sf=obs['semantic_frame'];image=Path(media['rgb']['path']).read_bytes()
        candidates=tuple(Candidate(c['candidate_id'],tuple(c['bbox_xyxy_pixels']),c['detector_score'],c['source_class_name'],c['truncated']) for c in sf['candidates'])
        frame=SemanticFrame(sf['session_id'],sf['target_generation'],sf['frame_id'],sf['frame_sequence'],sf['image_sha256'],*sf['image_size_wh'],sf['readback_receipt_monotonic_s'],sf['candidates_ready_monotonic_s'],candidates,image,json.dumps(sf['detector_report']))
        now=row['surface_started_monotonic_s'];folder=output/frame.frame_id;folder.mkdir()
        poses=[dict(position_local_ned_m=p['position_local_ned_m'],orientation_xyzw=p['orientation_xyzw'],world_from_local_translation_ned_m=row['world_from_local_translation_ned_m']) for p in row['response_camera_poses'][:2]]
        meta=dict(mode='historical_replay',frame_id=frame.frame_id,rgb_size_wh=sf['image_size_wh'],depth_size_wh=sf['image_size_wh'],rgb_K=row['scene_intrinsics']['K_nominal'],depth_K=row['scene_intrinsics']['K_nominal'],rgb_pose=poses[0],depth_pose=poses[1],rgb_timestamp_ns=int(row['response_camera_poses'][0]['render_readback_timestamp_ns']),depth_timestamp_ns=int(row['response_camera_poses'][1]['render_readback_timestamp_ns']),received_monotonic_s=row['capture_received_host_s'],timestamps_advanced=True,capture_errors=[],source_host_clock_preserved=True,rgb_path=media['rgb']['path'],depth_path=media['depth']['path'],image_sha256=sf['image_sha256'])
        depth=np.load(media['depth']['path'],allow_pickle=False);positions=[];verified=None;reasons=[]
        try:verified=verify_delayed_selection(frame,obs['temporal_identity'],row['description'],now)
        except ValueError as ex:reasons.append(str(ex))
        for c in frame.candidates:
            try:position=surface_from_box(c.bbox_xyxy_pixels,depth,meta,cfg,now)
            except ValueError as ex:position=dict(valid=False,reason=str(ex));reasons.append(str(ex))
            positions.append(dict(candidate_id=c.candidate_id,position=position,designated_in_original_replay=bool(verified and c.candidate_id==verified['selected_candidate_id'])))
        result=dict(mode='historical_replay',source_report=str(source),source_report_sha256=manifest['source_report_sha256'],frame_id=frame.frame_id,media=media,capture=meta,semantic_frame=sf,original_description=row['description'],historical_semantic_poll=obs['semantic_poll'],historical_semantic_request=obs['semantic_request'],historical_temporal_identity=obs['temporal_identity'],historical_transport_trace=obs['sanitized_transport_trace'],certificate_verification=verified,candidate_positions=positions,current_designated_instance_confirmed=False,planning_preview=dict(status='REJECTED',reason='REPLAY_SCENE_AND_DESCRIPTION_NOT_CURRENT_ONLINE_EVIDENCE',flight_authorized=False),rejection_reasons=reasons+['NOT_CURRENT_MAP_OR_CURRENT_TARGET_VERSION'],logical_replay_now=now,clock_rebased=False)
        save(folder/'perception.json',result)
        rendered=cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR)
        for c in frame.candidates:
            x,y,u,v=map(int,c.bbox_xyxy_pixels);cv2.rectangle(rendered,(x,y),(u,v),(0,255,255),2);cv2.putText(rendered,c.candidate_id[-8:],(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
        cv2.putText(rendered,'HISTORICAL REPLAY / ORIGINAL CLOCK AND DESCRIPTION',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1);cv2.imwrite(str(folder/'overlay.png'),rendered)
        indices.append(dict(frame_id=frame.frame_id,perception_path=str(folder/'perception.json'),media=media,original_clock=now,certificate_verified=verified is not None,surface_count=sum(p['position']['valid'] for p in positions)))
    with (output/'frame_index.jsonl').open('w',encoding='utf-8') as f:
        for i in indices:f.write(json.dumps(i,ensure_ascii=False)+'\n')
    if probe_vlx:
        from dotenv import load_dotenv
        from vlx_api_client import VLXSeekAPIClient
        from PIL import Image
        import base64, io
        load_dotenv(ROOT/'.env');probe=dict(mode='archived_image_transport_probe',current_description=cfg['target_description'],target_generation=cfg['target_generation'],frame_id=indices[0]['frame_id'],source_image_sha256=indices[0]['media']['rgb']['sha256'],requested_utc=utc(),request_host_monotonic_s=time.monotonic(),source_is_old=True,online_identity_valid=False,flight_authorized=False)
        save(output/'vlx_request.json',probe)
        try:
            client=VLXSeekAPIClient(timeout=cfg['vlx_http_timeout_s'],max_retries=1)
            upload=client.upload_image;convert=client._convert_response
            def bind_upload(image,format='PNG'):
                url=upload(image,format='PNG');encoded=base64.b64decode(url.split(',',1)[1]);(output/'vlx_transmitted.png').write_bytes(encoded)
                probe.update(transmitted_image_sha256=hashlib.sha256(encoded).hexdigest(),model=client.model,format='PNG');save(output/'vlx_request.json',probe);return url
            def bind_reply(response,boxes):save(output/'vlx_raw_reply.json',response);return convert(response,boxes)
            client.upload_image=bind_upload;client._convert_response=bind_reply
            result=client.ground(Image.open(indices[0]['media']['rgb']['path']).convert('RGB'),[],cfg['target_description'],lang='zh',temperature=0.,max_new_tokens=128)
            elapsed=time.monotonic()-probe['request_host_monotonic_s']
            reply=dict(status='TRANSPORT_RESPONSE_RECEIVED',result=result,elapsed_s=elapsed,received_utc=utc(),source_binding=probe,online_adoption='REJECTED_ARCHIVED_FRAME',online_deadline_met=elapsed<=cfg['vlx_response_timeout_s'],flight_authorized=False)
        except Exception as ex:reply=dict(status='TRANSPORT_PROBE_FAILED',reason=type(ex).__name__,online_adoption='REJECTED_ARCHIVED_FRAME',flight_authorized=False)
        save(output/'vlx_reply.json',reply)
    save(output/'summary.json',dict(frames=len(indices),certificates_verified=sum(i['certificate_verified'] for i in indices),surfaces=sum(i['surface_count'] for i in indices),current_online_confirmations=0,flight_commands_issued=False))
    return indices

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--probe-vlx',action='store_true');a=p.parse_args();print(json.dumps(replay(a.source,a.output,a.probe_vlx),ensure_ascii=False))
