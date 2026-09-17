"""Capture actual onboard PNGs and source-labelled poses, without flight commands."""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import airsim
import cv2
import numpy as np
from camera_geometry import scene_intrinsics

ROOT=Path(__file__).resolve().parents[1]
def vector(v): return [float(v.x_val),float(v.y_val),float(v.z_val)]
def quaternion(q): return [float(q.w_val),float(q.x_val),float(q.y_val),float(q.z_val)]
def pose(p): return {'position_m':vector(p.position),'orientation_wxyz':quaternion(p.orientation)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=int,default=10)
    ap.add_argument('--interval',type=float,default=.5)
    ap.add_argument('--ground-truth',action='store_true',help='Write separate simulator evaluation annotations')
    ap.add_argument('--scenario',choices=['baseline','low_sun','raised_waves','occlusion'])
    args=ap.parse_args()
    if args.pairs<1 or args.interval<0: ap.error('pairs >= 1 and interval >= 0 required')
    out=ROOT/'datasets'/'dual_camera'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    settings=(ROOT/'configs/maritime_harbor_settings.json').read_bytes()
    (out/'settings.json').write_bytes(settings)
    cfg=json.loads(settings)
    c=airsim.MultirotorClient(port=41452,timeout_value=30)
    if not {'DroneA','DroneB'}.issubset(c.listVehicles()): raise RuntimeError('Both vehicles required')
    # Require this live process to actually use the saved camera configuration.
    live=json.loads(c.getSettingsString())
    if live['Vehicles']!=cfg['Vehicles']: raise RuntimeError('Restart harbor to load updated camera settings')
    report={'status':'RUNNING','camera_metadata_version':2,'flight_commands_issued':False,'settings_sha256':hashlib.sha256(settings).hexdigest(),'pairing':'sequential_RPC_not_synchronized_stereo','pose_frame':'vehicle-specific local NED, metres; +X north +Y east +Z down; quaternion wxyz','pose_scope':'simulator camera/body pose, not an estimated target position','labels':'NOT_CREATED','detector':'NOT_RUN','pairs':[]}
    annotation_context=None
    try:
        if args.scenario:
            import maritime_experiments
            report['experiment']=maritime_experiments.apply(c,args.scenario)
            for filename in ('maritime_experiments.json','maritime_targets.json'):
                (out/filename).write_bytes((ROOT/'configs'/filename).read_bytes())
            assets=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/MaritimeHarborV2'
            provenance=[ROOT/'scripts/unreal/build_maritime_harbor.py',ROOT/'python/capture_dual_cameras.py',ROOT/'python/camera_geometry.py',ROOT/'python/maritime_annotations.py',ROOT/'python/maritime_experiments.py']+list(assets.rglob('*.uasset'))+list(assets.rglob('*.umap'))
            report['experiment']['asset_and_code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance}
            (out/'experiment.json').write_text(json.dumps(report['experiment'],indent=2))
        if args.ground_truth:
            import maritime_annotations
            annotation_context=maritime_annotations.setup(c,out)
            report['labels']='SIMULATOR_ANNOTATIONS_REQUIRE_REVIEW'
        with (out/'frames.jsonl').open('w',encoding='utf-8') as records:
            for i in range(args.pairs):
                stamps=[]; views=[]
                for name in ('DroneA','DroneB'):
                    camera='front_center'
                    before=c.simGetVehiclePose(vehicle_name=name)
                    host_start=time.time_ns()
                    r=c.simGetImages([airsim.ImageRequest(camera,airsim.ImageType.Scene,False,True)],vehicle_name=name)[0]
                    host_end=time.time_ns()
                    after=c.simGetVehiclePose(vehicle_name=name)
                    info=c.simGetCameraInfo(camera,vehicle_name=name)
                    data=bytes(r.image_data_uint8)
                    frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                    if r.message or frame is None or frame.shape[:2]!=(720,1280): raise RuntimeError('Invalid capture: '+name+' '+r.message)
                    cam_pose={'position_m':vector(r.camera_position),'orientation_wxyz':quaternion(r.camera_orientation)}
                    if not np.isfinite(cam_pose['position_m']+cam_pose['orientation_wxyz']).all(): raise RuntimeError('Nonfinite camera pose')
                    if not 0<info.fov<180 or r.time_stamp<=0: raise RuntimeError('Invalid FOV or timestamp')
                    folder=out/'images'/name; folder.mkdir(parents=True,exist_ok=True)
                    path=folder/('%06d.png'%i); path.write_bytes(data)
                    intrinsics=scene_intrinsics(info,r.width,r.height)
                    row={'pair_id':i,'vehicle':name,'camera':camera,'image':path.relative_to(out).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'width':r.width,'height':r.height,'timestamp_ns':int(r.time_stamp),'host_request_start_ns':host_start,'host_response_end_ns':host_end,'camera_pose_from_image':cam_pose,'vehicle_pose_before':pose(before),'vehicle_pose_after':pose(after),**intrinsics,'mount_config':cfg['Vehicles'][name]['Cameras'][camera],'projection_matrix_from_api':info.proj_mat.matrix}
                    records.write(json.dumps(row,allow_nan=False)+'\n'); records.flush()
                    if annotation_context: maritime_annotations.annotate(c,annotation_context,row,frame)
                    stamps.append(int(r.time_stamp)); views.append(frame)
                report['pairs'].append({'pair_id':i,'timestamp_delta_ms':abs(stamps[1]-stamps[0])/1e6})
                if i==0:
                    preview=np.hstack(views)
                    cv2.putText(preview,'DroneA', (25,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                    cv2.putText(preview,'DroneB', (1305,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                    cv2.imwrite(str(out/'preview.jpg'),preview)
                if i+1<args.pairs: time.sleep(args.interval)
        report['status']='PASS'
        report['images_saved']=args.pairs*2
    except Exception as exc:
        report.update(status='FAIL',error=str(exc)); raise
    finally:
        if args.ground_truth:
            try:
                import maritime_annotations
                maritime_annotations.cleanup(c)
            except Exception as exc:
                report.update(status='FAIL',annotation_cleanup_error=str(exc))
        if args.scenario:
            try:
                import maritime_experiments
                maritime_experiments.apply(c,'baseline')
                report['scene_restored_to_baseline']=True
            except Exception as exc:
                report.update(status='FAIL',restore_error=str(exc))
        (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(out)
    if report['status']!='PASS': raise RuntimeError('Capture cleanup failed; see manifest')
    (ROOT/'logs'/'dual_camera_latest.json').write_text(json.dumps({'output':str(out)}),encoding='utf-8')
if __name__=='__main__': main()
