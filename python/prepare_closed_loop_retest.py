"""Freeze the one-trial retest inputs and verify current ground state."""
import sys,shutil,platform,time,argparse
from pathlib import Path
import airsim
from fixed_map_perception import ROOT,read,save,sha,utc
from fixed_map_scene_binding import bind_scene_config
from fixed_map_resources import memory_headroom
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--config',default='configs/fixed_map_perception_yolo_cpu_v11.json');p.add_argument('--previous',default='reports/closed_loop_decode_retest_20261011_132735');args=p.parse_args()
root=Path(args.output).resolve();root.mkdir(exist_ok=False)
c=airsim.MultirotorClient(port=41454,timeout_value=3)
ground={}
for name in ['DroneA','DroneB']:
    a=c.getMultirotorState(vehicle_name=name);time.sleep(.05);b=c.getMultirotorState(vehicle_name=name)
    ground[name]=dict(landed=b.landed_state==airsim.LandedState.Landed,api_control=c.isApiControlEnabled(vehicle_name=name),timestamp_before=a.timestamp,timestamp_after=b.timestamp,fresh=b.timestamp>a.timestamp)
assert all(s['landed'] and not s['api_control'] and s['fresh'] for s in ground.values()),'GROUND_NOT_RELEASED_OR_NOT_FRESH'
cfg=bind_scene_config(read(ROOT/args.config),read(ROOT/'configs/fixed_map_scene_state.json'))
detector_config_path=ROOT/cfg['detector_config']
d=read(detector_config_path)
assert sha(d['weights'])==d['weights_sha256'],'FROZEN_SPECIALIST_HASH_MISMATCH'
assert sha(d['training_receipt'])==d['training_receipt_sha256'],'FROZEN_TRAINING_RECEIPT_HASH_MISMATCH'
archive=root/'source_before_trial';archive.mkdir();files=[]
sources=list((ROOT/'python').glob('fixed_map_*.py'))+[ROOT/'python/run_yolo_language_mission_v8.py',ROOT/'python/check_ground_yolo_cpu_v9.py',ROOT/'python/prepare_closed_loop_retest.py']
sources += [ROOT/'python'/name for name in ['specialist_balloon_detector.py','train_t1_balloon_yolo.py','t1_source_attribute_guard.py','t1_semantic_perception.py','t1_temporal_identity.py','t1_pose_geometry.py','camera_geometry.py','vlx_api_client.py']]
sources+=[ROOT/args.config,detector_config_path,ROOT/'configs/fixed_map_scene_state.json',ROOT/'configs/fixed_map_scene_runtime_v2.json',ROOT/'reports/fixed_map_flight_20261009/fixed_version/task.json',Path.home()/'Documents/AirSim/settings.json']
for p in sources:
    dest=archive/p.name;shutil.copyfile(p,dest);files.append(dict(source=str(p),snapshot=str(dest),sha256=sha(p)))
maps=[]
for name in ['EnclosedUAVLab_Manual.umap','EnclosedUAVLab_RuntimeRandom_v2.umap']:
    p=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/EnclosedUAVLabManual/Maps'/name;maps.append(dict(path=str(p),sha256=sha(p),copied=False))
assert maps[0]['sha256']=='d686934f7215219429f9fa9b053351a397383f6f42a0240b5e457e2aa4d485e5'
save(archive/'manifest.json',dict(utc=utc(),files=files,maps=maps,weights=dict(path=d['weights'],sha256=d['weights_sha256'],copied=False),python=platform.python_version()))
save(root/'RETEST_DECLARATION.json',dict(utc=utc(),previous=args.previous,scope='one real '+cfg['schema']+' retest after targeted interface ground checks; previous failure and test evidence retained',maximum_new_takeoffs=1,no_model_training=True,vlx_requests=0,no_truth_control_input=True,no_reset_or_aircraft_teleport=True,no_scene_randomization=True,config=cfg,ground=ground,memory=memory_headroom(),success='source-bound fresh online plan, actual sensor approach stable hover, return land and release; failures preserved',failure_policy='return safely; no repeated takeoff within this frozen trial',disturbances='only after basic pass and a separately frozen scene'))
print(str(root))
