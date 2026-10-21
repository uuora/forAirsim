"""Small real online ground collection; contains no motion/Actor scene calls."""
import sys,time,json
from pathlib import Path
from collections import Counter
import airsim
from fixed_map_perception import ROOT,read,save,utc
from fixed_map_scene_binding import bind_scene_config
from fixed_map_perception_process import PerceptionProcess
def main():
    root=Path(sys.argv[1]).resolve();label='online_ground_yolo_v8'+(sys.argv[2] if len(sys.argv)>2 else '');out=root/label
    if out.exists() or (root/(label+'_result.json')).exists():raise FileExistsError('GROUND_RECHECK_OUTPUT_ALREADY_USED')
    cfg=bind_scene_config(read(ROOT/'configs/fixed_map_perception_yolo_v8.json'),read(ROOT/'configs/fixed_map_scene_state.json'))
    save(root/(label+'_declaration.json'),dict(utc=utc(),scope='24 real online ground frames, no motion',config=cfg,scene_truth_not_read=True,no_reset=True,no_teleport=True,maximum_ground_rechecks=1))
    c=airsim.MultirotorClient(port=41454,timeout_value=2);service=PerceptionProcess(out,cfg);rows=[];issue=None;started=False
    try:
        for n in ['DroneA','DroneB']:
            if c.getMultirotorState(vehicle_name=n).landed_state!=airsim.LandedState.Landed or c.isApiControlEnabled(vehicle_name=n):raise ValueError('GROUND_NOT_RELEASED')
        service.start();started=True;deadline=time.monotonic()+65;last=None
        while service.process_count<24 and time.monotonic()<deadline:
            tick=time.monotonic();a=c.getMultirotorState(vehicle_name='DroneA');b=c.getMultirotorState(vehicle_name='DroneB');gap=tick-last if last is not None else 0.;last=tick
            row=dict(host_monotonic_s=tick,state_timestamp_ns=a.timestamp,gap_s=gap,DroneA_landed=a.landed_state==airsim.LandedState.Landed,DroneB_landed=b.landed_state==airsim.LandedState.Landed,DroneA_api_control=c.isApiControlEnabled(vehicle_name='DroneA'),DroneB_api_control=c.isApiControlEnabled(vehicle_name='DroneB'));rows.append(row)
            if gap>.75:raise ValueError('GROUND_STATE_GAP')
            if not row['DroneA_landed'] or not row['DroneB_landed'] or row['DroneA_api_control'] or row['DroneB_api_control']:raise ValueError('GROUND_STATE_CHANGED')
            if service.process.poll() is not None:raise ValueError('PERCEPTION_WORKER_EXIT')
            if any('INFERENCE_BLOCKED' in r for r in service.latest.get('rejection_reasons',[])):raise ValueError('INFERENCE_BLOCKED')
            time.sleep(max(0,.1-(time.monotonic()-tick)))
        if service.process_count<24:issue='GROUND_CAPTURE_DEADLINE'
    except Exception as ex:issue=type(ex).__name__+':'+str(ex)
    finally:
        if started:service.stop(wait=True)
    (root/(label+'_states.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    frames=[read(p) for p in sorted((out/'frames').glob('*/perception.json'))];reject=Counter(r for f in frames for r in f.get('rejection_reasons',[]))
    result=dict(utc=utc(),issue=issue,processed_frames=len(frames),online_valid_designated_frames=sum(f.get('valid',False) for f in frames),candidate_frames=sum(bool(f.get('semantic_frame',{}).get('candidates')) for f in frames),inference_max_s=max([f['inference_wall_s'] for f in frames if 'inference_wall_s' in f],default=None),max_state_gap_s=max([r['gap_s'] for r in rows],default=None),rejections=dict(reject),engine_identity=cfg['runtime_engine_identity'],scene_version=cfg['runtime_scene_version'],flight_commands_issued=False,ground_only=True)
    save(root/(label+'_result.json'),result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
