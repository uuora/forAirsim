"""Eight actual ground camera pairs with independent state reads; no model loading."""
import sys,time,json,threading
from pathlib import Path
import airsim
from fixed_map_perception import ROOT,PerceptionService,read,save,utc
from fixed_map_scene_binding import bind_scene_config
def main():
    root=Path(sys.argv[1]).resolve();cfg=bind_scene_config(read(ROOT/'configs/fixed_map_perception_ground_v7.json'),read(ROOT/'configs/fixed_map_scene_state.json'))
    s=PerceptionService(root/'raw_ground_samples',cfg);done=threading.Event();states=[];errors=[];captures=[]
    def watch():
        try:
            c=airsim.MultirotorClient(port=41454,timeout_value=2);last=None
            while not done.is_set():
                tick=time.monotonic();a=c.getMultirotorState(vehicle_name='DroneA');b=c.getMultirotorState(vehicle_name='DroneB')
                row=dict(host_monotonic_s=tick,DroneA_timestamp_ns=a.timestamp,DroneB_timestamp_ns=b.timestamp,DroneA_landed=a.landed_state==airsim.LandedState.Landed,DroneB_landed=b.landed_state==airsim.LandedState.Landed,DroneA_api=c.isApiControlEnabled(vehicle_name='DroneA'),DroneB_api=c.isApiControlEnabled(vehicle_name='DroneB'),gap_s=tick-last if last is not None else 0.);states.append(row);last=tick
                if not row['DroneA_landed'] or not row['DroneB_landed'] or row['DroneA_api'] or row['DroneB_api'] or row['gap_s']>.75:raise ValueError('GROUND_WATCH_FAILED')
                done.wait(max(0,.1-(time.monotonic()-tick)))
        except Exception as ex:errors.append(type(ex).__name__+':'+str(ex));done.set()
    t=threading.Thread(target=watch);t.start()
    try:
        c=airsim.MultirotorClient(port=41454,timeout_value=3)
        for seq in range(1,9):
            if done.is_set():break
            meta,depth=s.capture_once(c,seq);captures.append(meta)
            save(s.output/'frames'/meta['frame_id']/'inference_status.json',dict(mode='REAL_ONLINE_RAW_GROUND_ONLY',detector='NOT_RUN_MODEL_MEMORY_GATE_BLOCKED',identity='UNCONFIRMED',position_estimate_valid=False,planning_preview='NO_VALID_ONLINE_ESTIMATE',flight_commands_issued=False,reason='online_ground_v7_result.json records MODEL_LOAD_MEMORY_HEADROOM_BELOW_3GIB; no inference or target success fabricated'))
            done.wait(.25)
    finally:done.set();t.join(timeout=4)
    (s.output/'frame_index.jsonl').write_text(''.join(json.dumps(dict(frame_id=m['frame_id'],rgb=m['rgb_path'],depth=m['depth_path'],pose=str(Path(m['rgb_path']).parent/'capture.json'),image_sha256=m['image_sha256'],depth_sha256=m['depth_sha256'],rgb_timestamp_ns=m['rgb_timestamp_ns'],depth_timestamp_ns=m['depth_timestamp_ns'],mode='REAL_ONLINE_RAW_GROUND_NO_INFERENCE'))+'\n' for m in captures),encoding='utf-8')
    (root/'raw_ground_states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in states),encoding='utf-8')
    result=dict(utc=utc(),raw_pairs=len(captures),capture_errors=[m['capture_errors'] for m in captures if m['capture_errors']],max_rgb_depth_readback_skew_ms=max([m['rgb_depth_delta_ms'] for m in captures],default=None),max_state_gap_s=max([r['gap_s'] for r in states],default=None),watch_errors=errors,detector_not_run=True,flight_commands_issued=False,ground_only=True)
    save(root/'raw_ground_result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
