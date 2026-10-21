"""One brief read-only runtime gate before the v5 prospective flight."""
import sys,time,json
from pathlib import Path
import airsim
from fixed_map_perception import ROOT,read,save,utc
from fixed_map_perception_process import PerceptionProcess
out=Path(sys.argv[1]).resolve();suffix=sys.argv[2] if len(sys.argv)>2 else '';cfg=read(ROOT/'configs/fixed_map_perception_v5.json');scene=read(ROOT/'configs/fixed_map_scene_state.json');cfg.update(runtime_scene_version=scene['scene_version'],collection_context='ground_gate_v5')
c=airsim.MultirotorClient(port=41454,timeout_value=2)
for n in ['DroneA','DroneB']:
    if c.getMultirotorState(vehicle_name=n).landed_state!=airsim.LandedState.Landed or c.isApiControlEnabled(vehicle_name=n):raise ValueError('GROUND_NOT_RELEASED')
s=PerceptionProcess(out/('ground_v5'+suffix),cfg);s.start();rows=[];last=None;issue=None;start=time.monotonic()
try:
    while s.process_count<60 and time.monotonic()-start<75:
        tick=time.monotonic();a=c.getMultirotorState(vehicle_name='DroneA');b=c.getMultirotorState(vehicle_name='DroneB');now=time.monotonic();gap=now-last if last else 0.;last=now
        rows.append(dict(host_monotonic_s=now,state_timestamp_ns=int(a.timestamp),gap_s=gap,landed=a.landed_state==airsim.LandedState.Landed,DroneB_landed=b.landed_state==airsim.LandedState.Landed))
        if gap>.75:raise ValueError('GROUND_STATE_GAP')
        if s.process.poll() is not None:raise ValueError('GROUND_PERCEPTION_PROCESS_EXIT')
        if any('INFERENCE_BLOCKED' in x for x in s.latest.get('rejection_reasons',[])):raise ValueError('GROUND_INFERENCE_BLOCKED')
        time.sleep(max(0,.1-(time.monotonic()-tick)))
    if s.process_count<60:issue='GROUND_CAPTURE_DEADLINE'
except Exception as ex:issue=str(ex)
finally:
    s.stop(wait=True);(out/('ground_state_v5'+suffix+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    result=dict(utc=utc(),issue=issue,ground_only=True,processed_frames=s.process_count,max_state_gap_s=max(x['gap_s'] for x in rows),scene_version=scene['scene_version'],no_flight_commands=True)
    save(out/('ground_v5'+suffix+'_result.json'),result);print(json.dumps(result),flush=True)
