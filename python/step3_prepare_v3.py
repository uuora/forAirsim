"""Prerecord new policy and one ground CUDA/control-isolation recheck."""
import sys,time,json
from pathlib import Path
import airsim
from fixed_map_perception import read,save,utc,ROOT,sha
from fixed_map_perception_process import PerceptionProcess

def main():
    out=Path(sys.argv[1]).resolve();cfg=read(ROOT/'configs/fixed_map_perception_v1.json')
    cfg.update(schema='step3_delayed_attribute_perception_v3',target_description='带白色横条的红色气球',target_generation=2,detector_config='configs/step3_balloon_cuda1280_v2.json',static_geometry_version='v2',camera_reference_version='frozen_self_origin_v2',current_attribute_guard=True,source_attribute_guard=True,semantic_policy_version='delayed_chain_attributes_v3',vlx_response_timeout_s=6.,vlx_http_timeout_s=6.,semantic_source_max_age_s=8.,preview_mode='external_guarded_async',torch_threads=1,cuda_runtime_guard=True,surface_stability_max_m=.25,sensor_surface_exclusion_m=3.25,flight_authorized=True)
    save(ROOT/'configs/fixed_map_perception_v3.json',cfg)
    declaration=dict(utc=utc(),version='step3_delayed_attributes_v3',user_description_confirmed='带白色横条的红色气球',target_generation=2,basis='actual online v2 VLX replies about 3s exceed old2s; actual same-time AcceleratorError and NVIDIA kernel errors; retain all failures',semantic_response_timeout_s=6.,source_max_age_s=8.,current_max_age_s=.5,association_max_gap_s=.5,review_period_s=3.,confirmation='source bound RGB white band; chain adoption plus2 newer frames; exact current frame attribute guard; no age rebasing',runtime_repair='one isolated process; cuDNN disabled, TF32 disabled, allocator cap .5 of8GiB; auxiliary UI image process closed on released ground',position_error_bound_m=.5,unseen_balloon_extent_bound_m=.75,sensor_exclusion_m=3.25,standoff_m=3.5,risks=['8s semantic sources can be wrong: fresh attribute and continuous unique chain still required','driver failure may recur; one ground recheck gates flight','declared error/extent bounds are engineering assumptions, not statistical confidence or unseen obstacle coverage'],ground_recheck='180 processed fresh ground frames; parent control state gaps <=.75s; no accelerator failure; no API control changes; no driver event',subsequent_flight='at most one after ground passes; base only; stop further same-class attempts on failure',old2s_evidence_read_only=True,weights_sha256=read(ROOT/cfg['detector_config'])['weights_sha256'])
    save(out/'policy_runtime_v3_declaration.json',declaration)
    c=airsim.MultirotorClient(port=41454,timeout_value=2)
    for name in ['DroneA','DroneB']:
        if c.getMultirotorState(vehicle_name=name).landed_state!=airsim.LandedState.Landed or c.isApiControlEnabled(vehicle_name=name):raise RuntimeError('GROUND_NOT_RELEASED')
    service=PerceptionProcess(out/'ground_recheck_v3',cfg);service.start();rows=[];start=time.monotonic();deadline=start+115;last=None;maxgap=0;issue=None
    try:
        while time.monotonic()<deadline and service.process_count<180:
            tick=time.monotonic();s=c.getMultirotorState(vehicle_name='DroneA');b=c.getMultirotorState(vehicle_name='DroneB');received=time.monotonic()
            gap=0 if last is None else received-last;last=received;maxgap=max(maxgap,gap)
            rows.append(dict(host_monotonic_s=received,host_time_ns=time.time_ns(),state_timestamp_ns=int(s.timestamp),gap_s=gap,rpc_wall_s=received-tick,landed=s.landed_state==airsim.LandedState.Landed,DroneB_landed=b.landed_state==airsim.LandedState.Landed))
            if gap>.75:issue='GROUND_STATE_GAP_EXCEEDED';break
            if service.process.poll() is not None:issue='PERCEPTION_PROCESS_EXITED';break
            time.sleep(max(0,.1-(time.monotonic()-tick)))
        if service.process_count<180 and issue is None:issue='GROUND_RECHECK_DEADLINE'
        if any('INFERENCE_BLOCKED' in r for r in service.latest.get('rejection_reasons',[])):issue='GROUND_INFERENCE_BLOCKED'
    finally:
        service.stop(wait=True)
        (out/'ground_recheck_state.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        receipt=dict(utc=utc(),issue=issue,status='PASS_GROUND_PENDING_DRIVER_EVENT_AUDIT' if issue is None else 'FAIL',processed_count=service.process_count,max_state_gap_s=maxgap,elapsed_host_s=time.monotonic()-start,ground_only=True,flight_commands_issued=False,config_sha256=sha(ROOT/'configs/fixed_map_perception_v3.json'))
        save(out/'ground_recheck_v3_result.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
