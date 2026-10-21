"""Recover an already airborne unowned drone using prior recorded home. No takeoff."""
import sys,time,json
from pathlib import Path
import numpy as np,airsim
from fixed_map_runtime import configure_session
import fixed_map_mission as m
from fixed_map_perception_execution import PerceptionExecutor
out=Path(sys.argv[1]).resolve();configure_session(out/'unowned_recovery_runtime')
reference=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else m.ROOT/'reports/step3_development_20261010_101819/base_runtime_v4/flight_20261010_111800/plan.json'
e=PerceptionExecutor(dict(observation_goal_xyz_m=[30.,-9.,3.1]));e.api=airsim;e.c=airsim.MultirotorClient(port=41454,timeout_value=2)
e.calibrate();e.sample();e.home=np.array(m.read(reference)['home_xyz_m']);e.ground_b=e.state['DroneB_position_xyz_m']
e.run_dir=m.OUT/('safety_recovery_'+time.strftime('%Y%m%d_%H%M%S'));e.run_dir.mkdir();e.log=(e.run_dir/'timestamped.jsonl').open('w',encoding='utf-8')
m.save(e.run_dir/'source_binding.json',dict(utc=m.utc(),initial=e.state,home_reference=str(reference),home_xyz_m=e.home.tolist(),no_new_takeoff=True,reason='airborne API owner absent at continuation start; not a perception trial'))
e.collision_base=int(e.c.simGetCollisionInfo(vehicle_name='DroneA').time_stamp)
if not e.state['landed']:
    e.hold('UNOWNED_AIRBORNE_SAFE_RECOVERY','STOP_HOVER');e.sample();e.process_request(dict(action='return',actor='mandatory_unowned_recovery'))
    last=None;deadline=time.monotonic()+120
    while e.stage!='READY':
        tick=time.monotonic();e.sample()
        if last is not None and tick-last>.75:raise ValueError('RECOVERY_STATE_GAP')
        if tick>deadline:raise ValueError('RECOVERY_TIMEOUT')
        last=tick;e.control();time.sleep(max(0,.1-(time.monotonic()-tick)))
else:e.c.armDisarm(False,vehicle_name='DroneA');e.c.enableApiControl(False,vehicle_name='DroneA')
e.sample();e.planner.shutdown(wait=False)
receipt=dict(utc=m.utc(),state=e.state,api_control=e.c.isApiControlEnabled(vehicle_name='DroneA'),no_new_takeoff=True,run_dir=str(e.run_dir))
m.save(out/'unowned_recovery_result.json',receipt);print(json.dumps(receipt),flush=True)
