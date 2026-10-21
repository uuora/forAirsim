"""Static map guard on the child RPC owner, including feedback commands."""
import math
from t1_process_control_factories_v1 import make_airsim
from t1_obstacle_scene import validate_scene,command_guard

def make_obstacle_airsim(config,log):
    scene=validate_scene(config['obstacle_scene'])
    flight,policy=make_airsim(config,log)
    prior_issue=flight.issue
    def issue(xy,duration,purpose,*,gate=None):
        if math.hypot(*xy)>1e-12:
            state=flight.state();proof=command_guard(state['position_world_ned_m'],list(xy),scene)
            log(dict(kind='STATIC_OBSTACLE_COMMAND_CHECK',purpose=purpose,**proof))
            if not proof['allowed']:
                prior_issue([0.,0.],.12,'STATIC_OBSTACLE_REJECTION_ZERO')
                raise RuntimeError('STATIC_OBSTACLE_COMMAND_REJECTED')
        return prior_issue(xy,duration,purpose,gate=gate)
    flight.issue=issue
    prior_protection=flight.protection
    def protection(state):
        reasons=prior_protection(state)
        proof=command_guard(state['position_world_ned_m'],[0.,0.],scene)
        if not proof['allowed']:reasons.append('STATIC_BUILDING_CLEARANCE')
        collision=flight.client.simGetCollisionInfo(vehicle_name='DroneA')
        if collision.has_collided and str(collision.object_name).startswith('T1Building_'):reasons.append('BUILDING_COLLISION')
        return reasons
    flight.protection=protection
    return flight,policy
