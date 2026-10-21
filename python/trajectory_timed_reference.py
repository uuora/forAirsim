"""Time-indexed reference and a declared host-time freshness budget; no AirSim I/O."""
import math
import numpy as np
from perception_trajectory_planner import evaluate_plan


def command_ramp_interval(previous_request_host_s, evaluated_host_s, maximum_interval_s):
    """Use elapsed request time for a command-change rate, capped conservatively."""
    values = (evaluated_host_s, maximum_interval_s)
    if previous_request_host_s is not None:
        values += (previous_request_host_s,)
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
        raise ValueError('Invalid command-ramp clock')
    if maximum_interval_s <= 0:
        raise ValueError('Invalid command-ramp interval cap')
    elapsed = 0. if previous_request_host_s is None else float(evaluated_host_s) - float(previous_request_host_s)
    if elapsed < 0:
        raise ValueError('Command-ramp clock moved backwards')
    return dict(previous_request_host_s=previous_request_host_s, evaluated_host_s=float(evaluated_host_s),
                elapsed_since_previous_request_s=elapsed, effective_interval_s=min(elapsed, maximum_interval_s),
                maximum_interval_s=float(maximum_interval_s),
                scope='command change per actual request interval; not measured physical acceleration')


def timed_reference(plan, position, input_host_s, dispatch_host_s, previous_reference_s,
                    position_gain, xy_speed_cap):
    numbers = (input_host_s, dispatch_host_s, position_gain, xy_speed_cap)
    if any(isinstance(x, bool) or not math.isfinite(float(x)) for x in numbers):
        raise ValueError("Reference clock and budgets must be finite")
    elapsed = float(dispatch_host_s)-float(input_host_s)
    if previous_reference_s is not None and (isinstance(previous_reference_s,bool) or not math.isfinite(float(previous_reference_s))):
        raise ValueError("Invalid previous reference clock")
    if elapsed < 0 or (previous_reference_s is not None and elapsed+1e-9 < previous_reference_s):
        raise ValueError("Reference clock moved backwards")
    if position_gain < 0 or xy_speed_cap <= 0:
        raise ValueError("Invalid feedback budgets")
    actual = np.asarray(position, dtype=float)
    if actual.shape != (3,) or not np.isfinite(actual).all():
        raise ValueError("Invalid measured position")
    reference = evaluate_plan(plan, elapsed)  # Refuses failed plans and extrapolation.
    raw = reference['velocity']+position_gain*(reference['position']-actual)
    xy = raw[:2]
    norm = float(np.linalg.norm(xy))
    desired = xy*min(1.,xy_speed_cap/max(norm,1e-12))
    return {'reference_time_s':elapsed,'reference_input_host_s':float(input_host_s),
            'reference_dispatch_host_s':float(dispatch_host_s),
            'reference_position_world_ned_m':reference['position'].tolist(),
            'reference_velocity_ned_m_s':reference['velocity'].tolist(),
            'reference_acceleration_ned_m_s2':reference['acceleration'].tolist(),
            'tracking_position_error_ned_m':(reference['position']-actual).tolist(),
            'raw_tracking_velocity_ned_m_s':raw.tolist(),
            'desired_velocity_xy_m_s':desired.tolist(),'xy_norm_clipped':norm>xy_speed_cap,
            'scope':'actual host elapsed time; fixed world Z separately controlled; not exact tracking'}


def scheduled_quantum(age_s, maximum_age_s, request_s, minimum_s, reserve_s,
                      next_cycle_budget_s, rpc_overhead_budget_s, state_query_budget_s):
    values=(age_s,maximum_age_s,request_s,minimum_s,reserve_s,next_cycle_budget_s,
            rpc_overhead_budget_s,state_query_budget_s)
    if any(isinstance(x,bool) or not math.isfinite(float(x)) for x in values):
        raise ValueError("Invalid freshness scheduler value")
    if age_s < 0 or min(maximum_age_s,request_s,minimum_s) <= 0 or minimum_s > request_s:
        raise ValueError("Invalid freshness scheduler range")
    if min(reserve_s,next_cycle_budget_s,rpc_overhead_budget_s,state_query_budget_s) < 0:
        raise ValueError("Freshness reserves cannot be negative")
    live = maximum_age_s-age_s-reserve_s
    available = live-next_cycle_budget_s-rpc_overhead_budget_s-state_query_budget_s
    duration=min(request_s,available)
    status='EXECUTE' if duration >= minimum_s-1e-12 else ('BRAKE' if live < minimum_s else 'YIELD_FOR_NEW_OBSERVATION')
    return {'status':status,'age_proxy_s':float(age_s),'maximum_age_s':float(maximum_age_s),
            'reserve_s':float(reserve_s),'live_command_window_s':float(live),
            'next_cycle_budget_s':float(next_cycle_budget_s),
            'rpc_overhead_budget_s':float(rpc_overhead_budget_s),'state_query_budget_s':float(state_query_budget_s),
            'requested_s':float(request_s),'duration_s':max(0.,float(duration)) if status=='EXECUTE' else 0.,
            'scope':'declared host-time estimates; overruns checked after RPC; no latency guarantee'}
