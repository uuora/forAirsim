"""Replay one protected V6 transition; no simulator calls or new flight claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from trajectory_execution_guard import boundary_command
from trajectory_timed_reference import command_ramp_interval


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report_path, output):
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite: {output}')
    raw=json.loads(report_path.read_text(encoding='utf-8-sig'))
    failed=raw['rejected_boundary_envelopes'][-1]
    query=raw['execution_guard_samples'][-1]
    previous=raw['timed_command_records'][-1]
    if raw['protective_return_reason']!='NO_BOUNDARY_AND_RAMP_FEASIBLE_COMMAND' or query['guard_query_reason']!='foreground_pre_dispatch':
        raise ValueError('Not the diagnosed V6 boundary transition')
    cfg=raw['configuration'];p=cfg['planning'];g=p['execution_envelope']
    low=np.array(cfg['workspace_min_ned_m'])+p['execution_guard_margin_m']
    high=np.array(cfg['workspace_max_ned_m'])-p['execution_guard_margin_m']
    clock=command_ramp_interval(previous['dispatch_host_s'],query['host_monotonic_s'],.2)
    def replay(dt):
        return boundary_command(query['world_ned_m'],failed['desired_velocity_xy_m_s'],previous['command_velocity_xy_m_s'],
            low,high,p['execution_speed_cap_m_s'],p['execution_ramp_acceleration_m_s2'],dt,g['horizon_s'],g['position_reserve_m'])
    fixed=replay(p['execution_sample_dt_s']);elapsed=replay(clock['effective_interval_s'])
    if fixed!=failed:
        raise ValueError('Historical fixed-quantum rejection not reproduced exactly')
    if not elapsed['accepted']:
        raise ValueError('Elapsed-query proxy does not resolve this offline transition')
    nearest=np.clip(np.array(failed['previous_velocity_xy_m_s']),failed['lower_velocity_xy_m_s'],failed['upper_velocity_xy_m_s'])
    result=dict(kind='V6_PROTECTED_BOUNDARY_TRANSITION_OFFLINE_REPLAY',counts_as_flight_data=False,
        report_path=str(report_path.resolve()),report_sha256=digest(report_path),
        historical_rejection_exactly_reproduced=True,
        previous_request_host_s=previous['dispatch_host_s'],latest_guard_query_receipt_host_s=query['host_monotonic_s'],
        replay_time_scope='saved query receipt proxy for missing rejected-request clock; not actual actuation time',
        nearest_boundary_command_change_m_s=float(np.linalg.norm(nearest-np.array(failed['previous_velocity_xy_m_s']))),
        historical_fixed_quantum_budget_m_s=p['execution_ramp_acceleration_m_s2']*p['execution_sample_dt_s'],
        elapsed_query_proxy_budget_m_s=p['execution_ramp_acceleration_m_s2']*clock['effective_interval_s'],
        request_interval=clock,historical_fixed_quantum=fixed,elapsed_query_proxy=elapsed,
        physical_budget_changes='NONE; model-only clock correction, same geometry, delay, rate and speed',
        source=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),
        kernel_sha256={name:digest(Path(__file__).parent/name) for name in ('trajectory_execution_guard.py','trajectory_timed_reference.py')},
        limits=['Offline feasibility does not prove future task success or actual braking response',
                'V6 full sweep deferred; historical failed flight remains a started failure'],
        ai_assistance='Replay implementation assisted by AI; historical raw report preserved')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('historical_rejection_exactly_reproduced','nearest_boundary_command_change_m_s','historical_fixed_quantum_budget_m_s','elapsed_query_proxy_budget_m_s')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.report,args.output)
