"""Supplement only: reconstruct first actual zero for close-interrupted leases.

Frozen reports and their original qualification are retained. This does not
qualify a missing grid or infer a terminal endpoint from a partial trajectory.
"""
import math


def audit(report,events):
    rows=[];issues=[]
    commands=[(index,e) for index,e in enumerate(events) if e.get('kind')=='COMMAND' and e.get('stage')=='PERCEPTION_XY']
    for frame in report.get('frames',[]):
        command=frame.get('command',{});stop=frame.get('lease_stop',{})
        if command.get('purpose')!='CURRENT_TARGET_TRAJECTORY' or stop.get('lease_interrupted_by_close') is not True:continue
        try:
            matched=[(i,e) for i,e in commands if e.get('command_id')==command.get('command_id')]
            if len(matched)!=1 or any(matched[0][1].get(k)!=command.get(k) for k in command):
                raise ValueError('INTERRUPTED_TARGET_COMMAND_NOT_BOUND')
            index,target=matched[0]
            following=[(i,e) for i,e in commands if i>index]
            next_target=next((i for i,e in following if e['purpose']=='CURRENT_TARGET_TRAJECTORY'),len(events))
            zeros=[(i,e) for i,e in following if i<next_target and e['purpose'].startswith('OWN_STATE')
                   and math.hypot(*e['velocity_xy_m_s'])<=1e-12]
            if not zeros:raise ValueError('NO_ACTUAL_FIRST_TARGET_CLEAR_IN_SCOPE')
            _,first=zeros[0]
            if first['dispatch_monotonic_s']<target['dispatch_monotonic_s']:raise ValueError('FIRST_CLEAR_CLOCK_REGRESSION')
            clear=first['dispatch_monotonic_s'];deadline=command['target_end_deadline_monotonic_s']
            rows.append(dict(frame_id=frame['frame_id'],slot_index=frame['slot_index'],ticket_id=frame['ticket_id'],
                target_command_id=target['command_id'],first_actual_zero_command_id=first['command_id'],
                first_actual_target_clear_monotonic_s=clear,original_target_deadline_monotonic_s=deadline,
                actual_first_clear_margin_s=deadline-clear,actual_first_clear_within_original_deadline=clear<=deadline+1e-9,
                saved_shutdown_window_clear_monotonic_s=stop['target_setpoint_cleared_monotonic_s'],
                saved_clock_is_first_actual_target_clear=abs(stop['target_setpoint_cleared_monotonic_s']-clear)<=1e-9,
                original_lease_end_and_stop_deadline_not_modified=True))
            if clear>deadline+1e-9:raise ValueError('ACTUAL_FIRST_TARGET_CLEAR_EXCEEDS_ORIGINAL_DEADLINE')
        except (KeyError,TypeError,ValueError) as error:
            issues.append(dict(frame_id=frame.get('frame_id'),issue=str(error) if isinstance(error,ValueError) else type(error).__name__))
    return dict(status='PASS_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION' if rows and not issues else
                'FAIL_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION' if issues else 'NOT_APPLICABLE_NO_INTERRUPTED_TARGET_LEASE',
        rows=rows,issues=issues,supplemental_only=True,raw_report_modified=False,overall_experiment_qualification_changed=False,
        terminal_endpoint_inferred=False,formal_efficacy_results=False)
