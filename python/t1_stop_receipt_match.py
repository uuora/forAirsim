"""Match V1P4 pre-query command samples through ordered actual controller events."""
def match_state_sample(sample, state_events, command_events):
    candidates=[(i,r) for i,r in state_events if r['host_receipt_monotonic_s']==sample['receipt_monotonic_s']
        and r['state_timestamp_ns']==sample['state_timestamp_ns']
        and r['velocity_ned_m_s'][:2]==sample['velocity_xy_m_s']
        and r['position_world_ned_m']==sample['position_world_ned_m']]
    matches=[]
    for i,r in candidates:
        prior=[c for j,c in command_events if j<i]
        if (prior and prior[-1]['dispatch_monotonic_s']==sample['actual_last_command_dispatch_monotonic_s']
                and prior[-1]['velocity_xy_m_s']==sample['actual_last_command_xy_m_s']):
            matches.append((i,r))
    return dict(status='MATCHED_ACTUAL_PRE_QUERY_COMMAND' if matches else 'NO_MATCHING_ACTUAL_PRE_QUERY_COMMAND',
        state_event=matches[0] if matches else None,candidate_event_indexes=[i for i,_ in candidates],
        compatible_event_indexes=[i for i,_ in matches],time_or_velocity_changed=False)
