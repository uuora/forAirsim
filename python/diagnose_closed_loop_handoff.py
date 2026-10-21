"""Expose recorded plan handoff waits alongside measured aircraft state."""
import argparse
from pathlib import Path

from fixed_map_perception import read, save
from analyze_step3_development import lines, csv_save


def diagnose(root):
    root = Path(root).resolve()
    trial = root / 'online_trial'
    events = lines(trial / 'session_events.jsonl')
    states = sorted([s for p in trial.glob('flight_*/timestamped.jsonl')
                     for s in lines(p) if s['kind'] == 'state'],
                    key=lambda s:s['host_monotonic_s'])
    height = read(trial / 'fixed_version/task.json')['height_world_m']
    rows = []
    start = None
    for event in events:
        if event['kind'] == 'PERCEPTION_PLANNER_FINISHED':
            start = event['host_monotonic_s']
        if event['kind'] not in ['PERCEPTION_PLAN_WAITING_FRESH_FRAME', 'PERCEPTION_PLAN_REJECTED']:
            continue
        nearby = min(states, key=lambda s:abs(s['host_monotonic_s']-event['host_monotonic_s']))
        delta = nearby['host_monotonic_s']-event['host_monotonic_s']
        row = dict(utc=event['utc'], kind=event['kind'], reason=event['reason'],
                   stage=event['stage'],host_monotonic_s=event['host_monotonic_s'],
                   elapsed_since_planner_finished_s=None if start is None else event['host_monotonic_s']-start,
                   nearest_state_delta_s=delta,nearest_state_timestamp_ns=nearby['state_timestamp_ns'],
                   measured_height_m=nearby['position_xyz_m'][2],
                   measured_height_error_m=abs(nearby['position_xyz_m'][2]-height),
                   measured_speed_m_s=nearby['speed_m_s'],
                   state_match_within_0p3_s=abs(delta)<=.3,
                   interpretation='nearest logged state, not the exact internal revalidation input')
        rows.append(row)
    csv_save(root / 'audit/handoff_state_diagnostics.csv', rows)
    result = dict(wait_events=len(rows),terminal_rejections=[r for r in rows if r['kind']=='PERCEPTION_PLAN_REJECTED'],
                  caveat='Temporal adjacency is a diagnostic association; causal changes require a separately frozen test. Freshness and height checks are not relaxed.')
    save(root / 'HANDOFF_DIAGNOSTICS.json', result)
    print(result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    diagnose(parser.parse_args().root)
