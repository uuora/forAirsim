"""Replay saved observations without issuing flight commands."""
import json
from pathlib import Path
from collections import Counter
from visual_servo import command


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'logs/mission_a_hit_20260909_113349_501160/report.json'
    report = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    for frame in report['vision_records']:
        observation = frame['vehicles']['DroneA']
        cmd = command(observation.get('detected') and observation.get('tracking_state') == 'DETECTED',
                      observation.get('normalised_error_xy'), observation.get('depth_m'),
                      observation.get('truncated', False))
        rows.append({'time':frame['time'], 'phase':frame['phase'],
                     'velocity_body_ned_m_s':cmd.velocity,'reason':cmd.reason})
    result = {'source':str(source),'mode':'OFFLINE_REPLAY_NO_FLIGHT',
              'counts':dict(Counter(row['reason'] for row in rows)), 'rows':rows}
    destination = source.parent / 'visual_servo_replay.json'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(destination)
    print(result['counts'])


if __name__ == '__main__':
    main()
