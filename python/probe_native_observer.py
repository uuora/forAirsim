"""Record native observer and unchanged grounded vehicle after real UI actions.

This probe never arms, moves, pauses or resets the drone. It emits only the
observer's diagnostic console call and an optional native viewport export.
"""
import argparse
import json
from pathlib import Path
import time

import airsim
from dual_target_control.native_scene import reflected_command
from dual_target_control.simulator import ROOT


def pose_values(pose):
    return dict(position=pose.position.to_numpy_array().tolist(),
                orientation=pose.orientation.to_numpy_array().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage')
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/dual_target_control.json').read_text(encoding='utf8'))
    options = config['airsim']
    client = airsim.MultirotorClient(ip=options['host'], port=options['port'], timeout_value=3.)
    log = ROOT / 'logs/dual_target_control_ue.log'
    try:
        state = client.getMultirotorState(vehicle_name=options['vehicle'])
        api = client.isApiControlEnabled(vehicle_name=options['vehicle'])
        if state.landed_state != airsim.LandedState.Landed or api:
            raise RuntimeError('This probe is limited to a grounded session without API control.')
        offset = log.stat().st_size
        reflected_command(client, 'ke /Script/Blocks.DroneControlObserver DumpState', log)
        with log.open('rb') as stream:
            stream.seek(offset)
            rows = stream.read().decode('utf8', errors='replace').splitlines()
        observer = json.loads(next(row.split('DCO_STATE ', 1)[1] for row in rows if 'DCO_STATE ' in row))
        record = dict(stage=args.stage, timestamp=time.time(), observer=observer,
                      vehicle=pose_values(client.simGetVehiclePose(vehicle_name=options['vehicle'])),
                      front_camera=pose_values(client.simGetCameraInfo(options['camera'], vehicle_name=options['vehicle']).pose),
                      velocity=state.kinematics_estimated.linear_velocity.to_numpy_array().tolist(),
                      landed=state.landed_state, api=api)
        if args.screenshot:
            shot = args.screenshot.resolve()
            shot.parent.mkdir(parents=True, exist_ok=True)
            if not client.simRunConsoleCommand('HighResShot 1200x800 filename="' + shot.as_posix() + '"'):
                raise RuntimeError('Native viewport export was rejected.')
            record['screenshot'] = str(shot)
        output = ROOT / 'reports/dual_target_control/native_mouse_ground_probes.json'
        records = json.loads(output.read_text(encoding='utf8')) if output.exists() else []
        records.append(record)
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(record, ensure_ascii=False))
    finally:
        client.client.close()


if __name__ == '__main__':
    main()
