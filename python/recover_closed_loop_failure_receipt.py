"""Recover an explicitly failed terminal summary after a grounded driver crash.

This never reconstructs success or modifies raw flight/perception records.
"""
import argparse
import time
from pathlib import Path

import airsim
import psutil
from fixed_map_perception import read, save, sha, utc


def recover(root):
    root = Path(root).resolve()
    trial = root / 'online_trial'
    if (trial / 'result.json').exists():
        raise ValueError('ORIGINAL_RESULT_EXISTS')
    pid = int((root / 'driver_pid.txt').read_text().strip())
    if psutil.pid_exists(pid):
        raise ValueError('DRIVER_STILL_RUNNING')
    error_path = root / 'driver_stderr.txt'
    if 'PERCEPTION_PROCESS_STOP_TIMEOUT' not in error_path.read_text(encoding='utf-8'):
        raise ValueError('UNRECOGNIZED_DRIVER_EXCEPTION')
    if not read(trial / 'online/publication.json').get('closed'):
        raise ValueError('WORKER_NOT_CLOSED')
    status = read(trial / 'trial_status.json')
    if status['stage'] != 'READY' or not status['mission']['outcome'].startswith('FAILED_'):
        raise ValueError('NOT_A_GROUNDED_FAILED_MISSION')
    client = airsim.MultirotorClient(port=41454, timeout_value=3)
    ground = {}
    for name in ['DroneA', 'DroneB']:
        before = client.getMultirotorState(vehicle_name=name)
        time.sleep(.05)
        after = client.getMultirotorState(vehicle_name=name)
        ground[name] = dict(landed=after.landed_state == airsim.LandedState.Landed,
                            api_control=client.isApiControlEnabled(vehicle_name=name),
                            fresh=after.timestamp > before.timestamp,
                            timestamp_before=before.timestamp, timestamp_after=after.timestamp)
    if not all(s['landed'] and not s['api_control'] and s['fresh'] for s in ground.values()):
        raise ValueError('GROUND_NOT_FRESH_RELEASED')
    receipt = dict(utc=utc(), vehicles=ground,
                   scope='independent read-only postflight RPC; driver terminal write failed')
    save(root / 'POSTFAILURE_GROUND_RECEIPT.json', receipt)
    save(trial / 'result.json', dict(utc=utc(),
         issue='DRIVER_CLEANUP_PERCEPTION_PROCESS_STOP_TIMEOUT', mission=status['mission'],
         state=status['state'], final_ground=ground, flight_start_requested=True,
         perception_guidance_hover_passed=False,
         result_record_provenance='RECONSTRUCTED_FAILED_TERMINAL_SUMMARY_NOT_ORIGINAL_DRIVER_OUTPUT',
         reconstruction_sources=dict(last_status_sha256=sha(trial / 'trial_status.json'),
                                     stderr_sha256=sha(error_path),
                                     ground_receipt_sha256=sha(root / 'POSTFAILURE_GROUND_RECEIPT.json'),
                                     worker_closed_sha256=sha(trial / 'online/publication.json')),
         no_raw_record_changes=True, success_reconstruction_forbidden=True))
    print('FAILED_TERMINAL_RECEIPT_RECOVERED; no success evidence generated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    recover(parser.parse_args().root)
