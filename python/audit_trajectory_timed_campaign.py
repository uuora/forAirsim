"""Read-only V6 flight diagnostics; retain failures and frozen provenance.

Request/reference errors are descriptive host-clock quantities, not physical
tracking guarantees. Samples within a flight are not independent repeats.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_trajectory_matched_sweep import audit_sweep, descriptive
from perception_trajectory_planner import evaluate_plan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = list(values)
    result = descriptive(values)
    result['maximum'] = max(values, default=None)
    return result


def diagnostics(report):
    commands = report['timed_command_records']
    targets = [row for row in commands if row['target_used']]
    plans = report['planning_records']
    leases = [row['observation_age_at_dispatch_s'] + row['requested_duration_s']
              + report['configuration']['planning']['execution_envelope']['freshness_reserve_s']
              for row in targets]
    frames = {row['frame_id']: row for row in report['observations']}
    box_errors = []
    for frame in frames.values():
        if 'target_support_offsets_ned_m' not in frame:
            continue
        if frame.get('footprint_anchor_scope') != 'fixed_forward_projection_of_estimated_surface_point_not_detection_centroid':
            raise ValueError('Unexpected surface/box anchor scope')
        meta = frame['timestamped_observation']
        width, height = meta['projection']['rgb_size_wh']
        focal = width / (2 * math.tan(math.radians(meta['projection']['horizontal_fov_deg']) / 2))
        relative = np.array(frame['target_estimate_world_ned_m']) - np.array(frame['actual_camera_world_ned_m'])
        points = relative + np.array(frame['target_support_offsets_ned_m'])
        if np.any(points[:, 0] <= 0):
            raise ValueError('Box reconstruction behind declared forward camera')
        pixels = np.array([(width - 1) / 2, (height - 1) / 2]) + focal * points[:, 1:] / points[:, 0, None]
        x, y, w, h = meta['bbox_xywh']
        expected = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        box_errors.append(float(np.max(np.abs(pixels - expected))))
    post_errors, post_ages = [], []
    for plan in plans:
        for sample in plan.get('execution_samples', []):
            elapsed = sample['host_monotonic_s'] - plan['reference_input_host_s']
            reference = evaluate_plan(plan['plan'], elapsed)
            post_errors.append(float(np.linalg.norm(reference['position'] - np.array(sample['world_ned_m']))))
            post_ages.append(sample['observation_age_after_rpc_and_state_s'])
    stages = {}
    for name in sorted({row['stage'] for row in report['timing_events']}):
        events = [row for row in report['timing_events'] if row['stage'] == name]
        stages[name] = distribution(row['duration_s'] for row in events)
    return {
        'request_source_counts': dict(Counter(row['source'].split(':')[0] for row in commands)),
        'target_request_count': len(targets),
        'dispatch_age_proxy_s': distribution(row['observation_age_at_dispatch_s'] for row in targets),
        'age_plus_requested_lease_plus_reserve_s': distribution(leases),
        'reference_elapsed_s': distribution(row['reference']['reference_time_s'] for row in targets),
        'dispatch_position_error_norm_m': distribution(float(np.linalg.norm(row['reference']['tracking_position_error_ned_m'])) for row in targets),
        'post_query_position_reference_error_norm_m': distribution(post_errors),
        'post_query_age_proxy_s': distribution(post_ages),
        'capture_and_guard_stage_host_seconds': stages,
        'foreground_rpc_host_seconds': distribution(row['rpc_elapsed_host_s'] for row in targets if row['join_called']),
        'media_write_host_seconds': distribution(row['media_persistence']['finished_host_s'] - row['media_persistence']['started_host_s'] for row in frames.values()),
        'surface_box_reconstruction': {
            'assessed_planning_input_frames': len(box_errors),
            'unassessed_captured_frames': len(frames) - len(box_errors),
            'maximum_corner_error_px': max(box_errors, default=None),
            'within_1e_minus_8_px': bool(box_errors) and max(box_errors) < 1e-8,
            'scope': 'declared fixed-forward, same-axial-depth support rectangle; no rotated-camera/3D hull guarantee'},
        'execution_status_counts': dict(Counter(row.get('execution_status', 'NOT_RECORDED') for row in plans)),
        'freshness_event_status_counts': dict(Counter(row['status'] for row in report['freshness_replan_events'])),
        'limits': ['Host query receipt time is not exact actuation or exposure time',
                   'Requested leases can be superseded; never summed as actual executed time',
                   'Request and query summaries within one run are not independent experimental repeats']}


def main(sweep, output):
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite: {output}')
    summary, _ = audit_sweep(sweep)
    conditions = json.loads(Path(json.loads(sweep.read_text(encoding='utf-8-sig'))['conditions_path']).read_text(encoding='utf-8-sig'))
    expected = conditions['current_flight_source_sha256']
    rows, reasons = [], []
    for run in summary['runs']:
        if run['ledger_status'] != 'TERMINAL_REPORT':
            continue
        source = Path(run['report_path'])
        raw = json.loads(source.read_text(encoding='utf-8-sig'))
        info = diagnostics(raw)
        if raw['source_sha256'] != expected:
            reasons.append(f"{source}:SOURCE_NOT_EQUAL_TO_PRE_FLIGHT_FREEZE")
        if run['timed_execution_audit']['status'] != 'VERIFIED':
            reasons.append(f"{source}:COMMAND_OR_MEDIA_TRACE_INVALID")
        if not info['surface_box_reconstruction']['within_1e_minus_8_px']:
            reasons.append(f"{source}:SURFACE_BOX_RECONSTRUCTION_NOT_VERIFIED")
        rows.append(dict(condition_id=run['condition_id'], repetition=run['repetition'], policy=run['policy'],
                         report_path=str(source), report_sha256=digest(source),
                         task_success=run['task_success'], safe_landing_and_release=run['safe_landing_and_release'],
                         timed_execution_audit=run['timed_execution_audit'], diagnostics=info))
    result = dict(kind='V6_TIME_INDEXED_CAMPAIGN_READ_ONLY_DIAGNOSTICS',
                  generated_local=datetime.now().astimezone().isoformat(),
                  formal_thesis_effect_results=False, sweep_path=str(sweep.resolve()), sweep_sha256=digest(sweep),
                  audit_status='VERIFIED' if not reasons else 'INVALID', reasons=reasons,
                  all_started_denominator=dict(planned=summary['planned_flights'], known_started=summary['known_started_flights'],
                     not_started=summary['verified_not_started_flights'], unknown=summary['unknown_start_status_flights'],
                     missing_terminal=summary['missing_terminal_report_count']),
                  independent_safe_launches=summary['independent_launch_evidence_verified_session_count'],
                  pre_flight_source_sha256=expected, audited_source_module_count=len(expected), runs=rows,
                  analysis_source=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
                  ai_assistance='Read-only diagnostic code assisted by AI; inputs are actual saved AirSim reports')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), audit_status=result['audit_status'], denominator=result['all_started_denominator'],
                         target_requests=sum(row['diagnostics']['target_request_count'] for row in rows)), ensure_ascii=False))
    return 1 if reasons else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.sweep, args.output))
