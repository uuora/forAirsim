"""Freeze one calibration candidate; do not rewrite or count calibration as validation."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from braking_response_v1p1 import validate_acquisition
from trajectory_execution_guard import boundary_command
ROOT=Path(__file__).resolve().parents[1]
CALIBRATION=ROOT/'logs/braking_campaign_20261003_223639_9536884/analysis/summary.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    if output.exists():raise ValueError('Candidate freeze must be new')
    audit=json.loads(CALIBRATION.read_text(encoding='utf-8'));session=audit['sessions'][0]
    if audit['status']!='VERIFIED' or session['gate_passed'] or not session['safe_landing_verified']:
        raise ValueError('Expected intact calibration failure and safe closure')
    raw_path=Path(session['report_path']);raw=json.loads(raw_path.read_text(encoding='utf-8'))
    if sha(raw_path)!=session['report_sha256']:raise ValueError('Calibration raw bytes changed')
    pulses=session['pulses'];minimum_h=[(p['maximum_recorded_forward_displacement_m']-.02-p['entry_forward_speed_m_s']**2/(2*.15))/p['entry_forward_speed_m_s'] for p in pulses]
    # This is an equivalent allowance parameter, not an estimated actuation delay.
    # One candidate chosen from development data; subsequent flights are held out.
    candidate_h=1.0
    if max(minimum_h)+raw['configuration']['braking_measurement']['max_query_gap_s']>=candidate_h:
        raise ValueError('Frozen candidate lacks the declared extra query-gap allowance')
    cfg=raw['configuration'];margin=cfg['planning']['execution_guard_margin_m'];mismatches=0
    for row in raw['measurement_commands']:
        result=boundary_command(row['origin_sample']['world_ned_m'],row['desired_xy_m_s'],row['previous_command_velocity_xy_m_s'],
            np.array(cfg['workspace_min_ned_m'])+margin,np.array(cfg['workspace_max_ned_m'])-margin,
            cfg['planning']['execution_speed_cap_m_s'],cfg['planning']['execution_ramp_acceleration_m_s2'],
            max(row['ramp_timing']['effective_interval_s'],1e-12),candidate_h,.02)
        if not result['accepted'] or not np.allclose(result['command_velocity_xy_m_s'],row['command_velocity_xy_m_s'],rtol=0,atol=1e-12):mismatches+=1
    if mismatches:raise ValueError('Candidate changes calibration commands; redo low gate first')
    output.mkdir(parents=True);rows=[]
    for name in ('zero_short','low_gate','session_forward','session_reverse'):
        parent=ROOT/'configs/braking_response_v1p2'/(name+'.json');config=json.loads(parent.read_text(encoding='utf-8'))
        for source,digest in config['flight_source_sha256'].items():
            if sha(ROOT/'python'/source)!=digest:raise ValueError('Unchanged source freeze differs')
        config['protocol_id']='braking_response_v1p3_model_candidate_20261003'
        config['planning']['execution_envelope']['horizon_s']=candidate_h
        config['model_calibration']=dict(candidate_horizon_s=candidate_h,previous_horizon_s=.5,
            calibration_report=str(raw_path),calibration_report_sha256=sha(raw_path),
            interpretation='Equivalent allowance parameter, not precise actuation delay; only held-out low-speed sampled validation')
        config['evidence_limits'].append('Candidate h=1.0 s selected from V1.2 development data; current flight is held-out low-speed validation, not continuous safety certification')
        validate_acquisition(config);path=output/(name+'.json');path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rows.append(dict(configuration=str(path.resolve()),sha256=sha(path),mode=config['braking_measurement']['mode'],planned_pulses=len(config['braking_measurement']['pulses'])))
    receipt=dict(kind='BRAKING_MODEL_CANDIDATE_FREEZE',status='PREPARED_NOT_EXECUTED',prepared_local=datetime.now().astimezone().isoformat(),
        formal_thesis_effect_results=False,configurations=rows,candidate_horizon_s=candidate_h,minimum_h_for_each_calibration_pulse_s=minimum_h,
        calibration_audit=str(CALIBRATION),calibration_audit_sha256=sha(CALIBRATION),calibration_report=str(raw_path),calibration_report_sha256=sha(raw_path),
        offline_command_replay_count=len(raw['measurement_commands']),offline_command_mismatches=mismatches,offline_counts_as_new_flight=False,
        decision='One equivalent-window candidate 1.0 s exceeds all observed minimum windows plus the frozen 0.25 s query-gap cap. Keep a=.15 and reserve=.02; independent validation required.',
        gates='new zero short, then held-out forward full whose .03 pulses precede .06, audit before reverse full; separate low-only configuration remains not dispatched',
        full_planned_flights=2,full_planned_pulses=16,ai_assistance='Calibration candidate preparation assisted by AI; no validation result yet')
    (output/'conditions.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
