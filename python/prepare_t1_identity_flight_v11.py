"""Freeze V11 only after a source-bound real-spawn offline gate passes."""
import argparse
import json
from pathlib import Path
from prepare_t1_identity_flight_v10p2 import prepare as prior_prepare
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified
from t1_process_plan_v1 import ARCHITECTURE, PERSISTENCE, PROCESS, validate_plan

SOURCES=('python/t1_process_control_v1.py','python/t1_process_control_factories_v1.py',
    'python/t1_process_control_fixture_v1.py','python/audit_t1_process_control_v1.py',
    'python/t1_process_trace_v1.py','python/test_t1_process_control_v1.py',
    'python/run_t1_process_offline_v1.py','python/t1_process_plan_v1.py',
    'python/prepare_t1_identity_flight_v11.py','python/t1_identity_flight_v11.py',
    'python/analyze_t1_process_motion_v1.py','scripts/run-t1-identity-flight-v11.ps1')


def prepare(output,condition,session_id,offline_gate,device='cuda'):
    gate_path=Path(offline_gate).resolve()
    gate=json.loads(gate_path.read_text(encoding='utf-8'))
    if (gate['status']!='PASS_OFFLINE_SPAWN_INJECTIONS' or gate['source_unchanged_after_execution'] is not True
            or len(gate['conditions'])!=5 or gate['simulator_calls']!=0 or gate['provider_requests']!=0):
        raise ValueError('PASSED_SOURCE_BOUND_SPAWN_GATE_REQUIRED')
    for name,value in gate['source_sha256'].items():
        if digest(ROOT/'python'/name)!=value:raise ValueError('OFFLINE_TESTED_PROCESS_SOURCE_CHANGED')
    for row in gate['conditions']:
        if row['status']!='PASS':raise ValueError('ALL_DECLARED_OFFLINE_CONDITIONS_MUST_PASS')
        for key in ('final_receipt','audit'):
            if digest(row[key]['path'])!=row[key]['sha256']:raise ValueError('OFFLINE_EVIDENCE_CHANGED')
    plan=prior_prepare(output,condition,session_id,device);output=Path(output).resolve()
    for name in SOURCES:
        path=ROOT/name;sha=digest(path);snapshot=output/'source_bundle'/name
        copy_verified(path,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][name]=sha
    plan.update(control_architecture=ARCHITECTURE,bounded_persistence=dict(PERSISTENCE),process_control=dict(PROCESS),
        process_offline_gate=dict(path=str(gate_path),sha256=digest(gate_path),
                                 qualification='PASS_OFFLINE_SPAWN_INJECTIONS_ONLY'))
    validate_plan(plan);save(output/'plan.json',plan);return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition',choices=('zero','scale50','scale100'),required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--offline-gate',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');a=p.parse_args()
    r=prepare(a.output,a.condition,a.session_id,a.offline_gate,a.device);print(r['status'])
