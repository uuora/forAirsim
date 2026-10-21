"""Freeze V11 only after a source-bound real-spawn offline gate passes."""
import argparse
import json
from pathlib import Path
from prepare_t1_identity_flight_v10p2 import prepare as prior_prepare
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified
from t1_process_plan_v1p1 import ARCHITECTURE, PERSISTENCE, PROCESS, validate_plan

SOURCES=('python/t1_process_control_v1p1.py','python/t1_process_control_factories_v1.py',
    'python/t1_process_control_fixture_v1.py','python/audit_t1_process_control_v1p1.py',
    'python/t1_process_trace_v1.py','python/test_t1_process_control_v1p1.py',
    'python/run_t1_process_offline_v1p1.py','python/t1_process_plan_v1p1.py',
    'python/prepare_t1_identity_flight_v11p1.py','python/t1_identity_flight_v11p1.py',
    'python/analyze_t1_process_motion_v1p1.py','scripts/run-t1-identity-flight-v11p1.ps1','python/t1_async_control_v1p8.py','python/t1_capture_diagnostics_v1.py',
    'python/t1_obstacle_scene.py','python/t1_obstacle_control_factory.py')


def prepare(output,condition,session_id,offline_gate,device='cuda',scene=None):
    gate_path=Path(offline_gate).resolve()
    gate=json.loads(gate_path.read_text(encoding='utf-8'))
    if (gate['status']!='PASS_OFFLINE_SPAWN_INJECTIONS_V11P1' or gate['source_unchanged_after_execution'] is not True
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
                                 qualification='PASS_OFFLINE_SPAWN_INJECTIONS_V11P1_ONLY'))
    plan.update(capture_diagnostics='PAIRED_SINGLE_REQUEST_STAGE_TYPE_DIAGNOSTICS_V1',
        first_clear_clock_policy='PRESERVE_FIRST_CLEAR_SEPARATE_SHUTDOWN_WINDOW_V1P8')
    if scene:
        from t1_obstacle_scene import load_scene,scene_flight_geometry
        plan['obstacle_scene']=load_scene(scene)
        plan['scene']=scene_flight_geometry(plan['obstacle_scene'])
        plan['scene_scope']='operator static buildings; bounded motion development, not completed route or avoidance efficacy'
    validate_plan(plan);save(output/'plan.json',plan);return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition',choices=('zero','scale50','scale100'),required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--offline-gate',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');p.add_argument('--scene',type=Path);a=p.parse_args()
    r=prepare(a.output,a.condition,a.session_id,a.offline_gate,a.device,a.scene);print(r['status'])
