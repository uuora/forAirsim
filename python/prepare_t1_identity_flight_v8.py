"""Freeze source-fault conditions over the accepted V7P3 execution interface."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v7p3 import prepare_async as prepare_prior
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified, phase_grid
from t1_perturbation_contract_v1 import ARCHITECTURE, CONDITIONS, specification, validate_plan

SOURCES=('python/t1_perturbation_contract_v1.py','python/t1_reply_delivery_fault_v1.py',
    'python/t1_scene_fault_v1.py','python/t1_identity_flight_v8.py',
    'python/prepare_t1_identity_flight_v8.py','python/test_t1_source_perturbations.py',
    'python/analyze_t1_source_perturbations.py','scripts/run-t1-identity-flight-v8.ps1',
    'python/audit_t1_feedback_stop_v1p1.py','python/t1_stop_receipt_match.py',
    'python/analyze_t1_settlement_v1p4.py')


def prepare(output, mode, session_id, condition, device='cuda'):
    spec=specification(condition,'ZERO_GATE' if mode=='zero' else 'SHORT_XY')
    if any(not (ROOT/n).is_file() for n in SOURCES):raise ValueError('V8_SOURCE_MISSING')
    plan=prepare_prior(output,mode,session_id,device,60.)
    output=Path(output).resolve()
    for name in SOURCES:
        source=ROOT/name;sha=digest(source);snapshot=output/'source_bundle'/name
        copy_verified(source,snapshot,sha)
        if name not in plan['source_sha256']:
            plan['source_inventory'].append(dict(relative_path=name,path=str(source),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][name]=sha
    duration=sum(p['duration_s'] for p in spec['phases'])
    plan.update(control_architecture=ARCHITECTURE,source_perturbation=spec,
        phases=spec['phases'],planned_slots=phase_grid(spec['phases'],.4),planned_duration_s=duration,
        comparison_scope='source fault behavior on frozen execution; same-scale formal efficacy pending',
        duration_condition=dict(duration_s=duration,
            rationale='separate fixed 60-second source-fault condition; old 20/60-second trials unchanged'))
    plan['motion_scale_validation'].update(continuous_visible_duration_s=None if condition=='occlusion' else duration,
        planned_measured_slots=len(plan['planned_slots']))
    validate_plan(plan);save(output/'plan.json',plan);return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('zero','short'),required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');args=p.parse_args()
    plan=prepare(args.output,args.mode,args.session_id,args.condition,args.device)
    print(plan['status'],plan['control_architecture'],args.condition,len(plan['planned_slots']))
