"""Freeze a new source bundle for timed, bounded persistence V10 trials."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v9 import prepare as prepare_prior
from prepare_t1_identity_flight import ROOT,digest,save,copy_verified
from t1_pipeline_plan_v1 import ARCHITECTURE,PERSISTENCE,validate_plan

SOURCES=('python/t1_pipeline_plan_v1.py','python/prepare_t1_identity_flight_v10.py','python/t1_identity_flight_v10.py',
    'python/t1_bounded_evidence_v1.py','python/t1_timed_review_observer_v1.py','python/t1_async_control_v1p5.py',
    'python/t1_zero_hold_proof_v1.py','python/test_t1_zero_hold_proof.py','python/test_t1_bounded_evidence.py',
    'python/audit_t1_persistence_v1.py','python/analyze_t1_pipeline_motion.py','scripts/run-t1-identity-flight-v10.ps1')


def prepare(output,condition,session_id,device='cuda'):
    if any(not (ROOT/name).is_file() for name in SOURCES):raise ValueError('V10_SOURCE_MISSING')
    plan=prepare_prior(output,condition,session_id,device);output=Path(output).resolve()
    for name in SOURCES:
        path=ROOT/name;sha=digest(path);snapshot=output/'source_bundle'/name;copy_verified(path,snapshot,sha)
        if name not in plan['source_sha256']:
            plan['source_inventory'].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][name]=sha
    plan.update(control_architecture=ARCHITECTURE,bounded_persistence=dict(PERSISTENCE),
        comparison_scope='new bounded pipeline qualification; original source age, control limits, full deadline and scale remain')
    validate_plan(plan);save(output/'plan.json',plan);return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition',choices=('zero','scale50','scale100'),required=True);p.add_argument('--session-id',required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');a=p.parse_args()
    plan=prepare(a.output,a.condition,a.session_id,a.device);print(plan['status'],a.condition,len(plan['planned_slots']))
