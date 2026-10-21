"""Freeze continuous acquisition with strict stopping before a fresh owned run."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v7 import prepare_async as prepare_v7
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified
from t1_settlement_plan_v1p1 import ARCHITECTURE, TRANSITION, validate_plan

SOURCES=('python/t1_async_control_v1p3.py','python/test_t1_async_control_v1p3.py',
    'python/t1_identity_flight_v7p1.py','python/prepare_t1_identity_flight_v7p1.py',
    'python/t1_settlement_plan_v1p1.py','python/analyze_t1_settlement_v1p1.py',
    'scripts/run-t1-identity-flight-v7p1.ps1')


def prepare_async(output, mode, session_id, device='cuda'):
    if any(not (ROOT/n).is_file() for n in SOURCES): raise ValueError('V7P1_SOURCE_MISSING')
    plan=prepare_v7(output,mode,session_id,device); output=Path(output).resolve()
    for n in SOURCES:
        file=ROOT/n; sha=digest(file); snapshot=output/'source_bundle'/n
        copy_verified(file,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=n,path=str(file),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][n]=sha
    plan.update(control_architecture=ARCHITECTURE,acquisition_transition_policy=TRANSITION)
    validate_plan(plan); save(output/'plan.json',plan); return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('zero','short'),required=True); p.add_argument('--session-id',required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda'); args=p.parse_args()
    plan=prepare_async(args.output,args.mode,args.session_id,args.device)
    print(plan['status'],plan['control_architecture'],len(plan['planned_slots']))
