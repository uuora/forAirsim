"""Freeze new files after the failed V10 zero gate; preserve that source bundle."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v10 import prepare as prior_prepare
from prepare_t1_identity_flight import ROOT,digest,save,copy_verified
from t1_pipeline_plan_v1p1 import ARCHITECTURE,PERSISTENCE,validate_plan
SOURCES=('python/t1_async_control_v1p6.py','python/test_t1_async_control_v1p6.py','python/t1_pipeline_plan_v1p1.py',
    'python/prepare_t1_identity_flight_v10p1.py','python/t1_identity_flight_v10p1.py','python/analyze_t1_pipeline_motion_v1p1.py',
    'scripts/run-t1-identity-flight-v10p1.ps1')
def prepare(output,condition,session_id,device='cuda'):
    plan=prior_prepare(output,condition,session_id,device);output=Path(output).resolve()
    for name in SOURCES:
        path=ROOT/name;sha=digest(path);snapshot=output/'source_bundle'/name;copy_verified(path,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][name]=sha
    plan.update(control_architecture=ARCHITECTURE,bounded_persistence=dict(PERSISTENCE));validate_plan(plan);save(output/'plan.json',plan);return plan
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--condition',choices=('zero','scale50','scale100'),required=True);p.add_argument('--session-id',required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');a=p.parse_args();r=prepare(a.output,a.condition,a.session_id,a.device);print(r['status'])
