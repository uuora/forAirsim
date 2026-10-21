"""Freeze declared depth-first acquisition; control and rejection budgets stay."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v11p2 import prepare as prior_prepare
from prepare_t1_identity_flight import ROOT,digest,copy_verified,save

def prepare(output,condition,session_id,offline_gate,device='cpu',scene=None):
    plan=prior_prepare(output,condition,session_id,offline_gate,device,scene)
    output=Path(output).resolve()
    for name in ('python/t1_identity_flight_v11p3.py','python/prepare_t1_identity_flight_v11p3.py','scripts/run-t1-identity-flight-v11p3.ps1','python/t1_capture_diagnostics_v2.py','python/test_t1_capture_diagnostics_v2.py'):
        path=ROOT/name;sha=digest(path);snapshot=output/'source_bundle'/name;copy_verified(path,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)));plan['source_sha256'][name]=sha
    plan.update(runner_revision='DEPTH_FIRST_RESOURCE_TRACED_V11P3',capture_request_order=[2,0,5],
        pairing_maximum_readback_skew_s=.03,capture_transport_scope='single depth-first RPC; actual response stamps preserved; exposure simultaneity not established')
    save(output/'plan.json',plan);return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--condition',choices=('zero','scale50','scale100'),required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--offline-gate',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--scene',type=Path);a=p.parse_args()
    print(prepare(a.output,a.condition,a.session_id,a.offline_gate,a.device,a.scene)['status'])
