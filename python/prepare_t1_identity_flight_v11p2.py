"""Freeze CPU/other declared detector execution with allocation call-site tracing."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v11p1 import prepare as prior_prepare
from prepare_t1_identity_flight import ROOT,digest,copy_verified,save
def prepare(output,condition,session_id,offline_gate,device='cpu',scene=None):
    plan=prior_prepare(output,condition,session_id,offline_gate,device,scene)
    output=Path(output).resolve()
    for name in ('python/t1_identity_flight_v11p2.py','python/prepare_t1_identity_flight_v11p2.py','scripts/run-t1-identity-flight-v11p2.ps1'):
        path=ROOT/name;sha=digest(path);snapshot=output/'source_bundle'/name;copy_verified(path,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)));plan['source_sha256'][name]=sha
    plan.update(runner_revision='RESOURCE_TRACED_V11P2',resource_observation='physical available, parent RSS and swap; before evidence enqueue',
                failure_trace='file basename, line and function; exception bodies excluded')
    save(output/'plan.json',plan);return plan
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--condition',choices=('zero','scale50','scale100'),required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--offline-gate',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--scene',type=Path);a=p.parse_args()
    print(prepare(a.output,a.condition,a.session_id,a.offline_gate,a.device,a.scene)['status'])
