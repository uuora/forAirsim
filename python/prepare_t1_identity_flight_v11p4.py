"""Freeze a dedicated verified vertical client; all measured budgets retained."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v11p3 import prepare as prior_prepare
from prepare_t1_identity_flight import ROOT,digest,copy_verified,save

def prepare(output,condition,session_id,offline_gate,device='cuda',scene=None):
    plan=prior_prepare(output,condition,session_id,offline_gate,device,scene);output=Path(output).resolve()
    for name in ('python/t1_identity_flight_v11p4.py','python/prepare_t1_identity_flight_v11p4.py','scripts/run-t1-identity-flight-v11p4.ps1','python/t1_lifecycle_client_v1.py','python/test_t1_lifecycle_client_v1.py'):
        path=ROOT/name;sha=digest(path);snapshot=output/'source_bundle'/name;copy_verified(path,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)));plan['source_sha256'][name]=sha
    plan.update(runner_revision='VERIFIED_VERTICAL_LIFECYCLE_V11P4',lifecycle_rpc=dict(client_timeout_s=60.,
        task_timeouts_s=dict(takeoff=15.,move_to_z=40.,land=45.),future_completion='get, raise stored error, require True then actual state verification',
        current_image_rpc_timeout_unchanged=True,independent_control_rpc_timeout_unchanged=True,measured_control_and_stop_budgets_unchanged=True))
    save(output/'plan.json',plan);return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--condition',choices=('zero','scale50','scale100'),required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--offline-gate',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--scene',type=Path);a=p.parse_args()
    print(prepare(a.output,a.condition,a.session_id,a.offline_gate,a.device,a.scene)['status'])
