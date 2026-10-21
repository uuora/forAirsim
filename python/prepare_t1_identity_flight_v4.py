"""Freeze source-RGB descriptor guard before a new independent flight."""
import argparse
from dataclasses import asdict
from pathlib import Path
from prepare_t1_identity_flight_v3 import prepare_async as prepare_v3
from prepare_t1_identity_flight import ROOT,digest,save,copy_verified
from t1_source_attribute_guard import AttributeConfig

SOURCES=('python/t1_identity_flight_v4.py','python/prepare_t1_identity_flight_v4.py',
    'python/t1_source_attribute_guard.py','scripts/run-t1-identity-flight-v4.ps1')

def prepare_async(output,mode,session_id,device='cuda',delay_s=0.):
    plan=prepare_v3(output,mode,session_id,device,delay_s);output=Path(output).resolve()
    for name in SOURCES:
        file=ROOT/name;sha=digest(file);snapshot=output/'source_bundle'/name
        copy_verified(file,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(file),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][name]=sha
    plan.update(control_architecture='INDEPENDENT_CONTROL_SOURCE_ATTRIBUTE_V4',
        source_attribute_guard=dict(profile='RED_BALLOON_WITH_BROAD_WHITE_HORIZONTAL_BAND',
            configuration=asdict(AttributeConfig()),input_scope='bound source RGB and unique YOLO/source semantic boxes',
            rejected_reply='retained transport trace; no semantic anchor adoption',
            scope='controlled descriptor-specific development prerequisite; not general visual identity or calibrated probability'))
    save(output/'plan.json',plan);return plan

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=('zero','short'),required=True)
    parser.add_argument('--session-id',required=True);parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--delay-s',type=float,default=0.)
    args=parser.parse_args();plan=prepare_async(args.output,args.mode,args.session_id,args.device,args.delay_s)
    print(plan['status'],plan['control_architecture'])
