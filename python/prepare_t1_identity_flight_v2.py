"""Freeze independent-control pilots and a declared perception-stall injection."""
from __future__ import annotations
import argparse
from pathlib import Path
from prepare_t1_identity_flight import ROOT,prepare,digest,save,copy_verified

NEW_SOURCES=("python/t1_identity_flight_v2.py","python/t1_async_control.py",
             "python/prepare_t1_identity_flight_v2.py","scripts/run-t1-identity-flight-v2.ps1")

def prepare_async(output,mode,session_id,device="cuda",delay_s=0.):
    if delay_s not in (0.,.65):raise ValueError('DECLARED_STALL_MUST_BE_ZERO_OR_0P65')
    if delay_s and mode!='short':raise ValueError('STALL_REQUIRES_SHORT_OCCLUSION_PILOT')
    if any(not (ROOT/name).is_file() for name in NEW_SOURCES):raise ValueError('NEW_SOURCE_MISSING')
    plan=prepare(output,mode,session_id,device)
    output=Path(output).resolve()
    for name in NEW_SOURCES:
        source=ROOT/name;sha=digest(source);snapshot=output/'source_bundle'/name
        copy_verified(source,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(source),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][name]=sha
    plan.update(control_architecture='INDEPENDENT_OWN_STATE_THREAD_V2',
        control_rpc_timeout_setting_s=2.,
        control_rpc_timeout_scope='msgpack timeout polling is coarse; setting does not establish a subsecond return bound',
        injected_perception_delays=[dict(slot_index=14,duration_s=delay_s,
            boundary='AFTER_CONTROL_DECISION_BEFORE_MEDIA_SAVE'),dict(slot_index=26,duration_s=delay_s,
            boundary='AFTER_CAPTURE_BEFORE_OBSERVER')] if delay_s else [],
        lease_expiry_scope='explicit own-state zero dispatch, measured clearing and sampled stopping',
        timing_results_scope='development scheduling feasibility; full-cycle and formal efficacy are evaluated separately')
    save(output/'plan.json',plan)
    return plan

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('zero','short'),required=True)
    parser.add_argument('--session-id',required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--delay-s',type=float,default=0.)
    args=parser.parse_args()
    plan=prepare_async(args.output,args.mode,args.session_id,args.device,args.delay_s)
    print(plan['status'],plan['control_architecture'],len(plan['planned_slots']))
