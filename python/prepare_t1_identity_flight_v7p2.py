"""Freeze corrected command chronology and separately declared motion duration."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v7p1 import prepare_async as prepare_prior
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified, phase_grid
from t1_settlement_plan_v1p2 import ARCHITECTURE, validate_plan

SOURCES=('python/t1_async_control_v1p4.py','python/test_t1_async_control_v1p4.py',
    'python/t1_identity_flight_v7p2.py','python/prepare_t1_identity_flight_v7p2.py',
    'python/t1_settlement_plan_v1p2.py','python/analyze_t1_settlement_v1p2.py',
    'scripts/run-t1-identity-flight-v7p2.ps1')


def prepare_async(output, mode, session_id, device='cuda', duration_s=20.):
    if duration_s not in (20.,60.): raise ValueError('PREDECLARED_MOTION_DURATION_REQUIRED')
    if any(not (ROOT/n).is_file() for n in SOURCES): raise ValueError('V7P2_SOURCE_MISSING')
    plan=prepare_prior(output,mode,session_id,device); output=Path(output).resolve()
    for n in SOURCES:
        file=ROOT/n; sha=digest(file); snapshot=output/'source_bundle'/n; copy_verified(file,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=n,path=str(file),sha256=sha,snapshot_path=str(snapshot)))
        plan['source_sha256'][n]=sha
    duration=8. if mode=='zero' else duration_s; phases=[dict(phase_id='visible',duration_s=duration)]
    slots=phase_grid(phases,.4)
    plan.update(control_architecture=ARCHITECTURE,
        state_command_receipt_policy='actual last command before SELF_STATE query; later feedback cannot be relabeled as prior',
        planned_duration_s=duration,phases=phases,planned_slots=slots,
        motion_scale_validation=dict(plan['motion_scale_validation'],continuous_visible_duration_s=duration,
            planned_measured_slots=len(slots)),
        duration_condition=dict(duration_s=duration,longer_duration_is_a_separate_condition=(duration==60.),
            rationale='20-second strict-stop pilot reached 8 mm; 60-second scale validation retains the 20 mm threshold'))
    validate_plan(plan); save(output/'plan.json',plan); return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('zero','short'),required=True); p.add_argument('--session-id',required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda'); p.add_argument('--duration-s',choices=(20.,60.),type=float,default=20.)
    args=p.parse_args(); plan=prepare_async(args.output,args.mode,args.session_id,args.device,args.duration_s)
    print(plan['status'],plan['control_architecture'],len(plan['planned_slots']))
