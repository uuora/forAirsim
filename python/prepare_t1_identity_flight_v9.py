"""Freeze longer 50/100 mm qualification flights; historical plans unchanged."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v7p3 import prepare_async as prepare_prior
from prepare_t1_identity_flight import ROOT,digest,save,copy_verified,phase_grid
from t1_scaled_motion_plan_v1 import ARCHITECTURE,CONDITIONS,validate_plan

SOURCES=("python/t1_scaled_motion_plan_v1.py","python/prepare_t1_identity_flight_v9.py",
    "python/t1_identity_flight_v9.py","python/analyze_t1_scaled_motion.py",
    "python/test_t1_scaled_motion.py","scripts/run-t1-identity-flight-v9.ps1",
    "python/audit_t1_feedback_stop_v1p1.py","python/t1_stop_receipt_match.py",
    "python/t1_pending_review_audit_v1.py","python/test_t1_pending_review_audit.py",
    "python/analyze_t1_settlement_v1p4.py")


def prepare(output,condition,session_id,device="cuda"):
    if condition not in CONDITIONS:raise ValueError("UNKNOWN_SCALE_CONDITION")
    if any(not (ROOT/name).is_file() for name in SOURCES):raise ValueError("V9_SOURCE_MISSING")
    mode="zero" if condition=="zero" else "short"
    plan=prepare_prior(output,mode,session_id,device,60.);output=Path(output).resolve()
    for name in SOURCES:
        path=ROOT/name;sha=digest(path);snapshot=output/"source_bundle"/name
        copy_verified(path,snapshot,sha)
        if name not in plan["source_sha256"]:
            plan["source_inventory"].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)))
        plan["source_sha256"][name]=sha
    spec=CONDITIONS[condition];duration=spec["duration_s"]
    phases=[dict(phase_id="visible",duration_s=duration)];slots=phase_grid(phases,.4)
    plan.update(control_architecture=ARCHITECTURE,planned_duration_s=duration,
        phases=phases,planned_slots=slots,scaled_motion_condition=dict(name=condition,**spec,
            original_20mm_trials_unchanged=True,control_limits_unchanged=True,method_efficacy_comparison=False),
        comparison_scope="larger endpoint qualification with unchanged limits; not a fixed-duration algorithm comparison")
    plan["motion_scale_validation"].update(continuous_visible_duration_s=duration,planned_measured_slots=len(slots),
        minimum_endpoint_net_xy_displacement_m=None if condition=="zero" else spec["minimum_endpoint_mm"]/1000)
    plan["duration_condition"]=dict(duration_s=duration,rationale="fixed 160/320-second conditions declared before flights; original timing and 20 mm trials retained")
    validate_plan(plan);save(output/"plan.json",plan);return plan


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--condition",choices=CONDITIONS,required=True);p.add_argument("--session-id",required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda");a=p.parse_args()
    plan=prepare(a.output,a.condition,a.session_id,a.device)
    print(plan["status"],a.condition,len(plan["planned_slots"]))
