"""Predeclare 50/100 mm conditions without changing V7P3 control limits."""
import argparse
from copy import deepcopy
import hashlib,json
from pathlib import Path
from prepare_t1_identity_flight import phase_grid
from t1_settlement_plan_v1p3 import validate_plan as validate_common

ARCHITECTURE="INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V9"
CONDITIONS={"zero":dict(duration_s=8.,minimum_endpoint_mm=None),
            "scale50":dict(duration_s=160.,minimum_endpoint_mm=50.),
            "scale100":dict(duration_s=320.,minimum_endpoint_mm=100.)}


def validate_plan(plan):
    if plan.get("control_architecture")!=ARCHITECTURE:
        raise ValueError("V9_SCALED_MOTION_ARCHITECTURE_REQUIRED")
    declaration=plan.get("scaled_motion_condition",{})
    name=declaration.get("name")
    if name not in CONDITIONS:
        raise ValueError("FROZEN_SCALE_CONDITION_REQUIRED")
    spec=CONDITIONS[name]
    expected=dict(name=name,**spec,original_20mm_trials_unchanged=True,
        control_limits_unchanged=True,method_efficacy_comparison=False)
    if declaration!=expected:
        raise ValueError("EXACT_PREDECLARED_SCALE_AND_DURATION_REQUIRED")
    mode="ZERO_GATE" if name=="zero" else "SHORT_XY"
    duration=spec["duration_s"];phases=[dict(phase_id="visible",duration_s=duration)]
    slots=phase_grid(phases,.4);scale=plan.get("motion_scale_validation",{})
    minimum=None if name=="zero" else spec["minimum_endpoint_mm"]/1000
    if (plan.get("mode")!=mode or plan.get("planned_duration_s")!=duration
            or plan.get("phases")!=phases or plan.get("planned_slots")!=slots
            or scale.get("planned_measured_slots")!=len(slots)
            or scale.get("continuous_visible_duration_s")!=duration
            or scale.get("minimum_endpoint_net_xy_displacement_m")!=minimum):
        raise ValueError("COMPLETE_DECLARED_SCALE_GRID_REQUIRED")
    # Only the new grid and qualification minimum are adapted for validating
    # the inherited common command, source, acquisition and stop constraints.
    common=deepcopy(plan);common["control_architecture"]="INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7P3"
    old_duration=8. if name=="zero" else 60.
    common.update(planned_duration_s=old_duration,
        phases=[dict(phase_id="visible",duration_s=old_duration)],
        planned_slots=phase_grid([dict(phase_id="visible",duration_s=old_duration)],.4))
    common["motion_scale_validation"].update(planned_measured_slots=len(common["planned_slots"]),
        continuous_visible_duration_s=old_duration,
        minimum_endpoint_net_xy_displacement_m=None if name=="zero" else .020)
    validate_common(common)
    return dict(status="PASS_PREDECLARED_SCALED_MOTION_PLAN",architecture=ARCHITECTURE,
        expected_main_slots=len(slots),expected_main_duration_s=duration,
        minimum_endpoint_mm=spec["minimum_endpoint_mm"],simulator_calls=False,new_flights=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError("NEW_PREFLIGHT_RECEIPT_REQUIRED")
    receipt=dict(plan_path=str(a.plan.resolve()),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        simulator_calls=False,new_flights=0)
    try:receipt.update(validate_plan(json.loads(a.plan.read_text(encoding="utf-8-sig"))))
    except (ValueError,KeyError,TypeError) as exc:receipt.update(status="FAIL_SCALED_MOTION_PLAN",error=str(exc))
    a.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(receipt["status"]);raise SystemExit(0 if receipt["status"].startswith("PASS") else 2)
