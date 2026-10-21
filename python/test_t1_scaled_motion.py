"""Regress the new scale/duration contract, retaining the common control bounds."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from t1_scaled_motion_plan_v1 import validate_plan,ARCHITECTURE,CONDITIONS
from prepare_t1_identity_flight import phase_grid
from analyze_t1_scaled_motion import scale_status


class ScaledMotionTests(unittest.TestCase):
    def plan(self,name="scale50"):
        p=json.loads((Path(__file__).resolve().parents[1]/"configs/t1_settlement_visible60_v7p3_20261006/plan.json").read_text(encoding="utf-8"))
        spec=CONDITIONS[name];duration=spec["duration_s"];phases=[dict(phase_id="visible",duration_s=duration)]
        p.update(control_architecture=ARCHITECTURE,mode="ZERO_GATE" if name=="zero" else "SHORT_XY",
            phases=phases,planned_slots=phase_grid(phases,.4),planned_duration_s=duration,
            scaled_motion_condition=dict(name=name,**spec,original_20mm_trials_unchanged=True,
                control_limits_unchanged=True,method_efficacy_comparison=False))
        p["motion_scale_validation"].update(planned_measured_slots=len(p["planned_slots"]),
            continuous_visible_duration_s=duration,minimum_endpoint_net_xy_displacement_m=None if name=="zero" else spec["minimum_endpoint_mm"]/1000,
            maximum_endpoint_drift_m=.005 if name=="zero" else None,zero_gate_requires_no_nonzero_target_commands=name=="zero")
        return p

    def test_50_and_100_mm_have_exact_distinct_grids(self):
        for name,n in [("scale50",400),("scale100",800)]:
            self.assertEqual(validate_plan(self.plan(name))["expected_main_slots"],n)

    def test_cannot_relabel_old_20_mm_as_new_scale(self):
        self.assertEqual(scale_status("SHORT_XY",.023,50.,65),"FAIL_DECLARED_ENDPOINT_SCALE_NOT_REACHED")
        self.assertEqual(scale_status("SHORT_XY",.0999,100.,200),"FAIL_DECLARED_ENDPOINT_SCALE_NOT_REACHED")
        self.assertEqual(scale_status("SHORT_XY",.1,100.,200),"PASS_ENDPOINT_SCALED_MOTION")

    def test_changed_duration_or_missing_slot_refuses_before_flight(self):
        for mutate in [lambda p:p.update(planned_duration_s=200.),lambda p:p["planned_slots"].pop()]:
            p=self.plan();mutate(p)
            with self.assertRaises(ValueError):validate_plan(p)

    def test_common_source_and_command_limits_remain_enforced(self):
        for key,value in [("command_lease_s",.2),("command_speed_cap_m_s",.05),("period_s",.5)]:
            p=self.plan();p[key]=value
            with self.assertRaises(ValueError):validate_plan(p)

    def test_no_injected_stall_in_scale_qualification(self):
        p=self.plan();p["active_lease_stall"]["enabled"]=True
        with self.assertRaises(ValueError):validate_plan(p)

    def test_zero_gate_still_requires_zero_motion(self):
        self.assertEqual(validate_plan(self.plan("zero"))["expected_main_slots"],20)
        self.assertTrue(scale_status("ZERO_GATE",0.,None,0).startswith("PASS"))
        self.assertTrue(scale_status("ZERO_GATE",0.,None,1).startswith("FAIL"))


if __name__=="__main__":unittest.main()
