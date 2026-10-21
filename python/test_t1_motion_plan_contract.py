"""Regression cases for a stale phase denominator and flight plan tampering."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from t1_motion_plan_contract_v1 import ARCHITECTURE, DIAGNOSTICS, grid, validate_plan, validate_file


def plan(mode='SHORT_XY'):
    zero = mode == 'ZERO_GATE'
    duration = 8. if zero else 20.
    return dict(status='FROZEN_BEFORE_FLIGHT', description='带白色横条的红色气球',
        control_architecture=ARCHITECTURE, mode=mode, formal_efficacy_results=False,
        phases=[dict(phase_id='visible', duration_s=duration)], planned_duration_s=duration,
        planned_slots=grid(mode), period_s=.4, world_z_ned_m=-3., command_speed_cap_m_s=.03,
        command_lease_s=.12, self_monitor_period_s=.05, active_lease_stall=dict(enabled=False),
        initial_identity_acquisition=dict(maximum_duration_s=20., period_s=.4,
            maximum_planned_slots=50, desired_xy_m_s=[0., 0.]),
        idle_zero_scheduling=dict(refresh_s=.30, monitor_period_s=.05,
            maximum_previous_command_interval_s=.4, actual_command_clock_preserved=True),
        motion_scale_validation=dict(continuous_visible_duration_s=duration,
            planned_measured_slots=20 if zero else 50,
            minimum_endpoint_net_xy_displacement_m=None if zero else .020,
            maximum_endpoint_drift_m=.005 if zero else None,
            minimum_visible_target_mask_pixels=500, zero_gate_requires_no_nonzero_target_commands=zero,
            first_last_observation_window_s=1., response_diagnostics=DIAGNOSTICS))


class PlanContractTests(unittest.TestCase):
    def test_zero_and_motion_have_separate_complete_denominators(self):
        for mode, count in [('ZERO_GATE', 20), ('SHORT_XY', 50)]:
            with self.subTest(mode=mode):
                self.assertEqual(validate_plan(plan(mode))['expected_main_slots'], count)

    def test_stale_old_phase_count_cannot_authorize_motion(self):
        p=plan();p['motion_scale_validation']['planned_measured_slots']=41
        with self.assertRaisesRegex(ValueError, 'DECLARED_SLOT_COUNT'): validate_plan(p)

    def test_known_frozen_failure_is_reproduced(self):
        path=Path(__file__).resolve().parents[1]/'configs/t1_motion_visible20_v6_20261005/plan.json'
        p=json.loads(path.read_text(encoding='utf-8-sig'))
        p['control_architecture']=ARCHITECTURE
        self.assertEqual(len(p['planned_slots']),50)
        with self.assertRaisesRegex(ValueError, 'DECLARED_SLOT_COUNT'):validate_plan(p)

    def test_missing_duplicate_or_shifted_slots_are_rejected(self):
        for mutation in ('missing','duplicate','shifted'):
            with self.subTest(mutation=mutation):
                p=plan()
                if mutation=='missing':p['planned_slots'].pop()
                elif mutation=='duplicate':p['planned_slots'][12]=copy.deepcopy(p['planned_slots'][11])
                else:p['planned_slots'][12]['scheduled_start_s']+=.05
                with self.assertRaisesRegex(ValueError,'FULL_PLANNED_SLOT_GRID'):validate_plan(p)

    def test_do_not_relax_speed_or_lease_or_refresh(self):
        for key,value in [('command_speed_cap_m_s',.04),('command_lease_s',.2),('self_monitor_period_s',.12)]:
            with self.subTest(key=key):
                p=plan();p[key]=value
                with self.assertRaisesRegex(ValueError,'BUDGET_'):validate_plan(p)
        p=plan();p['idle_zero_scheduling']['refresh_s']=.4
        with self.assertRaisesRegex(ValueError,'IDLE_ACTUAL_COMMAND_CLOCK'):validate_plan(p)

    def test_threshold_cannot_be_lowered_or_swapped_for_drift(self):
        p=plan();p['motion_scale_validation']['minimum_endpoint_net_xy_displacement_m']=.006
        with self.assertRaisesRegex(ValueError,'ENDPOINT_MINIMUM'):validate_plan(p)
        p=plan('ZERO_GATE');p['motion_scale_validation']['zero_gate_requires_no_nonzero_target_commands']=False
        with self.assertRaisesRegex(ValueError,'ZERO_COMMAND_RULE'):validate_plan(p)

    def test_hidden_phase_or_stall_rejected(self):
        p=plan();p['phases'][0]['phase_id']='hidden'
        with self.assertRaisesRegex(ValueError,'CONTINUOUS_VISIBLE_DURATION'):validate_plan(p)
        p=plan();p['active_lease_stall']['enabled']=True
        with self.assertRaisesRegex(ValueError,'NO_INJECTIONS'):validate_plan(p)

    def test_acquisition_stays_zero_and_separate(self):
        p=plan();p['initial_identity_acquisition']['desired_xy_m_s']=[.001,0]
        with self.assertRaisesRegex(ValueError,'SEPARATE_ZERO_ACQUISITION'):validate_plan(p)

    def test_mode_cannot_fall_through_to_motion(self):
        p=plan();p['mode']='other'
        with self.assertRaisesRegex(ValueError,'MOTION_PLAN_MODE_INVALID'):validate_plan(p)

    def test_receipt_preserves_failure_without_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';out=Path(directory)/'receipt.json'
            p=plan();p['motion_scale_validation']['planned_measured_slots']=41
            path.write_text(json.dumps(p),encoding='utf-8')
            result=validate_file(path,out)
            self.assertEqual(result['status'],'FAIL_MOTION_PLAN_CONTRACT')
            self.assertFalse(result['simulator_calls']);self.assertEqual(result['new_flights'],0)
            self.assertEqual(len(result['plan_sha256']),64)
            self.assertTrue(out.is_file())


if __name__=='__main__':unittest.main()
