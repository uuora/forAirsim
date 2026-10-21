import copy
from dataclasses import asdict
import unittest

from test_t1_motion_plan_contract import plan as old_plan
from t1_control_settlement import SettlementConfig
from t1_settlement_plan_v1 import ARCHITECTURE, TIMING_POLICY, validate_plan


def plan():
    p = old_plan()
    p.update(control_architecture=ARCHITECTURE, own_state_settlement=asdict(SettlementConfig()),
        completion_timing_policy=copy.deepcopy(TIMING_POLICY),
        settled_endpoint_policy=dict(initial_wait_timeout_s=2.5, final_wait_timeout_s=2.5,
            endpoint='actual own state after strict sampled stop; observed tail retained'))
    return p


class StrictPlanTests(unittest.TestCase):
    def test_valid_before_simulator(self):
        self.assertFalse(validate_plan(plan())['simulator_calls'])

    def test_residual_threshold_and_dwell_cannot_relax(self):
        for k, v in (('stationary_xy_threshold_m_s', .005), ('minimum_post_clear_observation_s', .05)):
            p = plan(); p['own_state_settlement'][k] = v
            with self.assertRaises(ValueError): validate_plan(p)

    def test_old_full_deadline_and_freshness_are_retained(self):
        p = plan(); p['completion_timing_policy']['historical_full_deadline_s'] = .8
        with self.assertRaises(ValueError): validate_plan(p)
        p = plan(); p['own_state_settlement']['self_state_max_age_s'] = .5
        with self.assertRaises(ValueError): validate_plan(p)

    def test_old_distance_threshold_and_grid_cannot_relax(self):
        p = plan(); p['motion_scale_validation']['minimum_endpoint_net_xy_displacement_m'] = .005
        with self.assertRaises(ValueError): validate_plan(p)
        p = plan(); p['planned_slots'].pop()
        with self.assertRaises(ValueError): validate_plan(p)


if __name__ == '__main__': unittest.main()
