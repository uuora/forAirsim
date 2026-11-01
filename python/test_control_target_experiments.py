"""Pure control case/curve checks: no simulator and no optimizer fitting."""
import unittest
import numpy as np
from control_target_experiments import make_case,objective
from run_airsim_curve_replay_pilot import plan_curve,tracking_command


class TargetControlTests(unittest.TestCase):
    def test_seed_reproduction_and_noncolliding_target_contract(self):
        self.assertEqual(make_case(13),make_case(13))
        self.assertNotEqual(make_case(13)['boundary']['p1'],make_case(14)['boundary']['p1'])
        for seed in range(100):
            case=make_case(seed)
            self.assertEqual(case['obstacles'],[])
            self.assertEqual(case['boundary']['v0'],(0.,0.,0.))
            self.assertEqual(case['boundary']['v1'],(0.,0.,0.))
            self.assertLess(abs(case['boundary']['p1'][1]),6)

    def test_fixed_case_does_not_let_seed_change_target(self):
        self.assertEqual(make_case(1,[8,0,5])['boundary'],make_case(2,[8,0,5])['boundary'])

    def test_endpoint_states_and_same_physical_basis(self):
        case=make_case(12);boundary=case['boundary']
        plans=[dict(parameters=[.2,-.3,.1,6.],family='single_bump',representation=r,boundary=boundary) for r in ('bezier','hermite')]
        curves=[plan_curve(p) for p in plans]
        for d,names in enumerate((('p0','p1'),('v0','v1'),('a0','a1'))):
            for curve in curves:
                np.testing.assert_allclose(curve.sample([0,6],d),[boundary[k] for k in names],atol=1e-10)
            np.testing.assert_allclose(curves[0].sample(np.linspace(0,6,101),d),curves[1].sample(np.linspace(0,6,101),d),atol=1e-10)

    def test_model_validation_and_controller_use_actual_velocity(self):
        case=make_case(1101,[8,0,5]);result=objective([0,0,0,6.8],'hermite',case['boundary'])
        self.assertTrue(result['feasible'])
        plan=dict(parameters=[0,0,0,6.8],family='single_bump',representation='hermite',boundary=case['boundary'])
        curve=plan_curve(plan);ctrl=dict(position_gain=2.,velocity_error_gain=.8,acceleration_feedforward_s=.3)
        a=dict(position_zup_m=curve.sample(2),velocity_zup_mps=[0,0,0]);b=dict(a,velocity_zup_mps=[1,0,0])
        np.testing.assert_allclose(tracking_command(curve,2,a,ctrl)-tracking_command(curve,2,b,ctrl),[.8,0,0],atol=1e-12)

    def test_surrogate_outer_command_matches_airsim_equation(self):
        from control_target_experiments import PROXY_CONDITION,PROXY_CONTROLLER
        from trajectory_research.timed_tracking import run_timed_tracking
        case=make_case(31)
        curve=plan_curve(dict(parameters=[.2,.3,-.2,6.4],family='single_bump',representation='hermite',boundary=case['boundary']))
        rollout=run_timed_tracking(curve,PROXY_CONDITION,**PROXY_CONTROLLER)
        ctrl=dict(position_gain=PROXY_CONDITION.position_gain,**PROXY_CONTROLLER)
        for i in (0,20,70,140,220):
            state=dict(position_zup_m=rollout['actual_position_m'][i],velocity_zup_mps=rollout['actual_velocity_mps'][i])
            expected=tracking_command(curve,rollout['times_s'][i],state,ctrl)
            np.testing.assert_allclose(rollout['raw_velocity_command_mps'][i],expected,atol=1e-12)


if __name__=='__main__':unittest.main()
