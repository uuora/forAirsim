import unittest
import numpy as np
from control_target_experiments import make_case,objective
from trajectory_research.control_target_bound import target_quality_lower_bound


class ControlTargetBoundTests(unittest.TestCase):
    def test_bound_does_not_exceed_complete_score_on_edges_and_random_cases(self):
        rng=np.random.default_rng(410)
        for seed in (1101,1102,1105):
            boundary=make_case(seed)['boundary']
            probes=[[0,0,0,3],[0,0,0,7],[.8,1.4,.8,3],[-.8,-1.4,-.8,7]]
            probes+=rng.uniform([-.8,-1.4,-.8,3],[.8,1.4,.8,7],(12,4)).tolist()
            for representation in ('bezier','hermite'):
                for x in probes:
                    lower=target_quality_lower_bound(x,boundary,representation)
                    full=objective(x,representation,boundary)
                    self.assertLessEqual(lower,full['score'])

    def test_nonzero_endpoint_velocity_is_rejected(self):
        boundary=make_case(1101)['boundary'];boundary['v0']=[1,0,0]
        with self.assertRaises(ValueError):target_quality_lower_bound([0,0,0,5],boundary)


if __name__=='__main__':unittest.main()
