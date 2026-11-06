"""Offline numerical and evidence checks; fixtures are not flight results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .audit import frozen_plan_issues, measured_passages
from .campaign import PLANNER_CONFIG, TASK_CONFIG, run_campaign
from .planners import METHODS, construct, norm_peak, plan, common_fingerprint
from .reference import FrozenReference, build_reference


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads(TASK_CONFIG.read_text(encoding='utf-8'))
        cls.cfg=json.loads(PLANNER_CONFIG.read_text(encoding='utf-8'))

    def test_only_the_three_requested_planners_are_enabled(self):
        self.assertEqual(METHODS, ('bezier','hermite','bspline'))
        self.assertEqual(list(self.cfg['cases']), [m+'_pass' for m in METHODS])

    def test_hermite_and_bezier_are_equivalent_for_unequal_segment_times(self):
        points=self.spec['task']['points_ned'];times=[4.1,5.2,3.8]
        a=construct('bezier',points,times);b=construct('hermite',points,times)
        np.testing.assert_allclose(a,b,atol=2e-10,rtol=1e-10)
        c=construct('bspline',points,times)
        self.assertGreater(float(np.linalg.norm(c-a)), .1)

    def test_all_frozen_curves_pass_independent_constraints_and_are_c3(self):
        for case in self.cfg['cases']:
            with self.subTest(case=case):
                self.assertEqual(frozen_plan_issues(self.cfg,case), [])
                ref=build_reference(self.cfg,case)
                for t in ref.times[1:-1]:
                    left,right=ref.sample(t-1e-7),ref.sample(t+1e-7)
                    for k in ('p','v','a','jerk'):
                        np.testing.assert_allclose(left[k],right[k],atol=1e-5,rtol=0)
                    self.assertGreater(np.linalg.norm(ref.sample(t)['v']), .65-1e-6)

    def test_derivatives_match_centered_differences(self):
        for case in self.cfg['cases']:
            ref=build_reference(self.cfg,case)
            for t in [1.2,ref.times[1]+.3,ref.duration-.7]:
                a,b,r=ref.sample(t-1e-5),ref.sample(t+1e-5),ref.sample(t)
                for low,high in zip(('p','v','a','jerk'),('v','a','jerk','snap')):
                    np.testing.assert_allclose((np.array(b[low])-a[low])/2e-5,r[high],atol=1e-6,rtol=1e-5)

    def test_stationary_root_extrema_include_interior_peak(self):
        c=np.zeros((1,8,3));c[0,2,0]=3;c[0,3,0]=-2
        self.assertAlmostEqual(norm_peak(c,[1.],1),1.5,places=10)

    def test_local_time_search_is_feasible_and_reduces_initial_time(self):
        result=plan('hermite',self.spec['task'],self.spec['time_search'])
        self.assertTrue(result['planning']['success'])
        self.assertLess(sum(result['segment_durations']),sum(self.spec['task']['initial_segment_durations']))
        self.assertLessEqual(result['planning']['peak_speed_mps'],1.6+1e-6)
        self.assertIn('not global',result['planning']['optimality'])
        self.assertGreater(result['planning']['wall_s'],0)

    def test_tampered_plan_fails_independent_audit_and_decoder(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['cases']['bezier_pass']['frozen_reference']['coefficients'][0][7][0]+=.1
        self.assertTrue(frozen_plan_issues(cfg,'bezier_pass'))
        with self.assertRaisesRegex(ValueError,'interpolate'):
            FrozenReference(cfg['cases']['bezier_pass']['frozen_reference'])

    def test_missing_freeze_cannot_revert_to_legacy_reference(self):
        cfg=copy.deepcopy(self.cfg)
        del cfg['cases']['bezier_pass']['frozen_reference']
        self.assertTrue(frozen_plan_issues(cfg,'bezier_pass'))

    def test_measured_passages_require_actual_nonzero_speed_and_order(self):
        case='bezier_pass';ref=build_reference(self.cfg,case)
        rows=[dict(phase='TRACKING',t=float(t),position=ref.points[i].tolist(),velocity=[.7,0,0])
              for i,t in enumerate(ref.times[1:-1],1)]
        passed,receipts=measured_passages(rows,self.cfg,case)
        self.assertTrue(passed)
        self.assertEqual(len(receipts),2)
        rows[0]['velocity']=[0,0,0]
        passed,receipts=measured_passages(rows,self.cfg,case)
        self.assertFalse(passed)
        self.assertFalse(receipts[0]['passed'])
        self.assertFalse(receipts[1]['passed'])
        rows[0]['velocity']=[.7,0,0];rows[0]['t']+=2.
        self.assertFalse(measured_passages(rows,self.cfg,case)[0])

    def test_comparison_identity_changes_with_control_or_gates(self):
        old=common_fingerprint(self.cfg);cfg=copy.deepcopy(self.cfg)
        cfg['limits']['position_error_m']+=.01
        self.assertNotEqual(old,common_fingerprint(cfg))
        cfg=copy.deepcopy(self.cfg)
        cfg['cases']['hermite_pass']['segment_durations'][0]+=.1
        self.assertEqual(old,common_fingerprint(cfg))

    def test_campaign_stops_after_unverified_cleanup_without_retry(self):
        calls=[]
        def failed(*args,**kwargs):
            calls.append(args[1])
            folder=Path(args[2])/'synthetic';folder.mkdir(parents=True)
            (folder/'manifest.json').write_text('{}',encoding='utf-8')
            return dict(run_dir=str(folder),report=dict(normal_completion=False,accepted=False,
                        cleanup=dict(landed=True,disarmed=False,api_control_released=False)))
        with tempfile.TemporaryDirectory() as tmp, patch('geometric_validation.runtime.local_preflight',return_value={}):
            folder,rows=run_campaign(output=Path(tmp)/'campaign',runner=failed)
            self.assertEqual(calls,['bezier_pass'])
            self.assertEqual([r['status'] for r in rows],['RECORDED','NOT_RUN','NOT_RUN'])
            self.assertFalse(json.loads((folder/'summary.json').read_text(encoding='utf-8'))['all_accepted'])


if __name__=='__main__':
    unittest.main()
