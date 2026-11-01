"""Closed planning request validation; no planner, process or simulator runs."""
import unittest
from control_target_planner_service import ROOT,validate_request


class PlannerServiceRequestTests(unittest.TestCase):
    def request(self,**changes):
        r=dict(action='generate',output=str(ROOT/'reports/service_request_fixture'),seed=1101,goal=[8,0,5]);r.update(changes);return r

    def test_fixed_target_request_preserves_explicit_geometry(self):
        output,case=validate_request(self.request())
        self.assertEqual(case['boundary']['p1'],(8.,0.,5.))
        self.assertEqual(case['obstacles'],[])
        self.assertEqual(output,ROOT/'reports/service_request_fixture')

    def test_only_reports_children_can_be_created(self):
        for value in (str(ROOT),str(ROOT/'reports'),str(ROOT/'reports/../python/protected')):
            with self.assertRaises(ValueError):validate_request(self.request(output=value))

    def test_unrecognized_actions_extra_fields_and_boolean_seeds_are_rejected(self):
        for value in (self.request(action='exec'),self.request(seed=True),dict(self.request(),shell='anything')):
            with self.assertRaises(ValueError):validate_request(value)


if __name__=='__main__':unittest.main()
