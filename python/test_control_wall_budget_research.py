"""Deadline accounting tests using an explicitly synthetic slow oracle."""
import json,time,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from control_target_experiments import make_case
from run_control_wall_budget_research import worker


class WallBudgetAccountingTests(unittest.TestCase):
    def arguments(self,folder,budget,start):
        folder=Path(folder);case=folder/'case.json';case.write_text(json.dumps(make_case(13)),encoding='utf-8')
        return SimpleNamespace(start=start,budget=budget,case=case,method='coordinate',seed=13,output=folder/'result.json')

    def test_expired_startup_produces_no_invented_candidate_or_quality(self):
        with TemporaryDirectory() as folder:
            args=self.arguments(folder,.01,time.perf_counter()-1)
            with patch('control_target_experiments.objective') as evaluate:
                self.assertEqual(worker(args),0)
            evaluate.assert_not_called();result=json.loads(args.output.read_text())
            self.assertEqual(result['full_oracle_calls'],0)
            self.assertIsNone(result['primary_best_feasible_score'])
            self.assertFalse(result['primary_has_feasible'])
            self.assertGreater(result['overrun_s'],0)

    def test_late_complete_evaluation_is_retained_but_not_primary_quality(self):
        def slow(*args):
            time.sleep(.2)
            return dict(score=.4,feasible=True,metrics=dict(synthetic_unit_fixture=True))
        with TemporaryDirectory() as folder:
            args=self.arguments(folder,.1,time.perf_counter())
            with patch('control_target_experiments.objective',side_effect=slow):worker(args)
            result=json.loads(args.output.read_text())
            self.assertEqual(result['full_oracle_calls'],1)
            self.assertEqual(result['completed_after_deadline'],1)
            self.assertEqual(result['complete_calls_by_deadline'],0)
            self.assertIsNone(result['primary_best_feasible_score'])
            self.assertFalse(result['primary_has_feasible'])
            self.assertEqual(result['calls'][0]['score'],.4)


if __name__=='__main__':unittest.main()
