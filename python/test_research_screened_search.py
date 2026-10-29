"""Exact-selection and accounting tests for research lower-bound screening."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from trajectory_research.search import run_search
from trajectory_research.screened_search import run_screened_search


UPSTREAM = Path(__file__).resolve().parents[1] / "tmp/abom_deploy_probe/upstream"
HAS_NEURAL = importlib.util.find_spec("torch") is not None and (UPSTREAM / "abom_net.py").is_file()


def mixed_objective(x):
    feasible = bool(x[0] >= 0)
    return dict(score=float(x @ x) + (0.0 if feasible else 5.0), feasible=feasible)


def squared_norm(x):
    return float(x @ x)


class ScreenedSearchTests(unittest.TestCase):
    def assert_same_search(self, first, second):
        self.assertEqual(first["initial_population_normalized"], second["initial_population_normalized"])
        self.assertEqual(first["best_parameters"], second["best_parameters"])
        self.assertEqual(first["best_score"], second["best_score"])
        self.assertEqual([r["parameters"] for r in first["evaluations"]],
                         [r["parameters"] for r in second["evaluations"]])
        for a, b in zip(first["history"], second["history"]):
            for field in ("best", "scale", "success_rate", "population_diversity"):
                self.assertEqual(a.get(field), b.get(field))

    def test_original_stream_agreement_random_and_gaussian(self):
        for method in ("random", "gaussian"):
            for initialization in ("uniform", "lhs"):
                with self.subTest(method=method, initialization=initialization):
                    kwargs = dict(method=method, lower=[-2.] * 3, upper=[2.] * 3,
                                  seed=17, population=8, budget=128,
                                  anchor=[0.1, 0.1, 0.1], initialization=initialization)
                    original = run_search(mixed_objective, **kwargs)
                    baseline = run_screened_search(mixed_objective, **kwargs)
                    screened = run_screened_search(mixed_objective, lower_bound=squared_norm, **kwargs)
                    self.assert_same_search(original, baseline)
                    self.assert_same_search(original, screened)
                    self.assertEqual(baseline["final_population_normalized"], screened["final_population_normalized"])
                    self.assertEqual(baseline["final_scores"], screened["final_scores"])
                    self.assertGreater(screened["pruned_count"], 0)
                    self.assertTrue(any(not r["feasible"] for r in original["evaluations"]))
                    self.assertTrue(any(r["feasible"] for r in original["evaluations"]))

    @unittest.skipUnless(HAS_NEURAL, "pinned optional ABOM source and torch required")
    def test_original_stream_agreement_neural_methods(self):
        for method in ("abom", "abom_perturb", "abom_triggered"):
            with self.subTest(method=method):
                kwargs = dict(method=method, lower=[-2.] * 4, upper=[2.] * 4,
                              seed=9, population=8, budget=96, anchor=[0.] * 4,
                              upstream=UPSTREAM)
                original = run_search(mixed_objective, **kwargs)
                baseline = run_screened_search(mixed_objective, **kwargs)
                screened = run_screened_search(mixed_objective, lower_bound=squared_norm, **kwargs)
                self.assert_same_search(original, baseline)
                self.assert_same_search(original, screened)
                self.assertEqual(baseline["final_population_normalized"], screened["final_population_normalized"])
                self.assertGreater(screened["pruned_count"], 0)

    def test_worst_not_best_parent_is_required_for_valid_pruning(self):
        initial = np.array([[-1.], [.2], [.4], [.8]])
        children = np.array([[.1], [.3], [.6], [.95]])
        objective = lambda x: dict(score=float(x[0]), feasible=True)
        kwargs = dict(lower=[0.], upper=[1.], method="gaussian", population=4,
                      budget=8, initialization="lhs", lower_bound=lambda x: float(x[0]))
        with patch("trajectory_research.screened_search.latin_hypercube", return_value=initial.copy()), \
                patch("trajectory_research.screened_search.paired_perturbation", return_value=children):
            result = run_screened_search(objective, **kwargs)
        np.testing.assert_allclose(result["final_scores"], [0., .55, .6, .65], rtol=0, atol=1e-15)
        self.assertEqual([r["status"] for r in result["evaluations"][4:]],
                         ["EVALUATED", "EVALUATED", "EVALUATED", "PRUNED"])
        self.assertTrue(all(r["selection_threshold"] == .9 for r in result["evaluations"][4:]))

    def test_stable_equal_score_ties_preserve_parents_even_if_infeasible(self):
        kwargs = dict(lower=[-1., -1.], upper=[1., 1.], population=4, budget=20,
                      method="gaussian", seed=3)
        objective = lambda x: dict(score=-2.0, feasible=False)
        baseline = run_screened_search(objective, **kwargs)
        result = run_screened_search(objective, lower_bound=lambda x: -2.0, **kwargs)
        self.assert_same_search(baseline, result)
        self.assertEqual(result["full_evaluations"], 4)
        self.assertEqual(result["pruned_count"], 16)
        self.assertEqual(result["final_population_normalized"], result["initial_population_normalized"])

    def test_nullable_pruned_results_and_exact_counters(self):
        called = []
        def objective(x):
            called.append(x.tolist())
            return dict(score=0.0, feasible=True)
        result = run_screened_search(objective, [-1.], [1.], lower_bound=lambda x: 0.,
                                     population=4, budget=16)
        self.assertEqual(len(called), 4)
        self.assertEqual(result["proposals"], 16)
        self.assertEqual(result["proposal_budget"], 16)
        self.assertEqual(result["full_evaluations"], 4)
        self.assertEqual(result["pruned_count"], 12)
        self.assertEqual(result["lower_bound_calls"], 16)
        self.assertEqual([r["proposal_index"] for r in result["evaluations"]], list(range(1, 17)))
        for row in result["evaluations"][4:]:
            for key in ("score", "feasible", "fe"):
                self.assertIsNone(row[key])
            self.assertEqual(row["prune_kind"], "CHEAP_BOUND")
        json.dumps(result, allow_nan=False)

    def test_invalid_bounds_abort_visibly(self):
        objective = lambda x: dict(score=0., feasible=True)
        for value in (float("nan"), float("inf"), -float("inf"), True, "0", [0.], 1.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_screened_search(objective, [-1.], [1.], lower_bound=lambda x: value,
                                    population=4, budget=8)

    def test_bad_objective_results_abort(self):
        for result in (None, {}, dict(score=float("nan"), feasible=True),
                       dict(score=0., feasible=1), dict(score=True, feasible=True),
                       dict(status="UNRESOLVED", score=0., feasible=True)):
            with self.subTest(result=result), self.assertRaises(ValueError):
                run_screened_search(lambda x: result, [-1.], [1.], population=4, budget=8)

    def test_partial_objective_exact_preservation_and_accounting(self):
        calls = []
        def objective(x):
            return dict(score=float(x @ x) + 0.1, feasible=True)
        def bounded(x, cutoff):
            calls.append(cutoff)
            bound = float(x @ x)
            if cutoff is not None and bound >= cutoff:
                return dict(status="PRUNED", lower_bound=bound, score=None,
                            feasible=None, score_details={"completed_terms": 1})
            return objective(x)
        kwargs = dict(lower=[-1.] * 3, upper=[1.] * 3, method="gaussian",
                      seed=13, population=8, budget=96, anchor=[0.] * 3)
        baseline = run_screened_search(objective, **kwargs)
        result = run_screened_search(objective, bounded_objective=bounded, **kwargs)
        self.assert_same_search(baseline, result)
        self.assertTrue(all(x is None for x in calls[:8]))
        self.assertEqual(result["bounded_objective_calls"], 96)
        self.assertGreater(result["partial_evaluations"], 0)
        self.assertEqual(result["pruned_count"], result["partial_evaluations"])
        self.assertEqual(result["full_evaluations"] + result["partial_evaluations"], 96)
        self.assertAlmostEqual(result["evaluation_s"],
                               result["full_evaluation_s"] + result["partial_evaluation_s"], places=10)
        for row in result["evaluations"]:
            if row["status"] == "PRUNED":
                self.assertEqual(row["prune_kind"], "BOUNDED_OBJECTIVE")
                self.assertEqual(row["details"]["completed_terms"], 1)
                self.assertIsNone(row["feasible"])
        json.dumps(result, allow_nan=False)

    def test_invalid_partial_certificates_rejected(self):
        objective = lambda x: dict(score=1., feasible=True)
        invalid = [dict(status="PRUNED", lower_bound=0., score=None, feasible=None),
                   dict(status="PRUNED", lower_bound=2., score=2., feasible=None),
                   dict(status="PRUNED", lower_bound=2., score=None, feasible=False),
                   dict(status="PRUNED", lower_bound=float("inf"), score=None, feasible=None)]
        for bad in invalid:
            def bounded(x, cutoff):
                return objective(x) if cutoff is None else bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_screened_search(objective, [-1.], [1.], bounded_objective=bounded,
                                    population=4, budget=8)
        with self.assertRaises(ValueError):
            run_screened_search(objective, [-1.], [1.], population=4, budget=8,
                                bounded_objective=lambda x, c: invalid[0])

    def test_callback_exceptions_are_not_swallowed(self):
        def fail(x):
            raise RuntimeError("oracle failure")
        with self.assertRaisesRegex(RuntimeError, "oracle failure"):
            run_screened_search(fail, [-1.], [1.], population=4, budget=8)

    @unittest.skipUnless(HAS_NEURAL, "pinned optional ABOM source and torch required")
    def test_dropout_ablation_is_recorded_and_reproducible(self):
        kwargs = dict(lower=[-1.] * 3, upper=[1.] * 3, method="abom", seed=11,
                      population=8, budget=32, upstream=UPSTREAM, dropout=.5)
        first = run_screened_search(mixed_objective, **kwargs)
        second = run_screened_search(mixed_objective, lower_bound=squared_norm, **kwargs)
        self.assert_same_search(first, second)
        self.assertEqual(first["dropout"], .5)
        self.assertEqual(first["model_update_schedule"], "every_generation")

    def test_invalid_configuration(self):
        for kwargs in (dict(dropout=1.), dict(dropout=-.1), dict(dropout=True),
                       dict(lower_bound=1), dict(bounded_objective=1),
                       dict(population=3), dict(budget=7)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_screened_search(mixed_objective, [-1.], [1.], **kwargs)


if __name__ == "__main__":
    unittest.main()
