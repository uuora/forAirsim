"""Budget, subspace, restart and canonical-ledger tests for staged search."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from trajectory_research.blackbox import run_blackbox_search
from trajectory_research.parameterizations import parameter_bounds, default_parameters
from trajectory_research.staged_search import (STAGED_METHODS, FOUR_INDICES, embed_four,
    run_staged_search, second_stage_seed)
import run_staged_curve_research as runner


def sphere(p):
    return dict(score=float(p @ p), feasible=True)


class StagedTests(unittest.TestCase):
    def test_exact_budget_canonical_embedding_history_and_determinism(self):
        lower, upper = parameter_bounds("paired_bump")
        for method in STAGED_METHODS:
            calls = []
            def objective(p):
                calls.append(p.copy())
                return sphere(p)
            result = run_staged_search(objective, lower, upper, method=method, seed=3,
                population=4, budget=32, anchor=default_parameters("paired_bump"))
            repeat = run_staged_search(sphere, lower, upper, method=method, seed=3,
                population=4, budget=32, anchor=default_parameters("paired_bump"))
            self.assertEqual(len(calls), 32)
            self.assertEqual([r["fe"] for r in result["evaluations"]], list(range(1, 33)))
            self.assertTrue(all(len(r["parameters"]) == 7 for r in result["evaluations"]))
            self.assertEqual(result["best_parameters"], repeat["best_parameters"])
            np.testing.assert_allclose(calls, [r["parameters"] for r in result["evaluations"]])
            for stage in result["stages"]:
                if stage["dimension"] == 4:
                    chunk = result["evaluations"][stage["first_fe"] - 1:stage["last_fe"]]
                    self.assertTrue(all(row["parameters"][3:6] == [0., 0., 0.] for row in chunk))
            for row in result["history"]:
                self.assertEqual(row["best"], min(r["score"] for r in result["evaluations"][:row["fe"]]))
            self.assertEqual(result["evaluations"][16]["parameters"], result["stages"][0]["best_parameters"])

    def test_reconstruct_both_stage_seeds_and_initial_populations(self):
        lower, upper = parameter_bounds("paired_bump")
        anchor = default_parameters("paired_bump")
        result = run_staged_search(sphere, lower, upper, method="staged_4to7", seed=9, population=4, budget=32, anchor=anchor)
        first = run_blackbox_search(lambda p: sphere(embed_four(p)), lower[FOUR_INDICES], upper[FOUR_INDICES],
            method="coordinate", seed=9, population=4, budget=16, anchor=anchor[FOUR_INDICES])
        second = run_blackbox_search(sphere, lower, upper, method="coordinate", seed=second_stage_seed(9),
            population=4, budget=16, anchor=embed_four(first["best_parameters"]))
        np.testing.assert_allclose([r["parameters"] for r in result["evaluations"][:16]],
            [embed_four(r["parameters"]) for r in first["evaluations"]], atol=1e-15)
        np.testing.assert_allclose([r["parameters"] for r in result["evaluations"]][16:],
            [r["parameters"] for r in second["evaluations"]], atol=1e-15)
        self.assertEqual(result["stages"][1]["seed"], second_stage_seed(9))
        np.testing.assert_allclose(result["stages"][1]["initial_population_normalized"], second["initial_population_normalized"], atol=1e-15)

    def test_cross_schedule_initialization_difference_is_explicit(self):
        lower, upper = parameter_bounds("paired_bump")
        results = [run_staged_search(sphere, lower, upper, method=m, seed=3, population=4, budget=16,
            anchor=default_parameters("paired_bump")) for m in STAGED_METHODS]
        self.assertEqual(results[0]["initial_population_normalized"], results[1]["initial_population_normalized"])
        self.assertNotEqual(results[0]["initial_population_normalized"], results[2]["initial_population_normalized"])

    def test_invalid_half_budget_or_nonzero_four_dimensional_anchor_is_rejected(self):
        lower, upper = parameter_bounds("paired_bump")
        for budget in (15, 24, True):
            with self.assertRaises(ValueError):
                run_staged_search(sphere, lower, upper, population=8, budget=budget)
        anchor = default_parameters("paired_bump"); anchor[3] = .1
        with self.assertRaises(ValueError):
            run_staged_search(sphere, lower, upper, anchor=anchor)

    def test_tiny_real_runner_protocol_summary_cache_and_schedule_audit(self):
        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary) / "campaign"
            args = SimpleNamespace(families=["paired_bump"], layouts=["central"], methods=list(STAGED_METHODS),
                seeds=[3], population=4, budget=16, stress_seed=20261121, stress_count=2,
                upstream=Path(temporary) / "unused")
            runner.train(args, output)
            runner.base.validate(output)
            runner.base.summarize(output)
            audit = runner.audit(output)
            self.assertEqual(audit["runs"], 3)
            self.assertEqual(audit["training_candidates"], 48)
            runner.train(args, output)
            runner.base.validate(output)
            runner.base.summarize(output)
            self.assertEqual(audit, runner.audit(output))


if __name__ == "__main__":
    unittest.main()
