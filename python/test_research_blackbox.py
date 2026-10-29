"""Independent replay checks for budgeted differential evolution operators."""
import unittest

import numpy as np

from trajectory_research.blackbox import run_blackbox_search
from trajectory_research.search import reflect_box, run_search


def sphere(x):
    return dict(score=float(x @ x), feasible=True)


class BlackboxTests(unittest.TestCase):
    def test_shared_initialization_budget_bounds_and_replay(self):
        for initialization in ("uniform", "lhs"):
            kwargs = dict(lower=[-3., -2., 4.], upper=[4., 5., 8.], seed=7,
                          population=8, budget=64, anchor=[1., 1., 5.],
                          initialization=initialization)
            reference = run_search(sphere, method="random", **kwargs)
            for method in ("de", "jde", "coordinate"):
                with self.subTest(initialization=initialization, method=method):
                    first = run_blackbox_search(sphere, method=method, **kwargs)
                    second = run_blackbox_search(sphere, method=method, **kwargs)
                    self.assertEqual(first["initial_population_normalized"],
                                     reference["initial_population_normalized"])
                    self.assertEqual(first["best_parameters"], second["best_parameters"])
                    self.assertEqual(first["final_factors"], second["final_factors"])
                    self.assertEqual(len(first["evaluations"]), 64)
                    self.assertEqual([row["fe"] for row in first["evaluations"]], list(range(1, 65)))
                    self.assertEqual([row["fe"] for row in first["history"]], list(range(8, 65, 8)))
                    self.assertTrue(all(a["best"] >= b["best"] for a, b in
                                        zip(first["history"], first["history"][1:])))
                    points = np.array([row["parameters"] for row in first["evaluations"]])
                    self.assertTrue(np.all(points >= kwargs["lower"]))
                    self.assertTrue(np.all(points <= kwargs["upper"]))

    def test_proposal_and_control_inheritance_replay_from_frozen_parents(self):
        for method in ("de", "jde"):
            result = run_blackbox_search(sphere, [-1.] * 4, [1.] * 4, method=method,
                                        seed=13, population=8, budget=256)
            parents = np.array(result["initial_population_normalized"])
            # Replay recorded objective values exactly; independent reductions
            # may differ by one ULP across BLAS/NumPy platforms.
            costs = np.array([row["score"] for row in result["evaluations"][:8]])
            np.testing.assert_allclose(costs, np.sum(parents ** 2, axis=1),
                                       rtol=0., atol=1e-14)
            factors, rates = np.full(8, .5), np.full(8, .9)
            accepted_count = rejected_count = adapted_count = 0
            for start in range(8, 256, 8):
                frozen = parents.copy()
                frozen_costs = costs.copy()
                for record in result["evaluations"][start:start + 8]:
                    ev = record["de"]
                    i = ev["target_index"]
                    donors = ev["donor_indices"]
                    self.assertEqual(len(set(donors)), 3)
                    self.assertNotIn(i, donors)
                    self.assertTrue(ev["crossover_mask"][ev["forced_coordinate"]])
                    self.assertEqual(ev["previous_factor"], factors[i])
                    self.assertEqual(ev["previous_crossover_rate"], rates[i])
                    self.assertEqual(ev["parent_score"], frozen_costs[i])
                    a, b, c = frozen[donors]
                    trial = reflect_box(np.where(ev["crossover_mask"],
                                                 a + ev["factor"] * (b - c), frozen[i]))
                    np.testing.assert_allclose(record["parameters"], trial, atol=1e-15)
                    self.assertAlmostEqual(record["score"], float(trial @ trial), places=14)
                    accept = record["score"] <= frozen_costs[i]
                    self.assertEqual(ev["accepted"], accept)
                    self.assertTrue(.1 <= ev["factor"] <= 1.)
                    self.assertTrue(0. <= ev["crossover_rate"] <= 1.)
                    if accept:
                        accepted_count += 1
                        adapted_count += ev["factor"] != factors[i] or ev["crossover_rate"] != rates[i]
                        parents[i], costs[i] = trial, record["score"]
                        factors[i], rates[i] = ev["factor"], ev["crossover_rate"]
                    else:
                        rejected_count += 1
                self.assertGreaterEqual(np.min(frozen_costs), np.min(costs))
            np.testing.assert_allclose(result["final_population_normalized"], parents, atol=1e-15)
            np.testing.assert_allclose(result["final_scores"], costs, atol=1e-14)
            np.testing.assert_equal(result["final_factors"], factors)
            np.testing.assert_equal(result["final_crossover_rates"], rates)
            self.assertGreater(accepted_count, 0)
            self.assertGreater(rejected_count, 0)
            self.assertGreater(adapted_count, 0) if method == "jde" else self.assertEqual(adapted_count, 0)

    def test_equal_cost_accepts_trial_and_adapted_controls(self):
        result = run_blackbox_search(lambda x: dict(score=1., feasible=False), [-1.], [1.],
                                    method="jde", seed=2, population=4, budget=80)
        self.assertTrue(all(record["de"]["accepted"] for record in result["evaluations"][4:]))
        self.assertTrue(any(value != .5 for value in result["final_factors"]))
        self.assertTrue(all(row["best"] == 1. for row in result["history"]))

    def test_coordinate_proposals_replay_and_poll_all_axes(self):
        result = run_blackbox_search(sphere, [-1.] * 7, [1.] * 7,
                                    method="coordinate", seed=3, budget=128)
        best_score = min(row["score"] for row in result["evaluations"][:16])
        scale = .2
        for start in range(16, 128, 16):
            batch = result["evaluations"][start:start + 16]
            self.assertEqual(sorted({row["poll"]["axis"] for row in batch
                                     if row["poll"]["axis"] is not None}), list(range(7)))
            for first, second in zip(batch[::2], batch[1::2]):
                ev = first["poll"]
                self.assertEqual(ev["incumbent_score"], best_score)
                self.assertAlmostEqual(ev["step"], scale)
                self.assertEqual(first["poll"]["direction"], second["poll"]["direction"])
                self.assertEqual(first["poll"]["sign"], -second["poll"]["sign"])
                for row in (first, second):
                    ev = row["poll"]
                    centre = np.array(ev["centre_normalized"])
                    self.assertAlmostEqual(float(centre @ centre), best_score, places=14)
                    self.assertAlmostEqual(np.linalg.norm(ev["direction"]), 1., places=14)
                    trial = reflect_box(centre + ev["step"] * ev["sign"] * np.array(ev["direction"]))
                    np.testing.assert_allclose(row["parameters"], trial, atol=1e-15)
            new_best = min(best_score, min(row["score"] for row in batch))
            scale = min(.5, scale * 1.2) if new_best < best_score else max(1e-4, scale * .5)
            best_score = new_best
        self.assertAlmostEqual(best_score, result["best_score"])
        self.assertLess(result["best_score"], result["history"][0]["best"])

    def test_coordinate_rotates_axes_for_small_population_and_halves_stalled_step(self):
        result = run_blackbox_search(lambda x: dict(score=1., feasible=False), [-1.] * 7, [1.] * 7,
                                    method="coordinate", seed=3, population=4, budget=20)
        axes = {row["poll"]["axis"] for row in result["evaluations"][4:]}
        self.assertEqual(axes, set(range(7)))
        self.assertAlmostEqual(result["history"][-1]["scale"], .2 / 16)

    def test_one_generation_and_one_dimension(self):
        result = run_blackbox_search(sphere, [-1.], [1.], budget=4, population=4)
        self.assertEqual(len(result["evaluations"]), 4)
        self.assertEqual(result["best_score"], min(x["score"] for x in result["evaluations"]))
        self.assertEqual(len(result["history"]), 1)

    def test_invalid_inputs_and_objective_fail_visibly(self):
        for kwargs in (dict(budget=7), dict(budget=True), dict(seed=-1), dict(population=3),
                       dict(method="unknown"), dict(initialization="unknown"), dict(anchor=[2.])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_blackbox_search(sphere, [-1.], [1.], **kwargs)
        for lower, upper in (([1.], [1.]), ([], []), ([np.nan], [1.]), ([0., 1.], [1.])):
            with self.assertRaises(ValueError):
                run_blackbox_search(sphere, lower, upper)
        for invalid in (None, {}, dict(score=0), dict(score=np.inf, feasible=True),
                        dict(score=0., feasible="yes"), dict(score="bad", feasible=True)):
            with self.subTest(result=invalid), self.assertRaises(ValueError):
                run_blackbox_search(lambda x: invalid, [-1.], [1.])
        with self.assertRaisesRegex(RuntimeError, "actual objective failure"):
            run_blackbox_search(lambda x: (_ for _ in ()).throw(RuntimeError("actual objective failure")),
                                [-1.], [1.])


if __name__ == "__main__":
    unittest.main()
