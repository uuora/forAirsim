"""Known-prefix fixtures for training-only budget sensitivity and binding."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import analyze_parameter_search_budget as analysis


class ParameterBudgetTests(unittest.TestCase):
    def setUp(self):
        self.context = TemporaryDirectory()
        self.addCleanup(self.context.cleanup)
        self.root = Path(self.context.name)
        source = self.root / "source_snapshot" / "fixture.py"
        source.parent.mkdir()
        source.write_text("FROZEN = 1\n", encoding="utf-8")
        self.manifest = dict(schema=analysis.SCHEMA, families=["single_bump"], layouts=["central"],
            methods=["de", "abom"], seeds=[3, 4], population=4, budget=12,
            source_sha256={"fixture.py": hashlib.sha256(source.read_bytes()).hexdigest()},
            variables={"single_bump": dict(lower=[-1., 1.], upper=[1., 5.])})
        self.write("manifest.json", self.manifest)
        self.manifest_sha = self.sha("manifest.json")
        choices = []
        for method in self.manifest["methods"]:
            for seed in self.manifest["seeds"]:
                # Shared initial population and known minima at FE 2, 6, 10.
                scores = [.9, .8, .85, .95, .75, .6, .7, .65, .55, .4, .4, .5]
                if method == "de":
                    scores = [value - .1 if index >= 4 else value for index, value in enumerate(scores)]
                points = [[0., 4.]] * 4 + [[.1, 3.5]] * 4 + [[.2, 3.]] * 4
                records = []
                for fe, (score, point) in enumerate(zip(scores, points), 1):
                    records.append(dict(fe=fe, score=score, feasible=True, parameters=point,
                        details=dict(quality_term=score, duration_s=point[-1],
                                     jerk_energy_m2_s5=20. - fe, deformation_rms_m=.1 * fe)))
                name = "train/" + analysis._key("single_bump", "central", method, seed) + ".json"
                run = dict(family="single_bump", layout="central", method=method, seed=seed,
                    population=4, budget=12, manifest_sha256=self.manifest_sha,
                    initial_population_normalized=[[0., .5]] * 4,
                    best_score=min(scores), best_parameters=points[9], evaluations=records)
                self.write(name, run)
                choices.append(dict(family="single_bump", layout="central", method=method, seed=seed,
                    parameters=run["best_parameters"], training_score=run["best_score"],
                    training_file=name, training_sha256=self.sha(name)))
        self.write("selection.json", dict(manifest_sha256=self.manifest_sha,
            selection_basis="training_only", incumbents=choices))
        # This malformed validation file would fail if the analysis opened it.
        (self.root / "validation_summary.json").write_text("NOT JSON AND MUST NOT BE READ", encoding="utf-8")

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")

    def sha(self, relative):
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def rebind_training_hash(self, name):
        selection, _ = analysis.read_evidence(self.root / "selection.json")
        next(c for c in selection["incumbents"] if c["training_file"] == name)["training_sha256"] = self.sha(name)
        self.write("selection.json", selection)

    def test_known_prefix_minima_stable_ties_and_monotone_scores(self):
        report = analysis.analyze(self.root, [4, 8, 12])
        self.assertEqual(len(report["groups"]), 6)
        self.assertEqual(len(report["prefix_incumbents"]), 12)
        self.assertEqual(report["provenance"]["validation_access"], "none")
        rows = [r for r in report["prefix_incumbents"] if r["method"] == "de" and r["seed"] == 3]
        self.assertEqual([r["selected_fe"] for r in rows], [2, 6, 10])
        np.testing.assert_allclose([r["score"] for r in rows], [.8, .5, .3])
        self.assertEqual([r["duration_s"] for r in rows], [4., 3.5, 3.])
        # FE 10 and 11 tie; first-tie selection retains FE 10's jerk metric.
        self.assertEqual(rows[-1]["jerk_energy_m2_s5"], 10.)
        self.assertTrue(all(r["mean_delta"] <= 0 for r in report["adjacent_budget_changes"] if r["field"] == "score"))
        self.assertEqual(len(report["comparisons_versus_abom"]), 15)
        vs = next(r for r in report["comparisons_versus_abom"] if r["budget"] == 8 and r["field"] == "score")
        self.assertAlmostEqual(vs["mean_delta"], -.1)
        np.testing.assert_allclose(vs["exploratory_bootstrap_95_ci"], [-.1, -.1])

    def test_larger_unexecuted_budget_is_explicitly_skipped_or_rejected(self):
        report = analysis.analyze(self.root, [4, 8, 16])
        self.assertEqual([r["budget"] for r in report["skipped_prefixes"]], [16])
        self.assertEqual({r["budget"] for r in report["groups"]}, {4, 8})
        with self.assertRaisesRegex(ValueError, "exceed"):
            analysis.analyze(self.root, [4, 16], missing_budget="error")
        with self.assertRaisesRegex(ValueError, "No requested"):
            analysis.analyze(self.root, [16])
        for budgets in ([4, 4], [3], [6], [True], []):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                analysis.analyze(self.root, budgets)

    def test_nonconsecutive_or_missing_fe_and_nonmatching_run_budget_are_rejected(self):
        name = "train/" + analysis._key("single_bump", "central", "de", 3) + ".json"
        original, _ = analysis.read_evidence(self.root / name)
        mutations = (lambda run: run["evaluations"][1].update(fe=1),
                     lambda run: run["evaluations"].pop(), lambda run: run.update(budget=8))
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(original)
                mutate(changed)
                self.write(name, changed)
                self.rebind_training_hash(name)
                with self.assertRaises(ValueError):
                    analysis.analyze(self.root, [4, 8])
        self.write(name, original)

    def test_incomplete_or_duplicate_final_selection_is_rejected(self):
        original, _ = analysis.read_evidence(self.root / "selection.json")
        for duplicate in (False, True):
            changed = deepcopy(original)
            if duplicate:
                changed["incumbents"].append(deepcopy(changed["incumbents"][0]))
            else:
                changed["incumbents"].pop()
            self.write("selection.json", changed)
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "coverage"):
                analysis.analyze(self.root, [4])

    def test_training_bytes_and_source_snapshots_are_hash_bound(self):
        name = "train/" + analysis._key("single_bump", "central", "de", 3) + ".json"
        original = (self.root / name).read_bytes()
        (self.root / name).write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            analysis.analyze(self.root, [4])
        (self.root / name).write_bytes(original)
        (self.root / "source_snapshot" / "fixture.py").write_text("CHANGED = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot hash"):
            analysis.analyze(self.root, [4])

    def test_metrics_duration_must_match_selected_parameter(self):
        name = "train/" + analysis._key("single_bump", "central", "de", 3) + ".json"
        value, _ = analysis.read_evidence(self.root / name)
        value["evaluations"][1]["details"]["duration_s"] = 99.
        self.write(name, value)
        self.rebind_training_hash(name)
        with self.assertRaisesRegex(ValueError, "duration/parameter"):
            analysis.analyze(self.root, [4])

    def test_descriptive_and_paired_arithmetic_and_csv_output(self):
        stats = analysis.descriptive([1., 2., 3., 4.])
        self.assertEqual(stats, dict(mean=2.5, median=2.5, q1=1.75, q3=3.25, iqr=1.5))
        report = analysis.analyze(self.root, [4, 8, 12])
        destination = self.root / "analysis"
        analysis.write_outputs(report, destination)
        self.assertTrue((destination / "budget_analysis.json").is_file())
        self.assertEqual(len((destination / "prefix_incumbents.csv").read_text().splitlines()), 13)
        self.assertEqual(len((destination / "budget_groups.csv").read_text().splitlines()), 7)
        with self.assertRaises(ValueError):
            analysis.paired_change({3: {"score": 1}}, {4: {"score": 1}}, "score")


if __name__ == "__main__":
    unittest.main()
