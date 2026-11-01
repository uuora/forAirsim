"""Cache and evidence binding checks using real tiny curve-search campaigns.

The evaluator and all four searchers are real. A single harmless source fixture
stands in for the snapshot inventory so concurrent development of the runner
cannot invalidate the test corpus. No ABOM download, simulator, historical run,
or test-result substitution is used.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import run_parameter_perturbation_research as campaign


class ParameterCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_context = TemporaryDirectory()
        cls.addClassCleanup(cls.base_context.cleanup)
        cls.base = Path(cls.base_context.name)
        cls.source_root = cls.base / "source"
        cls.source_root.mkdir()
        (cls.source_root / "fixture.py").write_text("FROZEN_SOURCE = 1\n", encoding="utf-8")
        cls.frozen_output = cls.base / "corpus"
        cls.args = SimpleNamespace(
            families=["single_bump"], layouts=["central"],
            methods=["random", "de", "jde", "coordinate"], seeds=[3],
            population=16, budget=32, stress_seed=20261101, stress_count=2,
            upstream=cls.base / "unused-upstream",
        )
        with mock.patch.object(campaign, "ROOT", cls.source_root), \
                mock.patch.object(campaign, "SOURCES", ("fixture.py",)), \
                redirect_stdout(io.StringIO()):
            campaign.train(cls.args, cls.frozen_output)
            campaign.validate(cls.frozen_output)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(TemporaryDirectory()))
        self.output = self.temp / "campaign"
        shutil.copytree(self.frozen_output, self.output)
        self.stack.enter_context(mock.patch.object(campaign, "ROOT", self.source_root))
        self.stack.enter_context(mock.patch.object(campaign, "SOURCES", ("fixture.py",)))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def write(self, relative, value):
        path = self.output / relative
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")

    def edit(self, relative, mutation):
        value = campaign.read(self.output / relative)
        mutation(value)
        self.write(relative, value)

    def training_name(self, method="random"):
        return "train/" + campaign.key("single_bump", "central", method, 3) + ".json"

    def validation_name(self, method="random"):
        return "validation/" + campaign.key("single_bump", "central", method, 3) + ".json"

    def remove_generated_summary(self):
        for filename in ("summary.json", "summary.md"):
            path = self.output / filename
            if path.exists():
                path.unlink()

    def test_valid_real_chain_and_shared_initial_population_reuse_all_cache(self):
        initial = []
        for method in self.args.methods:
            run = campaign.read(self.output / self.training_name(method))
            self.assertEqual(len(run["evaluations"]), 32)
            initial.append(run["initial_population_normalized"])
        self.assertTrue(all(value == initial[0] for value in initial))
        selection_bytes = (self.output / "selection.json").read_bytes()
        validation_bytes = (self.output / "validation_summary.json").read_bytes()
        with mock.patch.object(campaign, "evaluate_parameters", side_effect=AssertionError("cache re-evaluated")), \
                mock.patch.object(campaign, "run_search", side_effect=AssertionError("cache re-searched")), \
                mock.patch.object(campaign, "run_blackbox_search", side_effect=AssertionError("cache re-searched")):
            campaign.train(self.args, self.output)
            campaign.validate(self.output)
            campaign.summarize(self.output)
        self.assertEqual((self.output / "selection.json").read_bytes(), selection_bytes)
        self.assertEqual((self.output / "validation_summary.json").read_bytes(), validation_bytes)
        report = campaign.read(self.output / "summary.json")
        self.assertEqual(report["training_runs"], 4)
        self.assertEqual(report["training_candidates"], 128)
        self.assertEqual(report["validation_conditions"], 10)  # 4 selected + initial, each 2
        self.assertEqual(len(report["initial_rows"]), 1)
        self.assertEqual({row["method"] for row in report["groups"]}, set(self.args.methods))

    def test_training_manifest_and_incumbent_corruption_are_rejected(self):
        original = campaign.read(self.output / self.training_name())
        for field, replacement in (("manifest_sha256", "other"), ("best_score", -999.),
                                   ("best_parameters", [0., 0., 0., 3.]), ("budget", 16)):
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed[field] = replacement
                self.write(self.training_name(), changed)
                with self.assertRaises(ValueError):
                    campaign.load_selection(self.output)
        self.write(self.training_name(), original)

    def test_missing_duplicate_or_nonconsecutive_training_evaluations_are_rejected(self):
        original = campaign.read(self.output / self.training_name())
        mutations = (
            lambda value: value["evaluations"].pop(),
            lambda value: value["evaluations"].append(deepcopy(value["evaluations"][0])),
            lambda value: value["evaluations"][1].update(fe=1),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(original)
                mutate(changed)
                self.write(self.training_name(), changed)
                with self.assertRaises(ValueError):
                    campaign.load_selection(self.output)
        self.write(self.training_name(), original)

    def test_changed_training_bytes_cannot_silently_rebind_frozen_selection(self):
        self.edit(self.training_name(), lambda value: value.update(elapsed_s=123.))
        with self.assertRaises(ValueError):
            campaign.validate(self.output)
        with self.assertRaises(ValueError):
            campaign.train(self.args, self.output)

    def test_nonidentical_cached_initial_population_is_rejected_on_train_resume(self):
        self.edit(self.training_name("de"), lambda value:
                  value["initial_population_normalized"][1].__setitem__(0, .1234567))
        with self.assertRaises(ValueError):
            campaign.train(self.args, self.output)

    def test_selection_parameters_and_missing_or_duplicate_coverage_are_rejected(self):
        original = campaign.read(self.output / "selection.json")
        mutations = (
            lambda value: value["incumbents"][0]["parameters"].__setitem__(0, .1234567),
            lambda value: value["incumbents"].pop(),
            lambda value: value["incumbents"].append(deepcopy(value["incumbents"][0])),
            lambda value: value.update(selection_basis="validation"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(original)
                mutate(changed)
                self.write("selection.json", changed)
                with self.assertRaises(ValueError):
                    campaign.validate(self.output)
        self.write("selection.json", original)

    def test_source_snapshot_mismatch_is_rejected_before_evaluation(self):
        (self.output / "source_snapshot" / "fixture.py").write_text("CHANGED = 2\n", encoding="utf-8")
        with mock.patch.object(campaign, "evaluate_parameters") as evaluate:
            with self.assertRaisesRegex(ValueError, "[Ss]ource changed"):
                campaign.validate(self.output)
            evaluate.assert_not_called()

    def test_current_source_mismatch_is_rejected_before_evaluation(self):
        real_digest = campaign.digest

        def altered_digest(path):
            return "0" * 64 if Path(path) == self.source_root / "fixture.py" else real_digest(path)

        with mock.patch.object(campaign, "digest", side_effect=altered_digest), \
                mock.patch.object(campaign, "evaluate_parameters") as evaluate:
            with self.assertRaisesRegex(ValueError, "[Ss]ource changed"):
                campaign.validate(self.output)
            evaluate.assert_not_called()

    def test_validation_cache_binding_and_case_count_changes_are_rejected(self):
        original = campaign.read(self.output / self.validation_name())
        mutations = (
            lambda value: value.update(selection_sha256="other"),
            lambda value: value.update(manifest_sha256="other"),
            lambda value: value.update(stress_sha256="other"),
            lambda value: value["parameters"].__setitem__(0, .1234567),
            lambda value: value["cases"].pop(),
            lambda value: value.update(failures=99),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(original)
                mutate(changed)
                self.write(self.validation_name(), changed)
                with self.assertRaises(ValueError):
                    campaign.validate(self.output)
        self.write(self.validation_name(), original)

    def test_validation_summary_values_must_match_frozen_result_evidence(self):
        original = campaign.read(self.output / "validation_summary.json")
        replacements = dict(quality=-999., duration_s=-999., failures=99, conditions=99,
                            all_conditions_pass="not-a-boolean", jerk_energy_m2_s5=-999.,
                            deformation_rms_m=-999.)
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed["rows"][0][field] = replacement
                self.write("validation_summary.json", changed)
                try:
                    with self.assertRaises(ValueError):
                        campaign.summarize(self.output)
                finally:
                    self.remove_generated_summary()
        self.write("validation_summary.json", original)

    def test_missing_duplicate_or_unknown_validation_rows_are_rejected(self):
        original = campaign.read(self.output / "validation_summary.json")
        initial_index = next(i for i, row in enumerate(original["rows"]) if row["method"] == "initial")
        mutations = (
            lambda value: value["rows"].pop(0),
            lambda value: value["rows"].append(deepcopy(value["rows"][0])),
            lambda value: value["rows"].pop(initial_index),
            lambda value: value["rows"].append(deepcopy(value["rows"][initial_index])),
            lambda value: value["rows"].append(dict(deepcopy(value["rows"][0]), method="unknown")),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(original)
                mutate(changed)
                self.write("validation_summary.json", changed)
                try:
                    with self.assertRaises(ValueError):
                        campaign.summarize(self.output)
                finally:
                    self.remove_generated_summary()
        self.write("validation_summary.json", original)

    def test_generated_summary_arithmetic_cannot_be_silently_replaced(self):
        campaign.summarize(self.output)
        self.edit("summary.json", lambda value: value["groups"][0].update(mean_quality=-999.))
        with self.assertRaises(ValueError):
            campaign.summarize(self.output)


if __name__ == "__main__":
    unittest.main()
