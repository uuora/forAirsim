"""Evidence binding tests with a tiny mocked evaluator and temporary campaigns.

No AirSim connection, physical rollout, ABOM network or historical report is
used. These tests exercise the campaign's persisted train/select/validate chain.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import run_hermite_abom_research as campaign


@dataclass(frozen=True)
class FixtureCase:
    name: str


class CampaignBindingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.output = self.root / "campaign"
        self.source = self.root / "frozen_source.py"
        self.source.write_text("FROZEN = 1\n", encoding="utf-8")
        self.args = SimpleNamespace(
            seeds=[7], methods=["random"], budget=8, population=4, dt=.05,
            initialization="uniform", injection_fraction=.5,
            upstream=self.root / "unused-upstream",
        )
        self.training_cases = tuple(FixtureCase(f"train_{i}") for i in range(3))
        self.validation_cases = tuple(FixtureCase(f"validation_{i}") for i in range(30))
        self.stack.enter_context(mock.patch.object(campaign, "ROOT", self.root))
        self.stack.enter_context(mock.patch.object(campaign, "SOURCE_FILES", (self.source.name,)))
        self.stack.enter_context(mock.patch.object(campaign, "make_training_cases", return_value=self.training_cases))
        self.stack.enter_context(mock.patch.object(campaign, "make_validation_cases", return_value=self.validation_cases))
        self.evaluate = self.stack.enter_context(mock.patch.object(
            campaign, "evaluate_parameters", side_effect=self.fake_evaluate))
        self.search = self.stack.enter_context(mock.patch.object(
            campaign, "run_search", side_effect=self.fake_search))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    @staticmethod
    def fake_evaluate(parameters, cases, *, dt):
        vector = (parameters.to_vector().tolist() if hasattr(parameters, "to_vector")
                  else list(parameters))
        return dict(
            parameters=vector, score=.25, feasible=True,
            score_details=dict(failed_cases=0, mean_elapsed_s=1.25,
                               mean_normalized_peak_command_jerk=.02),
            cases=[dict(passed=True) for _ in cases],
        )

    @staticmethod
    def fake_search(objective, lower, upper, **kwargs):
        # Calling the supplied objective verifies train's data path while the
        # patched evaluator makes this a small unit test rather than a rollout.
        value = objective(kwargs["anchor"])
        return dict(
            method=kwargs["method"], seed=kwargs["seed"],
            population=kwargs["population"], budget=kwargs["budget"],
            best_parameters=value["parameters"], best_score=value["score"],
            evaluations=[dict(score=value["score"]) for _ in range(kwargs["budget"])],
            elapsed_s=.01,
        )

    def train(self):
        campaign.train(self.args, self.output)

    def validate(self):
        campaign.validate(self.args, self.output)

    def edit_json(self, filename, mutate):
        path = self.output / filename
        value = campaign.read(path)
        mutate(value)
        campaign.save(path, value)

    def test_valid_chain_reuses_cache_without_evaluating_again(self):
        self.train()
        self.assertEqual(self.evaluate.call_args.args[1], self.training_cases)
        self.validate()
        self.assertEqual(self.evaluate.call_count, 3)  # train, baseline, selected
        self.assertTrue(all(call.args[1] == self.validation_cases
                            for call in self.evaluate.call_args_list[1:]))
        selection_bytes = (self.output / "selection.json").read_bytes()
        self.evaluate.reset_mock()
        self.search.reset_mock()
        self.train()
        self.validate()
        campaign.summarize(self.output)
        self.evaluate.assert_not_called()
        self.search.assert_not_called()
        self.assertEqual((self.output / "selection.json").read_bytes(), selection_bytes)
        report = campaign.read(self.output / "summary.json")
        self.assertEqual(report["budget_accounting"]["training_candidates"], 8)
        self.assertEqual(report["budget_accounting"]["training_rollout_conditions"], 24)
        self.assertEqual(report["budget_accounting"]["validation_rollout_conditions"], 60)

    def test_cached_baseline_from_another_manifest_is_rejected(self):
        self.train()
        self.validate()
        self.edit_json("validation_default.json", lambda value: value.update(manifest_sha256="other"))
        self.evaluate.reset_mock()
        with self.assertRaisesRegex(ValueError, "[Bb]aseline|manifest"):
            self.validate()
        self.evaluate.assert_not_called()

    def test_cached_baseline_must_use_the_default_parameters(self):
        self.train()
        self.validate()
        # Same campaign, different candidate: the baseline must retain its
        # declared default identity, even when its manifest hash is genuine.
        self.edit_json("validation_default.json",
                       lambda value: value["parameters"].__setitem__(0, .1))
        self.evaluate.reset_mock()
        with self.assertRaises(ValueError):
            self.validate()
        self.evaluate.assert_not_called()

    def test_cached_validation_parameters_must_match_selected_parameters(self):
        self.train()
        self.validate()
        self.edit_json("validation_random_s7.json",
                       lambda value: value["parameters"].__setitem__(0, .1))
        self.evaluate.reset_mock()
        with self.assertRaisesRegex(ValueError, "[Cc]ached|selected|parameters"):
            self.validate()
        self.evaluate.assert_not_called()

    def test_cached_validation_must_match_current_selection_hash(self):
        self.train()
        self.validate()
        self.edit_json("validation_random_s7.json",
                       lambda value: value.update(selection_sha256="other"))
        with self.assertRaisesRegex(ValueError, "[Cc]ached|selection|selected"):
            self.validate()

    def test_summary_from_another_selection_is_rejected(self):
        self.train()
        self.validate()
        self.edit_json("validation_summary.json",
                       lambda value: value.update(selection_sha256="other"))
        with self.assertRaisesRegex(ValueError, "[Ss]ummary|selection"):
            campaign.summarize(self.output)
        self.assertFalse((self.output / "summary.json").exists())

    def test_summary_used_fields_must_match_their_frozen_evidence(self):
        self.train()
        self.validate()
        path = self.output / "validation_summary.json"
        original = campaign.read(path)
        substitutions = dict(
            training_score=.01, failed_cases=99, jerk_proxy=.99, cases=999,
            held_out_seed_cases_passed=0, held_out_seed_cases=0,
        )
        for key, value in substitutions.items():
            with self.subTest(field=key):
                changed = deepcopy(original)
                changed["rows"][0][key] = value
                campaign.save(path, changed)
                with self.assertRaises(ValueError):
                    campaign.summarize(self.output)
        campaign.save(path, original)

    def test_changed_source_is_rejected_before_validation_evaluation(self):
        self.train()
        self.source.write_text("FROZEN = 2\n", encoding="utf-8")
        self.evaluate.reset_mock()
        with self.assertRaisesRegex(ValueError, "[Ss]ource changed"):
            self.validate()
        self.evaluate.assert_not_called()

    def test_resuming_training_cannot_replace_a_frozen_selection(self):
        self.train()
        selection_path = self.output / "selection.json"
        original_selection = selection_path.read_bytes()
        self.edit_json("train_random_s7.json",
                       lambda value: value.update(best_parameters=[.1, .3, .2, .45, 1.],
                                                  best_score=.1))
        with self.assertRaisesRegex(ValueError, "[Ff]rozen selection|selection changed"):
            self.train()
        self.assertEqual(selection_path.read_bytes(), original_selection)

    def test_selection_parameters_cannot_change_without_training_evidence(self):
        self.train()
        # Keep the manifest and training hashes valid. Only the selected values
        # are changed, so a hash-only check of the training file is insufficient.
        self.edit_json("selection.json", lambda value:
                       value["incumbents"][0]["best_parameters"].__setitem__(0, .1))
        self.evaluate.reset_mock()
        with self.assertRaises(ValueError):
            self.validate()
        self.evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
