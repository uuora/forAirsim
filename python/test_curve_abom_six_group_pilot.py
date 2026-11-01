"""Small contract tests; do not run the 320-FE six-group experiment."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_curve_abom_six_group_pilot as pilot


class SixGroupPilotTests(unittest.TestCase):
    def test_config_has_fixed_six_groups_and_common_search_settings(self):
        config=pilot.make_config()
        self.assertEqual(len(config["groups"]),6)
        self.assertEqual(len(set(config["frozen_execution_order"])),6)
        self.assertEqual([config[k] for k in ("population","proposal_budget","dropout","seed")],[20,80,.95,1101])
        self.assertFalse(config["screening_enabled"])
        self.assertEqual(config["expected_total_objective_calls"],326)
        self.assertEqual(sum(g["search_method"] is not None for g in config["groups"]),4)

    def test_canonical_domain_basis_preflight_uses_no_model_evaluations(self):
        with patch.object(pilot,"evaluate_timed_parameters",side_effect=AssertionError("unexpected oracle")):
            result=pilot.basis_preflight(pilot.make_config())
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["probes"],17)
        self.assertEqual(result["model_objective_calls"],0)

    def test_ordinary_anchor_calls_actual_model_once_and_exports_replay_schema(self):
        config=pilot.make_config()
        record=pilot.run_group(config["groups"][0],config,pilot.DEFAULT_UPSTREAM)
        self.assertEqual(record["parameters"],config["anchor_parameters"])
        self.assertIsNone(record["search"])
        self.assertEqual(record["counts"]["total_objective_calls"],1)
        self.assertEqual(record["counts"]["search_objective_calls"],0)
        self.assertIn("tracking_trace",record["selected_validation"])
        self.assertEqual(record["timings"]["search_wall_s"],0.)
        row=pilot.summary_row(record)
        self.assertEqual(set(row),set(pilot.SUMMARY_FIELDS))
        plan=pilot.export_replay_plan([record])
        self.assertEqual(plan["schema_version"],1)
        self.assertEqual(set(plan),{"schema_version","plans"})
        self.assertEqual(set(plan["plans"][0]),{"group_id","representation","family","parameters","seed","selection_provenance"})
        self.assertIn("model_feasible",plan["plans"][0]["selection_provenance"])

    def test_optimized_adapter_counts_search_and_extra_revalidation(self):
        config=pilot.make_config()
        group=next(g for g in config["groups"] if g["search_method"]=="abom_perturb")
        # The model and neural search are both mocked in this accounting test.
        # Synthetic records stay in memory; no claimed research evidence is saved.
        real=pilot.run_group(config["groups"][0],config,pilot.DEFAULT_UPSTREAM)["selected_validation"]
        def model(parameters,**kwargs):
            value=copy.deepcopy(real)
            if not kwargs.get("include_trace"):
                value.pop("tracking_trace",None)
            return value
        def fake_search(objective,lower,upper,**kwargs):
            self.assertIsNone(kwargs["lower_bound"])
            self.assertIsNone(kwargs["bounded_objective"])
            self.assertEqual(kwargs["method"],"abom_perturb")
            records=[]
            for i in range(80):
                result=objective(np.array(config["anchor_parameters"]))
                records.append(dict(status="EVALUATED",proposal_index=i+1,fe=i+1,
                    parameters=config["anchor_parameters"][:],score=result["score"],feasible=result["feasible"]))
            return dict(method=kwargs["method"],seed=1101,population=20,proposal_budget=80,
                initialization="uniform",dropout=.95,injection_fraction=.5,screening_enabled=False,
                pruned_count=0,partial_evaluations=0,lower_bound_calls=0,evaluations=records,proposals=80,
                full_evaluations=80,best_score=records[0]["score"],best_parameters=config["anchor_parameters"][:],
                neural_generation_s=0.,neural_update_s=0.,evaluation_s=0.)
        with patch.object(pilot,"evaluate_timed_parameters",side_effect=model) as mocked:
            record=pilot.run_group(group,config,pilot.DEFAULT_UPSTREAM,search_runner=fake_search)
        self.assertEqual(mocked.call_count,81)
        self.assertEqual(record["counts"]["total_objective_calls"],81)
        self.assertEqual(record["selection_provenance"]["proposal_index"],1)
        self.assertEqual(record["timings"]["search_plus_selected_validation_wall_s"],
            record["timings"]["search_wall_s"]+record["timings"]["selected_validation_wall_s"])

    def test_existing_directory_is_refused_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pilot,"evaluate_timed_parameters",side_effect=AssertionError("unexpected oracle")):
                with self.assertRaises(FileExistsError):
                    pilot.run(Path(directory))


if __name__=="__main__":
    unittest.main()
