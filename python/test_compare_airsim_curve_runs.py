"""Small offline comparison tests; never execute or contact a simulator."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_airsim_curve_runs as compare


class CompareAirSimCurveRunsTests(unittest.TestCase):
    def minimal_row(self,group="hermite_abom",**changes):
        value=dict(group_id=group,planned_slot=True,producer_status="COMPLETED",workflow_completed=True,
            included_in_numeric_summary=True,control_error_gates_passed=False,status="COMPLETED_CONTROL_FAIL",
            duplicate_evidence_of=None,**{name:1. for name in compare.METRICS})
        value.update(changes)
        return value

    def test_n1_has_no_sample_sd_and_control_failures_are_kept(self):
        summary=compare.summarize([self.minimal_row()])[0]
        self.assertEqual(summary["n_control_fail"],1)
        self.assertEqual(summary["n_valid_completed"],1)
        self.assertIsNone(summary["statistics"]["position_rmse_m"]["std"])
        self.assertNotIn("confidence_interval",summary)

    def test_repetitions_and_failures_are_counted_separately(self):
        rows=[self.minimal_row(position_rmse_m=1.),self.minimal_row(position_rmse_m=3.),
              self.minimal_row(status="FAILED_BEFORE_REFERENCE",producer_status="FAILED",workflow_completed=False,
                  included_in_numeric_summary=False,control_error_gates_passed=None),
              self.minimal_row(status="NOT_RUN",producer_status=None,workflow_completed=False,
                  included_in_numeric_summary=False,control_error_gates_passed=None)]
        summary=compare.summarize(rows)[0]
        self.assertEqual(summary["n_rows"],4)
        self.assertEqual(summary["n_valid_completed"],2)
        self.assertEqual(summary["n_execution_failed"],1)
        self.assertEqual(summary["n_not_run"],1)
        self.assertEqual(summary["statistics"]["position_rmse_m"]["mean"],2.)
        self.assertAlmostEqual(summary["statistics"]["position_rmse_m"]["std"],2**.5)

    def test_existing_output_is_refused_before_reading_input(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                compare.build_comparison([],directory)

    def test_different_conditions_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            out=base/"out"
            a=dict(scheduled=["a"],fingerprint="one",conditions=dict(map="map1"))
            b=dict(scheduled=["a"],fingerprint="two",conditions=dict(map="map2"))
            with patch.object(compare,"replay_directory",side_effect=lambda p:Path(p)), \
                 patch.object(compare,"inspect_run",side_effect=[a,b]):
                with self.assertRaisesRegex(ValueError,"map"):
                    compare.build_comparison([base/"run1",base/"run2"],out)
            self.assertFalse(out.exists())

    def test_exact_selection_does_not_expand_to_other_scheduled_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            run=dict(scheduled=["a","b","c"],fingerprint="same",conditions={})
            # Stop just after exact pair expansion so this checks selection
            # behavior without writing a synthetic research comparison.
            seen=[]
            def collect(run,group):
                seen.append(group)
                raise RuntimeError("selection checked")
            with patch.object(compare,"replay_directory",side_effect=lambda p:Path(p)), \
                 patch.object(compare,"inspect_run",return_value=run), \
                 patch.object(compare,"collect_record",side_effect=collect):
                with self.assertRaisesRegex(RuntimeError,"selection checked"):
                    compare.build_comparison([],base/"out",selections=[dict(run_path=str(base/"run"),group_ids=["b"])])
            self.assertEqual(seen,["b"])


if __name__=="__main__":
    unittest.main()
