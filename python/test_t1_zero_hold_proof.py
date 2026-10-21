"""Regression against the actual failed receipt; reject weaker hold evidence."""
import copy
import json
from pathlib import Path
import unittest

from t1_zero_hold_proof_v1 import stable_interval_proof

ROOT=Path(__file__).resolve().parents[1]


class StableProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=json.loads((ROOT/"logs/t1_scale_50_v9_rep3_20261007/flight/report.json").read_text(encoding="utf-8"))
        cls.saved=next(f["zero_held_stop_proof"] for f in raw["frames"] if f["frame_id"].endswith("_f240"))

    def test_actual_receipt_keeps_the_original_stable_window_without_preceding_feedback(self):
        before=copy.deepcopy(self.saved);proof=stable_interval_proof(before)
        self.assertEqual(before,self.saved)
        self.assertEqual(proof["configured_settlement"],before["configured_settlement"])
        self.assertEqual(proof["excluded_pre_stable_history_count"],1)
        self.assertEqual(len(proof["samples"]),4)
        self.assertEqual(proof["sampled_stop_started_monotonic_s"],before["sampled_stop_started_monotonic_s"])

    def test_feedback_command_inside_stable_interval_is_refused(self):
        proof=copy.deepcopy(self.saved);proof["samples"][-1]["actual_last_command_xy_m_s"]=[0.,.001]
        with self.assertRaises(ValueError):stable_interval_proof(proof)

    def test_insufficient_distinct_samples_or_dwell_is_refused(self):
        proof=copy.deepcopy(self.saved);proof["samples"]=proof["samples"][-2:]
        with self.assertRaises(ValueError):stable_interval_proof(proof)

    def test_changed_current_state_cannot_use_the_saved_proof(self):
        proof=copy.deepcopy(self.saved);proof["current_self_state"]["state_timestamp_ns"]+=1
        with self.assertRaises(ValueError):stable_interval_proof(proof)


if __name__=="__main__":unittest.main()
