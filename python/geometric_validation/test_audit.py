"""Synthetic evidence tests only: these fixtures are never flight evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geometric_validation.audit import audit
from geometric_validation.core import command
from geometric_validation.reference import build_reference
from geometric_validation.runtime import DEFAULT_CONFIG


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuditEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="geometric_synthetic_audit_")
        self.folder = Path(self.temp.name)
        self.cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.cfg["cases"] = {"hover": {"offsets": [[0, 0, 0]], "segment_durations": [.2]}}
        self.cfg["execution"].update(start_hold_s=.1, terminal_hold_s=.1)
        _write(self.folder / "config.json", self.cfg)
        source = self.folder / "source" / "synthetic.txt"
        source.parent.mkdir()
        source.write_text("Synthetic test fixture; no AirSim calls.\n", encoding="utf-8")
        self.release = 1_000_000_000
        self.manifest = {"protocol": self.cfg["protocol"], "case": "hover",
                         "config_sha256": _sha(self.folder / "config.json"),
                         "sources": {"synthetic.txt": _sha(source)},
                         "live": {"initial_state": {"collision_timestamp_ns": 0},
                                  "synthetic": True}}
        _write(self.folder / "manifest.json", self.manifest)
        self.rows = [self.row(float(t), "GEOMETRIC_SETTLING") for t in np.arange(-.12, .001, .02)]
        self.rows += [self.row(float(t), "TRACKING") for t in np.arange(.02, .321, .02)]
        self.events = [{"event": "RELEASE", "timestamp_ns": self.release},
                       {"event": "TERMINAL_STABLE", "t": .32}]
        self.report = {"normal_completion": True, "outcome": "COMPLETED", "actual_completion_time_s": .32,
                       "cleanup": {"landed": True}, "synthetic": True}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def row(self, t, phase):
        ref = build_reference(self.cfg, "hover").sample(max(0., t))
        cmd = command(ref["p"], ref["v"], [1, 0, 0, 0], ref, self.cfg["controller"])
        return {"timestamp_ns": self.release + round(t * 1e9), "t": max(0., t), "phase": phase,
                "position": list(ref["p"]), "velocity": list(ref["v"]), "acceleration": list(ref["a"]),
                "angular_velocity": [0., 0., 0.], "quaternion_wxyz": [1., 0., 0., 0.],
                "collision": False, "collision_timestamp_ns": 0, "landed_state": 1,
                "reference": ref, "command": cmd, "R_reference": cmd["R_des"],
                "attitude_reference_error_rad": 0., "command_sent": True}

    def save(self):
        (self.folder / "samples.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows), encoding="utf-8")
        _write(self.folder / "events.json", self.events)
        _write(self.folder / "report.json", self.report)

    def check(self):
        self.save()
        return audit(self.folder)

    def test_valid_synthetic_hover_and_repeatable_csv(self):
        first = self.check()
        self.assertTrue(first["data_audit_passed"], first["issues"])
        self.assertTrue(first["control_constraints_passed"], first["issues"])
        second = audit(self.folder)
        self.assertTrue(second["data_audit_passed"], second["issues"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_failed_task_preserved_without_claiming_control_pass(self):
        self.report.update(normal_completion=False, outcome="FAILED", actual_completion_time_s=None)
        self.events = [self.events[0], {"event": "FAILURE", "error": "Synthetic injected failure"}]
        self.rows[-1].update(failure="SYNTHETIC_FAILURE", command_sent=False)
        result = self.check()
        self.assertTrue(result["data_audit_passed"], result["issues"])
        self.assertFalse(result["control_constraints_passed"])
        self.assertFalse(result["normal_completion"])

    def test_csv_and_source_tampering_detected(self):
        self.check()
        csvpath = self.folder / "metrics_samples.csv"
        csvpath.write_text(csvpath.read_text(encoding="utf-8-sig").replace("False", "True", 1), encoding="utf-8-sig")
        (self.folder / "source" / "synthetic.txt").write_text("changed", encoding="utf-8")
        result = audit(self.folder)
        self.assertFalse(result["data_audit_passed"])
        self.assertTrue(any("CSV" in item for item in result["issues"]))
        self.assertTrue(any("Snapshot" in item for item in result["issues"]))

    def test_config_tamper_detected(self):
        changed = copy.deepcopy(self.cfg)
        changed["limits"]["position_error_m"] = .6
        _write(self.folder / "config.json", changed)
        result = audit(self.folder)
        self.assertFalse(result["data_audit_passed"])
        self.assertTrue(any("hash" in item.lower() for item in result["issues"]))

    def test_truncated_jsonl_produces_failed_audit_artifact(self):
        with (self.folder / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"timestamp_ns":')
        result = audit(self.folder)
        self.assertFalse(result["data_audit_passed"])
        self.assertTrue((self.folder / "audit.json").is_file())

    def test_missing_last_samples_cannot_claim_old_completion(self):
        self.rows = self.rows[:-4]
        result = self.check()
        self.assertFalse(result["data_audit_passed"])

    def test_early_completion_cannot_borrow_tracking_time_for_terminal_hold(self):
        self.cfg["cases"]["hover"]["segment_durations"] = [1.]
        _write(self.folder / "config.json", self.cfg)
        self.manifest["config_sha256"] = _sha(self.folder / "config.json")
        _write(self.folder / "manifest.json", self.manifest)
        result = self.check()
        self.assertFalse(result["data_audit_passed"], "Completed before frozen reference ended")

    def test_starting_hold_is_required_and_recomputed(self):
        self.rows = [r for r in self.rows if r["phase"] == "TRACKING"]
        result = self.check()
        self.assertFalse(result["data_audit_passed"], "Release had no verified starting hold")

    def test_logged_reference_is_bound_to_frozen_case(self):
        row = self.rows[-3]
        row["reference"]["p"][0] += .2
        row["position"][0] += .2
        result = self.check()
        self.assertFalse(result["data_audit_passed"], "A changed reference must not create a zero-error success")

    def test_first_tracking_gap_is_included(self):
        self.rows = [r for r in self.rows if r["phase"] != "TRACKING" or r["t"] >= .14]
        result = self.check()
        self.assertFalse(result["control_constraints_passed"], "Release-to-first-state gap was .14 seconds")

    def test_unsent_command_and_new_collision_fail_constraints(self):
        self.rows[-1]["command_sent"] = False
        self.rows[-1].update(collision=True, collision_timestamp_ns=self.release + 320_000_000)
        result = self.check()
        self.assertFalse(result["control_constraints_passed"])

    def test_reference_and_command_attitude_errors_stay_separate(self):
        # Reference stays level, while synthetic corrective desired attitude rolls .05 rad.
        from geometric_validation.core import euler_to_matrix
        for row in self.rows:
            if row["phase"] == "TRACKING":
                row["command"]["roll"] = .05
                row["command"]["R_des"] = euler_to_matrix(.05, 0, 0).tolist()
        result = self.check()
        self.assertTrue(result["data_audit_passed"], result["issues"])
        self.assertAlmostEqual(result["metrics"]["attitude_reference_error_rad"]["peak"], 0)
        self.assertAlmostEqual(result["metrics"]["attitude_command_error_rad"]["peak"], .05)


if __name__ == "__main__":
    unittest.main()
