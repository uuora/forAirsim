"""Provenance, session-group split and controlled-label export regressions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from build_t1_balloon_dataset import build, DatasetBuildError
from t1_ground_perception_probe import visible_mask


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class BalloonDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.palette = {"0": [0, 0, 0], "42": [92, 31, 106], "43": [207, 229, 90], "44": [125, 75, 48]}
        palette_path = self.root / "palette.txt"
        palette_path.write_text("\n".join(key + "\t" + json.dumps(value) for key, value in self.palette.items()), encoding="utf-8")
        settings = self.root / "settings.json"
        save(settings, {"SimMode": "Multirotor"})
        self.sources = {name: ("frozen-source-" + name).encode("ascii") for name in (
            "t1_ground_perception_probe.py", "setup_scene.py", "camera_geometry.py", "t1_balloon_data_probe.py")}
        self.launcher = b"# frozen launcher\n"
        self.plan = {"status": "FROZEN_BEFORE_COLLECTION_AND_TRAINING", "ground_only": True,
            "created_utc": "2026-10-01T00:00:00Z", "settings_path": str(settings), "settings_sha256": sha(settings),
            "segmentation_palette_path": str(palette_path), "cases_per_episode": 3, "frames_per_case": 2,
            "expected_source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in self.sources.items()},
            "launcher_sha256": hashlib.sha256(self.launcher).hexdigest(), "episodes": []}
        for index, split in enumerate(("train", "validation", "test"), 1):
            self.plan["episodes"].append(self.make_episode(index, split))
        self.plan_path = self.root / "plan.json"
        self.output = self.root / "dataset"
        self.sync_plan()

    def ground(self):
        return {name: {"landed": True, "api_control_enabled": False, "speed_m_s": 0.0}
                for name in ("DroneA", "DroneB")}

    def make_episode(self, index, split):
        folder = self.root / ("ep" + str(index))
        probe = folder / "probe"
        probe.mkdir(parents=True)
        snapshot = folder / "source_snapshot"
        snapshot.mkdir()
        for name, data in self.sources.items():
            (snapshot / name).write_bytes(data)
        (snapshot / "run-t1-balloon-data-campaign.ps1").write_bytes(self.launcher)
        protocol_path = folder / "protocol.json"
        seed = 100 + index
        cases = [{"case_id": "single", "target_world_ned_m": [0, 0, -3]},
                 {"case_id": "negative", "target_world_ned_m": [0, 0, -3], "occluder_world_ned_m": [-2, 0, -3]},
                 {"case_id": "pair", "target_world_ned_m": [0, -1, -3], "distractor_world_ned_m": [0, 1, -3]}]
        protocol = {"protocol_id": "fixture_" + str(index), "acquisition_seed": seed, "split": split,
                    "formal_efficacy_results": False, "frames_per_case": 2, "cases": cases}
        save(protocol_path, protocol)
        report_path = probe / "report.json"
        report = {"status": "PASS", "configuration": protocol, "protocol": str(protocol_path),
            "protocol_sha256": sha(protocol_path), "source_sha256": dict(self.plan["expected_source_sha256"]),
            "safe_ground_closure": True, "paused_after": True, "flight_commands_issued": False,
            "api_control_acquired": False, "independent_UE_sessions": 1, "formal_efficacy_results": False,
            "evaluation_segmentation_ids": {"T1ProbeTarget": 42, "T1ProbeDistractor": 43, "T1ProbeOccluder": 44},
            "segmentation_palette_sha256": sha(self.root / "palette.txt"), "segmentation_palette_rgb": self.palette,
            "ground_before": self.ground(), "ground_after": self.ground(), "frames": []}
        for case_index, case in enumerate(cases):
            for repeat in range(1, 3):
                frame_id = case["case_id"] + "_" + str(repeat).zfill(2)
                rgb = np.full((48, 64, 3), [index * 17, case_index * 21, repeat * 25], np.uint8)
                segmentation = np.zeros_like(rgb)
                if case["case_id"] == "negative":
                    segmentation[8:36, 8:36] = self.palette["44"][::-1]
                else:
                    segmentation[10:35, 2:30] = self.palette["42"][::-1]
                    if case["case_id"] == "pair":
                        segmentation[10:35, 34:62] = self.palette["43"][::-1]
                media_paths = {"rgb": probe / (frame_id + "_rgb.png"),
                               "segmentation": probe / (frame_id + "_seg.png"),
                               "depth": probe / (frame_id + "_depth.npy")}
                self.assertTrue(cv2.imwrite(str(media_paths["rgb"]), rgb))
                self.assertTrue(cv2.imwrite(str(media_paths["segmentation"]), segmentation))
                np.save(media_paths["depth"], np.full((48, 64), 4., dtype=np.float64), allow_pickle=False)
                observed = {name: visible_mask(segmentation, self.palette[str(segid)])
                            for name, segid in (("target", 42), ("distractor", 43), ("occluder", 44))}
                report["frames"].append({"frame_id": frame_id, "case_id": case["case_id"],
                    "session_id": "fixture_session_" + str(index), "scene_acceptance": "PASS_INTENDED_VISIBLE_INSTANCES",
                    "ground_state": self.ground(), "image_size_wh": [64, 48], "evaluation_only": observed,
                    "media": {kind: {"path": str(path), "sha256": sha(path)} for kind, path in media_paths.items()}})
        save(report_path, report)
        manifest_path = folder / "manifest.json"
        save(manifest_path, {"status": "PASS", "episode_id": "ep" + str(index), "split": split,
            "acquisition_seed": seed, "flight_requests": False, "formal_efficacy_results": False,
            "engine_closed_after_safe_completion": True, "engine_process_id": 1000 + index,
            "engine_started_utc": f"2026-10-04T00:00:0{index}.1234567Z",
            "protocol": str(protocol_path), "protocol_sha256": sha(protocol_path),
            "report": str(report_path), "report_sha256": sha(report_path), "saved_frames": len(report["frames"]),
            "settings": self.plan["settings_path"], "settings_sha256": self.plan["settings_sha256"],
            "launcher_sha256": self.plan["launcher_sha256"]})
        return {"episode_id": "ep" + str(index), "split": split, "acquisition_seed": seed,
                "protocol_path": str(protocol_path), "protocol_sha256": sha(protocol_path),
                "manifest_path": str(manifest_path), "report_path": str(report_path)}

    def sync_plan(self):
        save(self.plan_path, self.plan)

    def change_report(self, index, change):
        row = self.plan["episodes"][index]
        report_path = Path(row["report_path"])
        data = load(report_path)
        change(data)
        save(report_path, data)
        manifest = load(row["manifest_path"])
        manifest["report_sha256"] = sha(report_path)
        save(row["manifest_path"], manifest)

    def change_manifest(self, index, change):
        path = self.plan["episodes"][index]["manifest_path"]
        data = load(path)
        change(data)
        save(path, data)

    def rejected(self, match):
        with self.assertRaisesRegex(DatasetBuildError, match) as raised:
            build(self.plan_path, self.output)
        self.assertFalse(self.output.exists())
        receipt_path = raised.exception.receipt_path
        self.assertTrue(receipt_path.is_file())
        receipt = load(receipt_path)
        self.assertEqual(receipt["phase"], "PRE_EXPORT_VALIDATION")
        self.assertFalse(receipt["dataset_output_created"])
        return receipt_path

    def test_success_has_correct_split_counts_negatives_and_both_balloon_instances(self):
        result = build(self.plan_path, self.output)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["independent_UE_sessions"], 3)
        self.assertEqual(result["episode_counts_by_split"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(result["frame_count"], 18)
        self.assertEqual(result["visible_instance_labels"], 18)
        self.assertEqual(result["negative_frame_count"], 6)
        self.assertEqual(result["frames"], result["rows"])
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["training_executed"])
        self.assertTrue(result["train_val_test_ready"])
        self.assertEqual(result["plan_sha256"], sha(self.plan_path))
        for split, folder in (("train", "train"), ("validation", "val"), ("test", "test")):
            self.assertEqual(result["splits"][split]["frame_count"], 6)
            self.assertEqual(result["splits"][split]["negative_frame_count"], 2)
            self.assertEqual(len(list((self.output / "images" / folder).glob("*.png"))), 6)
        for row in result["rows"]:
            self.assertEqual(sha(row["image_path"]), row["source_rgb_sha256"])
            self.assertEqual(row["image_sha256"], row["exported_rgb_sha256"])
            self.assertEqual(sha(row["label_path"]), row["source_label_sha256"])
            text = Path(row["label_path"]).read_text(encoding="utf-8")
            self.assertEqual(len(text.splitlines()), row["visible_balloon_count"])
            if row["case_id"] == "negative":
                self.assertEqual(text, "")
            elif row["case_id"] == "pair":
                self.assertEqual(row["visible_boxes_xyxy_pixels"], [[2, 10, 30, 35], [34, 10, 62, 35]])
                self.assertTrue(all(line.startswith("0 ") for line in text.splitlines()))
        yaml = (self.output / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("val: images/val", yaml)
        self.assertNotIn("segmentation", yaml)

    def test_val_alias_is_normalized_and_counted_under_validation(self):
        self.plan["episodes"][1]["split"] = "val"
        self.sync_plan()
        result = build(self.plan_path, self.output)
        self.assertEqual(result["splits"]["validation"]["frame_count"], 6)
        self.assertTrue(all(row["split"] == "validation" for row in result["rows"] if row["episode_id"] == "ep2"))
        self.assertTrue((self.output / "labels/val").is_dir())

    def test_same_split_duplicate_rgb_is_retained_and_counted(self):
        def copy_frame(report):
            original, duplicate = report["frames"][:2]
            Path(duplicate["media"]["rgb"]["path"]).write_bytes(Path(original["media"]["rgb"]["path"]).read_bytes())
            duplicate["media"]["rgb"]["sha256"] = original["media"]["rgb"]["sha256"]
        self.change_report(0, copy_frame)
        result = build(self.plan_path, self.output)
        self.assertEqual(result["duplicate_rgb_frame_count"], 1)
        self.assertEqual(result["splits"]["train"]["frame_count"], 6)
        self.assertEqual(result["splits"]["train"]["unique_decoded_rgb_hashes"], 5)
        duplicate = result["rows"][1]
        self.assertTrue(duplicate["duplicate_rgb_within_split"])
        self.assertEqual(duplicate["duplicate_of"], "ep1__single_01")

    def test_identical_rgb_across_splits_is_rejected(self):
        original = load(self.plan["episodes"][0]["report_path"])["frames"][0]["media"]["rgb"]
        def copy_rgb(report):
            duplicate = report["frames"][0]["media"]["rgb"]
            Path(duplicate["path"]).write_bytes(Path(original["path"]).read_bytes())
            duplicate["sha256"] = original["sha256"]
        self.change_report(1, copy_rgb)
        self.rejected("identical RGB crosses|Identical RGB crosses")

    def test_same_decoded_rgb_reencoded_across_splits_is_rejected(self):
        source = load(self.plan["episodes"][0]["report_path"])["frames"][0]["media"]["rgb"]["path"]
        def reencode(report):
            receipt = report["frames"][0]["media"]["rgb"]
            pixels = cv2.imread(source)
            cv2.imwrite(receipt["path"], pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            receipt["sha256"] = sha(receipt["path"])
        self.change_report(1, reencode)
        self.rejected("identical RGB crosses|Identical RGB crosses")

    def test_duplicate_episode_and_same_process_cannot_cross_splits(self):
        self.plan["episodes"][1]["episode_id"] = "ep1"
        self.change_manifest(1, lambda data: data.update(episode_id="ep1"))
        self.sync_plan()
        self.rejected("episode is repeated")
        self.plan["episodes"][1]["episode_id"] = "ep2"
        first = load(self.plan["episodes"][0]["manifest_path"])
        self.change_manifest(1, lambda data: data.update(episode_id="ep2", engine_process_id=first["engine_process_id"],
                                                       engine_started_utc=first["engine_started_utc"]))
        self.sync_plan()
        self.rejected("UE process is shared")

    def test_reused_pid_with_different_start_is_an_independent_process(self):
        first = load(self.plan["episodes"][0]["manifest_path"])
        self.change_manifest(1, lambda data: data.update(engine_process_id=first["engine_process_id"]))
        self.assertEqual(build(self.plan_path, self.output)["independent_UE_sessions"], 3)

    def test_session_shared_by_independent_processes_is_rejected(self):
        self.change_report(1, lambda report: [frame.update(session_id="fixture_session_1") for frame in report["frames"]])
        self.rejected("rendered session is shared")

    def test_missing_or_repeated_case_frames_are_rejected(self):
        self.change_report(0, lambda report: report["frames"].pop())
        self.rejected("Complete registered cases")

    def test_missing_test_split_is_rejected(self):
        self.plan["episodes"].pop()
        self.sync_plan()
        self.rejected("Independent train, validation and test")

    def test_protocol_and_source_tampering_are_rejected_before_export(self):
        protocol = Path(self.plan["episodes"][0]["protocol_path"])
        old_bytes = protocol.read_bytes()
        protocol.write_bytes(old_bytes + b" ")
        self.rejected("Preregistered protocol hash mismatch")
        protocol.write_bytes(old_bytes)
        snapshot = Path(self.plan["episodes"][0]["manifest_path"]).parent / "source_snapshot/camera_geometry.py"
        snapshot.write_bytes(b"changed source")
        self.rejected("Source snapshot hash mismatch")

    def test_media_and_launcher_report_hash_tampering_are_rejected(self):
        source = Path(load(self.plan["episodes"][0]["report_path"])["frames"][0]["media"]["rgb"]["path"])
        original = source.read_bytes()
        source.write_bytes(original + b"changed")
        self.rejected("Raw media hash mismatch")
        source.write_bytes(original)
        self.change_manifest(0, lambda data: data.update(report_sha256="0" * 64))
        self.rejected("Launcher report hash/path mismatch")

    def test_stored_mask_truth_and_palette_tampering_are_rejected(self):
        self.change_report(0, lambda report: report["frames"][0]["evaluation_only"]["target"].update(visible_pixel_count=701))
        self.rejected("Stored instance masks")

    def test_strict_scene_recomputed_inactive_actor_not_hidden_by_pass_flag(self):
        def show_inactive(report):
            frame = report["frames"][0]
            path = frame["media"]["segmentation"]["path"]
            pixels = cv2.imread(path)
            pixels[0, 0] = self.palette["43"][::-1]
            cv2.imwrite(path, pixels)
            frame["media"]["segmentation"]["sha256"] = sha(path)
            frame["evaluation_only"]["distractor"] = visible_mask(pixels, self.palette["43"])
        self.change_report(0, show_inactive)
        self.rejected("Inactive distractor is visible")

    def test_full_occlusion_not_silent_negative_when_target_visible(self):
        def reveal_target(report):
            frame = report["frames"][2]
            path = frame["media"]["segmentation"]["path"]
            pixels = cv2.imread(path)
            pixels[0, 0] = self.palette["42"][::-1]
            cv2.imwrite(path, pixels)
            frame["media"]["segmentation"]["sha256"] = sha(path)
            frame["evaluation_only"]["target"] = visible_mask(pixels, self.palette["42"])
        self.change_report(0, reveal_target)
        self.rejected("Full target occlusion")

    def test_ground_closure_and_launcher_failures_reject_data(self):
        self.change_manifest(0, lambda data: data.update(engine_closed_after_safe_completion=False))
        self.rejected("Launcher PASS")
        self.change_manifest(0, lambda data: data.update(engine_closed_after_safe_completion=True))
        self.change_report(0, lambda report: report["ground_after"]["DroneA"].update(api_control_enabled=True))
        self.rejected("Ground after is not grounded")

    def test_independence_from_explicit_development_source_is_enforced(self):
        report = self.plan["episodes"][0]["report_path"]
        exclusion = self.root / "prior-development.json"
        save(exclusion, {"source_report": report, "rows": []})
        self.plan["excluded_development_manifests"] = [str(exclusion)]
        self.sync_plan()
        self.rejected("already used for development")

    def test_existing_output_preserved_and_failure_receipts_never_overwrite(self):
        self.output.mkdir()
        marker = self.output / "existing.txt"
        marker.write_text("preserve", encoding="utf-8")
        receipts = []
        for _ in range(2):
            with self.assertRaises(DatasetBuildError) as raised:
                build(self.plan_path, self.output)
            receipts.append(raised.exception.receipt_path)
        self.assertNotEqual(receipts[0], receipts[1])
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertTrue(all(path.exists() for path in receipts))
        self.assertFalse(load(receipts[0])["dataset_output_created"])

    def test_relative_paths_resolve_from_plan_directory_and_rows_alias(self):
        self.plan["rows"] = self.plan.pop("episodes")
        for row in self.plan["rows"]:
            for key in ("protocol_path", "manifest_path", "report_path"):
                row[key] = str(Path(row[key]).relative_to(self.root))
        self.sync_plan()
        self.assertEqual(build(self.plan_path, self.output)["frame_count"], 18)

    def test_plan_receipt_binds_collector_to_the_unchanged_plan(self):
        for index in range(3):
            self.change_manifest(index, lambda data: data.update(plan=str(self.plan_path), plan_sha256=sha(self.plan_path)))
        self.plan["extra_after_collection"] = True
        self.sync_plan()
        self.rejected("current frozen plan")

    def test_output_created_by_another_process_after_validation_is_preserved(self):
        from build_t1_balloon_dataset import _validate
        def raced_validation(plan_path, output):
            result = _validate(plan_path, output)
            output.mkdir()
            (output / "manifest.json").write_text("other process evidence", encoding="utf-8")
            return result
        with patch("build_t1_balloon_dataset._validate", side_effect=raced_validation):
            with self.assertRaises(DatasetBuildError) as raised:
                build(self.plan_path, self.output)
        self.assertEqual((self.output / "manifest.json").read_text(encoding="utf-8"), "other process evidence")
        self.assertFalse(load(raised.exception.receipt_path)["dataset_output_created"])


if __name__ == "__main__":
    unittest.main()
