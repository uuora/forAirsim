"""Review gate checks: simulator boxes alone must not certify visible labels."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_maritime_layout_review import validate


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = self.root / "frame.png"
        image.write_bytes(b"image bytes for integrity check")
        self.row = {
            "session_index": "1", "engine_process_id": "1234", "profile": "similar_red",
            "frame": "frame.png", "image_path": str(image),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "task_description": "the rightmost red balloon in the image",
            "expected_target_side": "right", "projected_target_box_reference": "[10,10,20,20]",
            "projected_distractor_box_reference": "[30,10,40,20]",
            "target_visible_bbox_xyxy": "", "distractor_visible_bbox_xyxy": "",
            "target_visibility": "", "distractor_visibility": "", "reviewer": "",
            "reviewed_at": "", "review_status": "PENDING_HUMAN",
        }
        (self.root / "collection_manifest.json").write_text(json.dumps({
            "frame_count": 1, "review_queue": str(self.root / "review_queue.csv")}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def write_row(self):
        with (self.root / "review_queue.csv").open("w", newline="", encoding="utf-8-sig") as out:
            writer = csv.DictWriter(out, fieldnames=self.row.keys())
            writer.writeheader()
            writer.writerow(self.row)

    def test_pending_rows_are_not_ready(self):
        self.write_row()
        result = validate(self.root / "collection_manifest.json")
        self.assertEqual(result["status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(result["verified"], 0)

    def test_verified_requires_human_metadata_and_visible_boxes(self):
        self.row["review_status"] = "VERIFIED"
        self.row["target_visibility"] = "VISIBLE"
        self.row["distractor_visibility"] = "VISIBLE"
        self.row["target_visible_bbox_xyxy"] = "[10,10,20,20]"
        self.row["distractor_visible_bbox_xyxy"] = "[30,10,40,20]"
        self.write_row()
        with self.assertRaisesRegex(ValueError, "reviewer and time"):
            validate(self.root / "collection_manifest.json")
        self.row["reviewer"] = "human-reviewer"
        self.row["reviewed_at"] = "2026-09-26T12:00:00+08:00"
        self.write_row()
        self.assertEqual(validate(self.root / "collection_manifest.json")["status"],
                         "READY_FOR_EVALUATION")

    def test_ai_review_is_only_exploratory_with_explicit_flag(self):
        self.row.update({
            "review_status": "AI_REVIEWED", "reviewer": "Codex AI visual review",
            "reviewed_at": "2026-09-26T12:00:00+08:00",
            "target_visibility": "VISIBLE", "distractor_visibility": "VISIBLE",
            "target_visible_bbox_xyxy": "[10,10,20,20]",
            "distractor_visible_bbox_xyxy": "[30,10,40,20]",
        })
        self.write_row()
        path = self.root / "collection_manifest.json"
        self.assertEqual(validate(path)["status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(validate(path, allow_ai_reviewed=True)["status"],
                         "READY_FOR_EXPLORATORY_EVALUATION")
        self.row["review_status"] = "VERIFIED"
        self.write_row()
        with self.assertRaisesRegex(ValueError, "cannot certify human"):
            validate(path)


if __name__ == "__main__":
    unittest.main()
