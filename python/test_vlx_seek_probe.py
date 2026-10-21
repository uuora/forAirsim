"""Contract checks for the saved-frame VLX handoff (without model weights)."""
import argparse
import json
import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vlx_api_client import VLXSeekAPIClient
from vlx_seek_probe import (DESCRIPTION, build_proposals, evaluate_proposals, run,
                            validated_vlx_box)


class FakeDetector:
    def detect(self, image):
        return {"candidate_count": 1, "candidates": [{
            "class_name": "balloon", "confidence": 0.8,
            "bbox_xyxy_pixels": [10.0, 10.0, 25.0, 25.0]}]}


class FakeWorker:
    def __init__(self, expected_description=DESCRIPTION):
        self.calls = []
        self.expected_description = expected_description

    def ground(self, image, boxes, description, **kwargs):
        self.calls.append((image, boxes, description, kwargs))
        assert description == self.expected_description
        assert len(boxes) == 2
        assert image.mode == "RGB"
        assert image.getpixel((60, 20)) == (255, 0, 0)
        x1, y1, x2, y2 = boxes[1]
        return {"answer": "<ground>red balloon</ground><objects><obj1></objects>",
                "result_bbox_list": [{"object_index": "<obj1>", "xmin": x1,
                                      "ymin": y1, "xmax": x2, "ymax": y2}]}


class VLXProbeTests(unittest.TestCase):
    def test_cloud_request_marks_proposals_as_not_transmitted(self):
        client = VLXSeekAPIClient(api_key="test-key", api_endpoint="https://example.invalid")
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "none"},
                                                    "result_bbox_list": []}]}
        client.session.post = Mock(return_value=response)
        image = Image.new("RGB", (40, 40), "red")
        client.ground(image, [[1, 2, 20, 25]], "red balloon")
        sent = client.session.post.call_args.kwargs["json"]
        self.assertEqual(client.request_mode, "image_text")
        self.assertFalse(client.proposals_transmitted)
        self.assertNotIn("bbox_list", sent)

    def test_region_proposal_recovers_partial_red_shape(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        cv2.rectangle(image, (51, 11), (68, 30), (0, 0, 255), -1)
        yolo = FakeDetector().detect(image)
        proposals = build_proposals(yolo, image, "yolo_red_regions")
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[1]["source"], "yolo_red_regions")
        self.assertTrue(evaluate_proposals(proposals, {"balloon_001": [50, 10, 69, 31]},
                                           "balloon_001")["target_proposal_recalled_iou_0_5"])

    def test_reject_box_that_was_not_proposed(self):
        with self.assertRaises(ValueError):
            validated_vlx_box({"result_bbox_list": [{"xmin": 0, "ymin": 0,
                "xmax": 3, "ymax": 3}]}, [{"bbox_xyxy": [10, 10, 20, 20]}])

    def test_saved_frame_pipeline_passes_only_image_boxes_and_description(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((60, 80, 3), dtype=np.uint8)
            cv2.rectangle(image, (51, 11), (68, 30), (0, 0, 255), -1)
            cv2.imwrite(str(root / "frame.png"), image)
            report = {"status": "PASS", "config": {"scene": "MaritimeHarborV2",
                      "target_id": "balloon_001"}, "conditions": [{"profile": "probe",
                      "frames": [{"frame": "frame.png", "simulator_boxes_xyxy": {
                          "balloon_001": [50, 10, 69, 31],
                          "balloon_002": [10, 10, 25, 25]}}]}]}
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "weights.pt").write_bytes(b"fake-weight-hash-only")
            worker = FakeWorker()
            args = argparse.Namespace(report=root / "report.json", output=root / "out.json",
                weights=root / "weights.pt", confidence=0.1, imgsz=1280, device="0",
                proposal_mode="yolo_red_regions", run_vlx=True, vlx_repo=None,
                model_path=None, vlx_device="cuda", description=DESCRIPTION,
                proposals_report=None)
            completed = run(args, detector=FakeDetector(), worker=worker)
            self.assertEqual(completed["summary_by_profile"]["probe"]["vlx_target"], 1)
            self.assertEqual(completed["vlx_request_mode"], "region_proposals")
            self.assertTrue(completed["vlx_proposals_transmitted"])
            self.assertEqual(len(worker.calls), 1)
            self.assertEqual(completed["frames"][0]["vlx_selection_evaluation"][
                "association"], "TARGET")

    def test_cached_proposals_run_without_yolo_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((60, 80, 3), dtype=np.uint8)
            cv2.rectangle(image, (51, 11), (68, 30), (0, 0, 255), -1)
            cv2.imwrite(str(root / "frame.png"), image)
            (root / "report.json").write_text(json.dumps({"status": "PASS",
                "config": {"scene": "MaritimeHarborV2", "target_id": "balloon_001"},
                "conditions": [{"profile": "probe", "frames": [{"frame": "frame.png",
                    "simulator_boxes_xyxy": {"balloon_001": [50, 10, 69, 31]}}]}]}),
                encoding="utf-8")
            weight = root / "weights.pt"
            weight.write_bytes(b"fake-weight-hash-only")
            args = argparse.Namespace(report=root / "report.json", output=root / "proposals.json",
                weights=weight, confidence=0.1, imgsz=1280, device="0",
                proposal_mode="yolo_red_regions", run_vlx=False, vlx_repo=None,
                model_path=None, vlx_device="cuda", description=DESCRIPTION,
                proposals_report=None)
            run(args, detector=FakeDetector())
            weight.unlink()
            args.proposals_report = args.output
            args.output = root / "vlx.json"
            args.run_vlx = True
            completed = run(args, worker=FakeWorker())
            self.assertEqual(completed["summary_by_profile"]["probe"]["vlx_target"], 1)
            self.assertEqual(completed["proposals_report"], str((root / "proposals.json").resolve()))

    def test_preregistered_frame_description_reaches_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((60, 80, 3), dtype=np.uint8)
            cv2.rectangle(image, (51, 11), (68, 30), (0, 0, 255), -1)
            cv2.imwrite(str(root / "frame.png"), image)
            description = "the leftmost red balloon in the image"
            (root / "report.json").write_text(json.dumps({"status": "PASS",
                "config": {"scene": "MaritimeHarborV2", "target_id": "balloon_001"},
                "conditions": [{"profile": "left", "frames": [{"frame": "frame.png",
                    "task_description": description,
                    "simulator_boxes_xyxy": {"balloon_001": [50, 10, 69, 31]}}]}]}),
                encoding="utf-8")
            weight = root / "weights.pt"
            weight.write_bytes(b"fake-weight-hash-only")
            args = argparse.Namespace(report=root / "report.json", output=root / "vlx.json",
                weights=weight, confidence=0.1, imgsz=1280, device="0",
                proposal_mode="yolo_red_regions", run_vlx=True, vlx_repo=None,
                model_path=None, vlx_device="cuda", description=None,
                proposals_report=None)
            completed = run(args, detector=FakeDetector(), worker=FakeWorker(description))
            self.assertEqual(completed["frames"][0]["task_description"], description)


if __name__ == "__main__":
    unittest.main()
