"""Meaningful contract tests; no training, test-set inference, or network."""
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from evaluate_t1_balloon_yolo import aggregate_frames, match_boxes
from specialist_balloon_detector import SpecialistBalloonDetector, require_balloon_names
from train_t1_balloon_yolo import (OfflineNetworkError, forbid_network,
    read_numeric_results_csv, sha256, validate_binding, validate_split_files, write_data_yaml)


class FakeModel:
    names = {0: "balloon"}
    def __init__(self, boxes=None, classes=None):
        self.calls = []
        boxes = boxes if boxes is not None else [[10, 20, 30, 40], [40, 10, 60, 30]]
        self.result = SimpleNamespace(names=dict(self.names), speed={"inference": .5},
            boxes=SimpleNamespace(xyxy=np.asarray(boxes).reshape(-1,4),
                cls=np.asarray(classes if classes is not None else [0]*len(boxes)),
                conf=np.asarray([.8]*len(boxes))))
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


class SpecialistContractTests(unittest.TestCase):
    def test_exact_class_head_rejects_coco(self):
        for names in ({0: "sports ball"}, {0:"balloon",1:"vessel"}, {"0":"balloon"}, {1:"balloon"}):
            with self.subTest(names=names), self.assertRaises(ValueError):
                require_balloon_names(names)

    def test_rgb_only_compatible_candidate_contract(self):
        fake = FakeModel()
        detector = SpecialistBalloonDetector(model=fake)
        image = np.zeros((80,100,3),np.uint8)
        result = detector.detect(image)
        self.assertIs(fake.calls[0]["source"], image)
        self.assertEqual(fake.calls[0]["conf"], .25)
        self.assertEqual(fake.calls[0]["iou"], .5)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["candidates"][0]["source_class_name"], "balloon")
        self.assertFalse(result["designated_instance_identity_established"])
        self.assertEqual(result["candidates"][0]["normalised_error_xy"], [-.6,-.25])

    def test_changed_head_rejected_at_runtime(self):
        fake = FakeModel(); detector = SpecialistBalloonDetector(model=fake)
        fake.names = {0:"sports ball"}
        with self.assertRaises(ValueError): detector.detect(np.zeros((80,100,3),np.uint8))

    def test_malformed_output_does_not_become_target(self):
        for box in ([0,0,0,1], [-1,0,5,5], [1,1,101,5], [1,1,float("nan"),5]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                SpecialistBalloonDetector(model=FakeModel([box])).detect(np.zeros((80,100,3),np.uint8))

    def test_nonzero_specialist_class_rejected(self):
        with self.assertRaises(ValueError):
            SpecialistBalloonDetector(model=FakeModel([[1,1,5,5]],[1])).detect(np.zeros((80,100,3),np.uint8))

    def test_empty_detection_retained(self):
        result = SpecialistBalloonDetector(model=FakeModel([])).detect(np.zeros((80,100,3),np.uint8))
        self.assertEqual(result["candidates"], [])

    def test_image_type_validation(self):
        detector = SpecialistBalloonDetector(model=FakeModel())
        for image in (None, np.zeros((5,5),np.uint8), np.zeros((5,5,3),float), np.zeros((0,5,3),np.uint8)):
            with self.subTest(shape=getattr(image,"shape",None)), self.assertRaises(ValueError): detector.detect(image)


class MatchingTests(unittest.TestCase):
    def test_duplicates_are_false_positive(self):
        result = match_boxes([[0,0,10,10],[0,0,10,10]],[[0,0,10,10]])
        self.assertEqual((result["true_positive"],result["false_positive"],result["false_negative"]),(1,1,0))

    def test_augmenting_match_avoids_greedy_miss(self):
        result = match_boxes([[3,0,13,10],[0,0,10,10]],[[0,0,10,10],[6,0,16,10]])
        self.assertEqual(result["true_positive"],2)

    def test_negative_and_no_output_denominators(self):
        negative = match_boxes([[0,0,10,10]], [])
        missing = match_boxes([], [[0,0,10,10],[20,20,30,30]])
        self.assertTrue(negative["negative_frame_false_positive"])
        summary = aggregate_frames([
            {"score":negative,"status":"RETURNED","elapsed_ns":123456},
            {"score":missing,"status":"ERROR","elapsed_ns":123456}])
        self.assertEqual(summary["frames"],2)
        self.assertEqual(summary["error_frames"],1)
        self.assertEqual(summary["two_visible_balloon_frames"],1)
        self.assertEqual(summary["both_balloons_detected_frames"],0)
        self.assertEqual(summary["recall"],0)
        self.assertEqual(summary["latency_ms_median"],.123456)


class DatasetAndOfflineTests(unittest.TestCase):
    def test_stdlib_results_csv_actual_format(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"results.csv"
            path.write_text("epoch,time,train/box_loss,metrics/mAP50-95(B),lr/pg0\n"
                            "1,6.05375,0.62321,0.88224,0.000305556\n"
                            "2,10.234,0.5,0.9,0.0007\n")
            values = read_numeric_results_csv(path)
            self.assertEqual(values["epoch"], [1,2])
            self.assertIs(type(values["epoch"][0]), int)
            self.assertEqual(values["time"], [6.05375,10.234])
            self.assertEqual(values["metrics/mAP50-95(B)"], [.88224,.9])

    def test_stdlib_results_csv_actual_failed_run(self):
        path = Path(__file__).resolve().parents[1]/"logs/t1_balloon_yolo_training_v1p1_20261004/training/fixed_protocol/results.csv"
        if not path.is_file():
            self.skipTest("Actual failed-run CSV not available")
        values = read_numeric_results_csv(path)
        self.assertEqual(values["epoch"], [1])
        self.assertEqual(len(values), 15)
        self.assertEqual(values["metrics/mAP50-95(B)"], [.88224])

    def test_stdlib_csv_failure_contract(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"results.csv"
            self.assertEqual(read_numeric_results_csv(path), {})
            path.write_text("epoch,time\n1\n")
            self.assertEqual(read_numeric_results_csv(path), {})
            path.write_text("epoch,time\n1,\n")
            self.assertEqual(read_numeric_results_csv(path), {"epoch":[1],"time":[None]})

    def test_training_yaml_never_contains_test(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            manifest={"rows":[{"split":split,"image_path":str(root/"images"/split/"one.png"),
                "image_sha256":"a"*64,"label_path":"unused"} for split in ("train","validation","test")]}
            path=write_data_yaml(root,manifest)
            data=json.loads(path.read_text())
            self.assertEqual(set(data),{"path","train","val","names"})
            self.assertFalse((root/"test_images.txt").exists())
            self.assertNotIn("/test/", (root/"train_images.txt").read_text())

    def test_offline_guard_denies_and_restores(self):
        original=socket.socket.connect
        with forbid_network():
            with self.assertRaises(OfflineNetworkError): socket.getaddrinfo("localhost",80)
            with socket.socket() as sock:
                with self.assertRaises(OfflineNetworkError): sock.connect(("127.0.0.1",1))
        self.assertIs(socket.socket.connect,original)

    def test_train_validation_files_do_not_read_test(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); (root/"images").mkdir(); (root/"labels").mkdir()
            image=root/"images"/"good.png"; Image.new("RGB",(100,80)).save(image)
            label=root/"labels"/"good.txt"; label.write_text("0 0.5 0.5 0.2 0.25\n")
            good={"episode_id":"train1","frame_id":"one","split":"train", "image_path":str(image),
                "image_sha256":sha256(image),"label_path":str(label),"label_sha256":sha256(label),
                "visible_balloon_count":1,"visible_boxes_xyxy_pixels":[[40,30,60,50]]}
            unreadable=dict(good,split="test",image_path=str(root/"missing.png"),label_path=str(root/"missing.txt"))
            checked=validate_split_files({"rows":[good,unreadable]},{"train","validation"})
            self.assertEqual(len(checked),1)
            label.write_text("0 .5 .5 .3 .25\n")
            with self.assertRaises(ValueError): validate_split_files({"rows":[good]},{"train"})

    def test_plan_hash_and_cross_split_leak_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); plan=root/"plan.json"; manifest=root/"manifest.json"
            plan.write_text(json.dumps({"episodes":[{"episode_id":key,"split":key} for key in ("train","validation","test")],
                "cases_per_episode":1,"frames_per_case":1}))
            rows=[{"episode_id":key,"split":key,"frame_id":key,"image_sha256":str(n)*64} for n,key in enumerate(("train","validation","test"))]
            value={"status":"PASS","train_val_test_ready":True,"plan_sha256":sha256(plan),"class_names":{"0":"balloon"},"rows":rows}
            manifest.write_text(json.dumps(value)); validate_binding(manifest,plan)
            value["rows"][1]["image_sha256"]=value["rows"][0]["image_sha256"]
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError): validate_binding(manifest,plan)
            value["plan_sha256"]="bad";manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError): validate_binding(manifest,plan)


if __name__ == "__main__":
    unittest.main()
