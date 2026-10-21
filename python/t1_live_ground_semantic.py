"""Actual asynchronous VLX with newer grounded camera frames; no flight API.

Simulator calls stay on the main thread. Only immutable RGB and description
reach the network worker. Separately recorded instance masks score outputs.
"""
from dataclasses import asdict
import os
from pathlib import Path
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from setup_scene import ROOT
from t1_semantic_perception import BoundedSemanticPolicy, PolicyConfig, YoloVlxBackend
from t1_ground_perception_probe import capture, digest, save, visible_mask
from evaluate_t1_ground_perception import identity_score


class GroundSemanticObserver:
    def __init__(self):
        task_config_dir = ROOT/"logs/t1_model_runtime/ultralytics"
        task_config_dir.mkdir(parents=True,exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(task_config_dir)
        os.environ["TORCH_HOME"] = str(ROOT/"logs/t1_model_runtime/torch")
        load_dotenv(ROOT/".env")  # Never serialize credentials or endpoint.
        self.backend = None
        self.policy = None
        self.sequence = 0
        self.generation = 0
        self.warmup_receipt = None
        self.config = PolicyConfig(current_frame_max_age_s=.5,semantic_max_age_s=2.,
                                   response_timeout_s=2.,min_request_interval_s=2.)
        self.source_hashes = {name:digest(ROOT/"python"/name) for name in (
            "t1_live_ground_semantic.py","t1_semantic_perception.py","yolo_detector.py","vlx_api_client.py")}

    def _new_frame(self, client, case_id, session_id, output, records):
        image, depth, segmentation, metadata = capture(client)
        self.sequence += 1
        frame_id = f"live_{case_id}_{self.sequence:04d}"
        paths={"rgb":output/(frame_id+"_rgb.png"),"depth":output/(frame_id+"_depth.npy"),
               "segmentation":output/(frame_id+"_seg.png")}
        for name,array in (("rgb",image),("segmentation",segmentation)):
            if not cv2.imwrite(str(paths[name]),array):raise ValueError("Live media persistence failed")
        np.save(paths["depth"],depth,allow_pickle=False)
        # Saved masks remain outside detect_frame, policy and cloud transport.
        evaluation={"target":visible_mask(segmentation,[92,31,106]),
                    "distractor":visible_mask(segmentation,[207,229,90])}
        frame = self.backend.detect_frame(paths["rgb"],session_id=session_id,target_generation=self.generation,
            frame_id=frame_id,frame_sequence=self.sequence,
            readback_receipt_monotonic_s=metadata["capture_received_host_s"],
            expected_image_sha256=digest(paths["rgb"]))
        record={"frame":frame.to_dict(),"capture":metadata,
                "media":{key:{"path":str(path),"sha256":digest(path)} for key,path in paths.items()},
                "evaluation_only":evaluation}
        records.append(record)
        return frame, record

    def observe(self, client, case_id, description, warmup_image, session_id, output):
        if self.backend is None:
            self.backend = YoloVlxBackend(weights=ROOT/"models/yolo11n.pt",device="cpu",imgsz=640,vlx_timeout_s=3.)
            self.warmup_receipt = self.backend.warmup(warmup_image)
        self.generation += 1
        if self.policy is None:
            self.policy = BoundedSemanticPolicy(self.backend,session_id=session_id,target_generation=self.generation,
                                                description=description,config=self.config)
        else:
            self.policy.reset(session_id=session_id,target_generation=self.generation,description=description)
        receipt={"status":"RUNNING","kind":"LIVE_GROUND_RGB_SEMANTIC_INTERFACE","case_id":case_id,"session_id":session_id,
            "target_generation":self.generation,"task_description":description,"policy_config":asdict(self.config),
            "source_sha256":self.source_hashes,"model_weights_sha256":digest(ROOT/"models/yolo11n.pt"),
            "warmup":self.warmup_receipt,"frames":[],"decisions":[],"flight_commands_issued":False,
            "formal_method_effect":False,"truth_used_by_policy":False,
            "request_mode":"image_text","proposals_transmitted":False}
        path=output/("semantic_"+case_id+".json")
        if path.exists():raise FileExistsError("Preserve existing semantic case receipt")
        save(path,receipt)
        frame, row = self._new_frame(client,case_id,session_id,output,receipt["frames"])
        decision = self.policy.request(frame)
        receipt["decisions"].append(decision)
        save(path,receipt)
        started=time.monotonic()
        # Retry only low-frequency gating before a request has actually started.
        while decision["status"]=="NOT_SUBMITTED" and time.monotonic()-started<2.5:
            time.sleep(.1)
            frame,row=self._new_frame(client,case_id,session_id,output,receipt["frames"])
            decision=self.policy.request(frame)
            receipt["decisions"].append(decision)
            save(path,receipt)
        if decision["status"]=="SUBMITTED":
            started=time.monotonic()
            while time.monotonic()-started<3.5:
                time.sleep(.1)
                frame,row=self._new_frame(client,case_id,session_id,output,receipt["frames"])
                decision=self.policy.poll(frame)
                receipt["decisions"].append(decision)
                save(path,receipt)
                if decision["status"]=="SEMANTIC_LOCAL_ASSOCIATION_VALID":break
                if decision["status"]=="UNCERTAIN" and not decision["transport_slot_occupied"]:break
            if decision["transport_slot_occupied"]:
                # No further case/request can start while a timed-out HTTP call remains live.
                raise ValueError("Semantic transport did not drain within its bounded HTTP timeout")
        box=decision.get("selected_candidate",{}).get("bbox_xyxy_pixels")
        receipt["final_decision"]=decision
        receipt["evaluation"]=identity_score(box,row["evaluation_only"])
        receipt["additional_live_capture_count"]=len(receipt["frames"])
        receipt["limits"]=["Grounded virtual camera; no flight or planning effect",
            "IoU association is not persistent instance tracking through occlusion",
            "Scene/generation change is a development context reset, not multi-target task allocation"]
        receipt["status"]="COMPLETE"
        save(path,receipt)
        print(f"Live semantic case {case_id}: {decision['status']} / {decision['reason']}",flush=True)
        return {"case_id":case_id,"path":str(path),"sha256":digest(path),
                "final_status":decision["status"],"reason":decision["reason"],
                "correct_visible_instance":receipt["evaluation"]["all_attempts_correct"],
                "additional_live_frames":len(receipt["frames"])}

    def close(self):
        if self.policy is not None:self.policy.close()
