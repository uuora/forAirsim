"""Saved-development-frame timing diagnostic; not new UE/accuracy trials."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from train_t1_balloon_yolo import configure_offline_runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(report_path,output):
    configure_offline_runtime()
    import cv2
    import torch
    from specialist_balloon_detector import SpecialistBalloonDetector
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads(Path(report_path).read_text(encoding="utf-8"))
    frames=report["frames"]
    images=[]
    for frame in (frames[0],frames[-1]):
        media=frame["media"]["rgb"]
        if digest(media["path"])!=media["sha256"]:
            raise ValueError("Development image binding changed")
        images.append(cv2.imread(media["path"]))
    config_path=Path("configs/t1_balloon_detector_v1p2.json")
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if digest(config["weights"])!=config["weights_sha256"]:
        raise ValueError("Checkpoint changed")
    rows=[]
    for device in (["cpu", "0"] if torch.cuda.is_available() else ["cpu"]):
        detector=SpecialistBalloonDetector(config["weights"],config["confidence"],
            config["imgsz"],device,nms_iou=config["nms_iou"])
        for index in range(3):
            detector.detect(images[index%2])
        durations=[]
        for index in range(16):
            start=time.monotonic()
            result=detector.detect(images[index%2])
            durations.append(time.monotonic()-start)
        rows.append({"device":device,"warmup_calls":3,"measured_calls":16,
            "median_wall_ms":statistics.median(durations)*1000,
            "maximum_wall_ms":max(durations)*1000,
            "wall_s":durations,"last_candidate_count":result["candidate_count"]})
        del detector
    receipt={"kind":"DEVELOPMENT_SAVED_FRAME_RUNTIME_DIAGNOSTIC", "new_UE_sessions":0,
        "new_flights":0,"cloud_requests":0,"source_report":str(Path(report_path).resolve()),
        "source_report_sha256":digest(report_path),"weights_sha256":config["weights_sha256"],
        "source_sha256":digest(Path(__file__)),"torch_version":torch.__version__,
        "cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows":rows,"limits":["Repeated two already-collected development frames; not detector accuracy or independent scene trials",
            "Wall timings on current hardware/load; flight deadline must be checked in new owned UE sessions"],
        "ai_assistance":"AI assisted timing diagnostic"}
    (output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cuda_available":receipt["cuda_available"],"medians_ms":{
        row["device"]:round(row["median_wall_ms"],2) for row in rows}}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    run(args.report,args.output)
