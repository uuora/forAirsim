"""One-frame hosted VLX-Seek smoke test (the public Space uses WeDetect).

This is deliberately separate from the YOLO-proposal adapter: the public
Space API has no bbox_list input and cannot validate YOLO -> VLX integration.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gradio_client import Client


SPACE = "hugging-apps/vlx-seek-1-5-10b-demo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--task", choices=("grounding_single", "vqa"), default="grounding_single")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = args.image.resolve()
    output = args.output.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    record = {
        "status": "SUBMITTED", "model_execution": "hosted community Space",
        "space": SPACE, "proposal_source": "Space internal WeDetect",
        "yolo_proposals_supplied": False, "image": str(image),
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "query": args.query, "task": args.task,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        client = Client(SPACE)
        job = client.submit(str(image), args.query, args.task, "en", 128,
                            0.0, 1.0, api_name="/run")
        regions, answer = job.result(timeout=240)
        record.update({"status": "RETURNED", "completed_utc": datetime.now(timezone.utc).isoformat(),
                       "model_answer": answer, "grounded_regions": regions})
    except Exception as error:
        record.update({"status": "ERROR", "completed_utc": datetime.now(timezone.utc).isoformat(),
                       "error_type": type(error).__name__, "error": str(error)[:1000]})
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": record["status"], "answer": record.get("model_answer"),
                      "error": record.get("error"), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
