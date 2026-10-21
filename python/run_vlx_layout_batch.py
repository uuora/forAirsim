"""Run actual VLX-Seek on all three cached maritime layout sessions.

Use in an environment with the official source, checkpoint, and enough memory.
The worker is loaded once; each session keeps a separate result report.
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path, PureWindowsPath

from vlx_seek_probe import load_worker, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True,
                        help="Folder containing collection_manifest.json and cached proposals")
    parser.add_argument("--logs-dir", type=Path, required=True,
                        help="Folder containing the three copied maritime_single_* captures")
    parser.add_argument("--vlx-repo", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlx-device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collection = args.collection.resolve()
    logs_dir = args.logs_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads((collection / "collection_manifest.json").read_text(encoding="utf-8"))
    if manifest["session_count"] != 3:
        raise ValueError("Expected the three-session layout collection")
    jobs = []
    for session in manifest["sessions"]:
        index = session["session_index"]
        capture_dir = PureWindowsPath(session["report_path"]).parent.name
        report = logs_dir / capture_dir / "report.json"
        proposals = collection / f"session_{index}_yolo_proposals.json"
        output = output_dir / f"session_{index}_vlx_actual.json"
        if not report.is_file() or not proposals.is_file():
            raise FileNotFoundError(f"Missing capture/proposal pair: {report}, {proposals}")
        if output.exists():
            raise FileExistsError(f"Preserving existing VLX result: {output}")
        jobs.append((report, proposals, output))
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = load_worker(args.vlx_repo, args.model_path, args.vlx_device)
    for report, proposals, output in jobs:
        options = Namespace(
            report=report, output=output, proposals_report=proposals,
            weights=Path("unused-yolo-weights"), confidence=0.1, imgsz=1280,
            device="0", proposal_mode="yolo_red_regions", description=None,
            run_vlx=True, vlx_repo=args.vlx_repo, model_path=args.model_path,
            vlx_device=args.vlx_device,
        )
        completed = run(options, worker=worker)
        print(json.dumps({"output": str(output), "frames": len(completed["frames"]),
                          "status": completed["status"], "vlx_executed": completed["vlx_executed"]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
