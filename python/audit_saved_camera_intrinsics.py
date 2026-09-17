"""Write additive corrections for historical CameraInfo-FOV-derived intrinsics."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from camera_geometry import scene_intrinsics

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "logs" / datetime.now().strftime("camera_intrinsics_audit_%Y%m%d_%H%M%S")
    output.mkdir()
    corrections, issues, counts = [], [], {"sessions": 0, "frames": 0, "already_correct": 0}
    for path in sorted((ROOT / "datasets/dual_camera").glob("*/frames.jsonl")):
        source = path.read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()
        counts["sessions"] += 1
        for number, line in enumerate(source.decode("utf-8").splitlines(), 1):
            counts["frames"] += 1
            try:
                row = json.loads(line)
                info = SimpleNamespace(fov=row.get("camera_info_fov_deg", row["horizontal_fov_deg"]),
                    proj_mat=SimpleNamespace(matrix=row["projection_matrix_from_api"]))
                corrected = scene_intrinsics(info, row["width"], row["height"])
                if np.allclose(row["K_nominal"], corrected["K_nominal"], rtol=1e-5):
                    counts["already_correct"] += 1
                    continue
                corrections.append({"source_frames": str(path.relative_to(ROOT)),
                    "source_frames_sha256": source_sha, "source_line_1_based": number,
                    "image": row["image"], "image_sha256_recorded": row["sha256"],
                    "old_K_nominal": row["K_nominal"], "corrected": corrected,
                    "reason": "Scene projection differs from viewer CameraInfo.fov; original file preserved"})
            except Exception as exc:
                issues.append({"source_frames": str(path.relative_to(ROOT)), "line": number, "error": str(exc)})
    with (output / "corrections.jsonl").open("w", encoding="utf-8") as stream:
        for row in corrections:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {**counts, "corrections": len(corrections), "issues": issues,
               "status": "COMPLETE" if not issues else "PARTIAL", "original_files_modified": False,
               "scope": "metadata reconstruction from recorded Scene matrices, not a new calibration"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
