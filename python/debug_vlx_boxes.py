"""Debug script to check VLX API box matching."""
import json
from pathlib import Path

# Read the partial result
result_path = Path("logs/vlx_api_full_test.json")
if result_path.exists():
    data = json.loads(result_path.read_text())

    print("Checking frames with VLX results...")
    for i, frame in enumerate(data.get("frames", [])):
        if "vlx_result" in frame and frame["vlx_result"]:
            vlx_boxes = frame["vlx_result"].get("result_bbox_list", [])
            proposals = frame.get("proposals", [])

            if vlx_boxes:
                vlx_box = [vlx_boxes[0]["xmin"], vlx_boxes[0]["ymin"],
                          vlx_boxes[0]["xmax"], vlx_boxes[0]["ymax"]]

                print(f"\nFrame {i} ({frame['frame']}):")
                print(f"  VLX returned: {vlx_box}")
                print(f"  Proposal count: {len(proposals)}")

                # Find closest match
                min_diff = float('inf')
                closest = None
                for p in proposals:
                    diff = sum(abs(a - b) for a, b in zip(vlx_box, p["bbox_xyxy"]))
                    if diff < min_diff:
                        min_diff = diff
                        closest = p["bbox_xyxy"]

                if closest:
                    print(f"  Closest proposal: {closest}")
                    print(f"  Difference (pixels): {min_diff:.1f}")

                if min_diff > 5:
                    print(f"  ⚠️ LARGE DIFFERENCE - likely VLX generated new box")
else:
    print("No partial result file found")
