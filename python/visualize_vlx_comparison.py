"""Visualize VLX API results with comparison to baselines."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def draw_box(image, box, color, label, thickness=2):
    """Draw bounding box with label."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 10, label_size[1] + 5)
    cv2.rectangle(image, (x1, label_y - label_size[1] - 5),
                  (x1 + label_size[0], label_y), color, -1)
    cv2.putText(image, label, (x1, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def create_comparison_grid(frame_data, image_dir, target_id):
    """Create 2x2 comparison grid for one frame."""
    # Load image
    image_path = image_dir / frame_data['frame']
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = original.shape[:2]

    # Create 4 copies
    img_ground_truth = original.copy()
    img_yolo = original.copy()
    img_rightmost = original.copy()
    img_vlx = original.copy()

    # Get ground truth box
    truth_boxes = frame_data.get('simulator_boxes_xyxy', {})
    target_box = truth_boxes.get(target_id)

    # 1. Ground Truth
    if target_box:
        draw_box(img_ground_truth, target_box, (0, 255, 0), "Ground Truth", 3)

    # 2. YOLO top balloon
    yolo_eval = frame_data.get('yolo_top_balloon_evaluation', {})
    yolo_candidates = frame_data.get('yolo_candidates', [])
    if yolo_candidates:
        yolo_box = yolo_candidates[0]['bbox_xyxy_pixels']
        is_correct = yolo_eval.get('association') == 'TARGET'
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        label = f"YOLO Top ({'✓' if is_correct else '✗'})"
        draw_box(img_yolo, yolo_box, color, label, 3)
        if target_box:
            draw_box(img_yolo, target_box, (128, 128, 128), "GT", 1)

    # 3. Rightmost balloon rule
    rightmost_eval = frame_data.get('rightmost_balloon_evaluation', {})
    yolo_balloons = [c for c in yolo_candidates if c['class_name'] == 'balloon']
    if yolo_balloons:
        rightmost = max(yolo_balloons, key=lambda c: sum(c['bbox_xyxy_pixels'][::2]))
        rightmost_box = rightmost['bbox_xyxy_pixels']
        is_correct = rightmost_eval.get('association') == 'TARGET'
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        label = f"Rightmost ({'✓' if is_correct else '✗'})"
        draw_box(img_rightmost, rightmost_box, color, label, 3)
        if target_box:
            draw_box(img_rightmost, target_box, (128, 128, 128), "GT", 1)

    # 4. VLX-Seek
    vlx_eval = frame_data.get('vlx_selection_evaluation', {})
    vlx_result = frame_data.get('vlx_result', {})
    if vlx_result and vlx_result.get('result_bbox_list'):
        vlx_boxes = vlx_result['result_bbox_list']
        if vlx_boxes:
            vlx_box = [vlx_boxes[0]['xmin'], vlx_boxes[0]['ymin'],
                      vlx_boxes[0]['xmax'], vlx_boxes[0]['ymax']]
            is_correct = vlx_eval.get('association') == 'TARGET'
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            label = f"VLX-Seek ({'✓' if is_correct else '✗'})"
            draw_box(img_vlx, vlx_box, color, label, 3)
            if target_box:
                draw_box(img_vlx, target_box, (128, 128, 128), "GT", 1)

    # Add titles
    def add_title(img, text, color=(255, 255, 255)):
        cv2.rectangle(img, (0, 0), (w, 40), (50, 50, 50), -1)
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    add_title(img_ground_truth, "Ground Truth")
    add_title(img_yolo, f"YOLO Top Confidence")
    add_title(img_rightmost, f"Rightmost Balloon Rule")
    add_title(img_vlx, f"VLX-Seek API")

    # Create 2x2 grid
    top_row = np.hstack([img_ground_truth, img_yolo])
    bottom_row = np.hstack([img_rightmost, img_vlx])
    grid = np.vstack([top_row, bottom_row])

    # Add frame info at bottom
    info_height = 60
    info_bar = np.zeros((info_height, grid.shape[1], 3), dtype=np.uint8)
    info_bar[:] = (50, 50, 50)

    profile = frame_data.get('profile', 'unknown')
    frame_name = frame_data.get('frame', 'unknown')

    cv2.putText(info_bar, f"Condition: {profile}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(info_bar, f"Frame: {frame_name}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Add method results
    yolo_ok = "✓" if yolo_eval.get('association') == 'TARGET' else "✗"
    right_ok = "✓" if rightmost_eval.get('association') == 'TARGET' else "✗"
    vlx_ok = "✓" if vlx_eval.get('association') == 'TARGET' else "✗"

    results_text = f"Results: YOLO {yolo_ok} | Rightmost {right_ok} | VLX {vlx_ok}"
    cv2.putText(info_bar, results_text, (grid.shape[1] - 400, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    grid = np.vstack([grid, info_bar])

    return grid

def main():
    """Generate comparison visualizations for all frames."""
    result_path = Path("logs/vlx_api_full_test.json")
    if not result_path.exists():
        print(f"Error: {result_path} not found")
        print("Please run vlx_seek_probe.py with --run-vlx --use-api first")
        return 1

    data = json.loads(result_path.read_text())
    image_dir = Path(data['source_report']).parent
    target_id = data['target_id_evaluation_only']

    output_dir = Path("logs/vlx_api_visualizations")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("生成 VLX-Seek API 对比可视化")
    print("=" * 70)
    print()

    profiles = {}
    for frame in data['frames']:
        profile = frame['profile']
        if profile not in profiles:
            profiles[profile] = []
        profiles[profile].append(frame)

    # Generate one image per condition (first frame)
    for profile, frames in sorted(profiles.items()):
        print(f"处理条件: {profile} ({len(frames)} 帧)")

        # Create grid for first frame
        frame_data = frames[0]
        try:
            grid = create_comparison_grid(frame_data, image_dir, target_id)
            output_path = output_dir / f"{profile}_comparison.png"
            cv2.imwrite(str(output_path), grid)
            print(f"  保存: {output_path}")
        except Exception as e:
            print(f"  错误: {e}")

    print()
    print("=" * 70)
    print(f"✅ 完成！可视化结果保存在: {output_dir}")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
