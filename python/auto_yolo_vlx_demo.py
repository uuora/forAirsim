"""Automatic YOLO + VLX pipeline demo.

Workflow:
1. Load image
2. YOLO automatically detects all objects (generates candidate boxes)
3. User provides natural language description
4. VLX-Seek selects the target from YOLO's results
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Load environment
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "python"))

from vlx_api_client import VLXSeekAPIClient
from yolo_detector import YoloTargetDetector


class AutoPipeline:
    """Automatic YOLO + VLX pipeline with OpenCV UI."""

    def __init__(self, yolo_weights, vlx_model="vlx-seek-1.5-3b"):
        print("Initializing pipeline...")
        self.yolo = YoloTargetDetector(yolo_weights, confidence=0.1, imgsz=1280)
        self.vlx = VLXSeekAPIClient(model=vlx_model)
        print("✓ YOLO loaded")
        print("✓ VLX API client ready")

        self.image = None
        self.display_image = None
        self.yolo_boxes = []
        self.vlx_result = None
        self.window_name = "Auto Pipeline: YOLO + VLX"

    def load_image(self, image_path):
        """Load image and run YOLO detection."""
        print(f"\nLoading image: {image_path}")
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(f"Cannot load: {image_path}")

        # Resize if too large
        h, w = self.image.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            self.image = cv2.resize(self.image, (new_w, new_h))
            print(f"Resized to {new_w}x{new_h}")

        # Run YOLO detection
        print("Running YOLO detection...")
        yolo_result = self.yolo.detect(self.image)

        # Extract boxes
        self.yolo_boxes = []
        for candidate in yolo_result["candidates"]:
            self.yolo_boxes.append({
                "bbox": candidate["bbox_xyxy_pixels"],
                "class": candidate["class_name"],
                "confidence": candidate["confidence"]
            })

        print(f"✓ YOLO detected {len(self.yolo_boxes)} objects")
        for i, box in enumerate(self.yolo_boxes):
            print(f"  [{i+1}] {box['class']} (conf: {box['confidence']:.2f})")

        self.vlx_result = None
        self.update_display()

    def run_vlx(self, description):
        """Run VLX-Seek on YOLO results."""
        if not self.yolo_boxes:
            print("Error: No YOLO detections! Load an image first.")
            return

        print(f"\nRunning VLX-Seek...")
        print(f"  Description: {description}")
        print(f"  Candidate boxes: {len(self.yolo_boxes)}")

        try:
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            # Prepare boxes for VLX
            boxes = [box["bbox"] for box in self.yolo_boxes]

            # Call VLX API
            result = self.vlx.ground(
                image=pil_image,
                boxes=boxes,
                description=description,
                lang="en",
                temperature=0.0,
                max_new_tokens=128
            )

            self.vlx_result = result
            bbox_list = result.get('result_bbox_list', [])

            print(f"  Answer: {result.get('answer', '')}")
            print(f"  Selected: {len(bbox_list)} object(s)")

            if bbox_list:
                selected = bbox_list[0]
                print(f"  → Label: {selected.get('label', '')}")
                print(f"  → Box: [{selected['xmin']:.0f}, {selected['ymin']:.0f}, "
                      f"{selected['xmax']:.0f}, {selected['ymax']:.0f}]")

            self.update_display()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def update_display(self):
        """Update display with YOLO boxes and VLX selection."""
        if self.image is None:
            return

        self.display_image = self.image.copy()
        h, w = self.display_image.shape[:2]

        # Draw YOLO boxes (green, thin)
        for i, box in enumerate(self.yolo_boxes):
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            label = f"{i+1}:{box['class']}"
            cv2.putText(self.display_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw VLX selection (red, thick)
        if self.vlx_result:
            bbox_list = self.vlx_result.get('result_bbox_list', [])
            if bbox_list:
                selected = bbox_list[0]
                x1 = int(selected['xmin'])
                y1 = int(selected['ymin'])
                x2 = int(selected['xmax'])
                y2 = int(selected['ymax'])

                # Thick red border
                cv2.rectangle(self.display_image, (x1-5, y1-5), (x2+5, y2+5),
                             (0, 0, 255), 8)

                # Large "VLX SELECTED" label
                label = "VLX SELECTED"
                font_scale = 1.0
                thickness = 3
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                label_y = max(y1 - 15, label_h + 10)
                cv2.rectangle(self.display_image,
                            (x1-5, label_y - label_h - 10),
                            (x1 + label_w + 10, label_y + 5),
                            (0, 0, 255), -1)
                cv2.putText(self.display_image, label, (x1, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Add info bar at top
        info_height = 80
        info_bar = np.zeros((info_height, w, 3), dtype=np.uint8)
        info_bar[:] = (50, 50, 50)

        text_y = 25
        cv2.putText(info_bar, f"YOLO Detections: {len(self.yolo_boxes)} objects",
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.vlx_result:
            bbox_list = self.vlx_result.get('result_bbox_list', [])
            if bbox_list:
                label = bbox_list[0].get('label', 'target')
                cv2.putText(info_bar, f"VLX Selected: {label}",
                           (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(info_bar, "VLX: Not run yet - press R to select target",
                       (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Add instruction bar at bottom
        inst_height = 30
        inst_bar = np.zeros((inst_height, w, 3), dtype=np.uint8)
        inst_bar[:] = (50, 50, 50)
        cv2.putText(inst_bar, "[R] Run VLX | [N] New image | [Q] Quit",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Combine
        self.display_image = np.vstack([info_bar, self.display_image, inst_bar])

        cv2.imshow(self.window_name, self.display_image)

    def run(self, initial_image=None):
        """Run the interactive pipeline."""
        cv2.namedWindow(self.window_name)

        if initial_image:
            self.load_image(initial_image)
        else:
            # Placeholder
            self.image = np.ones((600, 800, 3), dtype=np.uint8) * 200
            cv2.putText(self.image, "Press 'N' to load an image", (200, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            self.update_display()

        print("\n" + "=" * 70)
        print("Automatic YOLO + VLX Pipeline")
        print("=" * 70)
        print("Workflow:")
        print("  1. Load image → YOLO auto-detects objects (green boxes)")
        print("  2. Press R and enter description → VLX selects target (red box)")
        print()
        print("Controls:")
        print("  R: Run VLX-Seek (input description)")
        print("  N: Load new image (YOLO will auto-run)")
        print("  Q: Quit")
        print("=" * 70)
        print()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break

            elif key == ord('r'):  # Run VLX
                if not self.yolo_boxes:
                    print("\nError: No YOLO detections. Load an image first (press N).")
                    continue

                print("\n" + "-" * 70)
                description = input("Enter description (e.g., 'the rightmost red balloon'): ")
                if description.strip():
                    self.run_vlx(description)
                else:
                    print("Empty description, skipped.")
                print("-" * 70)

            elif key == ord('n'):  # New image
                print("\n" + "-" * 70)
                path = input("Enter image path: ").strip().strip('"')
                try:
                    self.load_image(path)
                except Exception as e:
                    print(f"Error: {e}")
                print("-" * 70)

        cv2.destroyAllWindows()
        print("\nPipeline closed.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Automatic YOLO + VLX Pipeline")
    parser.add_argument("--image", type=Path, help="Initial image")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo11n.pt",
                       help="YOLO weights")
    parser.add_argument("--vlx-model", default="vlx-seek-1.5-3b",
                       choices=["vlx-seek", "vlx-seek-1.5-3b"],
                       help="VLX model")
    args = parser.parse_args()

    try:
        pipeline = AutoPipeline(yolo_weights=args.weights, vlx_model=args.vlx_model)
        pipeline.run(initial_image=args.image)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
