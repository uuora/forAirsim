"""Interactive VLX-Seek demo with natural language grounding.

Similar to the paper demo: upload image, draw/select boxes, input description,
and VLX-Seek identifies the target.
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


class VLXDemoUI:
    """Interactive demo for VLX-Seek with OpenCV UI."""

    def __init__(self, model="vlx-seek-1.5-3b"):
        self.client = VLXSeekAPIClient(model=model)
        self.image = None
        self.display_image = None
        self.boxes = []
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.window_name = "VLX-Seek Interactive Demo"
        self.result_text = ""

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for box drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point[0], self.start_point[1], x, y)
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    # Normalize coordinates
                    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    self.boxes.append(box)
                    print(f"Added box {len(self.boxes)}: {box}")
                self.current_box = None
                self.update_display()

    def update_display(self):
        """Update display with current boxes and results."""
        if self.image is None:
            return

        self.display_image = self.image.copy()

        # Draw existing boxes
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.display_image, f"Box {i+1}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw current box being drawn
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw result text at top
        if self.result_text:
            # Create semi-transparent background for text
            overlay = self.display_image.copy()
            text_lines = self.result_text.split('\n')
            text_height = 30 * len(text_lines) + 20
            cv2.rectangle(overlay, (0, 0), (self.display_image.shape[1], text_height),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, self.display_image, 0.3, 0, self.display_image)

            y_offset = 25
            for line in text_lines:
                cv2.putText(self.display_image, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30

        # Add instructions at bottom
        h, w = self.display_image.shape[:2]
        instructions = [
            "Instructions: [L-Click & Drag] Draw box | [C] Clear boxes | [R] Run VLX | [N] New image | [Q] Quit"
        ]
        y_start = h - 30
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (0, y_start - 5), (w, h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.8, self.display_image, 0.2, 0, self.display_image)

        cv2.putText(self.display_image, instructions[0], (10, y_start + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow(self.window_name, self.display_image)

    def load_image(self, image_path):
        """Load image from file."""
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        # Resize if too large
        h, w = self.image.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            self.image = cv2.resize(self.image, (new_w, new_h))
            print(f"Resized image to {new_w}x{new_h}")

        self.boxes = []
        self.result_text = ""
        self.update_display()

    def run_vlx(self, description):
        """Run VLX-Seek with current boxes and description."""
        if not self.boxes:
            self.result_text = "Error: No boxes drawn! Draw at least one box first."
            self.update_display()
            return

        if not description.strip():
            self.result_text = "Error: Empty description! Please provide a description."
            self.update_display()
            return

        print(f"\nRunning VLX-Seek...")
        print(f"  Description: {description}")
        print(f"  Boxes: {len(self.boxes)}")

        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            # Call VLX API
            result = self.client.ground(
                image=pil_image,
                boxes=self.boxes,
                description=description,
                lang="en",
                temperature=0.0,
                max_new_tokens=128
            )

            # Process result
            answer = result.get('answer', '')
            bbox_list = result.get('result_bbox_list', [])

            print(f"  Answer: {answer}")
            print(f"  Detected: {len(bbox_list)} objects")

            # Redraw display first to clear any previous highlights
            self.update_display()

            if bbox_list:
                # Highlight the selected box
                selected_box = bbox_list[0]
                x1, y1 = int(selected_box['xmin']), int(selected_box['ymin'])
                x2, y2 = int(selected_box['xmax']), int(selected_box['ymax'])

                # Draw thick red highlight on top of everything
                cv2.rectangle(self.display_image, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 8)

                # Add large "SELECTED" label
                label = "SELECTED"
                font_scale = 1.0
                thickness = 3
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                font_scale, thickness)

                # Draw label background
                label_y = max(y1 - 15, label_h + 10)
                cv2.rectangle(self.display_image,
                            (x1-5, label_y - label_h - 10),
                            (x1 + label_w + 10, label_y + 5),
                            (0, 0, 255), -1)

                # Draw label text
                cv2.putText(self.display_image, label, (x1, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                self.result_text = f"VLX Selected: {selected_box.get('label', 'target')}\nBox location: [{x1}, {y1}, {x2}, {y2}]"
            else:
                self.result_text = "VLX: No selection made"

            # Force window update
            cv2.imshow(self.window_name, self.display_image)
            cv2.waitKey(1)

        except Exception as e:
            self.result_text = f"Error: {str(e)}"
            self.update_display()
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def run(self, initial_image=None):
        """Run the interactive demo."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        if initial_image:
            self.load_image(initial_image)
        else:
            # Create placeholder
            self.image = np.ones((600, 800, 3), dtype=np.uint8) * 200
            cv2.putText(self.image, "Press 'N' to load an image", (200, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            self.update_display()

        print("\n" + "=" * 70)
        print("VLX-Seek Interactive Demo")
        print("=" * 70)
        print("Controls:")
        print("  - Left-click and drag: Draw bounding box")
        print("  - C: Clear all boxes")
        print("  - R: Run VLX-Seek (will prompt for description)")
        print("  - N: Load new image (will prompt for path)")
        print("  - Q: Quit")
        print("=" * 70)
        print()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break

            elif key == ord('c'):  # Clear boxes
                self.boxes = []
                self.result_text = "Boxes cleared"
                self.update_display()
                print("Cleared all boxes")

            elif key == ord('r'):  # Run VLX
                print("\n" + "-" * 70)
                description = input("Enter description (e.g., 'the rightmost red balloon'): ")
                self.run_vlx(description)
                print("-" * 70)

            elif key == ord('n'):  # New image
                print("\n" + "-" * 70)
                path = input("Enter image path: ").strip().strip('"')
                try:
                    self.load_image(path)
                    print(f"Loaded: {path}")
                except Exception as e:
                    print(f"Error loading image: {e}")
                print("-" * 70)

        cv2.destroyAllWindows()
        print("\nDemo closed.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VLX-Seek Interactive Demo")
    parser.add_argument("--image", type=Path, help="Initial image to load")
    parser.add_argument("--model", default="vlx-seek-1.5-3b",
                       choices=["vlx-seek", "vlx-seek-1.5-3b"],
                       help="VLX model to use")
    args = parser.parse_args()

    try:
        demo = VLXDemoUI(model=args.model)
        demo.run(initial_image=args.image)
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
