"""Test VLX-Seek API client with a simple synthetic image."""
import os
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
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Load environment variables from .env file
load_dotenv(ROOT / ".env")

from vlx_api_client import VLXSeekAPIClient


def test_api_connection():
    """Test basic API connection and authentication."""
    print("Testing API connection...")

    # Check environment variables
    api_key = os.environ.get("OM_API_KEY")
    api_endpoint = os.environ.get("OM_API_ENDPOINT")

    if not api_key:
        print("ERROR: OM_API_KEY environment variable not set")
        print("Please set it with your Om AI Lab API key:")
        print("  $env:OM_API_KEY='your-api-key-here'")
        return False

    if not api_endpoint:
        print("ERROR: OM_API_ENDPOINT environment variable not set")
        print("Please set it with your API endpoint URL:")
        print("  $env:OM_API_ENDPOINT='https://your-endpoint'")
        return False

    print(f"✓ API key found: {api_key[:8]}...")
    print(f"✓ API endpoint: {api_endpoint}")

    try:
        client = VLXSeekAPIClient(model="vlx-seek-1.5-3b")
        print(f"✓ Client initialized with model: {client.model}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize client: {e}")
        return False


def test_simple_detection():
    """Test detection on a synthetic image with colored rectangles."""
    print("\n" + "="*60)
    print("Testing VLX-Seek API with synthetic image...")
    print("="*60)

    # Create a simple test image: white background with red and blue rectangles
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Draw a red rectangle (left side)
    cv2.rectangle(image, (50, 100), (150, 200), (0, 0, 255), -1)

    # Draw a blue rectangle (center)
    cv2.rectangle(image, (250, 100), (350, 200), (255, 0, 0), -1)

    # Draw another red rectangle (right side)
    cv2.rectangle(image, (450, 100), (550, 200), (0, 0, 255), -1)

    # Convert to PIL RGB
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Define bounding box proposals (we provide all three rectangles)
    proposals = [
        [50, 100, 150, 200],   # left red
        [250, 100, 350, 200],  # center blue
        [450, 100, 550, 200]   # right red
    ]

    # Test with a simple description
    description = "the rightmost red rectangle"

    print(f"\nImage size: {pil_image.size}")
    print(f"Number of proposals: {len(proposals)}")
    print(f"Task description: '{description}'")
    print("\nProposals:")
    for i, box in enumerate(proposals):
        print(f"  [{i}] {box}")

    try:
        client = VLXSeekAPIClient(model="vlx-seek-1.5-3b")

        print("\nCalling VLX-Seek API...")
        result = client.ground(
            image=pil_image,
            boxes=proposals,
            description=description,
            lang="en",
            temperature=0.0,
            max_new_tokens=128
        )

        print("\n" + "-"*60)
        print("API Response:")
        print("-"*60)
        print(f"Answer: {result['answer']}")
        print(f"\nDetected boxes: {len(result['result_bbox_list'])}")

        for i, bbox in enumerate(result['result_bbox_list']):
            print(f"\n[{i}] {bbox['object_index']}")
            print(f"    Box: [{bbox['xmin']}, {bbox['ymin']}, {bbox['xmax']}, {bbox['ymax']}]")
            if 'label' in bbox:
                print(f"    Label: {bbox['label']}")

        # Show usage statistics
        usage = client.get_usage_summary()
        print("\n" + "-"*60)
        print("Usage Statistics:")
        print("-"*60)
        print(f"Total requests: {usage['total_requests']}")
        print(f"Prompt tokens: {usage['total_prompt_tokens']}")
        print(f"Completion tokens: {usage['total_completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Total time: {usage['total_took_ms']} ms")
        print(f"Average time per request: {usage['average_took_ms']:.1f} ms")

        return True

    except Exception as e:
        print(f"\nERROR: API call failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("VLX-Seek API Client Test Suite")
    print("="*60)

    # Test 1: Connection
    if not test_api_connection():
        print("\n✗ Connection test failed. Please check your credentials.")
        return 1

    # Test 2: Simple detection
    if not test_simple_detection():
        print("\n✗ Detection test failed.")
        return 1

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nYou can now use the API client with your maritime frames.")
    print("See docs/VLX-API集成指南.md for usage instructions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
