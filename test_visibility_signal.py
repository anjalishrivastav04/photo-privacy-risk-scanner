"""
test_visibility_signal.py
Run this LOCALLY (needs internet the first time, to download the hand
landmark model) to answer the open question this upgrade was built to
answer: does MediaPipe's Tasks API actually give a useful per-landmark
confidence signal for hands, that can tell a genuinely visible fingertip
apart from a guessed/hallucinated one (e.g. a tucked thumb)?

Usage:
    python3 test_visibility_signal.py path/to/full_hand_photo.jpg path/to/tucked_thumb_photo.jpg

Pass one photo where all 5 fingers are clearly visible/extended, and one
photo (like your earlier peace-sign or open-palm shots) where the thumb (or
another finger) is tucked/not clearly visible. This prints the visibility
and presence values MediaPipe actually reports for each fingertip.

WHAT TO LOOK FOR:
- If the tucked finger's visibility/presence is noticeably LOWER than the
  clearly-visible fingers' values (e.g. 0.3 vs 0.9), the signal is real and
  useful -- risk_utils.py can be updated to filter/flag on it.
- If all values are similar regardless of whether a finger is actually
  visible (e.g. everything reads ~0.99, or everything reads 0.0), the
  signal is NOT meaningful for hands, and this is worth documenting as a
  confirmed limitation rather than pretending it's fixed.

Either outcome is a useful, honest result -- this script's job is to find
out which one is true, not to assume.
"""

import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

from hand_detector_v2 import detect_hands_v2
from risk_utils import FINGERTIP_IDS


def load_bgr(path):
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def report(path, label):
    print(f"\n=== {label}: {path} ===")
    img = load_bgr(path)
    hands = detect_hands_v2(img)

    if not hands:
        print("No hand detected by the Tasks API on this photo.")
        return

    for hi, pts in enumerate(hands):
        print(f"-- hand {hi + 1} --")
        for tip_idx, name in FINGERTIP_IDS.items():
            x, y, visibility, presence = pts[tip_idx]
            print(
                f"  {name:8s}: pos=({x:4d},{y:4d})  "
                f"visibility={visibility!r}  presence={presence!r}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    labels = ["photo 1 (pass a full-hand photo first)", "photo 2 (pass a tucked-finger photo second)"]
    for i, path in enumerate(sys.argv[1:3]):
        report(path, labels[i] if i < len(labels) else f"photo {i + 1}")

    print(
        "\nCompare the visibility/presence numbers for the tucked finger "
        "above against a clearly-visible finger in the same or other photo. "
        "See the top of this file for how to interpret the result."
    )
