"""
risk_utils.py
Core pipeline for the Photo Privacy Risk Scanner.

WHAT THIS DOES (and does NOT do):
- It detects hands/fingertips in a photo (MediaPipe Hands).
- For each fingertip, it estimates whether ridge-level detail is physically
  resolvable in that region (see detail_score_v2.py for the model).
- It does NOT extract, reconstruct, or match any actual fingerprint. There is
  no ridge-matching, no minutiae extraction, no biometric template of any
  kind produced or stored anywhere in this code. The output is a *risk
  score* for a region, not a fingerprint.

This intentional scope (detect risk, don't extract prints) is the whole
point of the project: it's the defensive/privacy-preserving version of the
capability, safe to build, run, and publish.

DETECTOR BACKEND: this used to run on MediaPipe's legacy `solutions.hands`
API. That API was dropped from recent MediaPipe releases (it's simply not
importable on newer versions), which breaks on newer Python installs where
only those newer MediaPipe wheels are available. This module now runs
entirely on MediaPipe's current "Tasks" API (`hand_detector_v2.py`).
"""

import numpy as np
import cv2

from gestures import classify_gesture
from hand_detector_v2 import detect_hands_v2
from detail_score_v2 import detail_score_v2, image_blockiness

# Fingertip landmark indices in MediaPipe Hands' 21-point model:
# thumb tip=4, index tip=8, middle tip=12, ring tip=16, pinky tip=20
FINGERTIP_IDS = {
    4: "thumb",
    8: "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}

# Risk thresholds on the normalized 0-1 detail score.
RISK_THRESHOLDS = {
    "low": 0.0,
    "medium": 0.35,
    "high": 0.62,
}


def detect_hands_raw(image_bgr, max_hands=4, min_detection_confidence=0.3):
    """Detect hands and return the full per-landmark tuples, including the
    visibility/presence fields from the Tasks API."""
    return detect_hands_v2(
        image_bgr, max_hands=max_hands, min_detection_confidence=min_detection_confidence
    )


def detect_hands(image_bgr, max_hands=4, min_detection_confidence=0.3):
    """Detect hands and return plain (x_px, y_px) pixel coordinates per
    landmark -- the format the rest of this pipeline expects."""
    raw_hands = detect_hands_raw(
        image_bgr, max_hands=max_hands, min_detection_confidence=min_detection_confidence
    )
    return [[(x, y) for (x, y, visibility, presence) in hand] for hand in raw_hands]


def _fingertip_crop_box(pts, tip_idx, image_shape, pad_ratio=0.55, max_frac=0.12):
    """Square crop box around a fingertip, sized relative to the distance
    between the fingertip and its neighbouring knuckle, so the box scales
    with how large the hand appears in frame.

    pad_ratio stays well under 1.0 so the crop hugs the fingertip pad rather
    than ballooning into the palm or background. max_frac caps the box at a
    fraction of the image's shorter side, so a noisy/hallucinated landmark
    can't pull in unrelated regions of the photo.
    """
    h, w = image_shape[:2]
    tip = np.array(pts[tip_idx])
    joint = np.array(pts[tip_idx - 1])
    finger_len = np.linalg.norm(tip - joint)

    box_half = int(finger_len * pad_ratio)
    max_half = int(min(h, w) * max_frac)
    box_half = int(np.clip(box_half, 12, max_half))

    x0 = max(tip[0] - box_half, 0)
    y0 = max(tip[1] - box_half, 0)
    x1 = min(tip[0] + box_half, w)
    y1 = min(tip[1] + box_half, h)
    return x0, y0, x1, y1


def classify(score):
    if score >= RISK_THRESHOLDS["high"]:
        return "high"
    if score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def analyze_image(image_bgr):
    """Full pipeline: detect hands -> score each fingertip -> aggregate."""
    hands = detect_hands(image_bgr)
    result = {"hands": [], "overall_score": 0.0, "overall_risk": "low"}

    # Compression level is a whole-image property (the JPEG 8px grid is
    # aligned to the frame origin), so measure it once and apply it to every
    # fingertip score.
    blockiness = image_blockiness(image_bgr)
    result["blockiness"] = blockiness

    all_scores = []
    for pts in hands:
        fingertip_results = []
        for tip_idx, name in FINGERTIP_IDS.items():
            box = _fingertip_crop_box(pts, tip_idx, image_bgr.shape)
            x0, y0, x1, y1 = box
            crop = image_bgr[y0:y1, x0:x1]
            score, parts = detail_score_v2(crop, blockiness=blockiness, return_parts=True)
            fingertip_results.append(
                {
                    "name": name,
                    "box": box,
                    "score": score,
                    "risk": classify(score),
                    "parts": parts,
                }
            )
            all_scores.append(score)
        result["hands"].append(
            {"fingertips": fingertip_results, "gesture": classify_gesture(pts)}
        )

    if all_scores:
        # Overall risk is driven by the worst (max) fingertip: one exposed,
        # in-focus fingertip is a real risk even if the rest are blurry.
        overall = float(max(all_scores))
        result["overall_score"] = overall
        result["overall_risk"] = classify(overall)

    return result