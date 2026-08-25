"""
hand_detector_v2.py
Hand detector using MediaPipe's current "Tasks" API
(mediapipe.tasks.python.vision.HandLandmarker) instead of the deprecated
legacy `mp.solutions.hands` API, which has been removed from recent
MediaPipe releases entirely.

WHY THIS EXISTS:
Testing (see README) found a real gap: when a finger is tucked/occluded
(most often the thumb), MediaPipe still outputs a landmark position for it
-- guessed, not real -- and the legacy solutions API gave no per-landmark
signal to tell "confidently detected" apart from "guessed because the model
always outputs 21 points." Several geometric heuristics were tried
(distance from wrist, which side of the palm, segment length) and none
reliably caught it.

The Tasks API's landmark objects carry `visibility` and `presence` fields,
which is the API-level attempt at exactly this signal. CONFIRMED RESULT:
for the Hand Landmarker model these come back as None for every landmark --
the fields exist in the shared landmark format but are not populated for
hands. See test_visibility_signal.py to reproduce. The resolution gate in
detail_score_v2.py mitigates most of this in practice, since an occluded
finger usually yields a sub-Nyquist crop that scores zero anyway.

Requires downloading a small (~10MB) model file on first use -- needs
internet access once; the file is cached locally after that.
"""

import os
import tempfile
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Cache the model next to the code, but fall back to a temp directory when the
# app folder is read-only -- some hosting platforms mount it that way.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = (
    os.path.join(_APP_DIR, "hand_landmarker.task")
    if os.access(_APP_DIR, os.W_OK)
    else os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
)

_detector = None


def _ensure_model():
    """Download the HandLandmarker model file if it isn't already cached
    locally. Needs internet access the first time this runs."""
    if os.path.exists(MODEL_PATH):
        return
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not download the hand landmark model from {MODEL_URL}. "
            f"This needs internet access on first run. Original error: {e}"
        ) from e


def _get_detector(num_hands=4, min_detection_confidence=0.3, min_presence_confidence=0.3):
    global _detector
    if _detector is not None:
        return _detector

    _ensure_model()
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_detection_confidence,
        running_mode=vision.RunningMode.IMAGE,
    )
    _detector = vision.HandLandmarker.create_from_options(options)
    return _detector


def detect_hands_v2(image_bgr, max_hands=4, min_detection_confidence=0.3):
    """Detect hands, returning each landmark as
    (x_px, y_px, visibility, presence).

    visibility/presence are whatever the model actually outputs -- for the
    hand model they are None (see module docstring). They are surfaced anyway
    so the limitation stays visible rather than silently assumed away.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detector = _get_detector(
        num_hands=max_hands, min_detection_confidence=min_detection_confidence
    )
    result = detector.detect(mp_image)

    hands_out = []
    for hand_landmarks in result.hand_landmarks:
        pts = []
        for lm in hand_landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            visibility = getattr(lm, "visibility", None)
            presence = getattr(lm, "presence", None)
            pts.append((x_px, y_px, visibility, presence))
        hands_out.append(pts)
    return hands_out