"""
gestures.py
Small bonus feature: a rule-based hand gesture label (peace sign, thumbs up,
open palm, fist, pointing, etc.), separate from the fingerprint-risk logic.

This is NOT machine-learned -- it's simple geometry on the 21 MediaPipe hand
landmarks: for each finger, we check whether the fingertip is farther from
the wrist than that finger's middle knuckle (PIP joint). If it is, the
finger is "extended"; otherwise it's "folded". The pattern of which fingers
are extended maps to a small set of common gesture labels.

Because this only looks at extended-vs-folded, not fine-grained orientation,
it can't distinguish "thumbs up" from "thumb pointing sideways" -- both just
register as "only the thumb is extended". That's a known, stated limitation,
not a bug.
"""

import numpy as np

# Landmark indices per MediaPipe's 21-point hand model.
WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _finger_extended(pts, tip_idx, pip_idx, wrist=WRIST, margin=1.05):
    """A non-thumb finger is 'extended' if its tip is meaningfully farther
    from the wrist than its own PIP (middle) joint -- true regardless of
    how the hand is rotated in frame, unlike a simple up/down y-check."""
    return _dist(pts[tip_idx], pts[wrist]) > _dist(pts[pip_idx], pts[wrist]) * margin


def _thumb_extended(pts, margin=1.1):
    """The thumb moves sideways relative to the palm rather than along the
    same axis as the other fingers, so it needs its own check: is the thumb
    tip meaningfully farther from the pinky's base than the thumb's own base
    is? (i.e. has the thumb swung away from the palm.)"""
    tip_to_pinky = _dist(pts[THUMB_TIP], pts[PINKY_MCP])
    base_to_pinky = _dist(pts[THUMB_MCP], pts[PINKY_MCP])
    return tip_to_pinky > base_to_pinky * margin


def _finger_states(pts):
    return {
        "thumb": _thumb_extended(pts),
        "index": _finger_extended(pts, INDEX_TIP, INDEX_PIP),
        "middle": _finger_extended(pts, MIDDLE_TIP, MIDDLE_PIP),
        "ring": _finger_extended(pts, RING_TIP, RING_PIP),
        "pinky": _finger_extended(pts, PINKY_TIP, PINKY_PIP),
    }


# Known extended-finger patterns -> label. Order matters: checked top to
# bottom, first match wins.
_PATTERNS = [
    ({"thumb", "index", "middle", "ring", "pinky"}, "Open palm ✋"),
    (set(), "Fist ✊"),
    ({"index", "middle"}, "Peace / victory sign ✌️"),
    ({"thumb"}, "Thumbs up/out 👍"),
    ({"index"}, "Pointing ☝️"),
    ({"thumb", "pinky"}, "Shaka / call me 🤙"),
    ({"index", "middle", "ring", "pinky"}, "Four fingers (no thumb)"),
    ({"index", "pinky"}, "Rock on 🤘"),
    ({"thumb", "index", "pinky"}, "Rock on 🤘"),
]


def classify_gesture(pts):
    """Return {"label": str, "extended": {finger_name: bool}} for one hand's
    21 landmarks."""
    states = _finger_states(pts)
    extended = {name for name, is_ext in states.items() if is_ext}

    label = None
    for pattern, name in _PATTERNS:
        if extended == pattern:
            label = name
            break

    if label is None:
        if extended:
            label = f"Other (extended: {', '.join(sorted(extended))})"
        else:
            label = "Fist ✊"

    return {"label": label, "extended": states}
