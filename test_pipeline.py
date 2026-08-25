"""
Sanity test for the pipeline that does NOT depend on MediaPipe's actual hand
detection (no real hand photo is available in this environment). Instead it:
  1. Builds a synthetic 400x400 test image with two regions of different
     texture (one sharp/detailed "in-focus" patch, one blurry flat patch).
  2. Manually supplies fake 21-point hand landmarks pointing at those two
     regions (bypassing detect_hands, which is the only part that needs a
     real photo of an actual hand).
  3. Runs the real scoring, classification, overlay, heatmap, and pixelation
     functions on it and asserts the results make sense: the sharp/detailed
     region should score higher risk than the flat/blurry one.

This proves the scoring math, aggregation, drawing, and blurring code all
run correctly end-to-end. Detection accuracy itself (MediaPipe finding real
fingertips in a real photo) should be checked with an actual photo once you
run `streamlit run app.py` locally or after deploying.
"""

import numpy as np
import cv2

import risk_utils
from visualize import draw_boxes, draw_heatmap, protect_image


def make_synthetic_image():
    img = np.full((400, 400, 3), 200, dtype=np.uint8)  # flat gray background

    # "high detail" patch: dense random noise + edges (simulates sharp,
    # in-focus fingertip skin texture) around (100, 100)
    detailed_patch = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    img[60:140, 60:140] = detailed_patch

    # "low detail" patch: flat/blurred region (simulates an out-of-focus or
    # distant fingertip) around (300, 300)
    flat_patch = np.full((80, 80, 3), 190, dtype=np.uint8)
    flat_patch = cv2.GaussianBlur(flat_patch, (15, 15), 5)
    img[260:340, 260:340] = flat_patch

    return img


def make_fake_landmarks():
    """21-point landmark layout, but only the points our code actually
    reads (tip indices 4/8/12/16/20 and their tip-1 neighbor) are placed
    meaningfully; the rest are filler."""
    pts = [(200, 200)] * 21

    # "index" fingertip (idx 8) sits in the high-detail patch, with its
    # neighbor joint (idx 7) close by so the crop-size math behaves normally.
    pts[8] = (100, 100)
    pts[7] = (90, 110)

    # "pinky" fingertip (idx 20) sits in the low-detail/blurry patch.
    pts[20] = (300, 300)
    pts[19] = (290, 310)

    return pts


def run():
    img = make_synthetic_image()
    fake_pts = make_fake_landmarks()

    # Monkeypatch detect_hands so analyze_image() uses our fake landmarks
    # instead of running real MediaPipe detection.
    risk_utils.detect_hands = lambda image_bgr, **kwargs: [fake_pts]

    analysis = risk_utils.analyze_image(img)

    assert len(analysis["hands"]) == 1, "expected one fake hand"
    fingertips = {t["name"]: t for t in analysis["hands"][0]["fingertips"]}

    index_score = fingertips["index"]["score"]
    pinky_score = fingertips["pinky"]["score"]

    print(f"index (sharp/detailed) score = {index_score:.3f} -> {fingertips['index']['risk']}")
    print(f"pinky (flat/blurry)   score = {pinky_score:.3f} -> {fingertips['pinky']['risk']}")

    assert index_score > pinky_score, (
        "expected the sharp/detailed synthetic patch to score higher risk "
        "than the flat/blurry patch"
    )
    assert analysis["overall_risk"] in ("low", "medium", "high")
    print(f"overall risk = {analysis['overall_risk']} (score {analysis['overall_score']:.3f})")

    # Exercise the visualization + protection functions end-to-end.
    boxed = draw_boxes(img, analysis)
    assert boxed.shape == img.shape

    heat = draw_heatmap(img, analysis)
    assert heat.shape == img.shape

    protected = protect_image(img, analysis, min_risk="medium")
    assert protected.shape == img.shape
    # The protected copy should actually differ from the original wherever
    # a medium/high risk region got pixelated.
    if fingertips["index"]["risk"] in ("medium", "high"):
        assert not np.array_equal(protected[60:140, 60:140], img[60:140, 60:140]), (
            "expected the high-detail region to be visibly altered by pixelation"
        )

    cv2.imwrite("test_output_boxed.png", boxed)
    cv2.imwrite("test_output_heatmap.png", heat)
    cv2.imwrite("test_output_protected.png", protected)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    run()
