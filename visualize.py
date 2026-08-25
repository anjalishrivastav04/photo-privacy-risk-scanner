"""
visualize.py
Turns the risk_utils.analyze_image() output into things a user can look at:
  - a bounding-box overlay colored by per-fingertip risk
  - a soft heatmap of "risk mass" across the image
  - an auto-blurred/pixelated copy that protects medium/high-risk regions
"""

import numpy as np
import cv2

RISK_COLOR_BGR = {
    "low": (80, 200, 80),      # green
    "medium": (0, 200, 255),   # amber
    "high": (0, 0, 255),       # red
}


def _overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def draw_boxes(image_bgr, analysis, thickness=3):
    """Return a copy of the image with fingertip risk boxes + labels drawn.

    Fingertips in a closed pose sit very close together, so naive label
    placement produces a pile of overlapping text. Labels are therefore
    nudged vertically until they clear every label already placed, and a
    short leader line is drawn back to the box when a label had to move far.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    placed = []

    # Draw all boxes first so labels always sit on top of every box.
    tips = [(tip, hand) for hand in analysis["hands"] for tip in hand["fingertips"]]
    for tip, _ in tips:
        x0, y0, x1, y1 = tip["box"]
        cv2.rectangle(out, (x0, y0), (x1, y1), RISK_COLOR_BGR[tip["risk"]], thickness)

    # Label the riskiest fingertips first so they get the best positions.
    for tip, _ in sorted(tips, key=lambda t: -t[0]["score"]):
        x0, y0, x1, y1 = tip["box"]
        color = RISK_COLOR_BGR[tip["risk"]]
        label = f'{tip["name"]}: {tip["risk"]} ({tip["score"]:.2f})'
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        bw, bh = tw + 6, th + 7

        lx = min(max(x0, 0), max(w - bw, 0))
        base_y = y0 - 4
        chosen = None
        for dy in list(range(0, 400, bh + 3)) + list(range(-bh, -400, -(bh + 3))):
            ly = base_y - dy if dy >= 0 else (y1 + bh - dy - bh)
            top = ly - bh
            if top < 0 or ly > h:
                continue
            rect = (lx, top, lx + bw, ly)
            if not any(_overlaps(rect, r) for r in placed):
                chosen = rect
                break
        if chosen is None:
            chosen = (lx, max(y0 - bh, 0), lx + bw, max(y0, bh))

        cx0, cy0, cx1, cy1 = chosen
        if abs(cy1 - y0) > bh + 6:
            cv2.line(out, (x0 + 4, y0), (cx0 + 4, cy1), color, 1, cv2.LINE_AA)

        cv2.rectangle(out, (cx0, cy0), (cx1, cy1), color, -1)
        cv2.putText(out, label, (cx0 + 3, cy1 - 5), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
        placed.append(chosen)

    return out


def draw_heatmap(image_bgr, analysis, alpha=0.45):
    """Return a copy of the image blended with a soft heatmap of risk score,
    built from Gaussian blobs centered on each fingertip box, scaled by
    that fingertip's score."""
    h, w = image_bgr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for hand in analysis["hands"]:
        for tip in hand["fingertips"]:
            x0, y0, x1, y1 = tip["box"]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            radius = max((x1 - x0), (y1 - y0)) // 2 + 1
            score = tip["score"]

            yy, xx = np.ogrid[:h, :w]
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            sigma2 = max(radius, 1) ** 2
            blob = np.exp(-dist2 / (2 * sigma2)) * score
            heat = np.maximum(heat, blob)

    if heat.max() > 0:
        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)

    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_bgr, 1 - alpha, heat_color, alpha, 0)
    return blended


def protect_image(image_bgr, analysis, min_risk="medium", pixelate_block=10):
    """Return a copy of the image with medium/high-risk fingertip regions
    pixelated, so the user can still share the photo safely."""
    order = {"low": 0, "medium": 1, "high": 2}
    out = image_bgr.copy()

    for hand in analysis["hands"]:
        for tip in hand["fingertips"]:
            if order[tip["risk"]] < order[min_risk]:
                continue
            x0, y0, x1, y1 = tip["box"]
            region = out[y0:y1, x0:x1]
            if region.size == 0:
                continue
            rh, rw = region.shape[:2]
            small = cv2.resize(
                region,
                (max(rw // pixelate_block, 1), max(rh // pixelate_block, 1)),
                interpolation=cv2.INTER_LINEAR,
            )
            pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            out[y0:y1, x0:x1] = pixelated

    return out
