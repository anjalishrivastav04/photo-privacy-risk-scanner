"""
detail_score_v2.py
Physically-grounded replacement for the original heuristic detail score.

WHY THE ORIGINAL WAS WRONG
The v1 score combined Laplacian-variance "sharpness", Canny edge density,
and a weak size term. Stress-testing found it was effectively *scale
invariant*: downscaling a photo to 400px wide RAISED its score (0.78 ->
0.81), and JPEG quality-15 compression also RAISED it (0.78 -> 0.83).
Both are wrong. A 400px-wide photo has ~30px fingertips, which physically
cannot resolve ridges; JPEG blocking creates fake edges that Canny counts
as "detail". v1 was measuring *relative local contrast inside the crop*,
not *resolvable ridge detail* -- two different claims.

THE PHYSICS THIS VERSION USES
Adult fingerprint ridges have a spatial period of roughly 0.4-0.6 mm.
A fingertip pad is roughly 13-17 mm across. So a fingertip spans on the
order of 25-40 ridge periods edge to edge.

For a fingertip that appears W pixels wide in an image:
  ridge period in pixels  ~=  W / 30
  Nyquist limit: you need >= 2 px per period to represent a ridge at all,
  and realistically >= 3 px/period before ridges are usable rather than
  aliased mush.

  => W < 60 px   : ridges are below Nyquist. NOT RESOLVABLE, full stop.
  => W 60-120 px : marginal; ridges may be aliased/ambiguous.
  => W > 120 px  : enough sampling for ridges to be genuinely readable.

This gives a HARD GATE on apparent size that no amount of sharpening,
contrast, or compression artifact can fake -- because it's a property of
sampling, not of pixel statistics.

WHAT IT MEASURES ON TOP OF THE GATE
Instead of generic edge density (which compression artifacts inflate), it
measures energy in the *ridge spatial-frequency band* specifically: the
annulus in the 2D FFT around ~30 cycles per fingertip width. Blur removes
energy there (correctly lowering the score), while JPEG 8x8 blocking puts
its energy at a different, higher frequency, so it no longer masquerades
as ridge detail.

Final score = resolution_gate * ridge_band_energy * compression_trust
The gate MULTIPLIES rather than adds, so an unresolvable fingertip cannot
score high regardless of how contrasty the crop is.
"""

import numpy as np
import cv2

# Ridge periods across a fingertip's width (see module docstring).
RIDGE_CYCLES_PER_FINGERTIP = 30.0

# Apparent fingertip width (px) thresholds from the Nyquist argument.
MIN_RESOLVABLE_PX = 60.0    # below this, ridges are below Nyquist -> gate ~0
GOOD_RESOLVABLE_PX = 130.0  # at/above this, sampling is comfortably sufficient


def _resolution_gate(fingertip_px):
    """0..1 multiplier for whether ridges are physically resolvable at this
    apparent fingertip size. Hard zero below Nyquist, ramping to 1.0 once
    there are ~4+ pixels per ridge period."""
    if fingertip_px <= MIN_RESOLVABLE_PX:
        return 0.0
    if fingertip_px >= GOOD_RESOLVABLE_PX:
        return 1.0
    return (fingertip_px - MIN_RESOLVABLE_PX) / (GOOD_RESOLVABLE_PX - MIN_RESOLVABLE_PX)


def _ridge_band_energy(gray):
    """Fraction of spectral energy sitting in the ridge frequency band.

    Ridges occupy ~RIDGE_CYCLES_PER_FINGERTIP cycles across the crop, so we
    integrate the 2D power spectrum over an annulus centred on that radial
    frequency (with generous tolerance for finger size / crop padding
    variation), normalised by total AC energy.

    Blur -> energy vanishes from this band -> low.
    JPEG blocking -> energy lands at the 8px-grid frequency, typically
    outside this band -> does not inflate the score the way Canny did.
    """
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 0.0

    win = np.outer(np.hanning(h), np.hanning(w))  # reduce edge leakage
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64) * win))
    power = np.abs(f) ** 2

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Cycles-per-image-width maps to FFT radius directly (radius r == r cycles
    # across the crop). Band covers plausible ridge densities.
    lo = RIDGE_CYCLES_PER_FINGERTIP * 0.55
    hi = RIDGE_CYCLES_PER_FINGERTIP * 1.7
    nyq = min(h, w) / 2.0
    hi = min(hi, nyq)
    if hi <= lo:
        return 0.0

    band = (radius >= lo) & (radius <= hi)
    ac = radius > 1.5  # exclude DC / very low frequency (shape, lighting)
    total = power[ac].sum()
    if total <= 0:
        return 0.0

    frac = power[band].sum() / total
    # Empirically, a well-resolved fingertip puts a few percent of AC energy
    # in this band; normalise so ~6% maps to 1.0.
    return float(np.clip(frac / 0.06, 0.0, 1.0))


# Blockiness (see image_blockiness) below CLEAN_BLOCKINESS is treated as an
# uncompressed/lightly-compressed image; at or above HEAVY_BLOCKINESS the
# image is so compressed that fine detail can't be trusted at all.
CLEAN_BLOCKINESS = 2.0
HEAVY_BLOCKINESS = 5.0
MIN_COMPRESSION_TRUST = 0.15


def image_blockiness(image_bgr):
    """No-reference JPEG blockiness estimate for a WHOLE image.

    Compares mean absolute gradient ACROSS 8-pixel block boundaries against
    the mean within blocks. ~1-2 for a clean image; grows steadily as JPEG
    quality drops (measured: q60~2.3, q40~2.6, q15~4.1, q8~6.2).

    Must be computed on the full frame, not a crop -- the 8px grid is
    aligned to the original image origin, and an arbitrary crop offset
    would smear the measurement.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    dh = np.abs(np.diff(gray, axis=1))
    dv = np.abs(np.diff(gray, axis=0))
    if dh.size == 0 or dv.size == 0:
        return 1.0

    col_boundary = dh[:, 7::8].mean() if dh[:, 7::8].size else 0.0
    col_interior = np.delete(dh, np.s_[7::8], axis=1).mean() if dh.size else 0.0
    row_boundary = dv[7::8, :].mean() if dv[7::8, :].size else 0.0
    row_interior = np.delete(dv, np.s_[7::8], axis=0).mean() if dv.size else 0.0

    boundary = (col_boundary + row_boundary) / 2.0
    interior = (col_interior + row_interior) / 2.0
    return float(boundary / interior) if interior > 0 else 1.0


def _compression_trust(blockiness):
    """0..1 multiplier: how much we trust 'fine detail' in an image with this
    much JPEG blocking. Heavy compression both destroys real ridge detail and
    fabricates edge-like artifacts, so a heavily blocked image should not be
    able to score high."""
    if blockiness <= CLEAN_BLOCKINESS:
        return 1.0
    span = HEAVY_BLOCKINESS - CLEAN_BLOCKINESS
    val = (HEAVY_BLOCKINESS - blockiness) / span
    return float(np.clip(val, MIN_COMPRESSION_TRUST, 1.0))


def detail_score_v2(crop_bgr, blockiness=None, return_parts=False):
    """Score 0..1 for how much genuinely resolvable ridge detail a fingertip
    crop carries. See module docstring for the model.

    blockiness: pass image_blockiness(full_image) so the compression penalty
    can be applied. If omitted, no compression penalty is applied (the score
    is then only gated on resolution and ridge-band energy).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return (0.0, {}) if return_parts else 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    fingertip_px = float(min(h, w))

    gate = _resolution_gate(fingertip_px)
    ridge = _ridge_band_energy(gray) if gate > 0 else 0.0
    trust = _compression_trust(blockiness) if blockiness is not None else 1.0
    score = float(np.clip(gate * ridge * trust, 0.0, 1.0))

    if return_parts:
        return score, {
            "fingertip_px": fingertip_px,
            "resolution_gate": gate,
            "ridge_band": ridge,
            "compression_trust": trust,
        }
    return score