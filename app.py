"""
Photo Privacy Risk Scanner — Streamlit app.

Checks an uploaded photo for two independent privacy risks:
  1. Fingerprint exposure -- how much fingertip ridge-level detail is
     physically resolvable, and *where* in the image that risk comes from.
  2. Metadata leakage -- GPS location, device model, and capture timestamp
     embedded in the file itself (EXIF), invisible in the image.

Run locally:
    streamlit run app.py
"""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import ui
from risk_utils import analyze_image
from visualize import draw_boxes, draw_heatmap, protect_image
from exif_privacy import analyze_exif

st.set_page_config(
    page_title="Photo Privacy Risk Scanner",
    page_icon="🖐️",
    layout="centered",
)

ui.inject_css()

# The status line depends on the analysis result, which isn't known until
# further down the script. Reserve the slot now, fill it once we know.
masthead_slot = st.empty()
with masthead_slot:
    ui.masthead("standby")

with st.expander("Method — and what this deliberately does not do"):
    st.markdown(
        "**Fingerprint exposure.** Hands and fingertips are located with "
        "MediaPipe. Each fingertip is then scored for how much *genuinely "
        "resolvable* ridge detail it carries:"
    )
    ui.formula("score = resolution_gate × ridge_band_energy × compression_trust")
    st.markdown(
        """
- **Resolution gate** — physics, not heuristics. Ridges sit ~0.5 mm apart on
  a ~15 mm fingertip, so a fingertip spans ~30 ridge periods. By Nyquist you
  need at least 2 pixels per period, so a fingertip under ~60 px wide
  *cannot* resolve ridges at any sharpness. It scores zero.
- **Ridge-band energy** — spectral energy in the specific frequency band
  where ridges live, rather than a generic edge count that compression fakes.
- **Compression trust** — heavy JPEG compression destroys real detail and
  fabricates edge-like artifacts, so it discounts the score.

**Metadata.** EXIF is read straight from the uploaded file before any
processing, since resizing or converting an image discards it.

**What it does not do.** It never extracts, reconstructs, or matches a
fingerprint. No biometric template is created or stored anywhere in this
code. The output is a risk score for a region of a photo — deliberately the
defensive side of this capability, not the offensive one.

**What it cannot tell you.** That visible ridges mean a fingerprint *could*
be cloned. Real spoofing needs sufficient minutiae, known scale, and physical
fabrication. This is a screening heuristic, not forensic proof.
"""
    )

uploaded = st.file_uploader(
    "Load image for analysis",
    type=["jpg", "jpeg", "png"],
    help="A front-facing peace sign or open palm, held far enough back that "
    "the whole hand is in frame, gives the most reliable reading.",
)

if uploaded is None:
    with masthead_slot:
        ui.masthead("system ready — awaiting input")
    st.info(
        "Load a photo to begin. For the most trustworthy result use an image "
        "straight from the camera — photos routed through WhatsApp or "
        "Instagram are recompressed and have their metadata stripped."
    )
    st.stop()

# ---------------------------------------------------------------- analysis --
pil_img = Image.open(uploaded)

# Read metadata BEFORE any transform below -- .convert()/.resize() build a new
# image and drop EXIF, so checking later would silently report "none found".
exif_report = analyze_exif(pil_img)

# Phone photos often carry EXIF orientation metadata that PIL does not apply
# automatically; without this a portrait selfie loads sideways and the hand
# detector misses it entirely.
pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")

MAX_SIDE = 1600
if max(pil_img.size) > MAX_SIDE:
    scale = MAX_SIDE / max(pil_img.size)
    pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)))

image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

slot = st.empty()
with slot:
    ui.scanning("analysing image")
analysis = analyze_image(image_bgr)
slot.empty()

hands = analysis["hands"]
n_hands = len(hands)

with masthead_slot:
    ui.masthead(
        f"scan complete — {n_hands} hand{'s' if n_hands != 1 else ''} located"
        if n_hands
        else "scan complete — no hand located"
    )

# ----------------------------------------------------------------- verdict --
if n_hands == 0:
    ui.verdict(
        "unknown",
        0.0,
        kicker="fingerprint exposure",
        note="No hand detected, so ridge exposure could not be assessed. "
        "<b>This is not a confirmation that the photo is safe.</b> Extreme "
        "close-ups, occluded hands, and hands holding objects are known "
        "blind spots of the detector.",
    )
else:
    risk = analysis["overall_risk"]
    score = analysis["overall_score"]
    worst = max((t for h in hands for t in h["fingertips"]), key=lambda t: t["score"])
    notes = {
        "low": "No fingertip in this image carries resolvable ridge detail.",
        "medium": f"Driven by the <b>{worst['name']}</b> fingertip — partial "
        "ridge detail is resolvable there.",
        "high": f"The <b>{worst['name']}</b> fingertip carries clearly "
        "resolvable ridge detail. Use the protected copy below.",
    }
    ui.verdict(risk, score, kicker="fingerprint exposure", note=notes.get(risk, ""))

st.write("")

# ------------------------------------------------------------ summary row --
col_a, col_b = st.columns(2)

with col_a:
    if n_hands == 0:
        ui.panel(
            "subject / vectors",
            "NO READING",
            ui.RISK_STYLES["unknown"]["color"],
            "No hand located in frame.",
        )
    else:
        total = n_hands * 5
        flagged = sum(1 for h in hands for t in h["fingertips"] if t["risk"] != "low")
        gestures = " · ".join(h["gesture"]["label"] for h in hands)
        noun = "vector" if flagged == 1 else "vectors"
        ui.panel(
            "subject / vectors",
            f"{flagged}/{total} {noun}",
            ui.RISK_STYLES[analysis["overall_risk"]]["color"],
            f"Fingertips scoring above low. Pose: {gestures}.",
        )

with col_b:
    ex_risk = exif_report["risk"]
    if exif_report["has_gps"]:
        headline, note = (
            "GPS EMBEDDED",
            "Exact coordinates readable from the original file by anyone who "
            "downloads it. Stripped in the protected copy.",
        )
    elif ex_risk == "medium":
        headline, note = (
            "DEVICE / TIME",
            "No GPS, but the file identifies your device and capture time.",
        )
    else:
        headline, note = ("CLEAN", "No GPS, device, or timestamp in this file.")
    ui.panel("metadata payload", headline, ui.RISK_STYLES[ex_risk]["color"], note)

st.write("")

# ------------------------------------------------------------- visual tabs --
if n_hands > 0:
    tab_map, tab_heat, tab_safe, tab_data = st.tabs(
        ["risk map", "heatmap", "protected copy", "telemetry"]
    )

    with tab_map:
        st.image(
            cv2.cvtColor(draw_boxes(image_bgr, analysis), cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
        st.caption(
            "Each box marks a fingertip, coloured by risk. Green fingertips "
            "carry no resolvable ridge detail."
        )

    with tab_heat:
        st.image(
            cv2.cvtColor(draw_heatmap(image_bgr, analysis), cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
        st.caption("Warmer regions carry more resolvable ridge-level detail.")

    with tab_safe:
        protected = protect_image(image_bgr, analysis, min_risk="medium")
        st.image(cv2.cvtColor(protected, cv2.COLOR_BGR2RGB), use_container_width=True)

        buf = io.BytesIO()
        # Image.fromarray() builds a new image from raw pixels only -- no
        # EXIF/IPTC/XMP block is ever attached, so this download is stripped of
        # GPS/device/timestamp metadata as well as having fingertips pixelated.
        Image.fromarray(cv2.cvtColor(protected, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        st.download_button(
            "↓ export protected image",
            data=buf.getvalue(),
            file_name="protected_photo.png",
            mime="image/png",
            use_container_width=True,
        )
        st.caption(
            "Both protections applied: medium/high-risk fingertips pixelated, "
            "all metadata removed."
        )

    with tab_data:
        ui.formula("score = resolution_gate × ridge_band_energy × compression_trust")
        for hi, hand in enumerate(hands):
            if len(hands) > 1:
                st.markdown(f"**hand {hi + 1}**")
            for tip in sorted(hand["fingertips"], key=lambda t: -t["score"]):
                p = tip.get("parts", {})
                color = ui.RISK_STYLES[tip["risk"]]["color"]
                ui.readouts(
                    [
                        ("score", tip["score"], color),
                        ("res gate", p.get("resolution_gate", 0.0), color),
                        ("ridge band", p.get("ridge_band", 0.0), color),
                        ("compression", p.get("compression_trust", 1.0), color),
                    ],
                    title=f"{tip['name']} — {tip['risk']} · "
                    f"{round(p.get('fingertip_px', 0))} px across",
                )
                st.write("")

        st.caption(
            f"Whole-image JPEG blockiness **{analysis.get('blockiness', 0):.2f}** "
            "(≈1–2 clean; 4+ heavily compressed, so fine detail is discounted). "
            "A fingertip under ~60 px scores 0 by the resolution gate no matter "
            "how sharp it looks. A tucked or occluded finger still gets a guessed "
            "position — its row is unreliable, though such fingers usually produce "
            "a small crop the gate zeroes anyway."
        )

# ------------------------------------------------------------ metadata det --
with st.expander("metadata findings in full"):
    for finding in exif_report["findings"]:
        st.write(f"- {finding}")