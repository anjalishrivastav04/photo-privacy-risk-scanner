# Fingerprint Privacy Risk Scanner

A computer-vision tool that scans a photo of a hand and flags **where** it
exposes enough fingertip detail that a fingerprint could plausibly be lifted
from it — before you post it, not after.

## Why this exists

Recent research and news coverage have shown that ordinary close-up photos
(a "peace sign" selfie, a hand-holding-an-ID shot, a food/product photo with
a visible fingertip) can carry enough resolvable ridge detail for a
fingerprint to be partially reconstructed. That's a real, largely
unaddressed privacy risk: biometric data leaking through completely
ordinary, non-biometric photos.

This project deliberately builds the **defensive** side of that capability,
not the extraction side.

- **User intent:** someone about to share a photo wants to know, before
  posting, whether it exposes fingerprint-level detail — and if so, where.
- **Business / product framing:** this is the kind of check a photo-sharing
  app, social platform, or even a phone camera app could run automatically,
  the same way platforms already scan screenshots for exposed PII. It
  reduces unintentional biometric data exposure by catching it before
  sharing, rather than after.
- **Output:** an overall risk rating (low / medium / high), a visual
  breakdown (bounding boxes and a heatmap) showing exactly which region of
  the photo is driving that risk, and a one-click auto-pixelated version
  that's safe to share.

## What it does NOT do

This is the most important design decision in the project, and it's worth
stating explicitly (including in an interview, if you're asked about it):

- It does not extract, reconstruct, or match fingerprints.
- It produces no biometric template of any kind.
- The model only answers "is there enough fine detail in this region that
  ridge lines would plausibly be resolvable" — a general image-quality /
  sharpness question, not a fingerprint-specific one.

That scoping is what makes this safe to build, run, demo, and publish on a
public GitHub repo, unlike an actual fingerprint-reconstruction tool.

## How it works

1. **Hand/fingertip detection** — [MediaPipe Hands](https://developers.google.com/mediapipe)
   locates 21 landmarks per hand; the 5 fingertip points are used to crop a
   small region around each fingertip, sized relative to how large the hand
   appears in frame.
2. **Detail scoring** (`risk_utils.py`) — each fingertip crop gets a 0-1
   score combining:
   - **Sharpness** (variance of the Laplacian — is this region in focus?)
   - **Edge density** (Canny edge fraction — is there fine-grained texture?)
   - **Apparent size** (pixel area — a tiny/distant fingertip can't carry
     ridge detail no matter how sharp the photo is)
3. **Classification** — each fingertip gets a low/medium/high risk label;
   the image's overall risk is the *worst* fingertip (one exposed, in-focus
   finger is enough to matter, even if the rest of the hand is blurry).
4. **Visualization** (`visualize.py`) — bounding boxes and a heatmap show
   which regions drove the score; a "protect" function pixelates
   medium/high-risk regions so the photo is still shareable.
5. **App** (`app.py`) — a Streamlit UI that ties it together: upload a
   photo, see the risk rating, the boxes/heatmap, and download a protected
   copy.

## Limitations & how you'd calibrate it further

- The risk thresholds in `risk_utils.RISK_THRESHOLDS` are heuristic
  starting points, not fit to a labeled dataset of real fingerprint-exposure
  cases (there isn't a public one — for good reason). If you extend this
  project, the honest next step is a small human-labeled set of your own
  photos at varying distances/focus, used to tune the thresholds and
  weightings.
- **Extreme crops fail detection entirely.** If a photo shows only 1-2
  fingers with most of the hand (palm, other fingers) out of frame,
  MediaPipe often can't recognize it as a hand at all — it's trained on
  photos showing a more complete hand structure. The app correctly reports
  "no hand detected" in this case rather than guessing.
- **Occluded/tucked fingers can get hallucinated positions.** MediaPipe's
  hand landmark model always outputs all 21 points, even for a finger
  (most often the thumb) that's tucked, folded, or angled away and not
  actually visible. The legacy API used here doesn't expose a per-landmark
  confidence score, so there's no reliable way to detect and discard a
  hallucinated point automatically — I tried two geometric plausibility
  checks (distance from wrist, which side of the palm it falls on) and
  neither reliably caught it in testing. **For accurate, trustworthy
  results, use photos where all fingers you care about are clearly
  extended and visible** — that's the case the underlying model was
  actually validated on. This is a real, worth-knowing failure mode of the
  pretrained model, not a bug that got papered over.
- Detection quality also depends generally on lighting, blur, and gloved or
  heavily overlapping fingers.
- This scores *apparent resolvable detail*, not an actual verified match —
  it's a screening heuristic, explicitly not a forensic tool.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints, upload a photo with a visible
hand (a close-up selfie works well), and try the three tabs (risk boxes,
heatmap, protected copy).

## Testing

`test_pipeline.py` validates the scoring/visualization code using a
synthetic image and mocked hand landmarks (this sandboxed environment
doesn't have network access to download a real sample photo). It confirms
a sharp/detailed region scores meaningfully higher risk than a flat/blurry
one, and that the box-drawing, heatmap, and pixelation functions all run
correctly end-to-end.

```bash
python3 test_pipeline.py
```

Before treating this as portfolio-ready, run it once against a handful of
your own real photos (different distances, lighting, focus) to sanity-check
that the risk labels match your intuition, and tune
`RISK_THRESHOLDS`/the weighting in `_detail_score` if needed.

## Detector backend: MediaPipe Tasks API

This project originally used MediaPipe's legacy `solutions.hands` API.
That API has been dropped from current MediaPipe releases entirely (it's
not importable on newer versions), which broke installation on newer
Python versions where only those newer MediaPipe wheels exist. The whole
pipeline now runs on MediaPipe's current, actively maintained **Tasks API**
(`hand_detector_v2.py`, wired into `risk_utils.py`) instead -- both for
broad compatibility, and because it's the fix attempt for a real gap found
during testing (below).

## Confirmed limitation: no usable per-landmark confidence for hands

Testing this project surfaced a real gap (see Limitations above): when a
finger like the thumb is tucked/not clearly visible, MediaPipe still
outputs a landmark position for it anyway -- guessed, not real. Several
geometric heuristics (distance from wrist, which side of the palm, segment
length) were tried against the legacy API and none reliably caught it.

The hypothesis was that MediaPipe's newer Tasks API would fix this, since
its landmark objects carry `visibility`/`presence` fields -- the mechanism
meant for exactly this, and one that's genuinely populated and useful for
MediaPipe's Pose model. **Tested directly** with `test_visibility_signal.py`
against a fully-visible-hand photo:

```
thumb : visibility=None  presence=None
index : visibility=None  presence=None
middle: visibility=None  presence=None
ring  : visibility=None  presence=None
pinky : visibility=None  presence=None
```

**Confirmed: MediaPipe's Hand Landmarker does not populate these fields at
all**, for any landmark, regardless of how visible the finger actually is.
The fields exist in the API (inherited from the landmark format shared
across Pose/Hand/Face) but simply aren't implemented for the hand model.
This closes off that fix path -- not an inconclusive result, a definitive
one.

The same test also confirmed a second finding: an extreme close-up crop
(only 1-2 fingers visible, most of the hand out of frame -- the classic
"viral risky selfie" case) still fails to be detected as a hand at all on
the Tasks API, exactly as it did on the legacy API. So that failure mode is
a genuine limitation of what MediaPipe's hand model was trained on, not an
artifact of which API version is calling it.

**What this means for the project, stated plainly:** reliably distinguishing
a genuinely-visible fingertip from a hallucinated one would need a
different approach entirely -- e.g. a small custom-trained occlusion
classifier run on each fingertip crop, or a hand-segmentation model used as
a sanity mask -- which is real future work, not a weekend fix. For now, the
honest and correct behavior is what's already implemented: score whatever
MediaPipe reports, and rely on **using photos where the hand you care about
is fully visible** for a trustworthy reading (documented in-app and above).

This is worth stating exactly this way in an interview: "I formed a
hypothesis for the fix, built it, tested it directly against real data, and
got a conclusive negative result -- which told me the actual limitation
lives one level deeper than I first thought." That's a stronger, more
specific engineering story than a demo that never surfaces the gap at all.

## Deploying (free)

- **Streamlit Community Cloud** — connect this repo at
  [streamlit.io/cloud](https://streamlit.io/cloud); it auto-installs
  `requirements.txt` and runs `app.py`.
- **Hugging Face Spaces** — create a new Space, choose the "Streamlit" SDK,
  and push this repo's files.

Either gives you a live shareable link, which matters more for a resume
project than the model itself — it proves you can ship a working product,
not just a notebook.

## Talking about this in an interview

A good way to frame it: *"I noticed a real privacy risk — fingerprints
being reconstructable from ordinary photos — and instead of building the
capability that causes the harm, I built the detector that prevents it.
The interesting engineering problem was scoring 'resolvable detail' as a
general image-quality signal rather than needing any actual fingerprint
data, which meant I could build and validate the whole thing without ever
handling sensitive biometric data."*

That's a stronger story than "I built a model that does X" — it shows
you thinking about the actual stakeholder, the actual risk, and making a
deliberate scope decision, which is exactly the judgment a DS/ML hiring
manager is trying to screen for.
