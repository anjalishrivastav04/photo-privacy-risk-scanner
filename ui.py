"""
ui.py
Presentation layer for the Photo Privacy Risk Scanner.

Visual direction: a forensic/security console. Near-black surface, hairline
panels with corner brackets, monospace readouts for anything numeric, and a
circular gauge as the headline result. The design commits to dark, so the
palette is declared outright rather than adapting to Streamlit's theme.
"""

import streamlit as st

# ---------------------------------------------------------------- palette --
BG = "#0a0e14"
PANEL = "#111823"
LINE = "#1e2a3a"
TEXT = "#c9d6e5"
DIM = "#5f7186"

RISK_STYLES = {
    "low": {"label": "LOW", "color": "#00e5a0", "glow": "rgba(0,229,160,0.35)"},
    "medium": {"label": "MEDIUM", "color": "#ffb020", "glow": "rgba(255,176,32,0.35)"},
    "high": {"label": "HIGH", "color": "#ff4d5e", "glow": "rgba(255,77,94,0.35)"},
    "unknown": {"label": "NO READING", "color": "#5f7186", "glow": "rgba(95,113,134,0.3)"},
}

MONO = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace"

CSS = f"""
<style>
  .stApp {{ background: {BG}; }}
  .block-container {{ padding-top: 1.4rem; max-width: 940px; }}
  .stApp, .stApp p, .stApp li, .stApp label {{ color: {TEXT}; }}
  /* Streamlit's own top bar defaults to light; blend it into the console. */
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbar"] {{ right: .6rem; }}

  /* ---------- masthead ---------- */
  .cns-bar {{
      display:flex; align-items:center; gap:.6rem;
      font-family:{MONO}; font-size:.68rem; letter-spacing:.22em;
      color:{DIM}; text-transform:uppercase; margin-bottom:.55rem;
  }}
  .cns-bar .dot {{
      width:7px; height:7px; border-radius:50%; background:#00e5a0;
      box-shadow:0 0 8px rgba(0,229,160,.9); animation:cns-pulse 2s infinite;
  }}
  @keyframes cns-pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.25}} }}

  .cns-title {{
      font-size:2.05rem; font-weight:750; letter-spacing:-.025em;
      line-height:1.08; margin:0 0 .5rem 0; color:#eaf2ff;
  }}
  .cns-title span {{ color:#00e5a0; }}
  .cns-sub {{ color:{DIM}; font-size:.93rem; line-height:1.6; max-width:64ch; }}

  .cns-rule {{
      height:1px; margin:1.5rem 0 1.3rem 0;
      background:linear-gradient(90deg,{LINE},transparent);
  }}

  /* ---------- panel ---------- */
  .cns-panel {{
      position:relative; background:{PANEL}; border:1px solid {LINE};
      border-radius:4px; padding:1.15rem 1.25rem; height:100%;
  }}
  .cns-panel::before, .cns-panel::after {{
      content:""; position:absolute; width:9px; height:9px;
      border-color:#2b3b50; border-style:solid;
  }}
  .cns-panel::before {{ top:-1px; left:-1px; border-width:1px 0 0 1px; }}
  .cns-panel::after  {{ bottom:-1px; right:-1px; border-width:0 1px 1px 0; }}

  .cns-ltitle {{
      font-family:{MONO}; font-size:.64rem; letter-spacing:.2em;
      text-transform:uppercase; color:{DIM}; margin-bottom:.7rem;
  }}
  .cns-val {{ font-size:1.22rem; font-weight:700; letter-spacing:-.01em; }}
  .cns-note {{ font-size:.85rem; color:{DIM}; line-height:1.55; margin-top:.35rem; }}

  /* ---------- verdict ---------- */
  .cns-verdict {{
      display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;
      background:{PANEL}; border:1px solid var(--c); border-radius:4px;
      padding:1.3rem 1.5rem; position:relative; overflow:hidden;
  }}
  .cns-verdict::after {{
      content:""; position:absolute; inset:0;
      background:radial-gradient(120% 100% at 0% 50%, var(--g), transparent 62%);
      pointer-events:none;
  }}
  .cns-vtext {{ position:relative; z-index:1; min-width:230px; flex:1; }}
  .cns-vlabel {{
      font-family:{MONO}; font-size:1.65rem; font-weight:700;
      letter-spacing:.06em; color:var(--c); line-height:1.1;
  }}
  .cns-vkicker {{
      font-family:{MONO}; font-size:.63rem; letter-spacing:.22em;
      color:{DIM}; text-transform:uppercase; margin-bottom:.3rem;
  }}
  .cns-vnote {{ font-size:.9rem; color:{TEXT}; opacity:.82; margin-top:.5rem; line-height:1.55; }}

  /* ---------- readout rows ---------- */
  .cns-row {{
      display:flex; align-items:center; gap:.8rem; font-family:{MONO};
      font-size:.76rem; padding:.32rem 0; border-bottom:1px dashed rgba(30,42,58,.75);
  }}
  .cns-row:last-child {{ border-bottom:0; }}
  .cns-k {{ color:{DIM}; width:112px; letter-spacing:.08em; text-transform:uppercase; flex:none; }}
  .cns-n {{ color:{TEXT}; width:44px; text-align:right; flex:none; }}
  .cns-track {{ flex:1; height:5px; background:#182231; border-radius:2px; overflow:hidden; }}
  .cns-fill {{ height:100%; border-radius:2px; }}

  /* ---------- scanning state ---------- */
  .cns-scan {{
      font-family:{MONO}; font-size:.72rem; letter-spacing:.2em; color:#00e5a0;
      text-transform:uppercase; border:1px solid {LINE}; border-radius:4px;
      padding:.85rem 1rem; position:relative; overflow:hidden; background:{PANEL};
  }}
  .cns-scan::after {{
      content:""; position:absolute; top:0; left:-45%; width:45%; height:100%;
      background:linear-gradient(90deg,transparent,rgba(0,229,160,.16),transparent);
      animation:cns-sweep 1.15s linear infinite;
  }}
  @keyframes cns-sweep {{ to {{ left:105%; }} }}

  /* ---------- streamlit chrome ---------- */
  .stTabs [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{
      font-family:{MONO}; font-size:.72rem; letter-spacing:.14em;
      text-transform:uppercase; color:{DIM};
  }}
  .stTabs [aria-selected="true"] {{ color:#00e5a0 !important; }}
  /* Streamlit's selected-tab underline (react-aria) defaults to red. */
  .stTabs .react-aria-SelectionIndicator {{ background-color:#00e5a0 !important; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color:#00e5a0 !important; }}
  div[data-testid="stExpander"] details {{
      background:{PANEL}; border:1px solid {LINE}; border-radius:4px;
  }}
  .stDownloadButton button {{
      font-family:{MONO}; letter-spacing:.12em; text-transform:uppercase;
      font-size:.74rem; border:1px solid #00e5a0; color:#00e5a0;
      background:rgba(0,229,160,.07); border-radius:3px;
  }}
  .stDownloadButton button:hover {{ background:rgba(0,229,160,.16); color:#eaf2ff; }}
  section[data-testid="stFileUploaderDropzone"] {{
      background:{PANEL}; border:1px dashed #2b3b50; border-radius:4px;
  }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def masthead(status="system ready"):
    # Single-line HTML: indented lines would be parsed as a markdown code block.
    st.markdown(
        f'<div class="cns-bar"><span class="dot"></span>{status}</div>'
        f'<div class="cns-title">PHOTO PRIVACY <span>RISK SCANNER</span></div>'
        f'<div class="cns-sub">Two independent exposure checks on a single '
        f"image — resolvable fingerprint ridge detail, and location/device "
        f"metadata embedded in the file. All processing is local; nothing is "
        f'uploaded or retained.</div>'
        f'<div class="cns-rule"></div>',
        unsafe_allow_html=True,
    )


def scanning(text="analysing image"):
    return st.markdown(f'<div class="cns-scan">▚ {text}…</div>', unsafe_allow_html=True)


def _gauge_svg(score, color, size=124):
    """Circular progress gauge drawn as inline SVG.

    NOTE: emitted as ONE line with no leading whitespace. Streamlit renders
    markdown before HTML, and any line indented four or more spaces becomes a
    fenced code block instead of markup.
    """
    r = size / 2 - 11
    circ = 2 * 3.14159265 * r
    filled = circ * max(0.0, min(1.0, score))
    c = size / 2
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'style="flex:none;position:relative;z-index:1;">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="#1b2635" stroke-width="7"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="7" '
        f'stroke-linecap="round" stroke-dasharray="{filled:.2f} {circ:.2f}" '
        f'transform="rotate(-90 {c} {c})"/>'
        f'<text x="{c}" y="{c + 2}" text-anchor="middle" dominant-baseline="middle" '
        f'fill="{color}" font-family="{MONO}" font-size="26" font-weight="700">{score:.2f}</text>'
        f'<text x="{c}" y="{c + 26}" text-anchor="middle" fill="{DIM}" '
        f'font-family="{MONO}" font-size="8.5" letter-spacing="2">SCORE</text>'
        f"</svg>"
    )


def verdict(risk, score=None, kicker="fingerprint exposure", note=""):
    s = RISK_STYLES.get(risk, RISK_STYLES["unknown"])
    gauge = _gauge_svg(score if score is not None else 0.0, s["color"])
    st.markdown(
        f'<div class="cns-verdict" style="--c:{s["color"]};--g:{s["glow"]};">'
        f"{gauge}"
        f'<div class="cns-vtext">'
        f'<div class="cns-vkicker">{kicker}</div>'
        f'<div class="cns-vlabel">{s["label"]}</div>'
        f'<div class="cns-vnote">{note}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def panel(title, value, value_color, note):
    st.markdown(
        f'<div class="cns-panel">'
        f'<div class="cns-ltitle">{title}</div>'
        f'<div class="cns-val" style="color:{value_color};">{value}</div>'
        f'<div class="cns-note">{note}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def readouts(rows, title=None):
    """rows: list of (label, numeric_value_0_1, color)."""
    head = f'<div class="cns-ltitle">{title}</div>' if title else ""
    body = "".join(
        f'<div class="cns-row">'
        f'<span class="cns-k">{k}</span>'
        f'<span class="cns-n">{v:.2f}</span>'
        f'<span class="cns-track"><span class="cns-fill" '
        f'style="width:{max(0.0, min(1.0, v)) * 100:.0f}%; background:{c};"></span></span>'
        f"</div>"
        for k, v, c in rows
    )
    st.markdown(f'<div class="cns-panel">{head}{body}</div>', unsafe_allow_html=True)


def rule():
    st.markdown('<div class="cns-rule"></div>', unsafe_allow_html=True)


def formula(text):
    st.markdown(
        f'<div style="font-family:{MONO}; font-size:.78rem; color:#00e5a0; '
        f'background:rgba(0,229,160,.06); border:1px solid rgba(0,229,160,.22); '
        f'border-radius:3px; padding:.55rem .8rem; display:inline-block; '
        f'margin:.2rem 0 .7rem 0; letter-spacing:.03em;">{text}</div>',
        unsafe_allow_html=True,
    )