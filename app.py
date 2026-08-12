"""
app.py
======
CNC Tool Wear Prediction & Scrap Risk Signal System -- v2 Streamlit Dashboard.

Four-tab architecture:
  Tab 1: Live Operations & Diagnostics  -- synchronized inputs, metric tiles,
         tool health badge, conformalized uncertainty band, SHAP charts, bulletin
  Tab 2: Mathematical Derivations       -- every formula in LaTeX with a live
         numeric substitution walkthrough for the current inputs
  Tab 3: CNC Machining Knowledge Base   -- wear types, ISO insert codes,
         machinability reference, troubleshooting guide
  Tab 4: Maintenance Logs & PDF Exporter -- session log, CSV export, PDF
         maintenance work-order export

Design note: every numeric process parameter defaults to 0 / 0.0 on first
load and is driven by a synchronized slider + manual number-input pair (see
`synced_input()`). Because 0 is not a physically valid cutting parameter,
Tab 1 explicitly detects this "unconfigured" state and shows a guided
placeholder instead of attempting a physics/ML computation on zero inputs.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import json

import log_database as logdb

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from feature_engineering import engineer_features_for_single_observation
from llm_narrative_agent import NarrativeAgent, build_payload
from ml_models import train_full_pipeline
from pdf_report import generate_pdf_report
from physics_engine import (
    apply_life_modifiers,
    check_coating_compatibility,
    cutting_energy_kj,
    cutting_power_kw,
    cutting_speed_m_min,
    feed_per_tooth_mm,
    material_removal_rate_cm3_min,
    spindle_load_pct,
    surface_roughness_ra_um,
    table_feed_rate_mm_min,
    taylor_ratio as taylor_ratio_fn,
    taylor_tool_life,
)
from shap_engine import explain_scrap_risk, explain_wear_prediction

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CNC SENTINEL - A CNC Tool Wear & Scrap Risk Signal System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logdb.init_db()

PALETTE = {
    "bg": "#0f172a",
    "card": "#1e293b",
    "card_alt": "#243247",
    "border": "#334155",
    "text": "#f1f5f9",
    "text_dim": "#94a3b8",
    "blue": "#3b82f6",
    "green": "#10b981",
    "amber": "#f59e0b",
    "red": "#ef4444",
}

STATUS_COLORS = {
    config.HEALTH_STATUS_OPTIMAL: PALETTE["green"],
    config.HEALTH_STATUS_WARNING: PALETTE["amber"],
    config.HEALTH_STATUS_CRITICAL: PALETTE["red"],
}
RISK_TIER_COLORS = {"Low": PALETTE["green"], "Medium": PALETTE["amber"], "High": "#f97316", "Critical": PALETTE["red"]}

MATERIAL_PLACEHOLDER = "-- Select Material --"
COOLANT_PLACEHOLDER = "-- Select Coolant --"
COATING_PLACEHOLDER = "-- Select Coating --"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', -apple-system, sans-serif; }}
    .stApp {{ background-color: {PALETTE['bg']}; }}
    [data-testid="stHeader"] {{ background-color: {PALETTE['bg']}; }}

    .signal-header {{
        display: flex; align-items: baseline; gap: 0.6rem;
        border-bottom: 2px solid {PALETTE['blue']}; padding-bottom: 0.6rem; margin-bottom: 0.2rem;
    }}
    .signal-header h1 {{ font-size: 1.6rem; font-weight: 700; color: {PALETTE['text']}; margin: 0; }}
    .signal-eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: {PALETTE['blue']};
        letter-spacing: 0.1em; text-transform: uppercase;
    }}
    .guardrail-banner {{
        background-color: #1c2942; border: 1px solid {PALETTE['blue']}; border-left: 4px solid {PALETTE['blue']};
        border-radius: 4px; padding: 0.55rem 0.9rem; font-size: 0.85rem; color: {PALETTE['text']};
        margin: 0.7rem 0 1rem 0;
    }}
    .metric-card {{
        background-color: {PALETTE['card']}; border: 1px solid {PALETTE['border']}; border-radius: 6px;
        padding: 0.7rem 0.9rem; height: 100%;
    }}
    .metric-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: {PALETTE['text_dim']}; }}
    .metric-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 600; color: {PALETTE['text']}; }}
    .metric-sub {{ font-size: 0.7rem; color: {PALETTE['text_dim']}; margin-top: 0.1rem; }}
    .section-label {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; letter-spacing: 0.09em; text-transform: uppercase;
        color: {PALETTE['text_dim']}; margin: 1.2rem 0 0.4rem 0; border-bottom: 1px solid {PALETTE['border']}; padding-bottom: 0.3rem;
    }}
    .status-badge {{
        display: block; text-align: center; font-family: 'IBM Plex Mono', monospace; font-weight: 700;
        font-size: 1.15rem; padding: 0.55rem; border-radius: 6px; letter-spacing: 0.06em; color: #0f172a;
    }}
    .bulletin-box {{
        background-color: {PALETTE['card_alt']}; border: 1px solid {PALETTE['border']}; border-left: 4px solid {PALETTE['blue']};
        border-radius: 4px; padding: 0.9rem 1.1rem; font-size: 0.95rem; line-height: 1.55; color: {PALETTE['text']};
    }}
    .bulletin-source {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: {PALETTE['text_dim']}; margin-top: 0.5rem; text-transform: uppercase; }}
    .idle-panel {{
        background-color: {PALETTE['card']}; border: 1px dashed {PALETTE['border']}; border-radius: 8px;
        padding: 1.4rem; text-align: center; color: {PALETTE['text_dim']}; margin-top: 0.5rem;
    }}
    .kb-card {{
        background-color: {PALETTE['card']}; border: 1px solid {PALETTE['border']}; border-radius: 6px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
    }}
    .kb-card h4 {{ color: {PALETTE['blue']}; margin-top: 0; }}
    .safety-footer {{
        margin-top: 2rem; padding: 0.8rem 1rem; background-color: #2a1515; border: 1px solid {PALETTE['red']};
        border-radius: 4px; font-size: 0.78rem; color: {PALETTE['text_dim']}; line-height: 1.5;
    }}
    /* Selectboxes render an internal text input for search-filtering, which
       shows the browser's default I-beam cursor even though the whole
       control is click-to-select, not type-to-edit -- override on the
       widget shell and every element BaseWeb mounts inside it. */
    div[data-baseweb="select"], div[data-baseweb="select"] *,
    div[role="combobox"], div[role="combobox"] *,
    div[role="listbox"], div[role="option"],
    .stSelectbox, .stSlider, .stRadio label, .stButton button {{
        cursor: pointer !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Synchronized slider + manual number-input helper
# ---------------------------------------------------------------------------
def _get_persistent(key, default):
    """Reads from an app-managed plain dict in session_state -- NOT a widget's
    own state -- so it survives Streamlit's widget-state cleanup, which fires
    whenever a keyed widget stops being rendered for a run (exactly what
    happens to Live-Ops-page widgets while another sidebar page is active)."""
    return st.session_state.setdefault("persistent_params", {}).get(key, default)


def _set_persistent(key, value):
    st.session_state.setdefault("persistent_params", {})[key] = value


def synced_input(label, key_base, min_value, max_value, step, help_text, is_int=False, num_format=None):
    """Renders a slider and a manual number-input bound to the SAME
    underlying value: dragging the slider updates the number box and typing
    in the number box updates the slider, via on_change callbacks that copy
    the just-changed widget's value into the other widget's session_state
    key before the next rerun. Both widgets default to `min_value` (0 for
    every parameter in this app) on first load.

    The true current value is mirrored into `_get/_set_persistent`, a plain
    dict entry immune to widget-lifecycle clearing, and used to RE-SEED the
    widget on every call -- so switching sidebar pages away from Live Ops
    and back does not silently reset the user's configured parameters."""
    slider_key, number_key = f"{key_base}_slider", f"{key_base}_number"
    caster = int if is_int else float

    persisted_value = caster(_get_persistent(key_base, min_value))
    if slider_key not in st.session_state:
        st.session_state[slider_key] = persisted_value
        st.session_state[number_key] = persisted_value

    def _from_slider():
        st.session_state[number_key] = st.session_state[slider_key]
        _set_persistent(key_base, st.session_state[slider_key])

    def _from_number():
        st.session_state[slider_key] = st.session_state[number_key]
        _set_persistent(key_base, st.session_state[number_key])

    c1, c2 = st.columns([3.2, 1])
    with c1:
        st.slider(
            label, min_value=caster(min_value), max_value=caster(max_value), step=caster(step),
            key=slider_key, help=help_text, on_change=_from_slider,
        )
    with c2:
        st.number_input(
            label, min_value=caster(min_value), max_value=caster(max_value), step=caster(step),
            key=number_key, help=help_text, on_change=_from_number, label_visibility="collapsed",
            format=num_format,
        )
    _set_persistent(key_base, st.session_state[slider_key])
    return st.session_state[slider_key]


def persistent_selectbox(label, key_base, options, help_text, format_func=None):
    """Same persistent-state rationale as synced_input: an st.selectbox keyed
    on `key_base` whose selection survives being conditionally un-rendered
    when the sidebar navigates to a different page."""
    select_key = f"{key_base}_select"
    persisted_value = _get_persistent(key_base, options[0])
    if select_key not in st.session_state or st.session_state[select_key] not in options:
        st.session_state[select_key] = persisted_value if persisted_value in options else options[0]

    def _on_change():
        _set_persistent(key_base, st.session_state[select_key])

    kwargs = {"format_func": format_func} if format_func is not None else {}
    st.selectbox(label, options, key=select_key, help=help_text, on_change=_on_change, **kwargs)
    _set_persistent(key_base, st.session_state[select_key])
    return st.session_state[select_key]


# ---------------------------------------------------------------------------
# Tool insert schematic (static illustrative SVG)
# ---------------------------------------------------------------------------
def tool_insert_schematic_svg() -> str:
    return f"""
    <svg viewBox="0 0 440 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;">
      <rect width="440" height="320" fill="{PALETTE['bg']}"/>
      <text x="72" y="26" fill="{PALETTE['text']}" font-size="15" font-family="sans-serif" font-weight="bold">Single-Point Tool Insert — Cross Section</text>
      <rect x="30" y="170" width="380" height="110" fill="{PALETTE['card']}" stroke="{PALETTE['border']}" stroke-width="1.5"/>
      <text x="345" y="295" fill="{PALETTE['text_dim']}" font-size="12" font-family="sans-serif">Workpiece</text>
      <line x1="220" y1="170" x2="220" y2="50" stroke="{PALETTE['border']}" stroke-width="1" stroke-dasharray="4,3"/>
      <polygon points="220,170 190.2,66.2 238.8,63.6" fill="{PALETTE['card_alt']}" stroke="{PALETTE['blue']}" stroke-width="2" stroke-linejoin="round"/>
      <line x1="220" y1="170" x2="223.8" y2="148.3" stroke="{PALETTE['red']}" stroke-width="6" stroke-linecap="round"/>
      <path d="M 220 132 A 38 38 0 0 0 209.5 133.5" fill="none" stroke="{PALETTE['blue']}" stroke-width="1.5"/>
      <line x1="205" y1="128" x2="178" y2="112" stroke="{PALETTE['blue']}" stroke-width="1"/>
      <text x="100" y="110" fill="{PALETTE['blue']}" font-size="13" font-family="sans-serif">Rake Angle (γ)</text>
      <path d="M 220 132 A 38 38 0 0 1 226.6 132.6" fill="none" stroke="{PALETTE['green']}" stroke-width="1.5"/>
      <line x1="229" y1="129" x2="270" y2="140" stroke="{PALETTE['green']}" stroke-width="1"/>
      <text x="273" y="144" fill="{PALETTE['green']}" font-size="13" font-family="sans-serif">Clearance Angle (α)</text>
      <line x1="223.8" y1="158" x2="255" y2="185" stroke="{PALETTE['red']}" stroke-width="1"/>
      <text x="258" y="190" fill="{PALETTE['red']}" font-size="13" font-family="sans-serif" font-weight="bold">Vb — Flank Wear</text>
      <circle cx="220" cy="170" r="3.5" fill="{PALETTE['amber']}"/>
      <line x1="220" y1="170" x2="150" y2="205" stroke="{PALETTE['amber']}" stroke-width="1"/>
      <text x="60" y="212" fill="{PALETTE['amber']}" font-size="12" font-family="sans-serif">Cutting Edge</text>
      <line x1="199.3" y1="97.9" x2="150" y2="70" stroke="{PALETTE['text']}" stroke-width="1"/>
      <text x="88" y="66" fill="{PALETTE['text']}" font-size="12" font-family="sans-serif">Rake Face</text>
      <line x1="233.0" y1="96.1" x2="285" y2="72" stroke="{PALETTE['text']}" stroke-width="1"/>
      <text x="288" y="68" fill="{PALETTE['text']}" font-size="12" font-family="sans-serif">Flank Face</text>
    </svg>
    """


# ---------------------------------------------------------------------------
# Tool health status (3-state badge)
# ---------------------------------------------------------------------------
def compute_health_status(wear_q50: float, band_width: float, scrap_tier: str, coating_compatible: bool) -> str:
    if not coating_compatible:
        return config.HEALTH_STATUS_CRITICAL
    if scrap_tier in ("High", "Critical") or wear_q50 >= config.WEAR_BAND_EDGES_MM[2]:
        return config.HEALTH_STATUS_CRITICAL
    if band_width > config.UNCERTAINTY_ALERT_SPREAD_MM or scrap_tier == "Medium" or wear_q50 >= config.WEAR_BAND_EDGES_MM[0]:
        return config.HEALTH_STATUS_WARNING
    return config.HEALTH_STATUS_OPTIMAL


@st.cache_data(show_spinner=False)
def _iso_code_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Position": "1 — Shape", "Example": "C, D, S, T, V, W", "Meaning": "Insert shape (C=80° rhombic, D=55° rhombic, S=square, T=triangle, V=35° rhombic, W=trigon)"},
        {"Position": "2 — Clearance Angle", "Example": "N, C, P, B", "Meaning": "Relief/clearance angle of the insert flank (N=0°, C=7°, P=11°, B=5°, etc.)"},
        {"Position": "3 — Tolerance Class", "Example": "M, G, U", "Meaning": "Dimensional tolerance on inscribed circle, thickness, and hole position"},
        {"Position": "4 — Type", "Example": "G, M, R", "Meaning": "Fixing/chipbreaker style (G=hole+chipbreaker both sides, M=hole+chipbreaker one side, R=no hole)"},
        {"Position": "5-6 — Size", "Example": "12", "Meaning": "Nominal inscribed-circle size (mm, tenths) or cutting-edge length code"},
        {"Position": "7 — Thickness", "Example": "04", "Meaning": "Insert thickness (mm, tenths)"},
        {"Position": "8 — Nose Radius", "Example": "08", "Meaning": "Corner/nose radius (mm, hundredths) -- e.g. 08 = 0.8 mm"},
    ])


@st.cache_data(show_spinner=False)
def _materials_reference_table() -> pd.DataFrame:
    rows = [
        {
            "Material": name, "Family": m.family, "Hardness (HB)": f"{m.hardness_hb[0]:.0f}-{m.hardness_hb[1]:.0f}",
            "Machinability (%)": m.machinability_pct, "Taylor C": round(m.C, 4), "n": m.n, "m": m.m, "p": m.p,
            "Specific Energy (J/mm³)": m.specific_energy_j_mm3,
        }
        for name, m in config.MATERIALS.items()
    ]
    return pd.DataFrame(rows).sort_values("Machinability (%)", ascending=False)


@st.cache_data(show_spinner=False)
def _troubleshooting_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Symptom": "Chatter marks / vibration lines on surface", "Likely Cause": "Excessive overhang, low process rigidity, or cutting speed near a resonant frequency",
         "Remedy": "Reduce axial/radial depth and/or overhang first; then adjust speed away from the resonant band; increase clamping rigidity"},
        {"Symptom": "Thermal cracking (comb cracks perpendicular to edge)", "Likely Cause": "Cyclic thermal loading, often from interrupted cuts combined with coolant thermal shock",
         "Remedy": "Switch to dry or MQL machining for interrupted cuts, or a coating/grade rated for thermal shock resistance"},
        {"Symptom": "Poor surface finish despite correct feed/nose-radius math", "Likely Cause": "Built-up edge, tool deflection, or wear-flattened nose radius no longer matching the theoretical Ra formula",
         "Remedy": "Raise cutting speed to clear BUE; verify insert edge condition; confirm actual vs. programmed nose radius"},
        {"Symptom": "Rapid, early flank wear far under Taylor-predicted life", "Likely Cause": "Coolant/coating mismatch for the material, or parameters outside the calibrated envelope",
         "Remedy": "Check the Tab 1 compatibility and envelope warnings; confirm coolant condition matches what's physically running"},
        {"Symptom": "Sudden catastrophic edge fracture (not gradual wear)", "Likely Cause": "Mechanical shock/chipping, often axial depth too aggressive for a brittle coating (e.g. Ceramic) or an interrupted cut",
         "Remedy": "Reduce axial depth and feed on entry; for interrupted cuts, prefer a tougher coating (TiAlN/AlTiN) over brittle Ceramic"},
    ])


def _shap_bar_figure(records, title):
    records = list(reversed(records))
    names = [r["display_name"] for r in records]
    values = [r["shap_value"] for r in records]
    colors_ = [PALETTE["blue"] if v > 0 else PALETTE["green"] for v in values]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors_))
    fig.update_layout(
        height=230, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=PALETTE["card"], paper_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"], size=11, family="IBM Plex Sans"),
        title=dict(text=title, font=dict(size=12, color=PALETTE["text_dim"])),
        xaxis=dict(title="SHAP value", gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    )
    return fig

# ---------------------------------------------------------------------------
# Cached model training -- two tiers
# ---------------------------------------------------------------------------
# FAST-BOOT (default): a small, real quantile+conformal+calibrated-classifier
# fit on a micro synthetic corpus, tuned to finish in well under a second so
# the app never shows a long loading screen. Conformal calibration is still
# genuine -- with less calibration data it correctly widens the band rather
# than falsely narrowing it. See config.py Section 10 for the profiling
# rationale and exact parameters.
#
# FULL-PRECISION (opt-in, via the sidebar button): the original
# 1400-run / 350-estimator / 4-fold pipeline, unchanged from before. Kept
# available for anyone who wants the tighter, better-validated band and is
# willing to wait ~20-30s for it once per session.
@st.cache_resource(show_spinner=False)
def get_fast_boot_models():
    from data_generator import generate_dataset
    from feature_engineering import build_modeling_table

    raw = generate_dataset(n_runs=config.FAST_BOOT_N_RUNS, seed=config.RANDOM_SEED)
    table = build_modeling_table(raw)
    return train_full_pipeline(
        table, backend=config.FAST_BOOT_BACKEND, n_estimators=config.FAST_BOOT_N_ESTIMATORS,
        max_depth=config.FAST_BOOT_MAX_DEPTH, cv_folds=config.FAST_BOOT_CV_FOLDS,
        random_state=config.RANDOM_SEED,
    )


@st.cache_resource(show_spinner=False)
def get_full_precision_models(backend: str):
    from data_generator import generate_dataset
    from feature_engineering import build_modeling_table

    raw = generate_dataset(n_runs=config.FULL_PRECISION_N_RUNS, seed=config.RANDOM_SEED)
    table = build_modeling_table(raw)
    return train_full_pipeline(
        table, backend=backend, n_estimators=config.FULL_PRECISION_N_ESTIMATORS,
        max_depth=config.FULL_PRECISION_MAX_DEPTH, cv_folds=config.FULL_PRECISION_CV_FOLDS,
        random_state=config.RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Header (always visible, above sidebar-driven page content)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="signal-header">
        <span style="font-size:1.55rem;">⚙️</span>
        <h1>CNC Tool Wear &amp; Scrap Risk Signal System</h1>
    </div>
    <div class="signal-eyebrow">Predictive Maintenance &middot; Physics + ML Hybrid &middot; Decision Support Only</div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="guardrail-banner">
        ⚠ <b>Decision support only.</b> This system does not control, stop, or override any
        machine tool. Every prediction below carries quantified uncertainty. Maintenance and
        production decisions require human review.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar -- navigation + model tier + Azure status
# ---------------------------------------------------------------------------
PAGE_LIVE_OPS = "🎛️ Live Operations & Diagnostics"
PAGE_MATH = "📐 Mathematical Derivations & Calculations"
PAGE_KNOWLEDGE = "📚 CNC Knowledge Base (A-to-Z Guide)"
PAGE_LOGS = "📊 Maintenance Logs & PDF Exporter"

if "model_tier" not in st.session_state:
    st.session_state["model_tier"] = "fast"

with st.sidebar:
    st.markdown('<div class="signal-eyebrow">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate", [PAGE_LIVE_OPS, PAGE_MATH, PAGE_KNOWLEDGE, PAGE_LOGS],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="signal-eyebrow">Model Tier</div>', unsafe_allow_html=True)
    if st.session_state["model_tier"] == "fast":
        st.caption("⚡ **Fast-Boot** active — instant startup, a wider (more conservative) "
                   "conformal uncertainty band fit on a small calibration corpus.")
        upgrade_backend = st.selectbox("Backend for full-precision fit", ["xgboost", "lightgbm"], index=0)
        if st.button("🚀 Train Full-Precision Model", width="stretch"):
            with st.spinner(f"Training full-precision model on {config.FULL_PRECISION_N_RUNS} runs "
                             f"(~20-30s, one time per session)..."):
                get_full_precision_models(upgrade_backend)
            st.session_state["model_tier"] = "full"
            st.session_state["active_backend"] = upgrade_backend
            st.rerun()
    else:
        st.success(f"🎯 **Full-Precision** active ({st.session_state.get('active_backend', 'xgboost')}) "
                   f"— tighter, better-validated uncertainty band.")
        if st.button("⚡ Revert to Fast-Boot", width="stretch"):
            st.session_state["model_tier"] = "fast"
            st.rerun()

    st.markdown("---")
    llm_agent = NarrativeAgent()
    if llm_agent.is_configured:
        st.success("Azure OpenAI: connected", icon="✅")
    else:
        st.info("Azure OpenAI: offline fallback active. See .env.template.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Resolve the active model tier for this run
# ---------------------------------------------------------------------------
if st.session_state["model_tier"] == "full":
    models, training_metrics = get_full_precision_models(st.session_state.get("active_backend", "xgboost"))
    backend_choice = st.session_state.get("active_backend", "xgboost")
else:
    models, training_metrics = get_fast_boot_models()
    backend_choice = config.FAST_BOOT_BACKEND
wear_model, scrap_model = models["wear_model"], models["scrap_model"]

# ===========================================================================
# TAB 1 — LIVE OPERATIONS & DIAGNOSTICS
# ===========================================================================
if page == PAGE_LIVE_OPS:
    st.markdown('<div class="section-label">Material &amp; Process Selection</div>', unsafe_allow_html=True)
    sel_col1, sel_col2, sel_col3 = st.columns(3)
    with sel_col1:
        material_name = persistent_selectbox(
            "Material", "material", [MATERIAL_PLACEHOLDER] + list(config.MATERIALS.keys()),
            help_text="Workpiece material. Determines the Taylor tool-life constants (C, n, a, b), "
                       "specific cutting energy, hardness, and the safe RPM/feed/depth envelope used "
                       "for the guardrail check below.",
        )
    with sel_col2:
        coolant_key = persistent_selectbox(
            "Coolant Condition", "coolant", [COOLANT_PLACEHOLDER] + list(config.COOLANT_CONDITIONS.keys()),
            format_func=lambda k: config.COOLANT_CONDITIONS[k].label if k in config.COOLANT_CONDITIONS else k,
            help_text="Cutting-fluid delivery method. Dry machining runs hottest and shortens tool "
                       "life the most; High-Pressure Coolant (HPC) extends it the most by improving "
                       "chip evacuation and heat removal at the cutting zone.",
        )
    with sel_col3:
        coating_key = persistent_selectbox(
            "Tool Coating / Substrate", "coating", [COATING_PLACEHOLDER] + list(config.TOOL_COATINGS.keys()),
            format_func=lambda k: config.TOOL_COATINGS[k].label if k in config.TOOL_COATINGS else k,
            help_text="Cutting-edge coating/substrate. Multiplies base tool life (up to 4x for "
                       "Diamond/PCD). Diamond/PCD is chemically incompatible with ferrous, "
                       "nickel-based, and titanium materials -- select CBN or Ceramic for those instead.",
        )

    st.markdown('<div class="section-label">Cutting Parameters</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        spindle_rpm = synced_input(
            "Spindle Speed (RPM)", "rpm", 0, 4500, 10,
            "Rotational speed of the spindle, in revolutions per minute. Combined with "
            "workpiece diameter this sets the cutting speed Vc = πDN/1000. Higher RPM raises "
            "cutting temperature and accelerates thermal (flank/crater) wear; too low can "
            "promote built-up edge on gummy materials.",
        )
        feed_rate = synced_input(
            "Feed Rate (mm/rev)", "feed", 0.0, 0.45, 0.005,
            "Linear tool advance per spindle revolution, in mm/rev. The single strongest "
            "lever on both tool life (via the Taylor equation's f^(1/m) term) and theoretical "
            "surface finish (Ra scales with f^2) -- doubling feed roughly quadruples Ra.",
            num_format="%.3f",
        )
        workpiece_diameter = synced_input(
            "Workpiece Diameter (mm)", "diameter", 0, 120, 1,
            "Diameter of the workpiece at the current cutting position, in mm. Converts "
            "spindle RPM into actual surface (cutting) speed: Vc = πDN/1000 -- the same RPM "
            "cuts much faster on a large-diameter part than a small one.",
        )
    with p2:
        axial_depth = synced_input(
            "Axial Depth of Cut, ap (mm)", "depth", 0.0, 3.2, 0.05,
            "Engagement of the cutting edge into the workpiece measured ALONG the tool axis, "
            "in mm. Deeper ap raises cutting force, mechanical (chipping/fracture) wear risk, "
            "and cutting power -- and is one of the two volume terms in MRR = ap*ae*vf/1000.",
            num_format="%.2f",
        )
        radial_width = synced_input(
            "Radial Width of Cut, ae (mm)", "radial_width", 0.0, 3.6, 0.05,
            "Engagement of the cutting edge measured PERPENDICULAR to the tool axis (the "
            "radial step-over), in mm. The second volume term in MRR = ap*ae*vf/1000 -- wider "
            "ae removes more material per pass but raises radial force and chatter risk.",
            num_format="%.2f",
        )
        num_flutes = synced_input(
            "Number of Flutes / Engaged Edges (Z)", "flutes", 0, 8, 1,
            "Number of cutting edges simultaneously engaged in the cut. Divides the table "
            "feed rate across edges (feed per tooth, fz = vf/(N*Z)) -- more flutes at the same "
            "table feed means a lighter, gentler load per edge but finer chips.",
            is_int=True,
        )
    with p3:
        num_spindles = synced_input(
            "Number of Spindles", "spindles", 0, config.NUM_SPINDLES_RANGE[1], 1,
            "Number of simultaneously-cutting spindles on this machine (multi-spindle lathes "
            "or gang-tooled setups). Total volumetric removal scales linearly: "
            "MRR_total = N_spindles * MRR_single -- wear/life physics per edge are unchanged.",
            is_int=True,
        )
        nose_radius = synced_input(
            "Tool Nose Radius (mm)", "nose", 0.0, 1.6, 0.05,
            "Radius of the cutting-edge tip, in mm. Larger nose radii give a smoother "
            "theoretical surface finish (Ra falls with radius) and a stronger, more "
            "chip-resistant edge, at the cost of higher radial cutting force.",
            num_format="%.2f",
        )
        runtime_min = synced_input(
            "Elapsed Runtime (min)", "runtime", 0, 400, 5,
            "Cumulative cutting time on this tool/insert since it was installed. Combined "
            "with the Taylor-predicted life this forms the Taylor Ratio, the single strongest "
            "driver of predicted wear in this system.",
        )
        _steadiness_key = "steadiness_slider"
        if _steadiness_key not in st.session_state:
            st.session_state[_steadiness_key] = int(_get_persistent("steadiness", 100))

        def _sync_steadiness():
            _set_persistent("steadiness", st.session_state[_steadiness_key])

        process_steadiness = st.slider(
            "Process Steadiness (%)", 0, 100, key=_steadiness_key, on_change=_sync_steadiness,
            help="100% = a perfectly steady process. Lower values simulate chatter/instability "
                 "by widening the rolling-variance features fed to the model -- use this to see "
                 "how an unstable process elevates scrap risk independent of wear level.",
        )
        _set_persistent("steadiness", process_steadiness)

    params_configured = (
        material_name != MATERIAL_PLACEHOLDER
        and coolant_key != COOLANT_PLACEHOLDER
        and coating_key != COATING_PLACEHOLDER
        and spindle_rpm > 0 and feed_rate > 0 and axial_depth > 0 and radial_width > 0
        and workpiece_diameter > 0 and num_flutes >= 1 and num_spindles >= 1 and nose_radius > 0
    )

    if not params_configured:
        idle_col1, idle_col2 = st.columns([1, 1])
        with idle_col1:
            st.markdown(
                """<div class="idle-panel">
                        <h3 style="color:#f1f5f9;">Awaiting machining parameters</h3>
                        <p>Select a material, coolant condition, and tool coating above, then set every
                        slider above 0 to see live wear prediction, scrap risk, and SHAP diagnostics.</p>
                   </div>""",
                unsafe_allow_html=True,
            )
        with idle_col2:
            st.markdown(tool_insert_schematic_svg(), unsafe_allow_html=True)
    else:
        material = config.MATERIALS[material_name]

        out_of_envelope = []
        if not (material.rpm_range[0] <= spindle_rpm <= material.rpm_range[1]):
            out_of_envelope.append("spindle speed")
        if not (material.feed_range_mm_rev[0] <= feed_rate <= material.feed_range_mm_rev[1]):
            out_of_envelope.append("feed rate")
        if not (material.ap_range_mm[0] <= axial_depth <= material.ap_range_mm[1]):
            out_of_envelope.append("axial depth of cut")
        if not (material.ae_range_mm[0] <= radial_width <= material.ae_range_mm[1]):
            out_of_envelope.append("radial width of cut")
        if out_of_envelope:
            st.warning(
                f"⚠ **Outside recommended envelope for {material_name}:** {', '.join(out_of_envelope)}. "
                f"Safe reference range -- RPM {material.rpm_range[0]:.0f}-{material.rpm_range[1]:.0f}, "
                f"feed {material.feed_range_mm_rev[0]:.2f}-{material.feed_range_mm_rev[1]:.2f} mm/rev, "
                f"ap {material.ap_range_mm[0]:.2f}-{material.ap_range_mm[1]:.2f} mm, "
                f"ae {material.ae_range_mm[0]:.2f}-{material.ae_range_mm[1]:.2f} mm.",
                icon="⚠️",
            )

        coating_profile = config.TOOL_COATINGS[coating_key]
        is_compatible, compat_warning = check_coating_compatibility(material, coating_profile)
        if not is_compatible:
            st.error(f"🚫 **Incompatible tooling pairing.** {compat_warning}", icon="🚫")

        cutting_speed = cutting_speed_m_min(spindle_rpm, workpiece_diameter)
        base_taylor = taylor_tool_life(cutting_speed, feed_rate, axial_depth, material)
        life_mod = apply_life_modifiers(
            base_taylor.predicted_life_min, material, coolant_key, coating_key, allow_incompatible=True
        )
        effective_life = life_mod.modified_life_min
        t_ratio = taylor_ratio_fn(runtime_min, effective_life)
        vf = table_feed_rate_mm_min(feed_rate, spindle_rpm)
        mrr = material_removal_rate_cm3_min(axial_depth, radial_width, vf, num_spindles=num_spindles)
        cum_energy = cutting_energy_kj(mrr, runtime_min, material.specific_energy_j_mm3)
        fz = feed_per_tooth_mm(vf, spindle_rpm, num_flutes)
        ra = surface_roughness_ra_um(feed_rate, nose_radius)
        power_kw = cutting_power_kw(mrr, material.specific_energy_j_mm3)
        load_pct = spindle_load_pct(power_kw)

        if base_taylor.is_extrapolated:
            st.info(
                f"ℹ The Taylor tool-life equation is extrapolating beyond its calibrated range at "
                f"these parameters (raw base estimate {base_taylor.raw_life_min:,.0f} min, clipped to "
                f"{base_taylor.predicted_life_min:,.0f} min). Treat the physics-derived life as "
                f"directional; the ML quantile band below still reflects trained model behavior.",
                icon="ℹ️",
            )

        instability_frac = (100 - process_steadiness) / 100.0
        rolling_rpm_std = instability_frac * 0.08 * spindle_rpm
        rolling_feed_std = instability_frac * 0.10 * feed_rate
        rolling_depth_std = instability_frac * 0.15 * axial_depth

        live_observation = {
            "material_code": config.MATERIAL_TO_CODE[material_name],
            "coolant_code": config.COOLANT_TO_CODE[coolant_key],
            "coating_code": config.COATING_TO_CODE[coating_key],
            "spindle_rpm": spindle_rpm, "feed_rate_mm_rev": feed_rate, "axial_depth_mm": axial_depth,
            "radial_width_mm": radial_width, "workpiece_diameter_mm": workpiece_diameter,
            "num_flutes": num_flutes, "num_spindles": num_spindles, "nose_radius_mm": nose_radius,
            "cutting_speed_m_min": cutting_speed, "runtime_min": runtime_min,
            "taylor_predicted_life_min": effective_life, "taylor_ratio": t_ratio,
            "mrr_cm3_min": mrr, "cumulative_energy_kj": cum_energy,
            "feed_per_tooth_mm": fz, "surface_roughness_ra_um": ra, "spindle_load_pct": load_pct,
            "rolling_rpm_std": rolling_rpm_std, "rolling_feed_std": rolling_feed_std, "rolling_depth_std": rolling_depth_std,
        }
        feature_row = engineer_features_for_single_observation(live_observation)
        X_live = pd.DataFrame([feature_row])[config.FEATURE_COLUMNS]

        wear_pred = wear_model.predict(X_live).iloc[0]
        scrap_proba = float(scrap_model.predict_proba(X_live)[0])
        risk_tier = scrap_model.risk_tier(scrap_proba)
        band_width = wear_pred["q90"] - wear_pred["q10"]
        uncertainty_flag = band_width > config.UNCERTAINTY_ALERT_SPREAD_MM
        health_status = compute_health_status(wear_pred["q50"], band_width, risk_tier, is_compatible)

        st.markdown(
            f'<div class="status-badge" style="background-color:{STATUS_COLORS[health_status]};">'
            f'TOOL HEALTH STATUS: {health_status}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Key Metric Tiles</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        tiles = [
            ("Vb Wear (Q50)", f"{wear_pred['q50']:.3f}", "mm"),
            ("Scrap Risk (P)", f"{scrap_proba * 100:.1f}", f"% · {risk_tier}"),
            ("Surface Roughness Ra", f"{ra:.2f}", "µm"),
            ("MRR (total)", f"{mrr:,.3f}", "cm³/min"),
            ("Spindle Load", f"{load_pct:.1f}", "%"),
        ]
        for col, (label, value, sub) in zip([m1, m2, m3, m4, m5], tiles):
            col.markdown(
                f"""<div class="metric-card"><div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label">Conformalized Uncertainty Bounds</div>', unsafe_allow_html=True)
        band_col, risk_col = st.columns([1.4, 1])
        with band_col:
            good, watch, warn = config.WEAR_BAND_EDGES_MM
            axis_max = max(0.5, wear_pred["q90"] * 1.25, warn * 1.4)
            fig = go.Figure()
            zone_bounds = [0, good, watch, warn, axis_max]
            zone_colors = [PALETTE["green"], PALETTE["amber"], "#f97316", PALETTE["red"]]
            for i in range(4):
                fig.add_vrect(x0=zone_bounds[i], x1=zone_bounds[i + 1], fillcolor=zone_colors[i], opacity=0.16, layer="below", line_width=0)
            fig.add_trace(go.Scatter(x=[wear_pred["q10"], wear_pred["q90"]], y=[0, 0], mode="lines",
                                      line=dict(color=PALETTE["blue"], width=14), hovertemplate="Band: %{x:.3f} mm<extra></extra>"))
            fig.add_trace(go.Scatter(x=[wear_pred["q50"]], y=[0], mode="markers",
                                      marker=dict(color=PALETTE["text"], size=16, symbol="diamond"),
                                      hovertemplate="Q50: %{x:.3f} mm<extra></extra>"))
            fig.add_vline(x=config.SCRAP_VB_THRESHOLD_MM, line_dash="dash", line_color=PALETTE["red"], line_width=1.5,
                          annotation_text="Scrap threshold", annotation_font_color=PALETTE["red"], annotation_font_size=10)
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=30), plot_bgcolor=PALETTE["card"],
                              paper_bgcolor=PALETTE["card"], font=dict(color=PALETTE["text"]),
                              xaxis=dict(title="Flank Wear VB (mm)", range=[0, axis_max], gridcolor=PALETTE["border"]),
                              yaxis=dict(visible=False, range=[-1, 1]), showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Q10 = {wear_pred['q10']:.3f} mm · Q50 = {wear_pred['q50']:.3f} mm · "
                      f"Q90 = {wear_pred['q90']:.3f} mm · ΔQ = {band_width:.3f} mm")
            if uncertainty_flag:
                st.error(f"🔍 **High uncertainty (ΔQ = {band_width:.3f} mm > {config.UNCERTAINTY_ALERT_SPREAD_MM:.2f} mm) "
                         f"— recommend manual visual inspection of the cutting edge.**", icon="🔍")
        with risk_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=scrap_proba * 100, number={"suffix": "%", "font": {"family": "IBM Plex Mono"}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_TIER_COLORS[risk_tier]}, "bgcolor": PALETTE["card"],
                       "steps": [{"range": [0, config.RISK_TIER_EDGES[0]*100], "color": "#0f2e22"},
                                 {"range": [config.RISK_TIER_EDGES[0]*100, config.RISK_TIER_EDGES[1]*100], "color": "#332411"},
                                 {"range": [config.RISK_TIER_EDGES[1]*100, config.RISK_TIER_EDGES[2]*100], "color": "#3a2410"},
                                 {"range": [config.RISK_TIER_EDGES[2]*100, 100], "color": "#3a1414"}]},
            ))
            gauge.update_layout(height=180, margin=dict(l=20, r=20, t=15, b=10), paper_bgcolor=PALETTE["card"], font=dict(color=PALETTE["text"]))
            st.plotly_chart(gauge, width="stretch")
            st.markdown(f'<div style="text-align:center;"><span class="status-badge" style="background-color:{RISK_TIER_COLORS[risk_tier]};display:inline-block;padding:0.3rem 0.8rem;font-size:0.9rem;">{risk_tier.upper()} RISK</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Explainable AI — SHAP Feature Drivers</div>', unsafe_allow_html=True)
        wear_explanation = explain_wear_prediction(wear_model, X_live)
        scrap_explanation = explain_scrap_risk(scrap_model, X_live)
        shap_c1, shap_c2 = st.columns(2)
        with shap_c1:
            st.plotly_chart(_shap_bar_figure(wear_explanation.as_records(5), "Wear Estimate (Q50) — Top Drivers"), width="stretch")
        with shap_c2:
            st.plotly_chart(_shap_bar_figure(scrap_explanation.as_records(5), "Scrap Risk — Top Drivers"), width="stretch")

        st.markdown('<div class="section-label">Operator Bulletin</div>', unsafe_allow_html=True)
        bulletin_payload = build_payload(
            material=material_name, coolant_label=config.COOLANT_CONDITIONS[coolant_key].label,
            coating_label=coating_profile.label, coating_compatible=is_compatible, coating_warning=compat_warning,
            runtime_min=runtime_min, taylor_predicted_life_min=effective_life, taylor_ratio_value=t_ratio,
            taylor_extrapolated=base_taylor.is_extrapolated,
            wear_quantiles={"q10": wear_pred["q10"], "q50": wear_pred["q50"], "q90": wear_pred["q90"]},
            scrap_probability=scrap_proba, scrap_risk_tier=risk_tier,
            wear_shap=wear_explanation, scrap_shap=scrap_explanation,
        )
        regenerate_clicked = st.button("🔄 Regenerate Bulletin")
        if "bulletin_result" not in st.session_state or regenerate_clicked:
            st.session_state["bulletin_result"] = llm_agent.generate_bulletin(bulletin_payload)
        bulletin_result = st.session_state["bulletin_result"]
        source_label = "Azure OpenAI (GPT-4o)" if bulletin_result.source == "azure_openai" else "Offline rule-based fallback"
        st.markdown(
            f'<div class="bulletin-box">{bulletin_result.bulletin_text}'
            f'<div class="bulletin-source">Source: {source_label}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Tool Insert Reference</div>', unsafe_allow_html=True)
        st.markdown(tool_insert_schematic_svg(), unsafe_allow_html=True)

        # Stash the full current snapshot in session_state for Tab 2 / Tab 4 to reuse.
        st.session_state["current_snapshot"] = {
            "material": material_name, "coolant": coolant_key, "coating": coating_key,
            "coolant_label": config.COOLANT_CONDITIONS[coolant_key].label, "coating_label": coating_profile.label,
            "spindle_rpm": spindle_rpm, "feed_rate_mm_rev": feed_rate, "axial_depth_mm": axial_depth,
            "radial_width_mm": radial_width, "workpiece_diameter_mm": workpiece_diameter,
            "num_flutes": num_flutes, "num_spindles": num_spindles, "nose_radius_mm": nose_radius,
            "cutting_speed_m_min": cutting_speed, "table_feed_rate_mm_min": vf, "runtime_min": runtime_min,
            "taylor_predicted_life_min": effective_life, "taylor_base_life_min": base_taylor.predicted_life_min,
            "taylor_extrapolated": base_taylor.is_extrapolated, "taylor_ratio": t_ratio,
            "mrr_cm3_min": mrr, "cumulative_energy_kj": cum_energy, "feed_per_tooth_mm": fz,
            "surface_roughness_ra_um": ra, "cutting_power_kw": power_kw, "spindle_load_pct": load_pct,
            "wear_q10": wear_pred["q10"], "wear_q50": wear_pred["q50"], "wear_q90": wear_pred["q90"],
            "scrap_probability": scrap_proba, "scrap_tier": risk_tier, "health_status": health_status,
            "coating_compatible": is_compatible, "coating_warning": compat_warning,
            "bulletin_text": bulletin_result.bulletin_text, "bulletin_source": bulletin_result.source,
            "wear_shap": wear_explanation.as_records(5), "scrap_shap": scrap_explanation.as_records(5),
            "material_obj": material,
        }

# ===========================================================================
# TAB 2 — MATHEMATICAL DERIVATIONS & CALCULATIONS
# ===========================================================================
elif page == PAGE_MATH:
    st.markdown("### Formula Reference &amp; Live Substitution")
    snap = st.session_state.get("current_snapshot")

    if snap is None:
        st.info("Configure parameters in **Tab 1** to see a live numeric substitution below each "
                "formula. The formulas themselves are shown either way.", icon="ℹ️")

    st.markdown("#### 1. Cutting Speed")
    st.latex(r"v = \frac{\pi \cdot D \cdot N}{1000} \quad [\text{m/min}]")
    if snap:
        st.latex(rf"v = \frac{{\pi \cdot {snap['workpiece_diameter_mm']:.1f} \cdot {snap['spindle_rpm']:.0f}}}{{1000}} "
                 rf"= {snap['cutting_speed_m_min']:.2f}\ \text{{m/min}}")

    st.markdown("#### 2. Table Feed Rate &amp; Feed per Tooth")
    st.latex(r"v_f = f \cdot N \qquad\qquad f_z = \frac{v_f}{N \cdot Z}")
    st.caption("vf (table feed, mm/min) is the intermediate quantity both MRR and fz are defined "
               "in terms of; since vf = f*N, fz algebraically reduces to f/Z, but the two-step form "
               "matches how vf is actually programmed at the machine control.")
    if snap:
        st.latex(rf"v_f = {snap['feed_rate_mm_rev']:.3f} \cdot {snap['spindle_rpm']:.0f} = {snap['table_feed_rate_mm_min']:.1f}\ \text{{mm/min}}")
        st.latex(rf"f_z = \frac{{{snap['table_feed_rate_mm_min']:.1f}}}{{{snap['spindle_rpm']:.0f} \cdot {snap['num_flutes']}}} "
                 rf"= {snap['feed_per_tooth_mm']:.4f}\ \text{{mm}}")

    st.markdown("#### 3. Material Removal Rate (axial/radial form, multi-spindle scaled)")
    st.latex(r"\text{MRR}_{\text{single}} = \frac{a_p \cdot a_e \cdot v_f}{1000} \qquad\quad \text{MRR}_{\text{total}} = N_{\text{spindles}} \cdot \text{MRR}_{\text{single}} \quad [\text{cm}^3/\text{min}]")
    st.caption("ap = axial depth of cut, ae = radial width of cut (both mm); dividing by 1000 converts mm³/min to cm³/min.")
    if snap:
        mrr_single = snap["mrr_cm3_min"] / max(1, snap["num_spindles"])
        st.latex(
            rf"\text{{MRR}}_{{\text{{single}}}} = \frac{{{snap['axial_depth_mm']:.2f} \cdot {snap['radial_width_mm']:.2f} \cdot {snap['table_feed_rate_mm_min']:.1f}}}{{1000}} "
            rf"= {mrr_single:.4f}\ \text{{cm}}^3/\text{{min}}"
        )
        st.latex(
            rf"\text{{MRR}}_{{\text{{total}}}} = {snap['num_spindles']} \cdot {mrr_single:.4f} = {snap['mrr_cm3_min']:.4f}\ \text{{cm}}^3/\text{{min}}"
        )

    st.markdown("#### 4. Theoretical Surface Roughness")
    st.latex(r"R_a \approx \frac{f^{2}}{32 \cdot r_{\epsilon}} \times 1000 \quad [\mu m]")
    if snap:
        st.latex(
            rf"R_a = \frac{{{snap['feed_rate_mm_rev']:.3f}^2}}{{32 \cdot {snap['nose_radius_mm']:.2f}}} \times 1000 "
            rf"= {snap['surface_roughness_ra_um']:.2f}\ \mu m"
        )

    st.markdown("#### 5. Extended Taylor Tool-Life Equation")
    st.latex(r"T = \frac{C_{\text{mat}} \cdot C_{\text{coolant}} \cdot C_{\text{coating}}}{v^{1/n} \cdot f^{1/m} \cdot d^{1/p}}")
    st.caption("n ∈ [0.20, 0.50] (speed sensitivity), m ∈ [0.15, 0.30] (feed sensitivity), "
               "p ∈ [0.10, 0.15] (depth sensitivity) -- material-specific constants in config.py. "
               "d here is axial depth (ap): the equation models tool-edge engagement depth, which "
               "for a single-point/indexable edge is the axial term.")
    if snap:
        m = snap["material_obj"]
        st.markdown(
            f"**Substituting current inputs** (material = {snap['material']}, "
            f"C={m.C:.4g}, n={m.n:g}, m={m.m:g}, p={m.p:g}):"
        )
        st.latex(
            rf"T_{{\text{{base}}}} = \frac{{{m.C:.4g}}}{{{snap['cutting_speed_m_min']:.2f}^{{1/{m.n:g}}} \cdot "
            rf"{snap['feed_rate_mm_rev']:.3f}^{{1/{m.m:g}}} \cdot {snap['axial_depth_mm']:.2f}^{{1/{m.p:g}}}}} "
            rf"= {snap['taylor_base_life_min']:.1f}\ \text{{min (base, pre coolant/coating)}}"
        )
        st.latex(
            rf"T_{{\text{{effective}}}} = T_{{\text{{base}}}} \cdot C_{{\text{{coolant}}}} \cdot C_{{\text{{coating}}}} "
            rf"= {snap['taylor_predicted_life_min']:.1f}\ \text{{min}}"
        )

    st.markdown("#### 6. Cumulative Cutting Energy")
    st.latex(r"E_{\text{cum}} = \int_0^{t} \big(\text{MRR}(\tau) \cdot k_{\text{specific}}\big)\, d\tau \;\approx\; \sum_i \text{MRR}_i \cdot k_{\text{specific}} \cdot \Delta t_i")
    st.caption("Implemented as a discretized Riemann sum over telemetry samples; the live dashboard evaluates it at constant current parameters over the full elapsed runtime.")
    if snap:
        st.latex(
            rf"E_{{\text{{cum}}}} \approx {snap['mrr_cm3_min']:.4f} \cdot 1000 \cdot {snap['material_obj'].specific_energy_j_mm3:g} \cdot "
            rf"{snap['runtime_min']:.1f} / 1000 = {snap['cumulative_energy_kj']:,.1f}\ \text{{kJ}}"
        )

    st.markdown("#### 7. Pinball (Quantile) Loss Function")
    st.latex(r"\mathcal{L}_{\tau}(y, \hat{y}) = \max\big(\tau (y - \hat{y}),\ (\tau - 1)(y - \hat{y})\big)")
    st.caption(
        "Used to train the Q10/Q50/Q90 wear regressors -- minimizing this asymmetric loss at "
        "τ=0.10/0.50/0.90 yields models targeting the true 10th/50th/90th conditional percentile "
        "of flank wear, not the conditional mean."
    )
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("**Worked example** (y = 0.30 mm actual, ŷ = 0.24 mm predicted):")
        for tau in (0.10, 0.50, 0.90):
            loss = max(tau * (0.30 - 0.24), (tau - 1) * (0.30 - 0.24))
            st.latex(rf"\mathcal{{L}}_{{{tau}}}(0.30, 0.24) = {loss:.4f}")
    with ex_col2:
        if snap:
            st.markdown(f"**This run's trained pinball losses** (backend={backend_choice}, held-out test set):")
            post = training_metrics["quantile_wear_model_post_conformal"]
            st.write(f"- Q10: {training_metrics['quantile_wear_model_pre_conformal']['pinball_loss_q10']:.4f}")
            st.write(f"- Q50: {training_metrics['quantile_wear_model_pre_conformal']['pinball_loss_q50']:.4f}")
            st.write(f"- Q90: {training_metrics['quantile_wear_model_pre_conformal']['pinball_loss_q90']:.4f}")
            st.write(f"- Post-conformal Q10-Q90 coverage: {post['q10_q90_empirical_coverage']*100:.1f}% (target 80%)")

# ===========================================================================
# TAB 3 — CNC MACHINING KNOWLEDGE BASE
# ===========================================================================
elif page == PAGE_KNOWLEDGE:
    st.markdown("### CNC Machining Knowledge Base")

    st.markdown(
        "The cutting mechanics governing every phenomenon below -- shear-plane angle, "
        "friction at the chip-tool interface, thermal partitioning -- are set by the "
        "**workpiece/tool material pair and cutting parameters (v, f, d)**, not by the host "
        "machine architecture. The physics is the same whether the cut is made on a CNC "
        "lathe, a Vertical Machining Center (VMC), or a Horizontal Machining Center (HMC); "
        "only the kinematics of how v/f/d are achieved differ."
    )

    st.markdown('<div class="section-label">Types of Tool Wear</div>', unsafe_allow_html=True)
    wear_types = [
        ("Flank Wear (VB)", "Abrasive wear on the relief/flank face below the cutting edge, from rubbing "
         "against the machined surface. The primary wear mode this system predicts. ISO 3685 sets "
         f"VB = {config.SCRAP_VB_THRESHOLD_MM:.2f} mm average flank wear as the standard tool-life-end "
         f"criterion for carbide tooling; this system raises a visual-inspection alert earlier, at "
         f"VB &ge; {config.VISUAL_INSPECTION_VB_ALERT_MM:.2f} mm, so an operator has a warning before the hard limit."),
        ("Crater Wear (KT)", "A concave crater on the rake face, set back slightly from the cutting edge, "
         "caused by chip friction and diffusion at high temperature. Grows fastest at high cutting speed "
         "on tough, high-strength materials; severe crater wear weakens the edge and can lead to fracture. "
         "Tracked qualitatively here (not a separate quantitative model output)."),
        ("Notch Wear", "Localized, accelerated wear at the depth-of-cut line where the tool exits the "
         "workpiece, driven by the hardened/work-hardened layer left by the previous pass -- especially "
         "pronounced on stainless steels and superalloys due to their strong work-hardening behavior."),
        ("Built-Up Edge (BUE)", "Workpiece material welding onto the rake face under high pressure and "
         "moderate temperature, common on gummy, ductile materials (aluminum, low-carbon steel) at low "
         "cutting speed. BUE periodically breaks away, tearing the finished surface and occasionally "
         "chipping the edge -- usually cured by raising cutting speed or improving coolant/coating."),
        ("Mechanical Chipping", "Sudden loss of a small edge fragment from mechanical shock -- interrupted "
         "cuts, excessive feed, or a brittle coating (Ceramic, PCD) under impact loading. Distinct from "
         "gradual wear: chipping is a step-change, not a curve, and calls for reducing entry feed/depth "
         "or switching to a tougher coating rather than adjusting the wear model's inputs."),
    ]
    for name, desc in wear_types:
        st.markdown(f'<div class="kb-card"><h4>{name}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">ISO Insert Identification Codes (ISO 1832)</div>', unsafe_allow_html=True)
    st.markdown(
        "Standard indexable-insert designations are built from an ordered code, e.g. **CNMG 12 04 08**:"
    )
    st.dataframe(_iso_code_table(), width="stretch", hide_index=True)

    st.markdown('<div class="section-label">Machinability &amp; Taylor Constants (this system\'s reference data)</div>', unsafe_allow_html=True)
    st.dataframe(_materials_reference_table(), width="stretch", hide_index=True)
    st.caption(
        "Machinability % is a relative-to-reference index (higher = easier to machine). C/n/m/p are "
        "the direct-exponent Taylor constants used in this system's tool-life equation (n=speed, "
        "m=feed, p=depth sensitivity) -- see Tab 2 for the formula. These are illustrative simulation "
        "defaults, not certified metallurgical data -- see the README for the recalibration procedure "
        "against real ISO 3685 tool-life trials."
    )

    st.markdown('<div class="section-label">Troubleshooting Guide</div>', unsafe_allow_html=True)
    st.dataframe(_troubleshooting_table(), width="stretch", hide_index=True)

    st.markdown('<div class="section-label">Academic Benchmark Literature</div>', unsafe_allow_html=True)
    st.markdown(
        "Reference literature for this problem class (CNC tool-wear prediction + scrap-risk signaling), "
        "organized by how each paper relates to this system's design. **Every citation below was "
        "independently checked against publisher/aggregator records rather than taken on trust** -- "
        "most originally-supplied DOIs did not resolve to the stated paper (wrong digits or an "
        "embedded-year mismatch); links below are the corrected DOIs, confirmed this session. Two "
        "entries are real papers whose exact DOI could not be pinned with confidence and are listed "
        "without a link rather than with an unverified guess -- please independently confirm before "
        "citing them externally."
    )
    citations_by_category = {}
    for c in config.CITATIONS:
        citations_by_category.setdefault(c["category"], []).append(c)
    for category, entries in citations_by_category.items():
        with st.expander(f"{category}  ({len(entries)})"):
            for c in entries:
                if c["verified"] and c["doi"]:
                    link = f' &middot; <a href="https://doi.org/{c["doi"]}" target="_blank" style="color:#3b82f6;">doi:{c["doi"]}</a>'
                else:
                    link = ' &middot; <span style="color:#f59e0b;">DOI not independently confirmed -- verify before citing</span>'
                st.markdown(
                    f'<div class="kb-card"><b>{c["authors"]} ({c["year"]})</b> — {c["title"]}. '
                    f'<i>{c["journal"]}</i>{link}</div>',
                    unsafe_allow_html=True,
                )

# ===========================================================================
# TAB 4 — MAINTENANCE LOGS & PDF EXPORTER
# ===========================================================================
elif page == PAGE_LOGS:
    st.markdown("### Maintenance Logs &amp; Work-Order Export")
    st.markdown(
        """<div class="guardrail-banner" style="border-color:{amber};border-left-color:{amber};background-color:#231c0f;">
        💾 Logs are stored in a local SQLite database (<code>logs.db</code>) -- they survive closing and
        reopening this browser tab. On <b>Streamlit Community Cloud specifically</b>, this file does
        <b>not</b> survive a redeploy or restart (Community Cloud's filesystem is ephemeral by design).
        Use the backup export below before a known redeploy if that matters to you.
        </div>""".format(amber=PALETTE["amber"]),
        unsafe_allow_html=True,
    )
    snap = st.session_state.get("current_snapshot")

    # ---- CREATE -------------------------------------------------------
    st.markdown('<div class="section-label">Log a New Entry</div>', unsafe_allow_html=True)
    if snap is None:
        st.info("Configure parameters in **Tab 1** to log a reading or export a work order.", icon="ℹ️")
    else:
        log_c1, log_c2 = st.columns([1, 2])
        with log_c1:
            operator_name_input = st.text_input("Operator Name", key="new_log_operator",
                                                  help="Recorded with this log entry; editable later.")
        with log_c2:
            operator_notes_input = st.text_input("Notes (optional)", key="new_log_notes",
                                                   help="Any context for this specific reading.")
        if st.button("📋 Log This Reading to Database"):
            new_id = logdb.insert_log(snap, operator_name=operator_name_input, operator_notes=operator_notes_input)
            st.toast(f"Saved as log #{new_id} to logs.db.")
            st.rerun()

    # ---- READ -----------------------------------------------------------
    st.markdown('<div class="section-label">Saved Logs</div>', unsafe_allow_html=True)
    all_logs = logdb.get_all_logs()

    if not all_logs:
        st.caption("No logs saved yet.")
    else:
        summary_df = pd.DataFrame([
            {
                "id": r["id"], "Work Order ID": r["work_order_id"], "Logged": r["created_at"],
                "Operator": r["operator_name"], "Material": r["material"], "Coating": r["coating"],
                "Vb Q50 (mm)": round(r["wear_q50"], 3) if r["wear_q50"] is not None else None,
                "Scrap Risk": r["scrap_tier"], "Status": r["work_order_status"],
            }
            for r in all_logs
        ])
        select_event = st.dataframe(
            summary_df, width="stretch", height=240, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="logs_table",
        )
        selected_rows = select_event.selection.rows if select_event and select_event.selection else []

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_bytes = logdb.export_csv_bytes()
            st.download_button("⬇ Export All Logs (CSV)", data=csv_bytes,
                                file_name=f"cnc_maintenance_logs_{dt.date.today().isoformat()}.csv", mime="text/csv")
        with exp_col2:
            backup_bytes = logdb.export_backup_bytes()
            st.download_button("⬇ Export Database Backup (.db)", data=backup_bytes,
                                file_name=f"logs_backup_{dt.date.today().isoformat()}.db", mime="application/octet-stream",
                                help="Full logs.db file -- re-import below after a redeploy resets Community Cloud storage.")

        with st.expander("⬆️ Restore from a Database Backup"):
            st.warning("Restoring OVERWRITES all logs currently in this database.", icon="⚠️")
            uploaded_backup = st.file_uploader("Choose a .db backup file", type=["db"], key="backup_uploader")
            if uploaded_backup is not None and st.button("Confirm Restore"):
                logdb.import_backup(uploaded_backup.read())
                st.success("Database restored. Reloading...")
                st.rerun()

        # ---- Detail view / UPDATE / DELETE for the selected row ----------
        if selected_rows:
            selected_id = int(summary_df.iloc[selected_rows[0]]["id"])
            record = logdb.get_log(selected_id)

            st.markdown('<div class="section-label">Log Detail — {}</div>'.format(record["work_order_id"]), unsafe_allow_html=True)

            with st.expander("🔍 View Full Snapshot", expanded=True):
                full = json.loads(record["full_snapshot_json"]) if record["full_snapshot_json"] else {}
                detail_c1, detail_c2 = st.columns(2)
                with detail_c1:
                    st.markdown("**Process Parameters**")
                    for label, key in [
                        ("Material", "material"), ("Coolant", "coolant"), ("Coating", "coating"),
                        ("Spindle RPM", "spindle_rpm"), ("Feed (mm/rev)", "feed_rate_mm_rev"),
                        ("Axial Depth ap (mm)", "axial_depth_mm"), ("Radial Width ae (mm)", "radial_width_mm"),
                        ("Flutes", "num_flutes"), ("Spindles", "num_spindles"), ("Nose Radius (mm)", "nose_radius_mm"),
                        ("Runtime (min)", "runtime_min"),
                    ]:
                        if key in full:
                            st.write(f"- **{label}:** {full[key]}")
                with detail_c2:
                    st.markdown("**Physics &amp; ML Outputs**")
                    for label, key in [
                        ("Cutting Speed (m/min)", "cutting_speed_m_min"), ("Taylor Life (min)", "taylor_predicted_life_min"),
                        ("Taylor Ratio", "taylor_ratio"), ("MRR (cm³/min)", "mrr_cm3_min"),
                        ("Surface Roughness Ra (µm)", "surface_roughness_ra_um"),
                        ("Wear Q10/Q50/Q90 (mm)", None), ("Scrap Probability", "scrap_probability"),
                        ("Health Status", "health_status"),
                    ]:
                        if key and key in full:
                            st.write(f"- **{label}:** {full[key]}")
                        elif label.startswith("Wear Q10"):
                            st.write(f"- **{label}:** {full.get('wear_q10')} / {full.get('wear_q50')} / {full.get('wear_q90')}")
                st.markdown("**AI Operator Bulletin**")
                st.info(full.get("bulletin_text", "(not recorded)"), icon="🤖")

            # ---- UPDATE ----
            with st.expander("✏️ Edit Log"):
                edit_operator = st.text_input("Operator Name", value=record["operator_name"] or "", key=f"edit_operator_{selected_id}")
                edit_status = st.selectbox("Work Order Status", logdb.WORK_ORDER_STATUSES,
                                            index=logdb.WORK_ORDER_STATUSES.index(record["work_order_status"])
                                            if record["work_order_status"] in logdb.WORK_ORDER_STATUSES else 0,
                                            key=f"edit_status_{selected_id}")
                edit_notes = st.text_area("Operator Notes", value=record["operator_notes"] or "", key=f"edit_notes_{selected_id}")
                if st.button("💾 Save Changes", key=f"save_{selected_id}"):
                    logdb.update_log(selected_id, operator_name=edit_operator, operator_notes=edit_notes,
                                      work_order_status=edit_status)
                    st.toast("Log updated.")
                    st.rerun()

            # ---- DELETE (two-step confirmation) ----
            del_col1, del_col2 = st.columns([1, 3])
            with del_col1:
                if st.button("🗑️ Delete Log", key=f"delete_{selected_id}"):
                    st.session_state["pending_delete_id"] = selected_id
            with del_col2:
                if st.session_state.get("pending_delete_id") == selected_id:
                    st.error(f"Permanently delete {record['work_order_id']}? This cannot be undone.", icon="⚠️")
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("✅ Confirm Delete", key=f"confirm_delete_{selected_id}"):
                            logdb.delete_log(selected_id)
                            st.session_state["pending_delete_id"] = None
                            st.toast("Log deleted.")
                            st.rerun()
                    with confirm_col2:
                        if st.button("Cancel", key=f"cancel_delete_{selected_id}"):
                            st.session_state["pending_delete_id"] = None
                            st.rerun()

            # ---- PDF export for the selected log ----
            if st.button("📄 Generate PDF Work Order (selected log)", key=f"pdf_{selected_id}"):
                full = json.loads(record["full_snapshot_json"]) if record["full_snapshot_json"] else {}
                pdf_path = f"/tmp/cnc_workorder_{record['work_order_id']}.pdf"
                generate_pdf_report(
                    snapshot=full, bulletin_text=full.get("bulletin_text", ""),
                    bulletin_source=full.get("bulletin_source", "fallback_template"),
                    wear_shap_records=full.get("wear_shap", []), scrap_shap_records=full.get("scrap_shap", []),
                    output_path=pdf_path,
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇ Download PDF Work Order", data=f.read(),
                                        file_name=f"{record['work_order_id']}.pdf", mime="application/pdf",
                                        key=f"dl_pdf_{selected_id}")
        else:
            st.caption("Select a row above to view full details, edit, delete, or export a PDF for that entry.")

st.markdown(
    """
    <div class="safety-footer">
        <b>SAFETY NOTICE —</b> This system provides decision support only. It does not issue control
        commands to any PLC, CNC controller, or machine tool, and cannot perform automatic shutdown.
        All wear and scrap-risk figures are statistical estimates with quantified uncertainty. Physics
        life estimates are simulation-calibrated defaults (see docs/ARCHITECTURE.md) and should be
        validated against real tool-life trial data (e.g. ISO 3685) before use in a live plant. Any
        maintenance, inspection, or production decision remains the responsibility of a qualified human
        operator or engineer.
    </div>
    """,
    unsafe_allow_html=True,
)
