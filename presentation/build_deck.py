"""
build_deck.py
=============
Builds the 10-slide academic presentation
"CNC Tool Wear Prediction And Scrap Risk Signal"
plus the three matplotlib charts it embeds.

Every quantitative figure embedded in the deck was measured by executing this
repository's own code (seed 42) during the independent validation audit --
see docs/VALIDATION_REPORT.md, Appendix B. Paper benchmarks are quoted from:
  [P1] Dong & Li,  ASME JCISE 22(5) 050903, 2022   doi:10.1115/1.4053919
  [P2] Fan,        DAPIC 2026                      doi:10.23919/DAPIC.2026.00023
  [P3] Munaro et al., JMMP 7(4) 129, 2023          doi:10.3390/jmmp7040129

Run from the repository root:  python presentation/build_deck.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

# ---------------------------------------------------------------- palette ---
BG    = RGBColor(0x0F, 0x17, 0x2A)
CARD  = RGBColor(0x1E, 0x29, 0x3B)
CARD2 = RGBColor(0x16, 0x20, 0x2F)
BORD  = RGBColor(0x33, 0x41, 0x55)
TEXT  = RGBColor(0xF1, 0xF5, 0xF9)
DIM   = RGBColor(0x94, 0xA3, 0xB8)
BLUE  = RGBColor(0x3B, 0x82, 0xF6)
GREEN = RGBColor(0x10, 0xB9, 0x81)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
RED   = RGBColor(0xEF, 0x44, 0x44)

HB, HC, HG, HT, HD, HBL, HGR, HAM, HRED = ("#0f172a", "#1e293b", "#334155",
                                           "#f1f5f9", "#94a3b8", "#3b82f6",
                                           "#10b981", "#f59e0b", "#ef4444")
FONT = "Calibri"

# ------------------------------------------- measured during the audit -----
MEASURED = dict(
    mae=0.0132, rmse=0.0177, wear_acc=93.1,
    cov_pre=62.8, cov_post=80.9, cov_fast=90.2,
    auc=0.873, brier=0.0974,
    lat_band=6.25, lat_scrap=8.75, lat_shap=160.0,
    size_full=2.92, size_fast=0.23, train_full=14.3, train_fast=0.38,
)

# ============================================================== CHARTS ======
plt.rcParams.update({
    "figure.facecolor": HB, "axes.facecolor": HC, "savefig.facecolor": HB,
    "axes.edgecolor": HG, "axes.labelcolor": HT, "xtick.color": HD,
    "ytick.color": HD, "text.color": HT, "grid.color": HG,
    "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.35,
    "axes.axisbelow": True,
})


def chart_wear_curve():
    from data_generator import generate_dataset
    raw = generate_dataset(n_runs=80, seed=42)
    counts = raw.groupby("run_id").size()
    run = counts.idxmax()
    r = raw[raw.run_id == run].sort_values("sample_index")
    t, vb = r.runtime_min.values, r.flank_wear_mm.values

    fig, ax = plt.subplots(figsize=(6.4, 3.55), dpi=200)
    ax.axhspan(0.00, 0.05, color=HGR, alpha=0.10)
    ax.axhspan(0.05, 0.15, color=HAM, alpha=0.10)
    ax.axhspan(0.15, 0.20, color=HRED, alpha=0.10)
    ax.plot(t, vb, color=HBL, lw=2.0, label="Simulated flank wear VB (3-stage)")
    for y, c, lab in [(0.05, HGR, "[P2] Initial 0.05"), (0.15, HAM, "[P2] Intermediate 0.15"),
                      (0.20, HRED, "[P2] Final 0.20"), (0.25, HAM, "visual alert 0.25"),
                      (0.30, HRED, "ISO 3685  0.30")]:
        ax.axhline(y, color=c, ls="--", lw=1.1, alpha=0.9)
        ax.text(t[-1] * 1.03, y, f" {lab}", color=c, va="center", ha="left", fontsize=7.5,
                bbox=dict(facecolor=HC, edgecolor="none", alpha=0.88, pad=1.3))
    ax.set_xlim(0, t[-1] * 1.42)
    ax.set_ylim(0, 0.34)
    ax.set_xlabel("Elapsed cutting time (min)")
    ax.set_ylabel("Flank wear VB (mm)")
    ax.set_title("Three-stage wear progression vs literature stage limits",
                 fontsize=9.5, color=HT, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.25)
    fig.tight_layout()
    p = os.path.join(ASSETS, "chart_wear_curve.png")
    fig.savefig(p); plt.close(fig)
    return p


def chart_coverage():
    m = MEASURED
    fig, ax = plt.subplots(figsize=(5.7, 3.35), dpi=200)
    labels = ["Raw quantile band\n(pre-conformal)", "Full-precision\npost-conformal", "Shipped fast-boot\npost-conformal"]
    vals = [m["cov_pre"], m["cov_post"], m["cov_fast"]]
    cols = [HD, HBL, HGR]
    bars = ax.bar(labels, vals, color=cols, width=0.62, edgecolor=HB, lw=1.2)
    ax.axhline(80, color=HAM, ls="--", lw=1.4)
    ax.text(-0.42, 82.0, "target 80 %", color=HAM, fontsize=8, ha="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f} %",
                ha="center", fontsize=9.5, color=HT, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Empirical Q10–Q90 coverage")
    ax.set_title("Conformal calibration delivers its coverage guarantee",
                 fontsize=9.5, color=HT, loc="left")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    p = os.path.join(ASSETS, "chart_coverage.png")
    fig.savefig(p); plt.close(fig)
    return p


def chart_benchmarks():
    m = MEASURED
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.35), dpi=200,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    models = ["This work*", "A-LWDRL\n(public)", "A-LWDRL\n(45# steel)", "LSTM", "SVM"]
    mae = [m["mae"], 0.023, 0.028, 0.038, 0.057]
    rmse = [m["rmse"], 0.031, 0.036, 0.049, 0.072]
    x = np.arange(len(models)); w = 0.38
    a1.bar(x - w / 2, mae, w, color=HBL, label="MAE (mm)", edgecolor=HB)
    a1.bar(x + w / 2, rmse, w, color=HGR, label="RMSE (mm)", edgecolor=HB)
    a1.set_xticks(x); a1.set_xticklabels(models, fontsize=7.6)
    a1.set_title("Point-estimate error vs [P2] Table I", fontsize=9.5, loc="left")
    a1.legend(fontsize=8, framealpha=0.25)
    a1.set_ylim(0, 0.085)

    mods2 = ["This\nwork*", "SVM", "LSTM", "CNN-\nLSTM", "DQN", "A-\nLWDRL"]
    acc = [m["wear_acc"], 82.3, 92.5, 94.1, 93.8, 98.7]
    cols = [HBL] + [HD] * 4 + [HGR]
    a2.axhspan(84, 100, color=HGR, alpha=0.10)
    b2 = a2.bar(mods2, acc, color=cols, width=0.66, edgecolor=HB)
    for b, v in zip(b2, acc):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center",
                fontsize=7.6, color=HT)
    a2.set_ylim(0, 108)
    a2.set_title("Wear-level accuracy vs [P2] baselines\n(shaded band = [P3] validated 84–100 %)",
                 fontsize=9.5, loc="left")
    a2.tick_params(axis="x", labelsize=7.6)
    fig.tight_layout()
    p = os.path.join(ASSETS, "chart_benchmarks.png")
    fig.savefig(p); plt.close(fig)
    return p


# ============================================================ PPTX HELPERS ==
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


def new_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def tb(slide, l, t, w, h):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text="", size=13, color=TEXT, bold=False, italic=False, first=False,
         space_before=0, space_after=5, align=PP_ALIGN.LEFT, font=FONT):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if text:
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
    return p


def bullet(tf, text, size=12.5, color=TEXT, glyph="▸", gcolor=BLUE, first=False,
           space_after=6, bold_prefix=None):
    p = para(tf, "", size=size, first=first, space_after=space_after)
    r = p.add_run(); r.text = glyph + "  "
    r.font.size = Pt(size); r.font.color.rgb = gcolor; r.font.bold = True; r.font.name = FONT
    if bold_prefix:
        rb = p.add_run(); rb.text = bold_prefix
        rb.font.size = Pt(size); rb.font.bold = True; rb.font.color.rgb = color; rb.font.name = FONT
    r2 = p.add_run(); r2.text = text
    r2.font.size = Pt(size); r2.font.color.rgb = color; r2.font.name = FONT
    return p


def header(slide, eyebrow, title, n):
    tf = tb(slide, 0.55, 0.32, 11.0, 1.15)
    para(tf, eyebrow, size=10, color=BLUE, bold=True, first=True, space_after=3)
    para(tf, title, size=26, color=TEXT, bold=True, space_after=0)
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.57), Inches(1.28),
                                Inches(2.1), Pt(2.2))
    ln.fill.solid(); ln.fill.fore_color.rgb = BLUE; ln.line.fill.background()
    ln.shadow.inherit = False
    footer(slide, n)


def footer(slide, n):
    tf = tb(slide, 0.55, 7.08, 9.5, 0.32)
    para(tf, "CNC Tool Wear Prediction And Scrap Risk Signal  ·  academic project  ·  Oct 2026",
         size=8.5, color=DIM, first=True, space_after=0)
    tf2 = tb(slide, 12.0, 7.08, 0.85, 0.32)
    para(tf2, f"{n} / 10", size=8.5, color=DIM, first=True, space_after=0, align=PP_ALIGN.RIGHT)


def card(slide, l, t, w, h, fill=CARD, line=BORD):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t),
                                Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.16); tf.margin_right = Inches(0.14)
    tf.margin_top = Inches(0.10); tf.margin_bottom = Inches(0.08)
    return tf


def cover_pic(slide, path, l, t, w, h):
    iw, ih = Image.open(path).size
    a_nat, a_box = iw / ih, w / h
    pic = slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))
    if a_nat > a_box:                      # image wider than box -> crop sides
        total = 1 - a_box / a_nat
        pic.crop_left = total / 2; pic.crop_right = total / 2
    else:                                  # image taller -> crop top/bottom
        total = 1 - a_nat / a_box
        pic.crop_top = total / 2; pic.crop_bottom = total / 2
    return pic


def set_cell(cell, text, size=10.5, bold=False, color=TEXT, fill=None,
             align=PP_ALIGN.LEFT):
    if fill is not None:
        cell.fill.solid(); cell.fill.fore_color.rgb = fill
    cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.06)
    cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT


def make_table(slide, data, l, t, w, col_w, row_h=0.34, hdr_fill=CARD,
               body_fill=CARD2, sizes=None, first_col_bold=False):
    rows, cols = len(data), len(data[0])
    gf = slide.shapes.add_table(rows, cols, Inches(l), Inches(t), Inches(w),
                                Inches(row_h * rows))
    tbl = gf.table
    tbl.first_row = False; tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(rows):
        tbl.rows[i].height = Inches(row_h)
        for j in range(cols):
            is_hdr = (i == 0)
            set_cell(tbl.cell(i, j), data[i][j],
                     size=(sizes[j] if sizes else 10.5),
                     bold=is_hdr or (first_col_bold and j == 0),
                     color=TEXT if is_hdr else (TEXT if j == 0 else DIM if False else TEXT),
                     fill=hdr_fill if is_hdr else (body_fill if i % 2 else CARD),
                     align=PP_ALIGN.LEFT)
            if is_hdr:
                tbl.cell(i, j).text_frame.paragraphs[0].runs[0].font.color.rgb = TEXT
    return tbl


def box(slide, l, t, w, h, title, body, fill=CARD, line=BORD, tcolor=BLUE,
        tsize=11, bsize=9.5):
    tf = card(slide, l, t, w, h, fill, line)
    para(tf, title, size=tsize, color=tcolor, bold=True, first=True, space_after=3)
    para(tf, body, size=bsize, color=TEXT, space_after=0)
    return tf


def arrow(slide, l, t, w, h, direction="right", color=DIM):
    shp = MSO_SHAPE.RIGHT_ARROW if direction == "right" else (
        MSO_SHAPE.UP_ARROW if direction == "up" else MSO_SHAPE.DOWN_ARROW)
    a = slide.shapes.add_shape(shp, Inches(l), Inches(t), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = color
    a.line.fill.background(); a.shadow.inherit = False
    return a


# ================================================================ SLIDE 1 ===
s = new_slide()
cover_pic(s, os.path.join(ASSETS, "cnc_hero.jpg"), 7.05, 0.0, SW - 7.05, SH)
veil = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.05), Inches(0), Inches(SW - 7.05), Inches(SH))
veil.fill.solid(); veil.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
try:
    _srf = veil.fill._xPr.find(qn('a:solidFill')).find(qn('a:srgbClr'))
    _srf.append(_srf.makeelement(qn('a:alpha'), {'val': '38000'}))
except Exception:
    pass
veil.line.fill.background(); veil.shadow.inherit = False

tf = tb(s, 0.7, 0.85, 6.1, 0.4)
para(tf, "ACADEMIC PROJECT PRESENTATION  ·  2026–27", size=11, color=BLUE, bold=True, first=True)
tf = tb(s, 0.7, 1.35, 6.2, 2.2)
para(tf, "CNC Tool Wear Prediction", size=34, color=TEXT, bold=True, first=True, space_after=0)
para(tf, "And Scrap Risk Signal", size=34, color=TEXT, bold=True, space_after=8)
para(tf, "A physics + machine-learning hybrid decision-support system for "
         "Industry 4.0 predictive maintenance", size=13, color=DIM, space_after=0)
ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.72), Inches(3.62), Inches(2.4), Pt(2.6))
ln.fill.solid(); ln.fill.fore_color.rgb = BLUE; ln.line.fill.background(); ln.shadow.inherit = False

tbl = make_table(s, [
    ["Team Member", "Roll Number"],
    ["Arvapalli Sai Adarsh", "1602-24-736-039"],
    ["A. Meghana", "1602-24-736-022"],
    ["M. Muzammil", "1602-24-736-027"],
], 0.7, 4.0, 5.6, [3.1, 2.5], row_h=0.42, first_col_bold=True)

tf = tb(s, 0.7, 6.15, 6.0, 0.9)
para(tf, "Validated against:  Dong & Li (ASME JCISE 2022)  ·  Fan (DAPIC 2026)  ·  "
         "Munaro, Attanasio & Del Prete (JMMP 2023)", size=9.5, color=DIM, first=True, space_after=2)
para(tf, "October 2026", size=9.5, color=DIM, space_after=0)

# ================================================================ SLIDE 2 ===
s = new_slide()
header(s, "SECTION 1 — PROBLEM STATEMENT", "Introduction & Motivation", 2)
tf = tb(s, 0.55, 1.62, 7.55, 3.3)
bullet(tf, "Progressive flank wear (VB) degrades dimensional accuracy and surface finish; "
           "undetected wear ends in scrap, unplanned downtime and tool breakage.", bold_prefix="The failure mode:  ", first=True)
bullet(tf, "cutting tooling is 3–12 % of total production cost (Teti 2002 via [P1]; Zhang et al. via [P3]).",
       bold_prefix="The cost:  ")
bullet(tf, "≈ 20 % of nonproduction time is due to tool wear [P1 §1]; tool failure causes "
           "> 35 % of production-line malfunctions [P2 §I].", bold_prefix="The downtime:  ")
bullet(tf, "a reliable TCM system yields 10–50 % higher cutting speed, 75 % less downtime and "
           "≈ 30 % maintenance-cost saving [P3 §1].", bold_prefix="The opportunity:  ")
bullet(tf, "periodic replacement wastes usable tool life; subjective engineer judgement loses "
           "> 40 % of achievable tool-cost savings [P1 §1].", bold_prefix="The gap:  ")

tf = card(s, 0.55, 5.0, 7.55, 1.85, fill=CARD, line=BLUE)
para(tf, "PROJECT OBJECTIVES", size=11, color=BLUE, bold=True, first=True, space_after=4)
for i, t in enumerate([
        "Estimate flank wear VB (mm) with a statistically calibrated uncertainty band — never a falsely-precise single number.",
        "Signal calibrated scrap-defect risk anchored to the ISO 3685 tool-life criterion (VB = 0.30 mm).",
        "Explain every prediction with SHAP and narrate it into a plain-language operator bulletin.",
        "Remain strictly decision-support: no machine control, no autonomous action."]):
    p = para(tf, "", size=11, space_after=3)
    r = p.add_run(); r.text = f"{i+1}.  "; r.font.size = Pt(11); r.font.bold = True
    r.font.color.rgb = GREEN; r.font.name = FONT
    r2 = p.add_run(); r2.text = t; r2.font.size = Pt(11); r2.font.color.rgb = TEXT; r2.font.name = FONT

cover_pic(s, os.path.join(ASSETS, "tool_wear_macro.jpg"), 8.45, 1.62, 4.35, 4.05)
tf = tb(s, 8.45, 5.78, 4.35, 1.0)
para(tf, "Flank wear land on a carbide insert — the abrasive wear band (VB) this system "
         "predicts and the metric all three reference papers converge on.",
     size=9, color=DIM, italic=True, first=True, space_after=0)

# ================================================================ SLIDE 3 ===
s = new_slide()
header(s, "SECTION 2 — GROUND TRUTH", "Literature Review — the three validating papers", 3)
cards = [
    ("[P1]  Dong & Li — Online Detection of Turning Tool Wear Based on Machine Vision",
     "ASME JCISE 22(5) 050903, 2022  ·  doi:10.1115/1.4053919",
     "Direct online vision measurement of VB-mean on a turning tool: crop → k-means segmentation → "
     "grayscale luminance Y = 0.3R + 0.59G + 0.11B → Robert's edge operator → Hough-transform width. "
     "Micron-scale agreement with manual measurement (0.064/0.086/0.096 mm vs 0.062/0.089/0.093 mm). "
     "Supplies the economic base: tooling 3–12 % of production cost, ≈20 % nonproduction time, >40 % sensor savings."),
    ("[P2]  Fan — CNC Tool Wear Prediction Integrating Deep Reinforcement Learning & Edge Data Processing",
     "DAPIC 2026  ·  doi:10.23919/DAPIC.2026.00023",
     "Indirect sensing (PCB 356A16 accelerometer + Kistler 9257B dynamometer, 1–5 kHz) → adaptive sliding-window "
     "denoising → attention features → lightweight DRL (A-LWDRL) on a Jetson Xavier NX. Benchmarks: MAE 0.023/0.028 mm, "
     "RMSE 0.031/0.036 mm, wear-level accuracy 98.7 %, 2.3 MB, 12.8 ms. Validates wear stages 0–0.05 / 0.05–0.15 / "
     "0.15–0.2 mm on PHM 2010 (Inconel 718, WC-Co) and real 45# steel."),
    ("[P3]  Munaro, Attanasio & Del Prete — Tool Wear Monitoring with AI Methods: A Review",
     "JMMP 7(4) 129, 2023  ·  doi:10.3390/jmmp7040129  ·  77 papers, 2010–2022",
     "Sensor ranking: cutting force (41 studies) > vibration (21) > acoustic emission (18) > current/power (16) > sound (7). "
     "Features: time domain most used (RMS, mean, std, kurtosis). AI: ANN 14, SVM 10, fuzzy 5, GA/BN 4… "
     "Accuracy 84–100 % for known wear, ≈73 % for new tools. ISO 3685 sets the VB tool-life limit at 0.3 mm."),
]
y = 1.55
for title, meta, body in cards:
    tf = card(s, 0.55, y, 12.25, 1.62)
    para(tf, title, size=12.5, color=TEXT, bold=True, first=True, space_after=2)
    para(tf, meta, size=9.5, color=BLUE, bold=True, space_after=4)
    para(tf, body, size=10.5, color=DIM, space_after=0)
    y += 1.76

# ================================================================ SLIDE 4 ===
s = new_slide()
header(s, "SECTION 3 — METHOD", "System Architecture — physics + ML hybrid pipeline", 4)
row1 = [
    ("Process & tooling inputs", "23 features: material, coolant, coating, RPM, feed, ap, ae, Ø, Z, nose radius, runtime…"),
    ("Physics engine", "Extended Taylor life, Vc = πDN/1000, MRR = ap·ae·vf/1000, cutting energy, Ra ≈ f²/32rε"),
    ("Feature engineering", "Rolling-window instability features (time domain, per [P3 §4.5.2])"),
    ("Quantile ML core", "GBM Q10/Q50/Q90 on pinball loss + split-conformal coverage guarantee"),
    ("Scrap-risk classifier", "Isotonic-calibrated probability anchored at ISO 3685 VB = 0.30 mm"),
    ("XAI + bulletin", "SHAP drivers → 3-sentence operator bulletin (GPT-4o / offline fallback)"),
]
bw, gap = 1.86, 0.245
x = 0.49
for i, (t, b) in enumerate(row1):
    box(s, x, 1.62, bw, 1.62, t, b, tsize=10.5, bsize=8.6)
    if i < 5:
        arrow(s, x + bw + 0.035, 2.28, gap - 0.07, 0.30)
    x += bw + gap
arrow(s, 2.35, 3.34, 0.30, 0.52, "up")
arrow(s, 6.55, 3.34, 0.30, 0.52, "down")
arrow(s, 10.75, 3.34, 0.30, 0.52, "down")
box(s, 0.55, 3.95, 3.90, 1.05, "Offline batch training (train_pipeline.py)",
    "Three-way split grouped by tool run — no tool's telemetry leaks across train / calibration / test; "
    "persists joblib artifacts.", tsize=10.5, bsize=8.8)
box(s, 4.75, 3.95, 3.90, 1.05, "Streamlit console — four tabs",
    "Live Operations & Diagnostics · Mathematical Derivations · CNC Knowledge Base · Maintenance Logs & PDF export.",
    tsize=10.5, bsize=8.8, tcolor=GREEN)
arrow(s, 8.68, 4.34, 0.24, 0.26, "right")
box(s, 8.95, 3.95, 3.88, 1.05, "Persistence & export",
    "SQLite work-order logs survive session resets; CSV export and PDF maintenance work-order dispatch tickets.",
    tsize=10.5, bsize=8.8, tcolor=AMBER)
tf = card(s, 0.55, 5.30, 12.28, 0.95, fill=CARD2, line=BLUE)
para(tf, "SAFETY GUARDRAIL — DECISION SUPPORT ONLY", size=10.5, color=BLUE, bold=True, first=True, space_after=3)
para(tf, "No PLC / CNC write path exists anywhere in the codebase. The LLM system prompt forbids implying autonomous "
         "action; a wide uncertainty band forces a “recommend manual visual inspection” bulletin. Every output carries "
         "quantified uncertainty; every decision remains human.", size=10, color=TEXT, space_after=0)
tf = tb(s, 0.55, 6.45, 12.25, 0.5)
para(tf, "Deployment: Streamlit cloud console (what-if mode) with an offline CLI training pipeline — standard "
         "train/serve separation. Edge deployment is roadmap, not current scope.", size=9.5, color=DIM, first=True)

# ================================================================ SLIDE 5 ===
s = new_slide()
header(s, "SECTION 3 — METHOD", "Physics Layer & Three-Stage Wear Model", 5)
tf = card(s, 0.55, 1.6, 6.15, 3.5, fill=CARD)
para(tf, "FIRST-PRINCIPLES EQUATIONS (live-substituted in the app)", size=10.5, color=BLUE, bold=True, first=True, space_after=6)
for eq in ["Vc = π · D · N / 1000   [m/min]",
           "T = C·C_coolant·C_coating / ( v^(1/n) · f^(1/m) · d^(1/p) )   — extended Taylor tool life",
           "MRR = ap · ae · vf / 1000  (× N_spindles)   [cm³/min]",
           "Ra ≈ f² / (32 · rε) × 1000   [µm]      E_cum = Σ MRR·k_specific·Δt",
           "P_cut = k_specific · MRR / 60   [kW]      Load% = P_cut / (P_rated·η)"]:
    p = para(tf, "", size=11, space_after=5)
    r = p.add_run(); r.text = eq; r.font.size = Pt(10.5); r.font.name = "Consolas"
    r.font.color.rgb = TEXT
para(tf, "n ∈ [0.20, 0.50], m ∈ [0.15, 0.30], p ∈ [0.10, 0.15] per material; clipped to the calibrated "
         "envelope with an explicit extrapolation flag.", size=9, color=DIM, space_after=0)

tf = card(s, 0.55, 5.28, 6.15, 1.62, fill=CARD)
para(tf, "THREE-STAGE WEAR PROGRESSION (simulated ground truth)", size=10.5, color=GREEN, bold=True, first=True, space_after=5)
p = para(tf, "", size=10.5, space_after=4)
r = p.add_run(); r.text = "VB(t) = 0.045·(1−e^(−t/6))  +  0.22·(t/T)·κ  +  3.2·max(0, t/T−0.78)³  +  instability  +  ε"
r.font.size = Pt(10); r.font.name = "Consolas"; r.font.color.rgb = TEXT
para(tf, "break-in / steady-state / catastrophic — the canonical curve shape validated in [P2 §III.C]; "
         "break-in ceiling 0.045 mm sits within 10 % of [P2]'s 0.05 mm Initial-stage boundary.",
     size=9, color=DIM, space_after=0)

pic = cover_pic(s, chart_wear_curve(), 6.95, 1.6, 5.85, 3.5)
tf = card(s, 6.95, 5.28, 5.85, 1.62, fill=CARD2)
para(tf, "MATERIAL & TOOLING COVERAGE — A SUPERSET OF [P3]'s CORPUS", size=10, color=AMBER, bold=True, first=True, space_after=4)
para(tf, "11 profiles: Al 6061/7075-T6, AISI 1045 (C45), 4140/42CrMo4, D2, 304, 316L, 17-4 PH, Ti-6Al-4V, "
         "Inconel 718, gray cast iron FG 15/CGI 450 · coatings: uncoated carbide, TiAlN, AlTiN, PCD, ceramic, CBN "
         "· coolant: dry / MQL / flood / HPC · PCD-on-ferrous hard-blocked (carbon diffusion / TiC graphitisation).",
     size=9.3, color=TEXT, space_after=0)

# ================================================================ SLIDE 6 ===
s = new_slide()
header(s, "SECTION 3 — METHOD", "Machine-Learning Core & Calibrated Uncertainty", 6)
tf = tb(s, 0.55, 1.6, 6.55, 4.6)
bullet(tf, "gradient-boosted quantile regressors (XGBoost / LightGBM) trained on the pinball loss at "
           "τ = 0.10 / 0.50 / 0.90 — the band, not a point, is the product.", bold_prefix="Q10 / Q50 / Q90:  ", first=True)
bullet(tf, "Romano–Patterson–Candès split-conformal calibration gives a finite-sample, distribution-free "
           "80 % marginal coverage guarantee on the Q10–Q90 band.", bold_prefix="Conformal guarantee:  ")
bullet(tf, "the scrap classifier is wrapped in isotonic CalibratedClassifierCV so “0.7” genuinely means "
           "≈70 % of similar validated runs scrapped — a number an operator reads as a probability.", bold_prefix="Calibrated probability:  ")
bullet(tf, "local SHAP attributions for both models (averaged across every calibration fold), fed verbatim "
           "into the bulletin payload — the LLM may never invent a driver.", bold_prefix="Explainability:  ")
bullet(tf, "train / calibration / test split by run_id, never by row — no tool's samples leak. Stricter than "
           "[P2]'s plain 7:3 / 8:2 split.", bold_prefix="Leakage control:  ", gcolor=GREEN)
bullet(tf, "two tiers: 0.38 s fast-boot fit on first load; opt-in full-precision fit (14.3 s) for the tighter band.",
       bold_prefix="Model tiers:  ")
bullet(tf, "deterministic offline bulletin generator mirrors the GPT-4o contract — narration is never a "
           "single point of failure.", bold_prefix="Resilience:  ")
pic = cover_pic(s, chart_coverage(), 7.35, 1.7, 5.45, 3.2)
tf = card(s, 7.35, 5.1, 5.45, 1.8, fill=CARD2)
para(tf, "WHY UNCERTAINTY IS THE POINT", size=10, color=BLUE, bold=True, first=True, space_after=4)
para(tf, "[P3 §5] names noisy industrial data and small training sets as the field's central obstacle. A guaranteed "
         "coverage interval answers that directly: when the band is wide the system says so and recommends manual "
         "inspection instead of asserting false precision.", size=9.3, color=TEXT, space_after=0)

# ================================================================ SLIDE 7 ===
s = new_slide()
header(s, "SECTION 4 — RESULTS", "Measured Performance vs Literature Benchmarks", 7)
m = MEASURED
tbl = make_table(s, [
    ["Metric (held-out tool runs, seed 42)", "This work", "Benchmark"],
    ["MAE — Q50 flank wear*", f"{m['mae']:.4f} mm", "≤ 0.023–0.028 [P2]"],
    ["RMSE — Q50 flank wear*", f"{m['rmse']:.4f} mm", "≤ 0.031–0.036 [P2]"],
    ["Wear-level accuracy (3-stage)*", f"{m['wear_acc']:.1f} %", "98.7 % [P2] · 84–100 % [P3]"],
    ["Q10–Q90 coverage (target 80 %)", f"{m['cov_post']:.1f} %", "not reported in [P2]/[P3]"],
    ["Scrap-risk ROC-AUC / Brier", f"{m['auc']:.3f} / {m['brier']:.4f}", "not reported in [P2]/[P3]"],
    ["Inference latency (band / scrap)", f"{m['lat_band']:.2f} / {m['lat_scrap']:.2f} ms", "≤ 12.8 ms [P2]"],
    ["Model size (full / fast-boot)", f"{m['size_full']:.2f} / {m['size_fast']:.2f} MB", "≤ 2.3 MB [P2]"],
    ["Training time (full / fast-boot)", f"{m['train_full']:.1f} / {m['train_fast']:.2f} s", "—"],
], 0.55, 1.6, 5.75, [2.75, 1.55, 1.45], row_h=0.415, sizes=[9.6, 9.6, 9.0])
tf = tb(s, 0.55, 5.5, 5.75, 1.3)
para(tf, "* synthetic ground truth — the target is this system's own physics-informed simulator, so these "
         "figures demonstrate internal consistency, not yet real-tool skill. External validation on PHM 2010 "
         "is the top roadmap item (slide 9).", size=9, color=AMBER, italic=True, first=True, space_after=4)
para(tf, "Latency beats [P2]'s edge budget on far weaker hardware; wear-level accuracy falls inside [P3]'s "
         "validated 84–100 % envelope for known-wear tools.", size=9, color=DIM, space_after=0)
cover_pic(s, chart_benchmarks(), 6.5, 1.75, 6.35, 2.6)
tf = card(s, 6.5, 4.6, 6.35, 2.25, fill=CARD)
para(tf, "READING THE COMPARISON HONESTLY", size=10.5, color=BLUE, bold=True, first=True, space_after=4)
bullet(tf, "the * bars beat [P2]'s A-LWDRL numerically only because the target is smooth simulated wear "
           "(r(taylor_ratio, VB) = 0.949); they are not yet comparable claims.", size=9.6, gcolor=AMBER, space_after=4)
bullet(tf, "93.1 % sits above [P2]'s SVM (82.3 %), LSTM (92.5 %) and DQN (93.8 %) rows and inside [P3]'s "
           "84–100 % band — but 5.6 pts below A-LWDRL's 98.7 %.", size=9.6, space_after=4)
bullet(tf, "the fast-boot tier ships a 0.23 MB model — 10× under [P2]'s 2.3 MB edge target.", size=9.6, gcolor=GREEN, space_after=0)

# ================================================================ SLIDE 8 ===
s = new_slide()
header(s, "SECTION 5 — INDEPENDENT VALIDATION", "Cross-Validation Against the Three Papers", 8)
tbl = make_table(s, [
    ["", "Finding", "Source"],
    ["✅", "Wear metric is flank wear VB in mm — the metric all three papers converge on", "[P1] abs. · [P2] §III.C · [P3] §6.2"],
    ["✅", "Replacement / scrap threshold = 0.30 mm — exact ISO 3685 match, correctly attributed", "[P3] §6.2, p.40"],
    ["✅", "Simulated break-in ceiling 0.045 mm ≈ validated Initial-stage boundary 0.05 mm", "[P2] §III.C"],
    ["✅", "Materials & tooling coverage is a superset of the 77-paper corpus (Inconel 718, Ti-6Al-4V, C45, 42CrMo4, 316, Al 6061/7075, FG 15)", "[P3] §4.2, Table 2"],
    ["✅", "Tool-run-grouped splitting + conformal & isotonic calibration exceeds [P2]'s experimental design", "[P2] §III.B"],
    ["⚠️", "Dashboard bands (0.15 / 0.25 / 0.30) do not match validated stages (0.05 / 0.15 / 0.20); simulated VB overshoots to 0.47 mm", "[P2] §III.C"],
    ["❌", "No measured sensor signal — 0.0083 Hz telemetry vs [P2]'s 1–5 kHz; spindle power is computed, not sensed", "[P3] §4.4 · [P2] §III.A"],
    ["❌", "MAE / RMSE computable but never displayed; nine citable economic figures unused", "[P2] Table I · [P3] §4.6.4, §1"],
], 0.55, 1.5, 12.25, [0.55, 8.35, 3.35], row_h=0.47, sizes=[11, 9.8, 9.3])
for i, sym in enumerate(["", "✅", "✅", "✅", "✅", "✅", "⚠️", "❌", "❌"]):
    c = tbl.cell(i, 0)
    col = TEXT if i == 0 else (GREEN if sym == "✅" else AMBER if sym == "⚠️" else RED)
    c.text_frame.paragraphs[0].runs[0].font.color.rgb = col
    c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
tf = card(s, 0.55, 6.18, 12.25, 0.74, fill=CARD2, line=AMBER)
para(tf, "AUDIT VERDICT:  ⚠ PARTIALLY VALIDATED — 2.7 / 5.   Core methods and the output contract align with the "
         "literature; structural gaps remain (sensing, external benchmarks). No instance was found in which the "
         "system contradicts any of the three papers.", size=11, color=TEXT, bold=True, first=True, space_after=0)

# ================================================================ SLIDE 9 ===
s = new_slide()
header(s, "SECTION 6 — CLOSING", "Conclusions & Future Work", 9)
tf = tb(s, 0.55, 1.6, 7.7, 2.6)
para(tf, "CONCLUSIONS", size=12, color=GREEN, bold=True, first=True, space_after=5)
bullet(tf, "A physics + ML hybrid MVP delivers operator-grade wear and scrap-risk signals with statistically "
           "honest, calibrated uncertainty — a stronger posture than point-estimate baselines.", gcolor=GREEN, space_after=5)
bullet(tf, "The output contract is literature-aligned: VB metric, exact ISO 3685 anchor, three-stage wear shape, "
           "material/tooling superset, leakage-safe splits.", gcolor=GREEN, space_after=5)
bullet(tf, "Remaining gaps are structural and openly disclosed: no measured sensing, self-referential accuracy, "
           "hidden MAE/RMSE, absent economic framing — each with a concrete fix.", gcolor=AMBER, space_after=0)
tf = tb(s, 0.55, 4.35, 7.7, 2.5)
para(tf, "FUTURE-WORK ROADMAP", size=12, color=BLUE, bold=True, first=True, space_after=5)
for i, t in enumerate([
        "Benchmark on the public PHM 2010 corpus (WC-Co on Inconel 718, 5 kHz) for a genuine comparison against [P2] Table I.",
        "Ingest one measured channel — spindle current/power, the cheapest validated sensor class [P3 §4.4.5, §6.2].",
        "Surface MAE / RMSE / wear-level accuracy in-app and overlay [P2]'s 0.05 / 0.15 / 0.20 stage boundaries.",
        "Add the economic-justification panel: 3–12 % tool cost, ≈20 % nonproduction time, >35 % malfunctions, 10–50 / 75 / 30 % TCM benefits.",
        "Persist full-precision artifacts and move toward edge deployment with live signal ingest."]):
    p = para(tf, "", size=10.8, space_after=4)
    r = p.add_run(); r.text = f"{i+1}.  "; r.font.size = Pt(10.8); r.font.bold = True
    r.font.color.rgb = BLUE; r.font.name = FONT
    r2 = p.add_run(); r2.text = t; r2.font.size = Pt(10.8); r2.font.color.rgb = TEXT; r2.font.name = FONT
cover_pic(s, os.path.join(ASSETS, "sensor_retrofit.jpg"), 8.55, 1.6, 4.25, 3.15)
tf = tb(s, 8.55, 4.85, 4.25, 1.9)
para(tf, "Roadmap item 2: retrofit condition monitoring. [P3] shows spindle current/power is non-invasive and "
         "“far less expensive” than dynamometry — the pragmatic first channel for a student lab.",
     size=9, color=DIM, italic=True, first=True, space_after=0)

# =============================================================== SLIDE 10 ===
s = new_slide()
header(s, "SECTION 7 — TEAM", "Project Team, References & Acknowledgements", 10)
tf = tb(s, 0.55, 1.62, 5.9, 0.5)
para(tf, "TEAM MEMBERS", size=12, color=BLUE, bold=True, first=True)
make_table(s, [
    ["Team Member", "Roll Number"],
    ["Arvapalli Sai Adarsh", "1602-24-736-039"],
    ["A. Meghana", "1602-24-736-022"],
    ["M. Muzammil", "1602-24-736-027"],
], 0.55, 2.05, 5.9, [3.3, 2.6], row_h=0.46, first_col_bold=True)
tf = tb(s, 0.55, 4.35, 5.9, 2.3)
para(tf, "ACKNOWLEDGEMENTS", size=12, color=BLUE, bold=True, first=True, space_after=5)
para(tf, "The authors thank their project supervisor for guidance and for supplying the three reference "
         "papers against which this system was validated, and the reviewers of ASME JCISE, IEEE DAPIC and "
         "MDPI JMMP whose published results form the ground truth of this study.", size=11, color=TEXT, space_after=8)
para(tf, "Thank you — questions & discussion are welcome.", size=12, color=GREEN, bold=True, space_after=0)
tf = tb(s, 6.85, 1.62, 5.95, 0.5)
para(tf, "REFERENCES", size=12, color=BLUE, bold=True, first=True)
tf = tb(s, 6.85, 2.05, 5.95, 4.6)
refs = [
    ("[1]", "X. Dong and Y. Li, “Online Detection of Turning Tool Wear Based on Machine Vision,” "
            "ASME J. Computing & Information Science in Engineering, vol. 22, no. 5, 050903, 2022. doi: 10.1115/1.4053919."),
    ("[2]", "Z. Fan, “A CNC Machine Tool Wear Prediction Model Integrating Deep Reinforcement Learning and "
            "Edge Data Processing,” Proc. DAPIC, 2026. doi: 10.23919/DAPIC.2026.00023."),
    ("[3]", "R. Munaro, A. Attanasio and A. Del Prete, “Tool Wear Monitoring with Artificial Intelligence "
            "Methods: A Review,” J. Manuf. Mater. Process., vol. 7, no. 4, 129, 2023. doi: 10.3390/jmmp7040129."),
    ("[4]", "ISO 3685 — tool-life criterion VB = 0.3 mm, as cited in [3, §6.2]."),
    ("[5]", "PHM Society 2010 Tool Wear Challenge dataset — external benchmark designated for future work."),
]
first = True
for tag, txt in refs:
    p = para(tf, "", size=10, first=first, space_after=8)
    first = False
    r = p.add_run(); r.text = tag + "  "; r.font.size = Pt(10); r.font.bold = True
    r.font.color.rgb = GREEN; r.font.name = FONT
    r2 = p.add_run(); r2.text = txt; r2.font.size = Pt(10); r2.font.color.rgb = TEXT; r2.font.name = FONT
tf = card(s, 6.85, 5.6, 5.95, 1.15, fill=CARD2, line=BORD)
para(tf, "REPRODUCIBILITY", size=10, color=AMBER, bold=True, first=True, space_after=3)
para(tf, "Source, audit report and measurement harness: github.com/ArvapalliSaiAdarsh39/cnc-tool-wear-system — "
         "docs/VALIDATION_REPORT.md and audit/ (every figure in this deck is re-runnable).", size=9.3, color=TEXT, space_after=0)

out = os.path.join(HERE, "CNC_Tool_Wear_Prediction_And_Scrap_Risk_Signal.pptx")
prs.save(out)
print("saved:", out)
print("slides:", len(prs.slides._sldIdLst))
