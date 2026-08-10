"""
pdf_report.py
=============
PDF maintenance work-order export for a single machining-run snapshot:
process parameters (including coolant/coating/tool-geometry), physics/ML
metrics, SHAP drivers, and the generated operator bulletin, laid out as a
work order suitable for a shift log, a scrap-risk audit trail, or a
maintenance dispatch ticket.

Built with reportlab's Platypus flowables (SimpleDocTemplate + Paragraph +
Table). Unicode superscript characters are avoided in favor of plain-ASCII
unit notation (e.g. "mm3" not "mm superscript 3") -- ReportLab's built-in
fonts render raw Unicode super/subscript glyphs outside Latin-1 as solid
black boxes; standard Latin-1 characters like the degree sign or micro sign
are safe and used directly where needed.
"""

from __future__ import annotations

import datetime as dt
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config

_STYLES = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle(
    "ReportTitle", parent=_STYLES["Title"], fontSize=18, spaceAfter=4, textColor=colors.HexColor("#0f172a")
)
_SUBTITLE_STYLE = ParagraphStyle(
    "ReportSubtitle", parent=_STYLES["Normal"], fontSize=10, textColor=colors.HexColor("#475569"), spaceAfter=14
)
_SECTION_STYLE = ParagraphStyle(
    "SectionHeader", parent=_STYLES["Heading2"], fontSize=12.5, spaceBefore=14, spaceAfter=6,
    textColor=colors.HexColor("#0f172a"),
)
_BODY_STYLE = ParagraphStyle("Body", parent=_STYLES["Normal"], fontSize=10, leading=14)
_STATUS_COLORS = {
    config.HEALTH_STATUS_OPTIMAL: colors.HexColor("#10b981"),
    config.HEALTH_STATUS_WARNING: colors.HexColor("#f59e0b"),
    config.HEALTH_STATUS_CRITICAL: colors.HexColor("#ef4444"),
}
_DISCLAIMER_STYLE = ParagraphStyle(
    "Disclaimer", parent=_STYLES["Normal"], fontSize=8.5, leading=11.5,
    textColor=colors.HexColor("#7A2E22"), borderColor=colors.HexColor("#7A2E22"),
    borderWidth=0.75, borderPadding=6, backColor=colors.HexColor("#FBEDE9"),
)

_HEADER_ROW_COLOR = colors.HexColor("#0f172a")
_ALT_ROW_COLOR = colors.HexColor("#f1f5f9")

_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_ROW_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _ALT_ROW_COLOR]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
)


def _process_table(snapshot: dict) -> Table:
    rows = [
        ["Process Parameter", "Value"],
        ["Material", snapshot["material"]],
        ["Coolant Condition", snapshot["coolant_label"]],
        ["Tool Coating", snapshot["coating_label"]],
        ["Spindle Speed", f"{snapshot['spindle_rpm']:.0f} RPM"],
        ["Feed Rate", f"{snapshot['feed_rate_mm_rev']:.3f} mm/rev"],
        ["Axial Depth of Cut (ap)", f"{snapshot['axial_depth_mm']:.2f} mm"],
        ["Radial Width of Cut (ae)", f"{snapshot['radial_width_mm']:.2f} mm"],
        ["Workpiece Diameter", f"{snapshot['workpiece_diameter_mm']:.1f} mm"],
        ["Number of Flutes/Edges (Z)", f"{snapshot['num_flutes']}"],
        ["Number of Spindles", f"{snapshot['num_spindles']}"],
        ["Tool Nose Radius", f"{snapshot['nose_radius_mm']:.2f} mm"],
        ["Elapsed Runtime", f"{snapshot['runtime_min']:.1f} min"],
    ]
    t = Table(rows, colWidths=[2.6 * inch, 3.4 * inch])
    t.setStyle(_TABLE_STYLE)
    return t


def _physics_table(snapshot: dict) -> Table:
    rows = [
        ["Physics-Derived Metric", "Value"],
        ["Cutting Speed (Vc)", f"{snapshot['cutting_speed_m_min']:.1f} m/min"],
        ["Material Removal Rate (total)", f"{snapshot['mrr_cm3_min']:.3f} cm3/min"],
        ["Feed per Tooth", f"{snapshot['feed_per_tooth_mm']:.4f} mm"],
        ["Theoretical Surface Roughness (Ra)", f"{snapshot['surface_roughness_ra_um']:.2f} um"],
        ["Cutting Power", f"{snapshot['cutting_power_kw']:.2f} kW"],
        ["Spindle Load", f"{snapshot['spindle_load_pct']:.1f} %"],
        ["Taylor-Predicted Tool Life (effective)", f"{snapshot['taylor_predicted_life_min']:.1f} min"
                                                     + (" (extrapolated)" if snapshot.get("taylor_extrapolated") else "")],
        ["Taylor Ratio", f"{snapshot['taylor_ratio']:.3f}"],
        ["Cumulative Cutting Energy", f"{snapshot['cumulative_energy_kj']:.1f} kJ"],
    ]
    t = Table(rows, colWidths=[2.6 * inch, 3.4 * inch])
    t.setStyle(_TABLE_STYLE)
    return t


def _wear_and_risk_table(snapshot: dict) -> Table:
    rows = [
        ["Prediction", "Value"],
        ["Tool Health Status", snapshot["health_status"]],
        ["Flank Wear - Q10 (optimistic)", f"{snapshot['wear_q10']:.3f} mm"],
        ["Flank Wear - Q50 (expected)", f"{snapshot['wear_q50']:.3f} mm"],
        ["Flank Wear - Q90 (pessimistic)", f"{snapshot['wear_q90']:.3f} mm"],
        ["Scrap Threshold (reference)", f"{config.SCRAP_VB_THRESHOLD_MM:.2f} mm"],
        ["Scrap Risk Probability (calibrated)", f"{snapshot['scrap_probability'] * 100:.1f}%"],
        ["Scrap Risk Tier", snapshot["scrap_tier"]],
    ]
    t = Table(rows, colWidths=[2.6 * inch, 3.4 * inch])
    t.setStyle(_TABLE_STYLE)
    return t


def _shap_table(records: List[dict]) -> Table:
    rows = [["Feature", "Value", "SHAP Contribution", "Effect"]]
    for rec in records:
        rows.append([rec["display_name"], f"{rec['feature_value']:.4g}", f"{rec['shap_value']:+.4f}", rec["direction"]])
    t = Table(rows, colWidths=[2.5 * inch, 1.1 * inch, 1.3 * inch, 1.1 * inch])
    t.setStyle(_TABLE_STYLE)
    return t


def generate_pdf_report(
    snapshot: dict,
    bulletin_text: str,
    bulletin_source: str,
    wear_shap_records: List[dict],
    scrap_shap_records: List[dict],
    output_path: str,
) -> str:
    """Renders a full maintenance work-order PDF to `output_path` and returns
    that path. `snapshot` must contain every key referenced in the table
    helpers above (see app.py's `_build_snapshot` for the exact shape)."""
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.65 * inch, bottomMargin=0.6 * inch, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        title="CNC Tool Wear Maintenance Work Order",
    )

    story = []
    story.append(Paragraph("CNC Tool Wear &amp; Scrap Risk -- Maintenance Work Order", _TITLE_STYLE))
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated {generated_at} &middot; Decision-support output, not a control action", _SUBTITLE_STYLE))

    status = snapshot.get("health_status", config.HEALTH_STATUS_WARNING)
    status_color = _STATUS_COLORS.get(status, colors.grey)
    status_style = ParagraphStyle(
        "StatusBadge", parent=_STYLES["Normal"], fontSize=13, textColor=colors.white,
        backColor=status_color, alignment=1, borderPadding=6,
    )
    story.append(Paragraph(f"TOOL HEALTH STATUS: {status}", status_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Process Parameters", _SECTION_STYLE))
    story.append(_process_table(snapshot))

    story.append(Paragraph("Physics-Derived Metrics", _SECTION_STYLE))
    story.append(_physics_table(snapshot))

    story.append(Paragraph("ML Wear Prediction &amp; Scrap Risk", _SECTION_STYLE))
    story.append(_wear_and_risk_table(snapshot))

    story.append(Paragraph("Top SHAP Drivers -- Wear Estimate", _SECTION_STYLE))
    story.append(_shap_table(wear_shap_records))

    story.append(Paragraph("Top SHAP Drivers -- Scrap Risk", _SECTION_STYLE))
    story.append(_shap_table(scrap_shap_records))

    story.append(Paragraph("Operator Bulletin", _SECTION_STYLE))
    source_label = "Azure OpenAI (GPT-4o)" if bulletin_source == "azure_openai" else "Offline rule-based fallback"
    story.append(Paragraph(f"<i>Source: {source_label}</i>", _BODY_STYLE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(bulletin_text, _BODY_STYLE))

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "SAFETY NOTICE: This system provides decision support only. It does not control, "
            "stop, or override any machine tool. All wear and scrap-risk estimates are "
            "statistical predictions with quantified uncertainty and require human review "
            "before any maintenance or production action is taken. Physics-derived tool-life "
            "estimates are calibrated against representative parameter ranges; values flagged "
            "'extrapolated' fall outside that calibrated envelope and should be treated with "
            "additional caution.",
            _DISCLAIMER_STYLE,
        )
    )

    doc.build(story)
    return output_path


if __name__ == "__main__":
    demo_snapshot = {
        "material": "Titanium Alloy (Ti-6Al-4V)", "coolant_label": "Flood Coolant", "coating_label": "Cubic Boron Nitride (CBN)",
        "spindle_rpm": 320, "feed_rate_mm_rev": 0.11, "axial_depth_mm": 0.8, "radial_width_mm": 1.0,
        "workpiece_diameter_mm": 45.0, "num_flutes": 2, "num_spindles": 1,
        "nose_radius_mm": 0.4, "cutting_speed_m_min": 45.2, "runtime_min": 142.0,
        "taylor_predicted_life_min": 168.4, "taylor_extrapolated": False, "taylor_ratio": 0.843,
        "mrr_cm3_min": 0.0028, "feed_per_tooth_mm": 0.055, "surface_roughness_ra_um": 0.95,
        "cutting_power_kw": 0.25, "spindle_load_pct": 2.0, "cumulative_energy_kj": 1120.5,
        "wear_q10": 0.221, "wear_q50": 0.268, "wear_q90": 0.331,
        "scrap_probability": 0.62, "scrap_tier": "High", "health_status": config.HEALTH_STATUS_WARNING,
    }
    demo_wear_shap = [
        {"display_name": "Taylor Ratio (runtime / predicted life)", "feature_value": 0.843, "shap_value": 0.041, "direction": "increases"},
        {"display_name": "Feed Rate Instability (rolling std)", "feature_value": 0.006, "shap_value": 0.018, "direction": "increases"},
    ]
    demo_scrap_shap = [
        {"display_name": "Taylor Ratio (runtime / predicted life)", "feature_value": 0.843, "shap_value": 0.88, "direction": "increases"},
    ]
    path = generate_pdf_report(
        demo_snapshot,
        "Expected flank wear on this Ti-6Al-4V run is 0.268 mm, but the uncertainty band is wide "
        "and scrap risk is currently rated High (62%).",
        "fallback_template", demo_wear_shap, demo_scrap_shap, "/tmp/demo_workorder.pdf",
    )
    import os
    print("Generated:", path, os.path.getsize(path), "bytes")
