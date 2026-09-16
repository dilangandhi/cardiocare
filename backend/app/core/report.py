"""Render a completed analysis as a PDF report."""

from __future__ import annotations

from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#14181C")
MUTED = colors.HexColor("#5A6570")
RULE = colors.HexColor("#C9D2D9")
ALERT = colors.HexColor("#B3341F")

SEVERITY_COLOR = {
    "normal": colors.HexColor("#0B7A6B"),
    "monitor": colors.HexColor("#8A6A10"),
    "urgent": colors.HexColor("#B3341F"),
    "critical": colors.HexColor("#8C1C10"),
}


def _styles():
    ss = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=ss["Title"], fontSize=17, leading=21,
                             textColor=INK, alignment=0, spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=ss["Normal"], fontSize=8.5, leading=11,
                              textColor=MUTED),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10, leading=13,
                             textColor=MUTED, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13.5,
                               textColor=INK),
        "small": ParagraphStyle("small", parent=ss["Normal"], fontSize=8, leading=11,
                                textColor=MUTED),
        "finding": ParagraphStyle("finding", parent=ss["Normal"], fontSize=15, leading=19,
                                  textColor=INK),
    }


def _fmt(v, unit="", nd=0):
    if v is None:
        return "not measurable"
    if isinstance(v, bool):
        return "present" if v else "not detected"
    try:
        return f"{float(v):.{nd}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def build_report(data: dict, buffer) -> None:
    """Write a PDF report of one analysis into ``buffer``."""
    S = _styles()
    finding = data.get("finding", {}) or {}
    m = data.get("measurements", {}) or {}
    source = data.get("source", {}) or {}

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="CardioCare ECG analysis report",
    )
    story = []

    story.append(Paragraph("ECG rhythm analysis", S["h1"]))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(
        f"CardioCare &middot; generated {generated} &middot; "
        f"source: {source.get('type', 'unknown')}",
        S["sub"],
    ))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", color=RULE, thickness=0.7))
    story.append(Spacer(1, 8))

    sev = finding.get("severity", "normal")
    colour = SEVERITY_COLOR.get(sev, INK)
    name = finding.get("name", "No finding")
    conf = finding.get("confidence", 0.0) or 0.0

    if finding.get("inconclusive"):
        story.append(Paragraph(
            f'<font color="{ALERT.hexval()}">Inconclusive &mdash; manual review required</font>',
            S["finding"],
        ))
        story.append(Paragraph(
            f"Highest-scoring class: {name} ({conf * 100:.0f}% confidence).", S["body"]
        ))
    else:
        story.append(Paragraph(
            f'<font color="{colour.hexval()}">{name}</font>', S["finding"]
        ))
        story.append(Paragraph(
            f"Confidence {conf * 100:.0f}% &middot; triage {sev}", S["small"]
        ))

    note = finding.get("agreement_note")
    if note:
        story.append(Spacer(1, 4))
        story.append(Paragraph(note, S["body"]))

    # Measurements
    story.append(Paragraph("MEASUREMENTS", S["h2"]))
    rows = [
        ["Heart rate", _fmt(m.get("heart_rate_bpm"), " bpm", 0),
         "QRS duration", _fmt(m.get("qrs_duration_ms"), " ms", 0)],
        ["Beats analysed", _fmt(m.get("beat_count"), "", 0),
         "Wide complexes", _fmt((m.get("qrs_wide_fraction") or 0) * 100, "%", 0)],
        ["RR mean", _fmt(m.get("rr_mean_ms"), " ms", 0),
         "RR variation", _fmt(m.get("rr_cv"), "", 3)],
        ["RR range", f"{_fmt(m.get('rr_min_ms'), '', 0)}-{_fmt(m.get('rr_max_ms'), ' ms', 0)}",
         "RMSSD", _fmt(m.get("rmssd_ms"), " ms", 0)],
        ["P waves", _fmt(m.get("p_wave_present")),
         "P consistency", _fmt(m.get("p_wave_consistency"), "", 2)],
        ["Atrial rate",
         _fmt(m.get("atrial_rate_bpm"), " /min", 0)
         if m.get("atrial_activity_organised") else "no organised activity",
         "Atrial organisation", _fmt(m.get("atrial_organisation"), "", 2)],
        ["Strip duration", _fmt(m.get("duration_s"), " s", 1),
         "Signal quality", _fmt((m.get("signal_quality") or 0) * 100, "%", 0)],
    ]
    t = Table(rows, colWidths=[34 * mm, 36 * mm, 38 * mm, 36 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("TEXTCOLOR", (3, 0), (3, -1), INK),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
    ]))
    story.append(t)

    # Evidence
    evidence = finding.get("evidence") or []
    if evidence:
        story.append(Paragraph("SUPPORTING CRITERIA", S["h2"]))
        for e in evidence:
            story.append(Paragraph(f"&bull;&nbsp; {e}", S["body"]))

    against = finding.get("contradicting") or []
    if against:
        story.append(Paragraph("CRITERIA NOT MET", S["h2"]))
        for e in against:
            story.append(Paragraph(f"&bull;&nbsp; {e}", S["small"]))

    # Differential
    diff = finding.get("differential") or []
    if diff:
        story.append(Paragraph("DIFFERENTIAL", S["h2"]))
        drows = [["Rhythm", "Probability"]] + [
            [d.get("name", d.get("key", "")), f"{(d.get('probability') or 0) * 100:.1f}%"]
            for d in diff
        ]
        dt = Table(drows, colWidths=[100 * mm, 30 * mm])
        dt.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("FONTNAME", (0, 1), (0, 1), "Helvetica-Bold"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ]))
        story.append(dt)

    # Provenance
    story.append(Paragraph("PROVENANCE", S["h2"]))
    model = finding.get("model", {}) or {}
    lines = [
        f"Analysis path: {finding.get('agreement', 'unknown').replace('_', ' ')}",
        f"Model: {model.get('name', 'none')}"
        + (f" (sha256 {model['sha256']})" if model.get("sha256") else ""),
    ]
    if model.get("note"):
        lines.append(model["note"])
    digi = (source.get("digitization") or {})
    if digi:
        lines.append(
            f"Digitisation: {digi.get('dpi', '?')} DPI, "
            f"{digi.get('coverage', 0) * 100:.0f}% trace coverage, "
            f"rotation corrected {digi.get('rotation_deg', 0)}°"
        )
        for w in digi.get("warnings", []):
            lines.append(f"Warning: {w}")
    for note in (m.get("quality_notes") or []):
        lines.append(f"Signal quality: {note}")
    for ln in lines:
        story.append(Paragraph(ln, S["small"]))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=RULE, thickness=0.7))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f'<font color="{ALERT.hexval()}"><b>Research use only. Not a medical device.</b></font> '
        "This report is generated by an automated research prototype and has not been "
        "reviewed or cleared by any regulatory authority. It must not be used for "
        "diagnosis or to guide treatment. Any clinical decision requires interpretation "
        "by a qualified clinician.",
        S["small"],
    ))

    doc.build(story)
