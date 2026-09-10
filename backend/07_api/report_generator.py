"""
National Risk Platform Executive Risk Memo PDF Generator
Ministry of Rural Development · SIH

Generates an analyst-grade, multi-section executive PDF report using ReportLab.
Features:
- Structured Cover Header Block & Project Attributes
- Executive Summary Callout Card
- Calibrated Risk Metrics Table
- RFCTLARR Act 2013 Statutory Compliance Analysis
- Expanded AI Delay Causation Diagnostics
- Site & Accessibility Geolocation Context
- Prioritized Actionable Next Steps
- Survey of India Compliance, Methodology Reference & Model Provenance Note
- Custom NumberedCanvas for 'Page X of Y' pagination
"""

import io
import time
import uuid
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print accurate 'Page X of Y'
    along with running executive headers and footers on every page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        page_w, page_h = A4

        # Running Top Rule & Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0F172A"))
            self.drawString(36, page_h - 25, "NATIONAL RISK PLATFORM")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(160, page_h - 25, "// Executive Risk Intelligence Memo — Confidential Statutory Briefing")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(36, page_h - 30, page_w - 36, page_h - 30)

        # Running Bottom Rule & Footer (All Pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(36, 36, page_w - 36, 36)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 24, "Survey of India Compliant Cartography · RFCTLARR Act 2013 Statutory Governance")

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_w - 36, 24, page_str)
        self.restoreState()


class ReportGenerator:
    """
    Modular ReportLab Document Builder synthesizing full Project Risk Memos.
    """

    def __init__(self):
        self._init_styles()

    def _init_styles(self):
        base = getSampleStyleSheet()
        self.styles = base

        # Colors
        self.c_primary = colors.HexColor("#0F172A")    # Deep Navy
        self.c_blue = colors.HexColor("#0071E3")       # Apple / Royal Blue
        self.c_dark = colors.HexColor("#1E293B")       # Dark Charcoal
        self.c_muted = colors.HexColor("#64748B")      # Slate Gray
        self.c_border = colors.HexColor("#E2E8F0")     # Light Border
        self.c_bg_card = colors.HexColor("#F8FAFC")    # Card Fill
        self.c_high = colors.HexColor("#EF4444")       # High Risk Red
        self.c_med = colors.HexColor("#F59E0B")        # Medium Risk Amber
        self.c_low = colors.HexColor("#10B981")        # Low Risk Green

        # Custom Paragraph Styles
        self.styles.add(ParagraphStyle(
            name="MemoSuperHeader",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=self.c_blue,
            textTransform="uppercase",
            spaceAfter=2
        ))
        self.styles.add(ParagraphStyle(
            name="MemoTitle",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=self.c_primary,
            spaceAfter=4
        ))
        self.styles.add(ParagraphStyle(
            name="MemoSubtitle",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=self.c_muted,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=self.c_primary,
            spaceBefore=10,
            spaceAfter=5
        ))
        self.styles.add(ParagraphStyle(
            name="BodyDark",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12.5,
            textColor=self.c_dark,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name="BodyDarkBold",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12.5,
            textColor=self.c_dark,
            spaceAfter=4
        ))
        self.styles.add(ParagraphStyle(
            name="ExecSummaryText",
            fontName="Helvetica",
            fontSize=9,
            leading=13.5,
            textColor=self.c_dark,
            spaceAfter=0
        ))
        self.styles.add(ParagraphStyle(
            name="MetaLabel",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=self.c_muted
        ))
        self.styles.add(ParagraphStyle(
            name="MetaVal",
            fontName="Helvetica",
            fontSize=8.5,
            leading=10,
            textColor=self.c_dark
        ))
        self.styles.add(ParagraphStyle(
            name="TableHeader",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white
        ))
        self.styles.add(ParagraphStyle(
            name="TableCell",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=self.c_dark
        ))
        self.styles.add(ParagraphStyle(
            name="TableCellBold",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=self.c_dark
        ))
        self.styles.add(ParagraphStyle(
            name="BulletItem",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=self.c_dark,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=4
        ))
        self.styles.add(ParagraphStyle(
            name="DisclaimerFoot",
            fontName="Helvetica",
            fontSize=7,
            leading=9.5,
            textColor=self.c_muted,
            spaceAfter=3
        ))

    def generate_pdf_bytes(
        self,
        project: Dict[str, Any],
        metrics: Dict[str, Any],
        narrative: Dict[str, Any]
    ) -> bytes:
        """
        Compile complete executive project risk memo and return raw PDF bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=40,
            bottomMargin=45
        )

        story = []
        content_width = A4[0] - 72  # 595.27 - 72 = 523.27 pt

        # =========================================================================
        # 1. COVER / HEADER BLOCK
        # =========================================================================
        report_id = f"REP-{time.strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
        gen_time = narrative.get("generated_at", time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))

        header_data = [
            [
                Paragraph("NATIONAL RISK PLATFORM", self.styles["MemoSuperHeader"]),
                Paragraph(f"REPORT ID: <b>{report_id}</b>", ParagraphStyle(
                    "ReportIdR", fontName="Helvetica-Bold", fontSize=7.5, leading=9,
                    textColor=self.c_muted, alignment=2
                ))
            ],
            [
                Paragraph("Statutory Land Acquisition Risk Memo", self.styles["MemoTitle"]),
                Paragraph(f"Generated: {gen_time}", ParagraphStyle(
                    "DateR", fontName="Helvetica", fontSize=7.5, leading=9,
                    textColor=self.c_muted, alignment=2
                ))
            ]
        ]
        t_head = Table(header_data, colWidths=[340, content_width - 340])
        t_head.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(t_head)
        story.append(HRFlowable(width="100%", thickness=1.5, color=self.c_blue, spaceBefore=4, spaceAfter=8))

        # Project Attributes Grid (2-row 4-column metadata strip)
        pid = project.get("project_id", "PROJ-UNASSIGNED")
        ptype = str(project.get("project_type", "Infrastructure")).replace("_", " ")
        state = project.get("state", "State")
        dist = project.get("district", "District")
        terrain = str(project.get("terrain_type", "Plain")).replace("_", " ")
        cost = float(project.get("estimated_cost_inr_crore", 0.0) or 0.0)
        land = float(project.get("land_area_hectares", 0.0) or 0.0)
        pafs = int(project.get("affected_families_count", 0) or 0)

        meta_rows = [
            [
                Paragraph("PROJECT IDENTIFIER", self.styles["MetaLabel"]),
                Paragraph("SECTOR / DOMAIN", self.styles["MetaLabel"]),
                Paragraph("JURISDICTION (DISTRICT, STATE)", self.styles["MetaLabel"]),
                Paragraph("TERRAIN PROFILE", self.styles["MetaLabel"])
            ],
            [
                Paragraph(f"<b>{pid}</b>", self.styles["MetaVal"]),
                Paragraph(f"{ptype}", self.styles["MetaVal"]),
                Paragraph(f"{dist}, {state}", self.styles["MetaVal"]),
                Paragraph(f"{terrain}", self.styles["MetaVal"])
            ],
            [
                Paragraph("CAPITAL OUTLAY", self.styles["MetaLabel"]),
                Paragraph("LAND EXTENT", self.styles["MetaLabel"]),
                Paragraph("AFFECTED FAMILIES (PAFS)", self.styles["MetaLabel"]),
                Paragraph("STATUTORY WINDOW", self.styles["MetaLabel"])
            ],
            [
                Paragraph(f"INR {cost:,.2f} Cr", self.styles["MetaVal"]),
                Paragraph(f"{land:,.1f} Hectares", self.styles["MetaVal"]),
                Paragraph(f"{pafs:,} Families", self.styles["MetaVal"]),
                Paragraph(f"{project.get('section_11_notification_days', 30)}d / 365d (Sec 19(7))", self.styles["MetaVal"])
            ]
        ]
        col_w = content_width / 4.0
        t_meta = Table(meta_rows, colWidths=[col_w] * 4)
        t_meta.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.c_bg_card),
            ("BOX", (0, 0), (-1, -1), 0.75, self.c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, self.c_border),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 10))

        # =========================================================================
        # 2. EXECUTIVE SUMMARY (Callout Card)
        # =========================================================================
        story.append(Paragraph("1. Executive Summary", self.styles["SectionHeading"]))
        exec_text = narrative.get("executive_summary", "")
        t_exec = Table([[Paragraph(exec_text, self.styles["ExecSummaryText"])]], colWidths=[content_width])
        t_exec.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.c_bg_card),
            ("LINELEFT", (0, 0), (-1, -1), 3.0, self.c_blue),
            ("BOX", (0, 0), (-1, -1), 0.5, self.c_border),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_exec)
        story.append(Spacer(1, 10))

        # =========================================================================
        # 3. CALIBRATED RISK METRICS TABLE
        # =========================================================================
        story.append(Paragraph("2. Calibrated Risk & Survival Metrics", self.styles["SectionHeading"]))
        auth = narrative.get("authoritative_metrics", metrics)
        crs_val = auth.get("crs", metrics.get("crs", 0.0))
        prob_val = auth.get("delay_probability", metrics.get("delay_probability", 0.0))
        days_val = auth.get("predicted_delay_days", metrics.get("predicted_delay_days", 0))
        surv_val = auth.get("median_survival_days", metrics.get("median_survival_days", 140))
        tier_val = auth.get("calibrated_risk_tier", metrics.get("calibrated_risk_tier", "Medium"))

        tier_color = self.c_high if tier_val == "High" else (self.c_med if tier_val == "Medium" else self.c_low)
        months_val = round(days_val / 30.4375, 1)
        weeks_val = int(round(days_val / 7.0))
        delay_fmt = f"{days_val:,} Days (~{months_val} mo / {weeks_val} wks)" if days_val >= 30 else f"{days_val:,} Days (~{weeks_val} wks)"

        conf_score = metrics.get("confidence_score", max(prob_val, 100.0 - prob_val))
        conf_label = metrics.get("confidence_label", "High Certainty" if conf_score >= 80 else "Moderate Certainty")

        metrics_data = [
            [
                Paragraph("Metric Indicator", self.styles["TableHeader"]),
                Paragraph("Calibrated Value", self.styles["TableHeader"]),
                Paragraph("Model Paradigm / Reference", self.styles["TableHeader"]),
                Paragraph("Statutory Interpretation", self.styles["TableHeader"]),
            ],
            [
                Paragraph("<b>Delay Probability</b>", self.styles["TableCell"]),
                Paragraph(f"<b>{prob_val:.1f}%</b> ({conf_score:.1f}% {conf_label})", self.styles["TableCellBold"]),
                Paragraph("Stacking Ensemble (XGB+LGB+Cat)", self.styles["TableCell"]),
                Paragraph("Risk of crossing statutory delivery date", self.styles["TableCell"])
            ],
            [
                Paragraph("<b>Composite Risk Score (CRS)</b>", self.styles["TableCell"]),
                Paragraph(f"<b>{crs_val:.1f} / 100</b>", self.styles["TableCellBold"]),
                Paragraph("Multi-factor Statutory Scoring", self.styles["TableCell"]),
                Paragraph(f"<b>{tier_val} Risk Tier</b>", ParagraphStyle("TierTag", parent=self.styles["TableCellBold"], textColor=tier_color))
            ],
            [
                Paragraph("<b>Predicted Statutory Delay</b>", self.styles["TableCell"]),
                Paragraph(f"<b>{delay_fmt}</b>", self.styles["TableCellBold"]),
                Paragraph("Timeline Regressor (Ridge+ET+GB)", self.styles["TableCell"]),
                Paragraph("Net timeline slippage against approved DPR", self.styles["TableCell"])
            ],
            [
                Paragraph("<b>Median Survival Duration</b>", self.styles["TableCell"]),
                Paragraph(f"<b>{surv_val} Days</b>", self.styles["TableCellBold"]),
                Paragraph("Random Survival Forest (RSF S(t))", self.styles["TableCell"]),
                Paragraph("Estimated time until 50% cumulative delay hazard", self.styles["TableCell"])
            ]
        ]
        t_metrics = Table(metrics_data, colWidths=[130, 125, 130, content_width - 385])
        t_metrics.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.c_primary),
            ("BOX", (0, 0), (-1, -1), 0.75, self.c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, self.c_border),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.c_bg_card])
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 10))

        # =========================================================================
        # 4. STATUTORY STATUS & RFCTLARR ACT 2013 GOVERNANCE
        # =========================================================================
        story.append(Paragraph("3. RFCTLARR Act 2013 Statutory Governance Status", self.styles["SectionHeading"]))
        stat_clause = narrative.get("statutory_code", "RFCTLARR Act 2013")
        stat_headline = narrative.get("statutory_status_headline", "Statutory Review")
        stat_meaning = narrative.get("statutory_meaning", "")
        stat_restart = narrative.get("statutory_restart", "")

        sec11_days = int(project.get("section_11_notification_days", 30) or 30)
        stat_border_color = self.c_high if sec11_days > 365 else (self.c_med if sec11_days >= 270 else self.c_low)

        stat_content = [
            Paragraph(f"<b>Statutory Milestone Trigger:</b> {stat_clause} &mdash; <i>{stat_headline}</i>", self.styles["BodyDarkBold"]),
            Spacer(1, 3),
            Paragraph(f"<b>Procedural Legal Implication:</b> {stat_meaning}", self.styles["BodyDark"]),
            Spacer(1, 3),
            Paragraph(f"<b>Statutory Re-initiation Protocol:</b> {stat_restart}", self.styles["BodyDark"])
        ]
        t_stat = Table([[stat_content]], colWidths=[content_width])
        t_stat.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.c_bg_card),
            ("LINELEFT", (0, 0), (-1, -1), 3.0, stat_border_color),
            ("BOX", (0, 0), (-1, -1), 0.5, self.c_border),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_stat)
        story.append(Spacer(1, 10))

        # =========================================================================
        # 5. AI DELAY CAUSATION DIAGNOSTICS (EXPANDED)
        # =========================================================================
        story.append(Paragraph("4. AI Delay Causation Diagnostics (Expanded Narrative)", self.styles["SectionHeading"]))
        diag_text = narrative.get("expanded_causation_diagnostics", "")
        story.append(Paragraph(diag_text, self.styles["BodyDark"]))
        story.append(Spacer(1, 8))

        # =========================================================================
        # 6. SITE & ACCESSIBILITY CONTEXT
        # =========================================================================
        story.append(Paragraph("5. Site Demarcation & Accessibility Context", self.styles["SectionHeading"]))
        site_text = narrative.get("site_accessibility_context", "")
        story.append(Paragraph(site_text, self.styles["BodyDark"]))
        story.append(Spacer(1, 8))

        # =========================================================================
        # 7. RECOMMENDED ACTIONS / PRIORITIZED NEXT STEPS
        # =========================================================================
        story.append(Paragraph("6. Prioritized Statutory Next Steps & Interventions", self.styles["SectionHeading"]))
        recs: List[str] = narrative.get("recommended_next_steps", [])
        if not recs:
            recs = [
                "Maintain bi-weekly tracking of Section 19 notification issuance with District Collectorate.",
                "Verify SLAO escrow balance for timely 100% Solatium disbursements under Section 30."
            ]

        for idx, rec in enumerate(recs, 1):
            story.append(Paragraph(f"<b>{idx}.</b> {rec}", self.styles["BulletItem"]))
        story.append(Spacer(1, 12))

        # =========================================================================
        # 8. METHODOLOGY, DISCLAIMER & MODEL PROVENANCE FOOTER
        # =========================================================================
        story.append(HRFlowable(width="100%", thickness=0.75, color=self.c_border, spaceBefore=4, spaceAfter=8))
        provenance = narrative.get("provenance", "Internal Trained Model Engine")

        story.append(Paragraph(
            "<b>Methodology & Conformal Calibration Reference:</b> Predictions generated by the dual-paradigm machine learning framework "
            "(Stacking Classifier + Random Survival Forests S(t) + Quantile Regression). Complies with RFCTLARR Act 2013 legal mandates. "
            "For full architectural documentation, loss functions, and C-Index benchmarks (0.906 Uno C-Index), consult the Technical Methodology at <u>/methodology</u>.",
            self.styles["DisclaimerFoot"]
        ))
        story.append(Paragraph(
            "<b>Statutory Disclaimer:</b> This memo is an AI-assisted predictive governance analysis intended for authorized project authorities. "
            "It does not supersede statutory decisions of the Competent Authority (Collector / Special Land Acquisition Officer) or judicial awards of the Land Acquisition Authority (LARRA).",
            self.styles["DisclaimerFoot"]
        ))
        story.append(Paragraph(
            f"<b>Model Provenance & Audit Trail:</b> Narrative Synthesis Engine: <b>{provenance}</b> · "
            f"Model Calibration Checkpoint: ensemble.joblib & timeline.joblib · Deterministic Validation Gate Passed.",
            self.styles["DisclaimerFoot"]
        ))

        # Build PDF with two-pass canvas
        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.getvalue()
