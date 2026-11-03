"""
generate_report_pdf.py
======================
Converts PROJECT_REPORT.md into a publication-grade academic PDF report
using ReportLab with embedded high-resolution figures, styled tables,
and Unicode Indic script font support (Nirmala UI).
"""

import os
import sys
import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "plots"
REPORT_MD_PATH = PROJECT_ROOT / "PROJECT_REPORT.md"
PDF_OUTPUT_PATH = PROJECT_ROOT / "PROJECT_REPORT.pdf"

# Register Unicode Font if available (Nirmala supports Latin, Devanagari, Kannada)
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_OBLIQUE = "Helvetica-Oblique"

try:
    nirmala_path = "C:/Windows/Fonts/Nirmala.ttc"
    if os.path.exists(nirmala_path):
        pdfmetrics.registerFont(TTFont("Nirmala", nirmala_path, subfontIndex=0))
        pdfmetrics.registerFont(TTFont("Nirmala-Bold", "C:/Windows/Fonts/NirmalaB.ttf" if os.path.exists("C:/Windows/Fonts/NirmalaB.ttf") else nirmala_path, subfontIndex=0))
        FONT_NAME = "Nirmala"
        FONT_BOLD = "Nirmala-Bold"
        print("✔ Registered Nirmala font for full Unicode/Indic script rendering.")
except Exception as e:
    print(f"Notice: Using standard Helvetica font fallback: {e}")

# ---------------------------------------------------------------------------
# Numbered Canvas for Running Headers & Footers
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
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

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont(FONT_NAME, 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "Machine Learning Project Report — Multi-Script Language Detection")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, text)
        self.drawString(54, 32, "English, Hindi, and Kannada • Phase 1, Phase 2 & Phase 3 Evaluation")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 44, letter[0] - 54, 44)
        self.restoreState()

# ---------------------------------------------------------------------------
# Text Processing & Helpers
# ---------------------------------------------------------------------------
def clean_markdown_inline(text):
    """Clean markdown inline formatting for ReportLab Paragraph XML."""
    if not text:
        return ""
    # Escape raw ampersands not part of entities
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos);)", "&amp;", text)
    # Convert markdown bold and italic
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.*?)`", r"<font name='Courier' color='#0f172a'><b>\1</b></font>", text)
    # Convert math symbols
    text = text.replace("$N = 1,050$", "N = 1,050").replace("$N$", "N").replace("$K=3$", "K = 3")
    text = text.replace("$\\rightarrow$", "→").replace("$\\alpha$", "α").replace("$\\approx$", "≈")
    text = text.replace("$\\le$", "≤").replace("$\\ge$", "≥").replace("$p = 0.333$", "p = 0.333")
    text = text.replace("$100\\%$", "100%").replace("$33.33\\%$", "33.33%")
    # Remove any remaining dollar signs
    text = text.replace("$", "")
    return text

def build_pdf_report():
    if REPORT_MD_PATH.exists():
        print(f"Generating PDF report from: {REPORT_MD_PATH}")
        with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
            md_text = f.read()
    else:
        # Fall back to docs reports if PROJECT_REPORT.md has been removed
        doc_paths = [PROJECT_ROOT / "docs" / "PHASE_1_REPORT.md", PROJECT_ROOT / "docs" / "PHASE_2_REPORT.md"]
        combined = []
        for p in doc_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    combined.append(f.read())
        md_text = "\n\n---\n\n".join(combined) if combined else "# Multi-Script Language Detection Project Report"
        print("Notice: Generating PDF report from compiled documentation.")

    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Define Hierarchy Styles
    title_style = ParagraphStyle(
        "DocTitle",
        fontName=FONT_BOLD,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=6
    )
    meta_style = ParagraphStyle(
        "MetaText",
        fontName=FONT_NAME,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155")
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        fontName=FONT_BOLD,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        fontName=FONT_BOLD,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    h3_style = ParagraphStyle(
        "Heading3_Custom",
        fontName=FONT_BOLD,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        fontName=FONT_NAME,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        fontName=FONT_NAME,
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=14,
        spaceAfter=3
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName=FONT_NAME,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName=FONT_BOLD,
        fontSize=8,
        leading=11,
        textColor=colors.white
    )
    code_block_style = ParagraphStyle(
        "CodeBlock",
        fontName="Courier",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # Title Banner
    story.append(Paragraph("Machine Learning Mini Project Report", title_style))
    story.append(Paragraph("<b>Project Title:</b> Multi-Script Language Detection for English, Hindi, and Kannada using Machine Learning", meta_style))
    story.append(Paragraph("<b>Domain:</b> Natural Language Processing (NLP) &amp; Multi-Class Text Classification &nbsp;|&nbsp; <b>Date:</b> October 2026", meta_style))
    story.append(Paragraph("<b>Evaluation Scheme:</b> Phase 1, Phase 2, and Phase 3 Complete Submission (20 Marks)", meta_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=12))

    # Parse lines of Markdown
    lines = md_text.splitlines()
    in_code_block = False
    code_buffer = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        
        # Determine number of columns
        col_count = max(len(r) for r in table_rows)
        # Pad shorter rows
        for r in table_rows:
            while len(r) < col_count:
                r.append("")

        # Calculate available width = 512 pt (letter width 612 - 100 margin)
        avail_width = letter[0] - 100
        col_width = avail_width / col_count

        formatted_data = []
        for i, row in enumerate(table_rows):
            row_paras = []
            for cell in row:
                st = table_header_style if i == 0 else table_cell_style
                row_paras.append(Paragraph(clean_markdown_inline(cell), st))
            formatted_data.append(row_paras)

        t = Table(formatted_data, colWidths=[col_width] * col_count)
        t_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]
        # Alternating row background
        for row_idx in range(1, len(formatted_data)):
            if row_idx % 2 == 0:
                t_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f8fafc")))
        t.setStyle(TableStyle(t_style))

        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 8))
        table_rows = []
        in_table = False

    skip_header = True
    for line in lines:
        stripped = line.strip()

        # Skip header metadata lines already processed
        if skip_header:
            if stripped.startswith("## 1. Problem Statement"):
                skip_header = False
            else:
                continue

        # Handle Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                # End of code block
                in_code_block = False
                code_content = "<br/>".join(code_buffer)
                code_table = Table([[Paragraph(code_content, code_block_style)]], colWidths=[letter[0] - 100])
                code_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(code_table)
                story.append(Spacer(1, 6))
                code_buffer = []
            else:
                in_code_block = True
                code_buffer = []
            continue

        if in_code_block:
            safe_code_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
            code_buffer.append(safe_code_line)
            continue

        # Handle Markdown Tables
        if "|" in stripped and not stripped.startswith("#"):
            # Check if separator row
            if re.match(r"^\|?\s*[-:]+[-|\s:]*$", stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            table_rows.append(cells)
            in_table = True
            continue
        elif in_table:
            flush_table()

        # Empty line
        if not stripped:
            continue

        # Horizontal Rule
        if stripped in ["---", "***", "___"]:
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
            continue

        # Headings
        if stripped.startswith("### "):
            text = clean_markdown_inline(stripped[4:])
            story.append(Paragraph(text, h2_style))
        elif stripped.startswith("#### "):
            text = clean_markdown_inline(stripped[5:])
            story.append(Paragraph(text, h3_style))
        elif stripped.startswith("## "):
            text = clean_markdown_inline(stripped[3:])
            story.append(Paragraph(text, h1_style))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            text = clean_markdown_inline(stripped[2:])
            story.append(Paragraph(f"• {text}", bullet_style))
        elif re.match(r"^\d+\.\s+", stripped):
            text = clean_markdown_inline(re.sub(r"^\d+\.\s+", "", stripped))
            num = stripped.split(".")[0]
            story.append(Paragraph(f"<b>{num}.</b> {text}", bullet_style))
        else:
            text = clean_markdown_inline(stripped)
            story.append(Paragraph(text, body_style))

        # Check if we should insert high-res figures right after specific sections!
        if "### 3.1 Exploratory Data Analysis" in stripped:
            eda_fig = PLOTS_DIR / "class_distribution.png"
            if eda_fig.exists():
                story.append(Spacer(1, 4))
                img = Image(str(eda_fig), width=5.2 * inch, height=2.4 * inch)
                img.hAlign = "CENTER"
                story.append(img)
                story.append(Paragraph("<i>Figure 1: Class Balance Verification (350 samples each for English, Hindi, and Kannada).</i>", table_cell_style))
                story.append(Spacer(1, 6))

        if "### 5.1 Benchmark Comparison Table" in stripped:
            comp_fig = PLOTS_DIR / "model_comparison_accuracy.png"
            if comp_fig.exists():
                story.append(Spacer(1, 4))
                img = Image(str(comp_fig), width=5.4 * inch, height=2.5 * inch)
                img.hAlign = "CENTER"
                story.append(img)
                story.append(Paragraph("<i>Figure 2: Empirical Model Accuracy Comparison across all 8 Evaluated Architectures.</i>", table_cell_style))
                story.append(Spacer(1, 6))

        if "## 6. Generated Visualizations" in stripped:
            roc_fig = PLOTS_DIR / "roc_curves.png"
            if roc_fig.exists():
                story.append(Spacer(1, 4))
                img = Image(str(roc_fig), width=5.2 * inch, height=2.4 * inch)
                img.hAlign = "CENTER"
                story.append(img)
                story.append(Paragraph("<i>Figure 3: Multi-Class One-vs-Rest Receiver Operating Characteristic (ROC) Curves (AUC = 1.0000).</i>", table_cell_style))
                story.append(Spacer(1, 6))

    if in_table:
        flush_table()

    # Build Document
    print("Building PDF document...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✔ Successfully generated publication-grade PDF: {PDF_OUTPUT_PATH} ({PDF_OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")

    # Also copy to Scrap folder if Scrap exists
    scrap_dest = Path("D:/Git Projects/Scrap/PROJECT_REPORT.pdf")
    if scrap_dest.parent.exists():
        try:
            import shutil
            shutil.copy2(PDF_OUTPUT_PATH, scrap_dest)
            print(f"✔ Mirrored PDF to: {scrap_dest}")
        except Exception as e:
            print(f"Notice: Could not mirror to Scrap folder (file may be open in viewer): {e}")

if __name__ == "__main__":
    build_pdf_report()
