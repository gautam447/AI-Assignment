#!/usr/bin/env python3
"""Build the polypharmacy synopsis PDF matching the reference format."""

from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent
ASSETS = Path("/tmp/pdf_assets")
UPLOAD_COVER = Path(
    "/home/ubuntu/.cursor/projects/workspace/uploads/"
    "Jeevan_Polypharmacy_in_Geriatric_patient-1_8e71.pdf"
)
OUT_PDF = ROOT / "A_Study_on_Polypharmacy_Among_Geriatric_Patients.pdf"
COVER_PNG = ASSETS / "cover_updated.png"
BODY_PDF = ASSETS / "body_pages.pdf"

TITLE = (
    "A STUDY ON POLYPHARMACY AMONG GERIATRIC PATIENTS "
    "AGED ≥65 YEARS IN A TERTIARY CARE HOSPITAL"
)

NAVY = (8, 21, 73)

pdfmetrics.registerFont(
    TTFont("LibSerif", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
)
pdfmetrics.registerFont(
    TTFont("LibSerif-Bold", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")
)
pdfmetrics.registerFont(
    TTFont(
        "LibSerif-Italic",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    )
)


def build_cover() -> Path:
    """Update the reference cover image with the corrected study title."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(UPLOAD_COVER))
    page = doc[0]
    xref = page.get_images(full=True)[0][0]
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    raw = ASSETS / "cover_img_0_20.png"
    pix.save(str(raw))
    doc.close()

    img = Image.open(raw).convert("RGB")
    pixels = img.load()
    w, _h = img.size
    bg = (252, 252, 252)
    y0, y1 = 700, 895
    for y in range(y0, y1):
        for x in range(80, w - 80):
            r, g, b = pixels[x, y]
            if not (r > 230 and g > 230 and b > 230):
                pixels[x, y] = bg

    draw = ImageDraw.Draw(img)
    font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
    title_lines = [
        "A STUDY ON POLYPHARMACY AMONG",
        "GERIATRIC PATIENTS AGED ≥65 YEARS",
        "IN A TERTIARY CARE HOSPITAL",
    ]
    max_width = w - 160
    size = 36
    font = ImageFont.truetype(font_path, size)
    while size > 18:
        font = ImageFont.truetype(font_path, size)
        widths = [draw.textlength(line, font=font) for line in title_lines]
        if max(widths) <= max_width:
            break
        size -= 1

    line_gap = int(size * 1.35)
    total_h = line_gap * len(title_lines)
    start_y = y0 + (y1 - y0 - total_h) // 2
    for i, line in enumerate(title_lines):
        tw = draw.textlength(line, font=font)
        x = (w - tw) / 2
        y = start_y + i * line_gap
        draw.text((x, y), line, font=font, fill=NAVY)

    img.save(COVER_PNG, quality=95)
    return COVER_PNG


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="StudyTitle",
            fontName="LibSerif-Bold",
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=18,
            spaceBefore=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            fontName="LibSerif-Bold",
            fontSize=13,
            leading=17,
            alignment=TA_LEFT,
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            fontName="LibSerif",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    return styles


def build_body_pdf() -> Path:
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(BODY_PDF),
        pagesize=letter,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title=TITLE,
        author="Jeevan Jyoti Mohanty",
    )
    story = [
        Paragraph(TITLE, styles["StudyTitle"]),
        Spacer(1, 8),
        Paragraph("INTRODUCTION AND RATIONALE:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide the introduction and rationale text.]",
            styles["BodyJust"],
        ),
        Paragraph("OBJECTIVES:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide primary/secondary objectives.]",
            styles["BodyJust"],
        ),
        Paragraph("REVIEW OF LITERATURE:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide the literature review.]",
            styles["BodyJust"],
        ),
        Paragraph("METHODOLOGY:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide methodology, inclusion/exclusion "
            "criteria, data collection, and analysis details.]",
            styles["BodyJust"],
        ),
        Paragraph("RESULTS:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide results, rates, and tables.]",
            styles["BodyJust"],
        ),
        Paragraph("DISCUSSION:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide the discussion.]",
            styles["BodyJust"],
        ),
        Paragraph("CONCLUSION:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide the conclusion.]",
            styles["BodyJust"],
        ),
        Paragraph("REFERENCES:-", styles["SectionHead"]),
        Paragraph(
            "[Content to be added — please provide the reference list.]",
            styles["BodyJust"],
        ),
    ]
    doc.build(story)
    return BODY_PDF


def build_pdf() -> Path:
    cover = build_cover()
    body = build_body_pdf()

    out = fitz.open()
    # Cover page from updated image
    cover_page = out.new_page(width=letter[0], height=letter[1])
    cover_page.insert_image(cover_page.rect, filename=str(cover))

    body_doc = fitz.open(str(body))
    out.insert_pdf(body_doc)
    body_doc.close()

    out.save(str(OUT_PDF))
    out.close()
    print(f"Wrote {OUT_PDF}")
    return OUT_PDF


if __name__ == "__main__":
    build_pdf()
