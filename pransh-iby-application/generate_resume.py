#!/usr/bin/env python3
"""Generate ATS-safe DOCX + PDF resume for Pransh Agarwal."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
NAVY_HEX = HexColor("#1F3A5F")
LINE_HEX = HexColor("#1F3A5F")
GRAY = HexColor("#333333")


def set_run_font(run, name="Calibri", size=11, bold=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_paragraph_format(p, before=0, after=0, line=1.08, align="left"):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def add_bottom_border(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1F3A5F")
    pBdr.append(bottom)
    pPr.append(pBdr)


def heading(doc, text):
    p = doc.add_paragraph()
    set_paragraph_format(p, before=6, after=1, line=1.0)
    run = p.add_run(text.upper())
    set_run_font(run, size=11, bold=True, color=NAVY)
    add_bottom_border(p)
    return p


def body(doc, text, size=10.5, bold=False, before=1, after=1, align="left"):
    p = doc.add_paragraph()
    set_paragraph_format(p, before=before, after=after, line=1.08, align=align)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def mixed_line(doc, parts, size=10, before=2, after=0, align="left"):
    p = doc.add_paragraph()
    set_paragraph_format(p, before=before, after=after, line=1.05, align=align)
    for text, bold in parts:
        run = p.add_run(text)
        set_run_font(run, size=size, bold=bold)
    return p


def bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    pf = p.paragraph_format
    pf.space_before = Pt(0.5)
    pf.space_after = Pt(0.5)
    pf.line_spacing = 1.07
    pf.left_indent = Inches(0.25)
    pf.first_line_indent = Inches(-0.15)
    # clear default runs
    if p.runs:
        p.runs[0].text = text
        set_run_font(p.runs[0], size=size)
    else:
        run = p.add_run(text)
        set_run_font(run, size=size)
    return p


def build_docx(path: Path):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(0.38)
    section.bottom_margin = Inches(0.38)

    styles = doc.styles["Normal"]
    styles.font.name = "Calibri"
    styles.font.size = Pt(10.5)
    styles._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    p = doc.add_paragraph()
    set_paragraph_format(p, before=0, after=0, line=1.0, align="center")
    run = p.add_run("PRANSH AGARWAL")
    set_run_font(run, size=18, bold=True, color=NAVY)

    body(
        doc,
        "MBA, Business Analytics  |  Indian Institute of Management, Sambalpur  |  Batch 2026–28",
        size=10.5,
        bold=True,
        before=1,
        after=0,
        align="center",
    )
    body(
        doc,
        "placementoffice@iimsambalpur.ac.in  |  Indian Institute of Management, Sambalpur, Odisha",
        size=10,
        before=0,
        after=0,
        align="center",
    )
    body(
        doc,
        "Target Role: AI Consultant (AI-Native New Business Development)",
        size=10,
        bold=True,
        before=1,
        after=2,
        align="center",
    )

    heading(doc, "Professional Summary")
    body(
        doc,
        "First-semester MBA in Business Analytics at IIM Sambalpur with a B.E. in Computer Science "
        "(9.34 CGPA) and 20 months of end-to-end operations ownership. Directed logistics, inventory, "
        "and quality across 500+ SKUs; cut leadership reporting time 40% with analytics dashboards; "
        "and launched a live 0-to-1 web product. Combine CS/ML literacy with operations execution to "
        "identify AI-native opportunities, define an MVP, run customer discovery, and own launch and revenue outcomes.",
        size=10,
        before=2,
        after=1,
    )

    heading(doc, "Education")
    mixed_line(
        doc,
        [
            ("MBA, Business Analytics", True),
            ("  |  Indian Institute of Management, Sambalpur  |  2026 – 2028", False),
        ],
        before=4,
        after=0,
    )
    mixed_line(
        doc,
        [
            ("B.E., Computer Science", True),
            ("  |  Chitkara University, Punjab  |  2024  |  93.40% (9.34 CGPA)", False),
        ],
        before=1,
        after=0,
    )
    mixed_line(
        doc,
        [
            ("Class XII (CBSE)", True),
            ("  |  Guru Tegh Bahadur Public School, Meerut  |  2020  |  74.83%", False),
        ],
        before=1,
        after=0,
    )
    mixed_line(
        doc,
        [
            ("Class X (CBSE)", True),
            ("  |  Dewan Public School, Meerut  |  2018  |  73.00%", False),
        ],
        before=1,
        after=2,
    )

    heading(doc, "Professional Experience")
    mixed_line(
        doc,
        [
            ("Vinay Enterprises", True),
            ("  —  Operations Manager  |  June 2024 – March 2026", False),
        ],
        before=4,
        after=1,
    )
    for t in [
        "Owned end-to-end daily operations (logistics, inventory, quality) across 500+ SKUs; held stock discrepancy below 2% in every quarterly audit by standardizing stock-movement workflows into explicit, auditable processes.",
        "Built automated Excel analytics dashboards (Pivot Tables, VLOOKUP, Macros) for executive leadership, reducing manual reporting time by 40% and enabling data-driven weekly performance tracking.",
        "Led a cross-functional team of 8+; set weekly Key Performance Indicators (KPIs) that raised on-time task completion by 25% and built accountability around process improvement.",
        "Conceived, engineered, and launched the company’s live responsive web application — a 0-to-1 product that improved mobile access and organic customer reach.",
    ]:
        bullet(doc, t)

    mixed_line(
        doc,
        [
            ("Hoping Minds", True),
            ("  —  Full Stack Developer Intern  |  December 2023 – June 2024", False),
        ],
        before=4,
        after=1,
    )
    for t in [
        "Shipped production web applications using React.js, Node.js, Express.js, and MongoDB, delivering 3+ client-facing features per sprint with stakeholders.",
        "Designed REST APIs with JWT authentication and Redux state management, improving API response times by 30%; sustained a >95% bug-free release record with zero critical downtime.",
    ]:
        bullet(doc, t)

    heading(doc, "Projects")
    mixed_line(
        doc,
        [("Internz — Business Intelligence and Analytics Platform", True), ("  |  2024", False)],
        before=4,
        after=1,
    )
    bullet(
        doc,
        "Built a Superset-style analytics web platform (MERN) so non-technical users could create real-time business dashboards; improved load time 25% with optimized MongoDB queries, JWT, and role-based access.",
    )

    mixed_line(
        doc,
        [("Blog Application — Full-Stack Content Management System", True), ("  |  2024", False)],
        before=2,
        after=0,
    )
    bullet(
        doc,
        "Launched a CMS with authentication, role-based access, and CRUD publishing; cut API latency 20% under peak-load testing.",
    )

    heading(doc, "Skills")
    mixed_line(
        doc,
        [
            ("Business and Go-to-Market: ", True),
            (
                "Customer Discovery, Market Research, Go-to-Market (GTM), Unit Economics, "
                "KPI Design, Process Improvement, Operations and Logistics, Stakeholder "
                "Communication, Cross-functional Leadership",
                False,
            ),
        ],
        before=4,
        after=1,
    )
    mixed_line(
        doc,
        [
            ("Analytics and AI: ", True),
            (
                "Business Analytics, Machine Learning (Regression, Classification, Clustering), "
                "Python, Excel Dashboards, Data Visualization, AI Literacy (LLMs, AI Agents)",
                False,
            ),
        ],
        before=1,
        after=1,
    )
    mixed_line(
        doc,
        [
            ("Product and Engineering: ", True),
            (
                "MVP Definition and Launch, React.js, Node.js, Express.js, MongoDB, REST APIs, "
                "JWT, Redux, Full-Stack Web Development",
                False,
            ),
        ],
        before=1,
        after=2,
    )

    heading(doc, "Certifications")
    bullet(
        doc,
        "IBM Full Stack Software Developer Professional Certificate (React, Node.js, DevOps, Cloud Native) — 2024",
    )
    bullet(
        doc,
        "University of Washington Machine Learning Specialization (Regression, Classification, Clustering, Information Retrieval; Python) — 2023",
    )

    heading(doc, "Leadership and Achievements")
    bullet(
        doc,
        "Skilled Sapiens x IIM Sambalpur (2026): Corporate mentor-driven leadership program — strategic leadership and real-world business case solving.",
    )
    bullet(
        doc,
        "National-level B-school case competitions (2026): Market-trend and operations analysis; executive pitch presentations for corporate problem statements.",
    )
    bullet(
        doc,
        "Top-percentile B.E. Computer Science graduate (9.34 CGPA); National Mathematics and Science Olympiad merit certificates.",
    )

    heading(doc, "Interests")
    body(
        doc,
        "Competitive chess, Sudoku, supply chain analytics, web product building.",
        size=10.5,
        before=3,
        after=0,
    )

    doc.save(path)
    print(f"Wrote {path}")


def pstyle(name, **kwargs):
    defaults = dict(
        fontName="Times-Roman",
        fontSize=9.4,
        leading=11.4,
        textColor=GRAY,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
    )
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)


def build_pdf(path: Path):
    styles = {
        "name": pstyle(
            "name",
            fontName="Times-Bold",
            fontSize=15,
            leading=17,
            alignment=TA_CENTER,
            textColor=NAVY_HEX,
            spaceAfter=1,
        ),
        "sub": pstyle(
            "sub",
            fontName="Times-Bold",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=GRAY,
            spaceAfter=0.5,
        ),
        "contact": pstyle(
            "contact",
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=GRAY,
            spaceAfter=0.5,
        ),
        "target": pstyle(
            "target",
            fontName="Times-Bold",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=NAVY_HEX,
            spaceAfter=2,
        ),
        "h": pstyle(
            "h",
            fontName="Times-Bold",
            fontSize=10,
            leading=12,
            textColor=NAVY_HEX,
            spaceBefore=5,
            spaceAfter=1,
        ),
        "role": pstyle(
            "role",
            fontName="Times-Bold",
            fontSize=9.5,
            leading=12,
            textColor=black,
            spaceBefore=3,
            spaceAfter=0.5,
        ),
        "body": pstyle(
            "body",
            fontName="Times-Roman",
            fontSize=9.2,
            leading=11.4,
            alignment=TA_JUSTIFY,
            spaceAfter=2,
        ),
        "edu": pstyle(
            "edu",
            fontName="Times-Roman",
            fontSize=9.2,
            leading=11.5,
            spaceAfter=0.5,
        ),
        "bullet": pstyle(
            "bullet",
            fontName="Times-Roman",
            fontSize=9.2,
            leading=11.4,
            leftIndent=11,
            firstLineIndent=-10,
            spaceAfter=1,
        ),
        "skill": pstyle(
            "skill",
            fontName="Times-Roman",
            fontSize=9.2,
            leading=11.4,
            spaceAfter=1,
        ),
    }

    def h(text):
        return KeepTogether(
            [
                Paragraph(text.upper(), styles["h"]),
                HRFlowable(width="100%", thickness=1, color=LINE_HEX, spaceBefore=0, spaceAfter=2),
            ]
        )

    def b(text):
        return Paragraph(f"•  {text}", styles["bullet"])

    story = [
        Paragraph("PRANSH AGARWAL", styles["name"]),
        Paragraph(
            "MBA, Business Analytics  |  Indian Institute of Management, Sambalpur  |  Batch 2026–28",
            styles["sub"],
        ),
        Paragraph(
            "placementoffice@iimsambalpur.ac.in  |  Indian Institute of Management, Sambalpur, Odisha",
            styles["contact"],
        ),
        Paragraph(
            "Target Role: AI Consultant (AI-Native New Business Development)",
            styles["target"],
        ),
        h("Professional Summary"),
        Paragraph(
            "First-semester MBA in Business Analytics at IIM Sambalpur with a B.E. in Computer Science "
            "(9.34 CGPA) and 20 months of end-to-end operations ownership. Directed logistics, inventory, "
            "and quality across 500+ SKUs; cut leadership reporting time 40% with analytics dashboards; "
            "and launched a live 0-to-1 web product. Combine CS/ML literacy with operations execution to "
            "identify AI-native opportunities, define an MVP, run customer discovery, and own launch and revenue outcomes.",
            styles["body"],
        ),
        h("Education"),
        Paragraph(
            "<b>MBA, Business Analytics</b>  |  Indian Institute of Management, Sambalpur  |  2026 – 2028",
            styles["edu"],
        ),
        Paragraph(
            "<b>B.E., Computer Science</b>  |  Chitkara University, Punjab  |  2024  |  93.40% (9.34 CGPA)",
            styles["edu"],
        ),
        Paragraph(
            "<b>Class XII (CBSE)</b>  |  Guru Tegh Bahadur Public School, Meerut  |  2020  |  74.83%",
            styles["edu"],
        ),
        Paragraph(
            "<b>Class X (CBSE)</b>  |  Dewan Public School, Meerut  |  2018  |  73.00%",
            styles["edu"],
        ),
        h("Professional Experience"),
        Paragraph(
            "Vinay Enterprises  —  Operations Manager  |  June 2024 – March 2026",
            styles["role"],
        ),
        b(
            "Owned end-to-end daily operations (logistics, inventory, quality) across 500+ SKUs; held stock discrepancy below 2% in every quarterly audit by standardizing stock-movement workflows into explicit, auditable processes."
        ),
        b(
            "Built automated Excel analytics dashboards (Pivot Tables, VLOOKUP, Macros) for executive leadership, reducing manual reporting time by 40% and enabling data-driven weekly performance tracking."
        ),
        b(
            "Led a cross-functional team of 8+; set weekly Key Performance Indicators (KPIs) that raised on-time task completion by 25% and built accountability around process improvement."
        ),
        b(
            "Conceived, engineered, and launched the company’s live responsive web application — a 0-to-1 product that improved mobile access and organic customer reach."
        ),
        Paragraph(
            "Hoping Minds  —  Full Stack Developer Intern  |  December 2023 – June 2024",
            styles["role"],
        ),
        b(
            "Shipped production web applications using React.js, Node.js, Express.js, and MongoDB, delivering 3+ client-facing features per sprint with stakeholders."
        ),
        b(
            "Designed REST APIs with JWT authentication and Redux state management, improving API response times by 30%; sustained a &gt;95% bug-free release record with zero critical downtime."
        ),
        h("Projects"),
        Paragraph("Internz — Business Intelligence and Analytics Platform  |  2024", styles["role"]),
        b(
            "Built a Superset-style analytics web platform (MERN) so non-technical users could create real-time business dashboards; improved load time 25% with optimized MongoDB queries, JWT, and role-based access."
        ),
        Paragraph("Blog Application — Full-Stack Content Management System  |  2024", styles["role"]),
        b(
            "Launched a CMS with authentication, role-based access, and CRUD publishing; cut API latency 20% under peak-load testing."
        ),
        h("Skills"),
        Paragraph(
            "<b>Business and Go-to-Market:</b> Customer Discovery, Market Research, Go-to-Market (GTM), "
            "Unit Economics, KPI Design, Process Improvement, Operations and Logistics, Stakeholder "
            "Communication, Cross-functional Leadership",
            styles["skill"],
        ),
        Paragraph(
            "<b>Analytics and AI:</b> Business Analytics, Machine Learning (Regression, Classification, Clustering), "
            "Python, Excel Dashboards, Data Visualization, AI Literacy (LLMs, AI Agents)",
            styles["skill"],
        ),
        Paragraph(
            "<b>Product and Engineering:</b> MVP Definition and Launch, React.js, Node.js, Express.js, MongoDB, "
            "REST APIs, JWT, Redux, Full-Stack Web Development",
            styles["skill"],
        ),
        h("Certifications"),
        b(
            "IBM Full Stack Software Developer Professional Certificate (React, Node.js, DevOps, Cloud Native) — 2024"
        ),
        b(
            "University of Washington Machine Learning Specialization (Regression, Classification, Clustering, Information Retrieval; Python) — 2023"
        ),
        h("Leadership and Achievements"),
        b(
            "Skilled Sapiens x IIM Sambalpur (2026): Corporate mentor-driven leadership program — strategic leadership and real-world business case solving."
        ),
        b(
            "National-level B-school case competitions (2026): Market-trend and operations analysis; executive pitch presentations for corporate problem statements."
        ),
        b(
            "Top-percentile B.E. Computer Science graduate (9.34 CGPA); National Mathematics and Science Olympiad merit certificates."
        ),
        h("Interests"),
        Paragraph(
            "Competitive chess, Sudoku, supply chain analytics, web product building.",
            styles["body"],
        ),
    ]

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.38 * inch,
        bottomMargin=0.38 * inch,
        title="Pransh Agarwal Resume",
        author="Pransh Agarwal",
    )
    doc.build(story)
    print(f"Wrote {path}")


if __name__ == "__main__":
    build_docx(OUT / "Pransh_Agarwal_ATS_Resume.docx")
    build_pdf(OUT / "Pransh_Agarwal_ATS_Resume.pdf")
