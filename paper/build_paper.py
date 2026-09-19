"""Render paper/Reflective_Synthesis_Paper.md to Reflective_Synthesis_Paper.pdf with reportlab."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "Reflective_Synthesis_Paper.md"
OUT = ROOT / "Reflective_Synthesis_Paper.pdf"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=styles["Title"], fontSize=15, leading=19, spaceAfter=6)
AUTHOR = ParagraphStyle("a", parent=styles["Normal"], alignment=1, fontSize=10, textColor="#444444", spaceAfter=14)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4, keepWithNext=True)
BODY = ParagraphStyle("b", parent=styles["Normal"], fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7)
REF = ParagraphStyle("r", parent=BODY, alignment=0, leftIndent=18, firstLineIndent=-18, spaceAfter=5, fontSize=9.5, leading=12.5)
CAPTION = ParagraphStyle("c", parent=styles["Italic"], fontSize=8.5, alignment=1, spaceAfter=10)


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier' size='9'>\1</font>", text)
    return re.sub(r"(https?://\S+)", r"<link href='\1' color='#1a4f8b'>\1</link>", text)


def build() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    story: list = []
    in_refs = False
    para: list[str] = []

    def flush() -> None:
        if para:
            story.append(Paragraph(inline(" ".join(para)), REF if in_refs else BODY))
            para.clear()

    for i, line in enumerate(lines):
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), TITLE))
        elif i == 2 and line.strip():
            story.append(Paragraph(inline(line), AUTHOR))
        elif line.startswith("## "):
            flush()
            heading = line[3:]
            in_refs = heading.strip().lower() == "references"
            if heading.startswith("3."):
                story.append(Image(str(ROOT / "diagrams" / "architecture.png"), width=6.6 * inch, height=3.55 * inch))
                story.append(Paragraph("Figure 1. Integrated architecture; colour indicates the originating capstone project.", CAPTION))
            story.append(Paragraph(inline(heading), H2))
        elif not line.strip():
            flush()
            if in_refs:
                pass
        else:
            if in_refs:
                flush()
                para.append(line)
                flush()
            else:
                para.append(line)
    flush()

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=1 * inch, rightMargin=1 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                            title="Reflective Synthesis Paper - Health-Investment Policy Advisor", author="Cameron Doelling")
    doc.build(story)
    body = SRC.read_text(encoding="utf-8").split("## References")[0]
    words = len(re.findall(r"[A-Za-z0-9'%.-]+", body))
    print(f"wrote {OUT.relative_to(ROOT)} ({words} words before References)")


if __name__ == "__main__":
    build()
