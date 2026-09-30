import html
import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["Export"])


class ExportRequest(BaseModel):
    format: str
    chat_history: List[Dict[str, Any]]


def _format_markdown_for_reportlab(text: str) -> str:
    """Sanitize and format markdown text into XML tags supported by ReportLab Paragraph."""
    if not text:
        return ""
    # XML Escape first
    escaped = html.escape(str(text))
    # Bold **text** or __text__
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)
    # Italic *text* or _text_
    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", escaped)
    # Inline code `code`
    escaped = re.sub(r"`(.+?)`", r'<font face="Courier" color="#312E81">\1</font>', escaped)
    # Normalize bullet points
    escaped = re.sub(r"^[ \t]*[-*][ \t]+", "&bull; ", escaped, flags=re.MULTILINE)
    # Line breaks
    escaped = escaped.replace("\n", "<br/>")
    return escaped


def _generate_reportlab_pdf(chat_history: List[Dict[str, Any]], pdf_path: str):
    """Generate high-fidelity research memorandum PDF using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=10,
    )
    user_header_style = ParagraphStyle(
        "UserHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=4,
    )
    user_bubble_style = ParagraphStyle(
        "UserBubble",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )
    ai_header_style = ParagraphStyle(
        "AiHeader",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4338CA"),
        spaceBefore=8,
        spaceAfter=4,
    )
    ai_text_style = ParagraphStyle(
        "AiText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )
    citation_style = ParagraphStyle(
        "CitationText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("OmniBrain Quant Research — Chat Export", title_style))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | OmniBrain Agentic Multi-Modal RAG",
            subtitle_style,
        )
    )

    # Top separator line
    divider = Table([[""]], colWidths=[540], rowHeights=[1])
    divider.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(divider)
    story.append(Spacer(1, 10))

    for idx, turn in enumerate(chat_history):
        q_num = idx + 1
        query_text = turn.get("query", "")
        response_text = turn.get("response", "")
        citations = turn.get("citations", [])

        # User Query Section
        story.append(Paragraph(f"Query #{q_num}", user_header_style))
        user_p = Paragraph(f"<b>User:</b> {_format_markdown_for_reportlab(query_text)}", user_bubble_style)
        user_table = Table([[user_p]], colWidths=[540])
        user_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(user_table)
        story.append(Spacer(1, 8))

        # AI Response Section
        story.append(Paragraph("OmniBrain Analysis & Quantitative Findings", ai_header_style))
        for block in response_text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            story.append(Paragraph(_format_markdown_for_reportlab(block), ai_text_style))

        # Citations & Evidence Section
        if citations:
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Verified Grounding Evidence & Citations:</b>", ai_header_style))
            for cit in citations:
                score_val = cit.get("relevance_score", cit.get("grounding_score", 1.0))
                score = round(score_val * 100)
                text = cit.get("text", cit.get("snippet", cit.get("executive_summary", str(cit))))
                title = cit.get("figure_title", cit.get("pdf_name", "Citation"))
                cit_p = Paragraph(
                    f"&bull; <b>{html.escape(str(title))}</b> [Faithfulness: {score}%]<br/>&nbsp;&nbsp;{_format_markdown_for_reportlab(text[:300])}",
                    citation_style,
                )
                story.append(cit_p)
                story.append(Spacer(1, 3))

        # Bottom turn divider
        story.append(Spacer(1, 8))
        story.append(divider)
        story.append(Spacer(1, 10))

    doc.build(story)


def _generate_fallback_pdf(markdown_content: str, pdf_path: str):
    """Fallback PDF generator using PyMuPDF (fitz) if ReportLab encounters an unexpected error."""
    import fitz

    doc = fitz.open()
    lines = markdown_content.splitlines()

    page = doc.new_page()
    pno = 0
    y = 50
    margin_x = 40
    line_height = 14

    for line in lines:
        if y > 750:
            page = doc.new_page()
            pno += 1
            y = 50

        # Basic heading check
        if line.startswith("# "):
            fontsize = 16
            clean_line = line.replace("# ", "")
        elif line.startswith("## "):
            fontsize = 13
            clean_line = line.replace("## ", "")
        else:
            fontsize = 10
            clean_line = line

        page.insert_text((margin_x, y), clean_line, fontsize=fontsize)
        y += line_height + (4 if fontsize > 10 else 0)

    doc.save(pdf_path)
    doc.close()


@router.post("/chat")
async def export_chat(req: ExportRequest):
    if req.format not in ["md", "pdf"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'md' or 'pdf'.")

    # Build Markdown Content
    markdown_content = (
        f"# OmniBrain Quant - Chat Export\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    for idx, turn in enumerate(req.chat_history):
        markdown_content += f"## Query {idx + 1}\n"
        markdown_content += f"**User:** {turn.get('query', '')}\n\n"
        markdown_content += f"**OmniBrain:**\n{turn.get('response', '')}\n\n"

        citations = turn.get("citations", [])
        if citations:
            markdown_content += "### Citations & Evidence\n"
            for cit in citations:
                score_val = cit.get("relevance_score", cit.get("grounding_score", 1.0))
                score = round(score_val * 100)
                text = cit.get("text", cit.get("snippet", cit.get("executive_summary", str(cit))))
                title = cit.get("figure_title", cit.get("pdf_name", "Citation"))
                markdown_content += (
                    f"- **{title}** (Faithfulness: {score}%)\n  > {text.replace(chr(10), ' ')}\n"
                )

        markdown_content += "\n---\n\n"

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    if req.format == "md":
        file_path = os.path.join(temp_dir, "OmniBrain_Chat_Export.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return FileResponse(
            path=file_path,
            filename="OmniBrain_Chat_Export.md",
            media_type="text/markdown",
        )

    if req.format == "pdf":
        pdf_path = os.path.join(temp_dir, "OmniBrain_Chat_Export.pdf")
        try:
            _generate_reportlab_pdf(req.chat_history, pdf_path)
        except Exception as e:
            logger.warning(f"ReportLab PDF generation encountered an error: {e}. Falling back to PyMuPDF...")
            try:
                _generate_fallback_pdf(markdown_content, pdf_path)
            except Exception as e2:
                logger.error(f"Both ReportLab and PyMuPDF fallback failed: {e2}")
                raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e2)}")

        return FileResponse(
            path=pdf_path,
            filename="OmniBrain_Chat_Export.pdf",
            media_type="application/pdf",
        )

