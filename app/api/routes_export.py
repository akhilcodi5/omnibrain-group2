import os
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
from datetime import datetime

# Attempt to import md2pdf
try:
    from md2pdf.core import md2pdf
    HAS_MD2PDF = True
except ImportError:
    HAS_MD2PDF = False

router = APIRouter(prefix="/api/v1/export", tags=["Export"])

class ExportRequest(BaseModel):
    format: str
    chat_history: List[Dict[str, Any]]

@router.post("/chat")
async def export_chat(req: ExportRequest):
    if req.format not in ['md', 'pdf']:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'md' or 'pdf'.")
        
    markdown_content = f"# OmniBrain Quant - Chat Export\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for idx, turn in enumerate(req.chat_history):
        markdown_content += f"## Query {idx + 1}\n"
        markdown_content += f"**User:** {turn.get('query', '')}\n\n"
        markdown_content += f"**OmniBrain:**\n{turn.get('response', '')}\n\n"
        
        citations = turn.get('citations', [])
        if citations:
            markdown_content += f"### Citations & Evidence\n"
            for cit in citations:
                score_val = cit.get('relevance_score', cit.get('grounding_score', 1.0))
                score = round(score_val * 100)
                text = cit.get('text', cit.get('snippet', cit.get('executive_summary', str(cit))))
                title = cit.get('figure_title', cit.get('pdf_name', 'Citation'))
                markdown_content += f"- **{title}** (Faithfulness: {score}%)\n  > {text.replace(chr(10), ' ')}\n"
        
        markdown_content += "\n---\n\n"

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    
    if req.format == 'md':
        file_path = os.path.join(temp_dir, "export.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return FileResponse(path=file_path, filename="OmniBrain_Export.md", media_type="text/markdown")
        
    if req.format == 'pdf':
        if not HAS_MD2PDF:
            # Fallback to MD if PDF generation library is missing
            raise HTTPException(status_code=501, detail="PDF generation library 'md2pdf' is not installed.")
            
        md_path = os.path.join(temp_dir, "temp.md")
        pdf_path = os.path.join(temp_dir, "export.pdf")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        try:
            md2pdf(pdf_path, md_content=markdown_content)
            return FileResponse(path=pdf_path, filename="OmniBrain_Export.pdf", media_type="application/pdf")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
