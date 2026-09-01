"""Agent query and chat endpoints for LangGraph supervisor orchestration."""

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.memo_synthesizer import MemoSynthesizer
from agents.supervisor import create_supervisor_graph
from app.models.vision_schemas import VisualAnalyticalMemoBlock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Agent Chat & Orchestration"])


class ChatQueryRequest(BaseModel):
    query: str = Field(..., description="The user or quantitative analyst research question.")
    pdf_name: Optional[str] = Field(None, description="Optional target PDF document to focus analysis.")
    referenced_images: List[str] = Field(default_factory=list, description="Optional list of image file paths or base64.")


class ChatQueryResponse(BaseModel):
    query: str
    final_response: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    visual_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    sql_results: Optional[List[Dict[str, Any]]] = None
    is_grounded: bool = True
    execution_time_seconds: float


class MemoGenerationRequest(BaseModel):
    company_name: str
    research_query: str
    visual_blocks: List[VisualAnalyticalMemoBlock] = Field(default_factory=list)
    text_snippets: List[str] = Field(default_factory=list)
    sql_metrics: Optional[Dict[str, Any]] = None


class MemoGenerationResponse(BaseModel):
    company_name: str
    investment_memo_markdown: str


@router.post("/query", response_model=ChatQueryResponse)
async def query_agent_orchestrator(req: ChatQueryRequest):
    """Execute multi-agent query routing across Search, Visual Analytics, and SQL via LangGraph."""
    start_time = time.time()
    try:
        graph = create_supervisor_graph()

        initial_state = {
            "messages": [],
            "query": req.query,
            "next_agent": None,
            "retrieved_docs": [],
            "visual_evidence": [],
            "referenced_images": req.referenced_images,
            "sql_query": None,
            "sql_results": None,
            "is_grounded": True,
            "retrieval_confidence": 1.0,
            "iteration_count": 0,
            "final_response": None,
            "citations": [],
        }

        # Execute LangGraph state machine
        final_state = await graph.ainvoke(initial_state)

        elapsed = round(time.time() - start_time, 2)

        return ChatQueryResponse(
            query=req.query,
            final_response=final_state.get("final_response") or "Analysis completed.",
            citations=final_state.get("citations", []),
            visual_evidence=final_state.get("visual_evidence", []),
            sql_results=final_state.get("sql_results"),
            is_grounded=final_state.get("is_grounded", True),
            execution_time_seconds=elapsed,
        )
    except Exception as e:
        logger.error(f"Error executing agent orchestrator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memo", response_model=MemoGenerationResponse)
async def generate_investment_memo_endpoint(req: MemoGenerationRequest):
    """Directly synthesize a multi-modal investment research memorandum."""
    try:
        memo_md = MemoSynthesizer.synthesize_investment_memo(
            company_name=req.company_name,
            analyst_query=req.research_query,
            visual_blocks=req.visual_blocks,
            text_context_snippets=req.text_snippets,
            sql_metrics=req.sql_metrics,
        )
        return MemoGenerationResponse(
            company_name=req.company_name,
            investment_memo_markdown=memo_md,
        )
    except Exception as e:
        logger.error(f"Error generating investment memo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
