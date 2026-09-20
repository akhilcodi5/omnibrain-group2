"""Agent query and chat endpoints for LangGraph supervisor orchestration with NeMo Guardrails and Langfuse Telemetry."""

import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.memo_synthesizer import MemoSynthesizer
from agents.supervisor import create_supervisor_graph
from app.core.telemetry import get_telemetry_manager
from app.models.vision_schemas import VisualAnalyticalMemoBlock
from eval.evaluator import MultiModalEvalReport, MultiModalRAGEvaluator
from guardrails.guardrail_service import get_guardrail_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Agent Chat & Orchestration"])


class ChatQueryRequest(BaseModel):
    query: str = Field(..., description="The user or quantitative analyst research question.")
    pdf_name: Optional[str] = Field(None, description="Optional target PDF document to focus analysis.")
    referenced_images: List[str] = Field(default_factory=list, description="Optional list of image file paths or base64.")
    thread_id: Optional[str] = Field(None, description="Thread ID for chat history persistence.")



class ChatQueryResponse(BaseModel):
    query: str
    final_response: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    visual_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    sql_results: Optional[List[Dict[str, Any]]] = None
    is_grounded: bool = True
    guardrails_triggered: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    telemetry_metrics: Optional[Dict[str, Any]] = None
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
    """Execute multi-agent query routing across Search, Visual Analytics, and SQL via LangGraph with NeMo Guardrails and Langfuse."""
    start_time = time.time()
    guardrails = get_guardrail_service()
    telemetry = get_telemetry_manager()
    evaluator = MultiModalRAGEvaluator()

    # 1. Create Langfuse Trace
    trace_id = telemetry.create_trace(
        name="MultiModal_Supervisor_Orchestration",
        metadata={"query": req.query, "pdf_name": req.pdf_name},
    )

    # 2. Enforce Input Guardrails (Topical boundaries & Jailbreak prevention)
    input_check = guardrails.check_input_query(req.query)
    if not input_check.is_allowed:
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Guardrail blocked input query: {input_check.flagged_reasons}")
        
        telemetry.log_agent_step(
            trace_id=trace_id,
            agent_name="NeMoGuardrails",
            action="BlockInput",
            input_data=req.query,
            output_data=input_check.refusal_message,
            latency_seconds=elapsed,
        )

        return ChatQueryResponse(
            query=req.query,
            final_response=input_check.refusal_message or "Query blocked by domain safety guardrails.",
            citations=[],
            visual_evidence=[],
            is_grounded=True,
            guardrails_triggered=input_check.applied_rails,
            trace_id=trace_id,
            execution_time_seconds=elapsed,
        )

    # 3. Execute LangGraph Multi-Agent State Machine
    try:
        graph = create_supervisor_graph()

        # Map frontend URL paths back to local file paths
        local_images = []
        for img in req.referenced_images:
            if img.startswith("/api/v1/images/"):
                basename = img.split("/")[-1]
                basename = urllib.parse.unquote(basename)
                local_images.append(f"storage/extracted_images/{basename}")
            else:
                local_images.append(img)

        initial_state = {
            "messages": [],
            "query": req.query,
            "next_agent": None,
            "retrieved_docs": [],
            "visual_evidence": [],
            "referenced_images": local_images,
            "sql_query": None,
            "sql_results": None,
            "is_grounded": True,
            "retrieval_confidence": 1.0,
            "iteration_count": 0,
            "final_response": None,
            "citations": [],
            "trace_id": trace_id,
        }

        # Initialize LangChain HumanMessage from user query
        from langchain_core.messages import HumanMessage
        initial_state["messages"] = [HumanMessage(content=req.query)]

        import uuid
        thread_id = req.thread_id or str(uuid.uuid4())
        logger.info(f"Executing LangGraph supervisor pipeline for thread: {thread_id}")

        final_state = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )


        # 4. Enforce Output Guardrails (Compliance & Grounding check)
        raw_memo = final_state.get("final_response") or "Analysis completed."
        is_grounded = final_state.get("is_grounded", True)
        guarded_output = guardrails.check_and_format_output(raw_memo, is_grounded=is_grounded)

        elapsed = round(time.time() - start_time, 2)

        # 5. Run Automated Multi-Modal RAG Evaluation
        eval_report = evaluator.evaluate_pipeline_output(
            generated_memo=guarded_output,
            retrieved_docs=final_state.get("retrieved_docs", []),
            visual_evidence=final_state.get("visual_evidence", []),
            citations=final_state.get("citations", []),
            trace_id=trace_id,
        )

        telemetry.log_agent_step(
            trace_id=trace_id,
            agent_name="LangGraphSupervisor",
            action="SynthesizeMemo",
            input_data=req.query,
            output_data=guarded_output[:300],
            prompt_tokens=450,
            completion_tokens=320,
            latency_seconds=elapsed,
        )

        trace_summary = telemetry.get_trace_summary(trace_id)

        return ChatQueryResponse(
            query=req.query,
            final_response=guarded_output,
            citations=final_state.get("citations", []),
            visual_evidence=final_state.get("visual_evidence", []),
            sql_results=final_state.get("sql_results"),
            is_grounded=is_grounded,
            guardrails_triggered=input_check.applied_rails,
            trace_id=trace_id,
            evaluation_summary=eval_report.model_dump(),
            telemetry_metrics=trace_summary,
            execution_time_seconds=elapsed,
        )
    except Exception as e:
        logger.error(f"Error executing agent orchestrator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memo", response_model=MemoGenerationResponse)
async def generate_investment_memo_endpoint(req: MemoGenerationRequest):
    """Directly synthesize a multi-modal investment research memorandum with compliance disclosures."""
    guardrails = get_guardrail_service()
    try:
        memo_md = MemoSynthesizer.synthesize_investment_memo(
            company_name=req.company_name,
            analyst_query=req.research_query,
            visual_blocks=req.visual_blocks,
            text_context_snippets=req.text_snippets,
            sql_metrics=req.sql_metrics,
        )
        guarded_memo = guardrails.check_and_format_output(memo_md, is_grounded=True)

        return MemoGenerationResponse(
            company_name=req.company_name,
            investment_memo_markdown=guarded_memo,
        )
    except Exception as e:
        logger.error(f"Error generating investment memo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
