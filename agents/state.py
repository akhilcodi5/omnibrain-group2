"""LangGraph AgentState schema definition for supervisor-worker graph execution."""

from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Shared state object passed between LangGraph supervisor and sub-agents."""
    
    # Conversation messages & memory
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Original user query and routing decisions
    query: str
    pdf_name: Optional[str]
    next_agent: Optional[str]
    trace_id: Optional[str]
    
    # Retrieval and context payloads
    retrieved_docs: List[Dict[str, Any]]
    
    # Multi-Modal Vision payloads (charts, tables, extracted images)
    visual_evidence: List[Dict[str, Any]]
    referenced_images: List[str]
    
    # Structured SQL results
    sql_query: Optional[str]
    sql_results: Optional[List[Dict[str, Any]]]
    
    # Self-RAG reflection & confidence metrics
    is_grounded: bool
    retrieval_confidence: float
    iteration_count: int
    
    # Final synthesized output and citations
    final_response: Optional[str]
    citations: List[Dict[str, Any]]
