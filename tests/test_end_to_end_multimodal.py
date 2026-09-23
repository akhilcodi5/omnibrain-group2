"""End-to-End Multi-Modal Integration Tests (Task 2B: Visual Analytics + Vector Store + Self-RAG)."""

import pytest
from PIL import Image

from agents.cross_modal_self_rag import CrossModalSelfRAG
from agents.memo_synthesizer import MemoSynthesizer
from agents.vision_agent import VisualAnalyticsIntegratorAgent
from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    VerificationStatus,
)
from storage.vector_store import VectorStore


@pytest.fixture
def mock_vector_store():
    """Create in-memory vector store populated with sample corporate report chunks."""
    vs = VectorStore(in_memory=True)
    vs.add_chunks(
        texts=[
            "Apex Holdings reported Q1 2024 revenue of $120.5 million with strong margin discipline.",
            "In the second quarter, Apex expanded quarterly revenue to $135.2 million.",
            "Q3 2024 operating revenue accelerated to $148.8 million, driven by cloud services expansion.",
            "Q4 2024 reached record quarterly revenue of $162.0 million, closing out a strong fiscal year.",
        ],
        metadatas=[
            {"pdf_name": "apex_annual_report.pdf", "page_number": 12, "chunk_id": "c1"},
            {"pdf_name": "apex_annual_report.pdf", "page_number": 13, "chunk_id": "c2"},
            {"pdf_name": "apex_annual_report.pdf", "page_number": 14, "chunk_id": "c3"},
            {"pdf_name": "apex_annual_report.pdf", "page_number": 15, "chunk_id": "c4"},
        ]
    )
    return vs


def test_cross_modal_self_rag_query_generation():
    """Test generating targeted visual search queries from chart series."""
    chart = ExtractedChartData(
        title="Operating Revenue",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Consolidated",
                data_points=[
                    DataPoint(label="Q1", value=120.5, raw_value="$120.5M"),
                    DataPoint(label="Q3", value=148.8, raw_value="$148.8M"),
                ]
            )
        ],
        summary="Quarterly revenue trajectory."
    )

    self_rag = CrossModalSelfRAG()
    queries = self_rag.generate_targeted_search_queries(chart, original_query="Analyze Apex revenue")

    assert len(queries) >= 2
    assert any("$120.5M" in q for q in queries)
    assert any("$148.8M" in q for q in queries)


def test_cross_modal_self_rag_fact_check_loop(mock_vector_store):
    """Test iterative vector store retrieval and verification of visual numbers."""
    chart = ExtractedChartData(
        title="Operating Revenue",
        chart_type=ChartType.BAR,
        series=[
            ChartSeries(
                series_name="Revenue",
                data_points=[
                    DataPoint(label="Q1", value=120.5, raw_value="$120.5M"),
                    DataPoint(label="Q3", value=148.8, raw_value="$148.8M"),
                    DataPoint(label="Q4", value=162.0, raw_value="$162.0M"),
                ]
            )
        ],
        summary="Q1 to Q4 revenue expansion."
    )

    self_rag = CrossModalSelfRAG(vector_store=mock_vector_store)
    # Start with empty text context so Self-RAG must query vector store
    report = self_rag.execute_visual_fact_check_loop(
        chart_data=chart,
        original_query="What were the quarterly revenue numbers?",
        initial_text_context="",
    )

    assert report.total_metrics_evaluated == 3
    assert report.matched_metrics_count == 3
    assert report.discrepancy_count == 0
    assert report.grounding_score == 1.0


@pytest.mark.asyncio
async def test_end_to_end_agent_pipeline(mock_vector_store):
    """Test the full VisualAnalyticsIntegratorAgent pipeline synthesizing a memo block."""
    agent = VisualAnalyticsIntegratorAgent(vector_store=mock_vector_store)
    img = Image.new("RGB", (600, 400), color="white")

    memo_block = await agent.analyze_and_verify_figure(
        image_input=img,
        query_context="Analyze quarterly financial performance",
        page_number=14,
        use_self_rag_loop=True,
    )

    assert memo_block.figure_title is not None
    assert len(memo_block.trend_analytics) > 0
    assert memo_block.verification_report is not None
    assert memo_block.citation_tag == f"[{memo_block.figure_title} - Page 14]"
    assert "### 📊 Visual Analytical Evidence" in memo_block.markdown_formatted_block

    # Test synthesizing into final investment memo
    memo = MemoSynthesizer.synthesize_investment_memo(
        company_name="Apex Holdings",
        analyst_query="Analyze quarterly performance",
        visual_blocks=[memo_block],
    )
    assert "# 📑 Investment Research Memorandum: Apex Holdings" in memo
    assert "## 2. Multi-Modal Visual Analytics & Trend Analysis" in memo
