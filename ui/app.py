"""Streamlit Dashboard for OmniBrain Multi-Modal RAG Orchestrator (Task 2B Integration)."""

import base64
import json
import streamlit as st
from PIL import Image

from agents.cross_modal_verifier import CrossModalVerifier
from agents.memo_synthesizer import MemoSynthesizer
from agents.visual_analytics import VisualAnalyticsEngine
from agents.visual_comparator import VisualComparator
from agents.visual_routing_evaluator import VisualRoutingEvaluator
from agents.visual_tools import VisualMemoFormatter
from app.models.vision_schemas import (
    BoundingBox,
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    VerificationStatus,
)
from app.services.citation_renderer import CitationRenderer
from ui.components.visual_analytics_view import render_visual_analytics_panel

# Page Configuration
st.set_page_config(
    page_title="OmniBrain | Multi-Modal Visual Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 OmniBrain: Multi-Modal Visual Analytics & RAG")
st.caption("Pod Role 2B: Visual Analytics, Cross-Modal Grounding Verification & Investment Memo Synthesis")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Analyst Workspace")
    doc_file = st.file_uploader("Upload Corporate Financial PDF", type=["pdf", "png", "jpg"])
    
    st.divider()
    st.subheader("🎯 Query Intent Classifier")
    sample_query = st.text_area(
        "Enter Analyst Research Prompt",
        value="Verify if the CEO's revenue growth claim of 20% in the earnings release matches the Q1-Q4 bar chart on page 14.",
        height=80,
    )
    
    if st.button("Evaluate Routing Intent", use_container_width=True):
        intent_score = VisualRoutingEvaluator.evaluate_query(sample_query)
        st.write(f"**Primary Intent**: `{intent_score.primary_intent.value}`")
        st.write(f"**Confidence**: `{int(intent_score.confidence * 100)}%`")
        st.write(f"**Routing Recommendation**: {intent_score.reasoning}")

# Demo Visual Data Fixture for Interactive Testing
sample_chart = ExtractedChartData(
    title="FY24 Quarterly Operating Revenue & Margin Expansion",
    chart_type=ChartType.BAR,
    x_axis_label="Fiscal Quarter",
    y_axis_label="USD (Millions)",
    series=[
        ChartSeries(
            series_name="Operating Revenue",
            data_points=[
                DataPoint(label="Q1 FY24", value=120.5, raw_value="$120.5M", unit="USD Millions"),
                DataPoint(label="Q2 FY24", value=135.2, raw_value="$135.2M", unit="USD Millions"),
                DataPoint(label="Q3 FY24", value=148.8, raw_value="$148.8M", unit="USD Millions"),
                DataPoint(label="Q4 FY24", value=162.0, raw_value="$162.0M", unit="USD Millions"),
            ]
        ),
        ChartSeries(
            series_name="Operating Margin %",
            data_points=[
                DataPoint(label="Q1 FY24", value=22.5, raw_value="22.5%", unit="%"),
                DataPoint(label="Q2 FY24", value=23.8, raw_value="23.8%", unit="%"),
                DataPoint(label="Q3 FY24", value=25.0, raw_value="25.0%", unit="%"),
                DataPoint(label="Q4 FY24", value=27.2, raw_value="27.2%", unit="%"),
            ]
        )
    ],
    summary="Consolidated quarterly revenue demonstrated consistent expansion across FY24 from $120.5M to $162.0M with expanding operating margins.",
    confidence_score=0.98,
    bounding_box=BoundingBox(ymin=0.15, xmin=0.10, ymax=0.65, xmax=0.90, is_normalized=True),
)

sample_text_context = (
    "In fiscal year 2024, our revenue started strongly in Q1 with $120.5 million, growing steadily to $135.2 million in Q2. "
    "Third-quarter performance accelerated to $148.8 million, culminating in a record $162.0 million for Q4. "
    "Operating margins expanded to 27.2% by year-end."
)

# Compute Derived Intelligence
trends = VisualAnalyticsEngine.analyze_chart_dataset(sample_chart)
verification = CrossModalVerifier.verify_chart_against_text(sample_chart, sample_text_context)

# Render Citation Payload
mock_canvas = Image.new("RGB", (800, 1000), color=(245, 247, 250))
citation_payload = CitationRenderer.render_citation_overlay(
    image=mock_canvas,
    bounding_box=sample_chart.bounding_box or BoundingBox(ymin=0.2, xmin=0.2, ymax=0.8, xmax=0.8),
    figure_title=sample_chart.title,
    page_number=14,
    status=VerificationStatus.VERIFIED_MATCH if verification.discrepancy_count == 0 else VerificationStatus.DISCREPANCY_DETECTED,
    grounding_confidence=verification.grounding_score,
)

memo_block = VisualMemoFormatter.format_memo_block(
    figure_id="rev_margin_fy24",
    figure_title=sample_chart.title,
    chart_data=sample_chart,
    trends=trends,
    verification=verification,
    page_number=14,
)
memo_block.citation_payload = citation_payload

# Main Interface Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 Synthesized Investment Memo",
    "📊 Visual Analytics & Trend Explorer",
    "🛡️ Cross-Modal Grounding Audit",
    "🔍 Interactive Citation Bounding Box",
])

with tab1:
    st.subheader("Autonomous Multi-Modal Investment Memorandum")
    memo_markdown = MemoSynthesizer.synthesize_investment_memo(
        company_name="Apex Holdings Corp (NYSE: APEX)",
        analyst_query=sample_query,
        visual_blocks=[memo_block],
        text_context_snippets=[sample_text_context],
        sql_metrics={
            "Ticker": "APEX",
            "Current Stock Price": "$184.50",
            "52-Week High/Low": "$192.00 / $128.40",
            "P/E Ratio (TTM)": "24.6x",
        }
    )
    st.markdown(memo_markdown)

with tab2:
    st.subheader("Downstream Quantitative Trend Derivations")
    render_visual_analytics_panel(memo_block)

with tab3:
    st.subheader("Cross-Modal Discrepancy & Verification Engine")
    st.write("Cross-referencing parsed visual numbers from financial figures against textual citations:")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Parsed Visual Numbers**:")
        for s in sample_chart.series:
            st.write(f"**{s.series_name}**: " + ", ".join([f"{p.label}: {p.raw_value}" for p in s.data_points]))
    with col_b:
        st.markdown("**Document Text Citation**:")
        st.info(sample_text_context)

    st.markdown("#### Verification Results Table")
    ver_table = []
    for cr in verification.cross_references:
        ver_table.append({
            "Metric": cr.metric_name,
            "Visual Value": cr.visual_raw_string,
            "Text Value": str(cr.text_value),
            "Status": cr.status.value.upper(),
            "Variance %": f"{cr.variance_percentage}%" if cr.variance_percentage is not None else "0.0%",
            "Explanation": cr.explanation,
        })
    st.table(ver_table)

with tab4:
    st.subheader("Interactive Visual Citation & Document Grounding Snip")
    st.write(f"Clickable Citation Anchor: `{memo_block.citation_tag}`")
    if citation_payload.highlighted_page_base64:
        img_bytes = base64.b64decode(citation_payload.highlighted_page_base64)
        st.image(img_bytes, caption=f"PDF Page 14 Highlight: {sample_chart.title}", width=600)
