"""Streamlit Visual Analytics & Citation Preview UI Component (Task 2B)."""

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional
import streamlit as st
from PIL import Image

from app.models.vision_schemas import (
    CrossModalVerificationReport,
    VisualAnalyticalMemoBlock,
    VisualTrendAnalysis,
)


def render_visual_analytics_panel(memo_block: VisualAnalyticalMemoBlock):
    """Render an interactive visual analytical card within the Streamlit dashboard."""
    with st.container():
        st.markdown(f"### 📊 {memo_block.figure_title}")
        st.caption(f"Citation Anchor: `{memo_block.citation_tag}` | PDF Page: `{memo_block.page_number or 'N/A'}`")

        # 1. Executive Summary & KPI Metrics
        st.info(memo_block.executive_summary)

        if memo_block.trend_analytics:
            cols = st.columns(min(4, len(memo_block.trend_analytics) * 2))
            for idx, trend in enumerate(memo_block.trend_analytics):
                col_idx = (idx * 2) % len(cols)
                with cols[col_idx]:
                    direction_icon = "📈" if trend.trend_direction.value == "upward" else "📉" if trend.trend_direction.value == "downward" else "⚖️"
                    st.metric(
                        label=f"{trend.series_name} Trajectory",
                        value=f"{direction_icon} {trend.trend_direction.value.upper()}",
                        delta=f"{trend.total_percentage_change}%" if trend.total_percentage_change else None,
                    )
                if trend.cagr_percentage is not None and col_idx + 1 < len(cols):
                    with cols[col_idx + 1]:
                        st.metric(
                            label=f"{trend.series_name} CAGR",
                            value=f"{trend.cagr_percentage}%",
                        )

        # 2. Cross-Modal Grounding Verification Status
        if memo_block.verification_report:
            ver = memo_block.verification_report
            grounding_pct = int(ver.grounding_score * 100)

            if ver.discrepancy_count == 0:
                st.success(f"🟢 **100% Grounded in Document**: All {ver.matched_metrics_count} visual numbers matched text context.")
            else:
                st.warning(f"🟡 **Cross-Modal Verification ({grounding_pct}% Grounded)**: {ver.matched_metrics_count} matched, {ver.discrepancy_count} discrepancy flagged.")

            if ver.critical_discrepancies:
                with st.expander("⚠️ View Flagged Contradictions & Discrepancies", expanded=True):
                    for disc in ver.critical_discrepancies:
                        st.error(disc)

        # 3. Interactive Visual Citation & Page Overlay Preview
        with st.expander("🔍 Interactive Visual Citation & Bounding Box Overlay", expanded=False):
            if memo_block.citation_payload and memo_block.citation_payload.highlighted_page_base64:
                img_data = base64.b64decode(memo_block.citation_payload.highlighted_page_base64)
                st.image(img_data, caption=f"PDF Page Highlight: {memo_block.figure_title}", use_container_width=True)
            elif memo_block.citation_payload and memo_block.citation_payload.thumbnail_snippet_base64:
                img_data = base64.b64decode(memo_block.citation_payload.thumbnail_snippet_base64)
                st.image(img_data, caption=f"Extracted Figure Snip: {memo_block.figure_title}", width=400)
            else:
                st.write("Visual asset preview available upon PDF ingestion.")

        # 4. Anomaly Warnings
        if memo_block.risk_warnings:
            for w in memo_block.risk_warnings:
                st.warning(f"⚠️ {w}")

        st.divider()
