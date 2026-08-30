"""Streamlit component for displaying interactive visual page citations, bounding-box snips, and extracted figures."""

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional
import streamlit as st
from PIL import Image

from app.models.vision_schemas import (
    VerificationStatus,
    VisualCitationPayload,
)


def render_citation_badge(citation: Dict[str, Any]):
    """Render a compact clickable citation chip."""
    tag = citation.get("citation_tag", "[Citation]")
    grounding = citation.get("grounding_score", 1.0)
    score_pct = int(grounding * 100)
    
    badge_color = "🟢" if score_pct >= 80 else "🟡" if score_pct >= 50 else "🔴"
    st.markdown(f"`{tag}` {badge_color} **{score_pct}% Grounded**")


def render_citation_gallery(citations: List[VisualCitationPayload]):
    """Render an interactive gallery grid of visual citations with bounding-box highlights."""
    if not citations:
        st.info("No visual citations attached to the current analysis.")
        return

    st.markdown("#### 🖼️ Referenced Document Figures & Citations")
    
    cols = st.columns(min(3, len(citations)))
    for idx, cit in enumerate(citations):
        col_idx = idx % len(cols)
        with cols[col_idx]:
            status_emoji = (
                "🟢 Verified" if cit.status == VerificationStatus.VERIFIED_MATCH
                else "🔴 Discrepancy" if cit.status == VerificationStatus.DISCREPANCY_DETECTED
                else "🟡 Visual Only"
            )
            
            st.markdown(f"**{cit.figure_title}**")
            st.caption(f"Page {cit.page_number or 'N/A'} | `{status_emoji}`")

            # Render Thumbnail
            if cit.thumbnail_snippet_base64:
                thumb_bytes = base64.b64decode(cit.thumbnail_snippet_base64)
                st.image(thumb_bytes, caption=cit.citation_tag, use_container_width=True)

            with st.popover(f"🔍 Drill Down: Page {cit.page_number or ''}"):
                st.write(f"### {cit.figure_title}")
                st.write(f"**Grounding Confidence**: `{int(cit.grounding_confidence * 100)}%`")
                if cit.highlighted_page_base64:
                    full_page_bytes = base64.b64decode(cit.highlighted_page_base64)
                    st.image(full_page_bytes, caption=f"Highlighted PDF Page {cit.page_number}", use_container_width=True)
                st.markdown(f"**Anchor**: `{cit.citation_tag}`")


def render_citation_sidebar_list(citations: List[Dict[str, Any]]):
    """Render list of active citations in Streamlit sidebar."""
    if not citations:
        return
    
    st.sidebar.markdown("### 📌 Active Citations")
    for cit in citations:
        tag = cit.get("citation_tag") or cit.get("citation_id") or "Citation"
        source = cit.get("source") or cit.get("pdf_name") or "Document"
        page = cit.get("page_number", "")
        st.sidebar.caption(f"• `{tag}` — {source} (P. {page})")
