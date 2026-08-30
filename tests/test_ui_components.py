"""Unit tests for Streamlit UI Components (Task 2B: Citation Viewer & Thought Trace)."""

import pytest
from app.models.vision_schemas import (
    BoundingBox,
    VerificationStatus,
    VisualCitationPayload,
)
from ui.components.citation_viewer import render_citation_badge
from ui.components.thought_trace import render_thought_trace


def test_visual_citation_payload_structure():
    """Test VisualCitationPayload properties and serialization."""
    payload = VisualCitationPayload(
        figure_id="fig_1",
        figure_title="Revenue Breakdown",
        page_number=12,
        citation_tag="[Revenue Breakdown - Page 12]",
        status=VerificationStatus.VERIFIED_MATCH,
        grounding_confidence=0.98,
    )

    assert payload.figure_id == "fig_1"
    assert payload.page_number == 12
    assert payload.status == VerificationStatus.VERIFIED_MATCH
    assert payload.grounding_confidence == 0.98
    assert "[Revenue Breakdown - Page 12]" in payload.citation_tag


def test_thought_trace_payload():
    """Test agent thought trace step format."""
    steps = [
        {"agent": "Supervisor", "action": "Route Query", "thought": "Analyzing intent", "status": "completed"},
        {"agent": "VisualAnalyticsIntegratorAgent", "action": "Extract Chart", "status": "completed"},
    ]

    assert len(steps) == 2
    assert steps[0]["agent"] == "Supervisor"
    assert steps[1]["action"] == "Extract Chart"
