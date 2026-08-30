"""Streamlit component for rendering agent multi-step thought process, tool calls, and execution telemetry."""

from typing import Any, Dict, List, Optional
import streamlit as st


def render_thought_trace(
    agent_steps: List[Dict[str, Any]],
    execution_time_seconds: Optional[float] = None,
):
    """Render a step-by-step collapsible thought trace of LangGraph agent reasoning."""
    with st.expander("🧠 Agent Thought Process & Tool Execution Trace", expanded=False):
        if execution_time_seconds:
            st.caption(f"⚡ Total Multi-Agent Pipeline Latency: **{execution_time_seconds:.2f}s**")

        if not agent_steps:
            st.info("No active thought steps recorded.")
            return

        for idx, step in enumerate(agent_steps, 1):
            agent_name = step.get("agent", "Supervisor")
            action = step.get("action", "Reasoning")
            thought = step.get("thought", "")
            tool_called = step.get("tool")
            status = step.get("status", "completed")

            status_icon = "✅" if status == "completed" else "🔄" if status == "running" else "⚠️"
            
            st.markdown(f"**Step {idx}: {status_icon} [{agent_name}] → {action}**")
            
            if thought:
                st.markdown(f"> *{thought}*")
                
            if tool_called:
                st.code(f"Tool Invoked: {tool_called}\nArgs: {step.get('args', {})}", language="json")

            if idx < len(agent_steps):
                st.divider()


def render_live_step_indicator(current_agent: str, current_action: str):
    """Display an active live execution toast/spinner indicator."""
    return st.status(f"🤖 **{current_agent}** is executing: *{current_action}*...", expanded=True)
