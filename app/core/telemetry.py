"""Langfuse Observability & Telemetry Service for multi-agent tracing and evaluation."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Manages Langfuse distributed tracing, token consumption logging, and evaluation metrics."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.is_enabled = bool(
            self.public_key
            and self.secret_key
            and not self.public_key.startswith("your_")
        )

        self._langfuse_client = None
        self._in_memory_traces: Dict[str, Dict[str, Any]] = {}

        if self.is_enabled:
            try:
                from langfuse import Langfuse
                self._langfuse_client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                logger.info("Langfuse telemetry client initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse SDK ({e}). Falling back to in-memory tracing.")
                self.is_enabled = False
        else:
            logger.info("Langfuse unconfigured or using placeholder keys. Operating in in-memory telemetry mode.")

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Initialize a new parent execution trace for a user query."""
        trace_id = f"trace_{int(time.time() * 1000)}_{name}"
        
        trace_data = {
            "trace_id": trace_id,
            "name": name,
            "user_id": user_id or "default_analyst",
            "start_time": time.time(),
            "metadata": metadata or {},
            "generations": [],
            "scores": [],
        }

        self._in_memory_traces[trace_id] = trace_data

        if self.is_enabled and self._langfuse_client:
            try:
                self._langfuse_client.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id,
                    metadata=metadata,
                )
                logger.info(f"Telemetry sent to Langfuse: Trace '{name}' initialized with id={trace_id}")
            except Exception as e:
                logger.error(f"Langfuse trace creation error: {e}")

        return trace_id

    def log_agent_step(
        self,
        trace_id: str,
        agent_name: str,
        action: str,
        model: str = "gpt-4o",
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_seconds: float = 0.0,
    ):
        """Record an agent execution step, token counts, and latency."""
        step_record = {
            "agent_name": agent_name,
            "action": action,
            "model": model,
            "input": str(input_data)[:200] if input_data else None,
            "output": str(output_data)[:200] if output_data else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": round(latency_seconds, 3),
            "timestamp": time.time(),
        }

        if trace_id in self._in_memory_traces:
            self._in_memory_traces[trace_id]["generations"].append(step_record)

        if self.is_enabled and self._langfuse_client:
            try:
                self._langfuse_client.generation(
                    trace_id=trace_id,
                    name=f"{agent_name}_{action}",
                    model=model,
                    input=input_data,
                    output=output_data,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                )
            except Exception as e:
                logger.error(f"Langfuse generation logging error: {e}")
                
        # Always log payload explicitly to terminal
        logger.info(
            f"--- [AGENT STEP: {agent_name}] ---\n"
            f"Action: {action}\n"
            f"Input: {str(input_data)[:500] if input_data else 'None'}\n"
            f"Output: {str(output_data)[:500] if output_data else 'None'}\n"
            f"Tokens: {prompt_tokens} prompt / {completion_tokens} completion\n"
            f"Latency: {round(latency_seconds, 3)}s\n"
            f"Trace ID: {trace_id}\n"
            f"--------------------------"
        )

    def log_span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        level: str = "DEFAULT",
    ):
        """Record a functional span (e.g., non-LLM python logic, DB extraction)."""
        if self.is_enabled and self._langfuse_client:
            try:
                self._langfuse_client.span(
                    trace_id=trace_id,
                    name=name,
                    input=input_data,
                    output=output_data,
                    level=level,
                )
            except Exception as e:
                logger.error(f"Langfuse span logging error: {e}")

        # Always log to terminal
        logger.info(
            f"--- [SPAN: {name}] ---\n"
            f"Input: {str(input_data)[:500] if input_data else 'None'}\n"
            f"Output: {str(output_data)[:500] if output_data else 'None'}\n"
            f"Trace ID: {trace_id}\n"
            f"--------------------------"
        )

    def log_event(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        level: str = "DEFAULT",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a point-in-time event (e.g., app startup, metadata extraction)."""
        if self.is_enabled and self._langfuse_client:
            try:
                self._langfuse_client.event(
                    trace_id=trace_id,
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Langfuse event logging error: {e}")
                
        # Always log to terminal
        import json
        meta_str = json.dumps(metadata, indent=2) if metadata else "{}"
        logger.info(
            f"--- [EVENT: {name}] ---\n"
            f"Metadata:\n{meta_str}\n"
            f"Trace ID: {trace_id}\n"
            f"--------------------------"
        )

    def log_evaluation_score(
        self,
        trace_id: str,
        metric_name: str,
        score: float,
        comment: Optional[str] = None,
    ):
        """Record an evaluation or grounding metric score to Langfuse."""
        score_record = {
            "name": metric_name,
            "value": score,
            "comment": comment,
            "timestamp": time.time(),
        }

        if trace_id in self._in_memory_traces:
            self._in_memory_traces[trace_id]["scores"].append(score_record)

        if self.is_enabled and self._langfuse_client:
            try:
                self._langfuse_client.score(
                    trace_id=trace_id,
                    name=metric_name,
                    value=score,
                    comment=comment,
                )
                logger.info(f"Telemetry sent to Langfuse: Score logged metric={metric_name} value={score} (trace_id={trace_id})")
            except Exception as e:
                logger.error(f"Langfuse score logging error: {e}")

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Retrieve aggregated token usage, cost, and latency for a trace."""
        trace = self._in_memory_traces.get(trace_id, {})
        generations = trace.get("generations", [])
        
        total_prompt_tokens = sum(g.get("prompt_tokens", 0) for g in generations)
        total_completion_tokens = sum(g.get("completion_tokens", 0) for g in generations)
        total_tokens = total_prompt_tokens + total_completion_tokens
        total_latency = sum(g.get("latency_seconds", 0) for g in generations)

        # Approximate GPT-4o cost ($5.00/1M prompt, $15.00/1M completion)
        est_cost_usd = (total_prompt_tokens * 0.000005) + (total_completion_tokens * 0.000015)

        return {
            "trace_id": trace_id,
            "name": trace.get("name", "unknown"),
            "step_count": len(generations),
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "estimated_cost_usd": round(est_cost_usd, 6),
            "total_latency_seconds": round(total_latency, 3),
            "scores": trace.get("scores", []),
        }

    def flush(self):
        """Flush pending events to Langfuse cloud."""
        if self.is_enabled and self._langfuse_client:
            try:
                self._langfuse_client.flush()
            except Exception as e:
                logger.error(f"Langfuse flush error: {e}")


_default_telemetry = None


def get_telemetry_manager() -> TelemetryManager:
    """Singleton getter for TelemetryManager."""
    global _default_telemetry
    if _default_telemetry is None:
        _default_telemetry = TelemetryManager()
    return _default_telemetry
