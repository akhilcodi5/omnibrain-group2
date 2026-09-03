"""NeMo Guardrails & Safety Service for OmniBrain Multi-Modal Orchestrator (Week 3)."""

import logging
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GuardrailCheckResult(BaseModel):
    """Container for input/output safety validation checks."""
    is_allowed: bool = True
    refusal_message: Optional[str] = None
    applied_rails: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    flagged_reasons: List[str] = Field(default_factory=list)


class GuardrailService:
    """Manages input/output guardrails, topical boundaries, and hallucination prevention."""

    OFF_TOPIC_PATTERNS = [
        r"\b(weather|forecast|rain|temperature)\b",
        r"\b(joke|funny|riddle|comedy)\b",
        r"\b(soccer|football|nfl|nba|world cup|olympics)\b",
        r"\b(recipe|cook|bake|pasta|pizza|cake)\b",
        r"\b(poem|poetry|rhyme|song lyrics)\b",
        r"\b(capital of|president of|celebrity|gossip)\b",
    ]

    JAILBREAK_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"you are now (dan|unrestricted)",
        r"bypass (all )?safety",
        r"system prompt",
        r"reveal (your )?instructions",
    ]

    DISCLAIMER_TEXT = (
        "\n\n> **Compliance Notice**: This investment memorandum is generated autonomously by "
        "OmniBrain for quantitative analytical research. It does not constitute formal legal, "
        "tax, or individualized investment advice."
    )

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or os.path.join(os.getcwd(), "guardrails")
        self._rails_app = None
        self._init_nemo_rails()

    def _init_nemo_rails(self):
        """Attempt to load native NeMo Guardrails if environment and dependencies permit."""
        try:
            from nemoguardrails import RailsConfig, LLMRails
            if os.path.exists(self.config_dir):
                config = RailsConfig.from_path(self.config_dir)
                self._rails_app = LLMRails(config)
                logger.info("NeMo Guardrails engine initialized successfully.")
        except Exception as e:
            logger.info(f"NeMo Guardrails native runtime unconfigured ({e}). Operating in deterministic policy mode.")

    def check_input_query(self, query: str) -> GuardrailCheckResult:
        """Validate incoming user queries against topical boundaries and jailbreak attempts."""
        q_lower = query.lower()

        # 1. Jailbreak and Prompt Injection Rail
        for pattern in self.JAILBREAK_PATTERNS:
            if re.search(pattern, q_lower):
                return GuardrailCheckResult(
                    is_allowed=False,
                    refusal_message="I cannot comply with requests that attempt to override system policies or security guardrails.",
                    applied_rails=["check_jailbreak"],
                    confidence_score=0.99,
                    flagged_reasons=["Jailbreak or prompt injection attempt detected."],
                )

        # 2. Out-of-Domain / Off-Topic Rail
        for pattern in self.OFF_TOPIC_PATTERNS:
            if re.search(pattern, q_lower):
                return GuardrailCheckResult(
                    is_allowed=False,
                    refusal_message=(
                        "I am OmniBrain, an agentic multi-modal research assistant specialized in financial document "
                        "and visual chart intelligence. I can only assist with questions regarding your uploaded corporate documents, "
                        "visual figures, and structured financial data."
                    ),
                    applied_rails=["check_off_topic"],
                    confidence_score=0.95,
                    flagged_reasons=["Query is outside the financial document intelligence domain."],
                )

        # Query passes all input rails
        return GuardrailCheckResult(
            is_allowed=True,
            applied_rails=["check_topical_domain", "check_jailbreak"],
            confidence_score=1.0,
        )

    def check_and_format_output(
        self,
        generated_response: str,
        is_grounded: bool = True,
        append_disclaimer: bool = True,
    ) -> str:
        """Enforce output guardrails, compliance notices, and grounding validation on generated memos."""
        output = generated_response

        # If grounding check failed, prefix with grounding warning
        if not is_grounded:
            output = (
                "> ⚠️ **Grounding Warning**: Some visual figures or text claims in this analysis could not be "
                "fully corroborated against the primary source PDF document.\n\n" + output
            )

        # Append compliance notice for investment memos
        if append_disclaimer and self.DISCLAIMER_TEXT not in output:
            output += self.DISCLAIMER_TEXT

        return output


_default_guardrail_service = None


def get_guardrail_service() -> GuardrailService:
    """Singleton getter for GuardrailService."""
    global _default_guardrail_service
    if _default_guardrail_service is None:
        _default_guardrail_service = GuardrailService()
    return _default_guardrail_service
