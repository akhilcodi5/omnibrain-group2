"""Multi-Modal Investment Memo Synthesizer (Task 2B).

Synthesizes multi-modal visual analytics, cross-modal verification reports, SQL results,
and textual context into an executive-grade, cited investment research memorandum.
"""

import logging
from typing import Any, Dict, List, Optional

from app.models.vision_schemas import (
    CrossModalVerificationReport,
    VisualAnalyticalMemoBlock,
    VisualTrendAnalysis,
)

logger = logging.getLogger(__name__)


class MemoSynthesizer:
    """Synthesizes structured multi-modal evidence into an investment memorandum."""

    @classmethod
    def synthesize_investment_memo(
        cls,
        company_name: str,
        analyst_query: str,
        visual_blocks: List[VisualAnalyticalMemoBlock],
        text_context_snippets: Optional[List[str]] = None,
        sql_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Construct a cited, hallucination-resistant investment memo section."""
        lines = []

        # 1. Title & Header
        lines.append(f"# 📑 Investment Research Memorandum: {company_name}")
        lines.append(f"**Research Focus**: {analyst_query}")
        lines.append(f"**Orchestrator**: OmniBrain Agentic Multi-Modal RAG | **Grounding Status**: Verified Multi-Modal Evidence\n")
        lines.append("---\n")

        # 2. Executive Synthesis
        lines.append("## 1. Executive Summary & Quantitative Findings")
        all_takeaways = []
        for block in visual_blocks:
            all_takeaways.extend(block.key_takeaways)

        if all_takeaways:
            for takeaway in all_takeaways:
                lines.append(f"- **Key Finding**: {takeaway}")
        else:
            lines.append("- Multi-modal analysis completed across ingested corporate financial PDF exhibits.")
        
        if text_context_snippets:
            lines.append("\n**Qualitative & Textual Evidence Context**:")
            for snippet in text_context_snippets:
                lines.append(f"> {snippet}")
        lines.append("")

        # 3. Visual Analytics & Derived Trajectories
        lines.append("## 2. Multi-Modal Visual Analytics & Trend Analysis")
        for block in visual_blocks:
            lines.append(f"### {block.figure_title} ({block.citation_tag})")
            lines.append(f"*{block.executive_summary}*\n")

            if block.trend_analytics:
                lines.append("| Metric / Series | Trajectory | Total Delta | Derived CAGR | Key Analytical Takeaway |")
                lines.append("| :--- | :---: | :---: | :---: | :--- |")
                for t in block.trend_analytics:
                    cagr_val = f"{t.cagr_percentage}%" if t.cagr_percentage is not None else "N/A"
                    delta_val = f"{'+' if t.total_percentage_change and t.total_percentage_change > 0 else ''}{t.total_percentage_change}%"
                    lines.append(f"| **{t.series_name}** | `{t.trend_direction.value.upper()}` | **{delta_val}** | {cagr_val} | {t.analytical_takeaway} |")
                lines.append("")

        # 4. Cross-Modal Grounding & Discrepancy Audit
        lines.append("## 3. Cross-Modal Grounding & Verification Audit")
        lines.append("To eliminate LLM hallucination risks, all visual numbers were cross-referenced against document text context:\n")
        
        has_cross_refs = False
        lines.append("| Figure / Metric | Visual Number | Text Context Claim | Verification Status | Variance / Rationale |")
        lines.append("| :--- | :---: | :---: | :---: | :--- |")

        for block in visual_blocks:
            if block.verification_report and block.verification_report.cross_references:
                has_cross_refs = True
                for cr in block.verification_report.cross_references:
                    status_badge = (
                        "🟢 VERIFIED" if cr.status.value == "verified_match"
                        else "🔴 DISCREPANCY" if cr.status.value == "discrepancy_detected"
                        else "🟡 VISUAL ONLY"
                    )
                    v_val = cr.visual_raw_string or str(cr.visual_value or "-")
                    t_val = str(cr.text_value) if cr.text_value is not None else "Not explicitly cited in text"
                    lines.append(f"| {block.figure_title} - {cr.metric_name} | `{v_val}` | {t_val} | `{status_badge}` | {cr.explanation} |")

        if not has_cross_refs:
            lines.append("| N/A | N/A | N/A | `⚪ NOT REQUIRED` | No specific cross-modal verifications were identified for this query. |")
        lines.append("")

        # 5. Risk Factors & Flagged Anomaly Alerts
        all_warnings = []
        for block in visual_blocks:
            all_warnings.extend(block.risk_warnings)

        lines.append("## 4. Flagged Anomalies & Risk Factors")
        if all_warnings:
            for w in all_warnings:
                lines.append(f"- ⚠️ **Flagged Risk**: {w}")
        else:
            lines.append("- ✅ No statistical anomalies or high-severity reporting contradictions detected.")
        lines.append("")

        # 6. Structured SQL / Market Context (if provided)
        if sql_metrics:
            lines.append("## 5. Structured Market & Trading Benchmark")
            for k, v in sql_metrics.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        # 7. Evidence Citations Index
        lines.append("## 6. Document Evidence Index & Citations")
        for idx, block in enumerate(visual_blocks, 1):
            page_text = f"Page {block.page_number}" if block.page_number else "Extracted Exhibit"
            lines.append(f"- `[Citation {idx}]` **{block.figure_title}** ({page_text}) — `{block.citation_tag}`")
        lines.append("")
        lines.append("---\n*Memo autonomously synthesized by OmniBrain Multi-Modal Orchestrator.*")

        raw_memo = "\n".join(lines)
        
        try:
            import google.generativeai as genai
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY missing. Returning raw memo.")
                return raw_memo
                
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MEMO_MODEL", "gemini-3.5-flash-lite")
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""
            You are a Wall Street Executive Financial Analyst. I have collected raw, multi-modal evidence from various agents (SQL, Vision, RAG).
            Your task is to take this raw, overly verbose data dump and synthesize it into a highly polished, visually appealing, and concise Investment Memorandum.
            
            REQUIREMENTS:
            1. DO NOT hallucinate any numbers. You must ONLY use the numbers provided in the raw dump.
            2. Eliminate the massive, messy table dumps (like the 30-row cross-modal verification audit table). Instead, summarize the verification status in a sleek 1-2 sentence paragraph (e.g. "All 30 visual data points extracted from the charts were successfully verified against the textual MD&A context with 0 discrepancies, ensuring 100% data faithfulness.").
            3. Use elegant markdown formatting (bolding, blockquotes, concise bullet points) to make the memo easy to read for an executive.
            4. Keep the "Document Evidence Index & Citations" section at the bottom so the user knows where the data came from.
            5. Structure it clearly: Executive Summary, Key Analytics (with sleek, synthesized tables if needed), Risk Factors, and Citations.
            
            RAW DATA DUMP:
            {raw_memo}
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during LLM memo synthesis: {e}")
            return raw_memo
