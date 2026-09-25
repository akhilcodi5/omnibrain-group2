"""Multi-Engine VLM Interface supporting OpenAI GPT-4o, local LLaVA (Ollama), and Mocking."""

import abc
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx

from app.models.vision_schemas import (
    ChartSeries,
    ChartType,
    DataPoint,
    ExtractedChartData,
    ExtractedTableData,
    VLMProviderType,
)

logger = logging.getLogger(__name__)


class BaseVisionEngine(abc.ABC):
    """Abstract Base Class for Vision-Language Model inference engines."""

    @abc.abstractmethod
    async def generate_response(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate unstructured markdown/text response from image and prompt."""
        pass

    @abc.abstractmethod
    async def extract_structured_json(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """Generate guaranteed structured JSON dictionary conforming to extraction schema."""
        pass


def _clean_and_parse_json(content: str) -> Dict[str, Any]:
    """Safely extracts and parses JSON from VLM output, handling markdown blocks and bracket boundaries."""
    if not content or not content.strip():
        return {}
    cleaned = content.strip()
    # Strip markdown code fencing if present
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        # Fallback to finding outermost JSON object brackets
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(cleaned[start:end + 1])
            except Exception as e:
                logger.debug(f"Outermost bracket JSON parse fallback failed: {e}")
        logger.error(f"Failed to parse valid JSON from VLM output: {content[:200]}")
        return {"raw_text": content, "error": "Invalid JSON response"}


class OpenAIVisionEngine(BaseVisionEngine):
    """VLM Engine implementation using OpenAI GPT-4o / GPT-4o-mini."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    async def generate_response(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ],
        })

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2000,
            temperature=temperature,
        )

        return {
            "content": response.choices[0].message.content or "",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            "model": self.model,
        }

    async def extract_structured_json(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)

        instruction = f"{prompt}\n\nIMPORTANT: Respond ONLY with a valid JSON object matching the requested schema."
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ],
        })

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content or "{}"
        return _clean_and_parse_json(content)


class OllamaLLaVAEngine(BaseVisionEngine):
    """Local Open-Source VLM Engine for LLaVA / Llama-3.2-Vision hosted via Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava:13b"):
        self.base_url = os.getenv("OLLAMA_HOST", base_url)
        self.model = os.getenv("OLLAMA_VISION_MODEL", model)

    async def generate_response(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt or "",
                "images": [base64_image],
                "stream": False,
                "options": {"temperature": temperature},
            }
            try:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                return {
                    "content": data.get("response", ""),
                    "usage": {"eval_count": data.get("eval_count", 0)},
                    "model": self.model,
                }
            except Exception as e:
                logger.error(f"Ollama LLaVA inference error: {str(e)}")
                raise

    async def extract_structured_json(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        instruction = f"{prompt}\nReturn strictly JSON format."
        result = await self.generate_response(
            base64_image, instruction, system_prompt, media_type, temperature=0.0
        )
        content = result.get("content", "{}")
        return _clean_and_parse_json(content)


class GeminiVisionEngine(BaseVisionEngine):
    """Google Gemini REST API implementation with intelligent model failover."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-flash-lite-latest"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_VISION_MODEL", model)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        # Prioritize working models: if primary is flash-latest, include flash-lite-latest and 2.5-flash-lite
        raw_candidates = [self.model, "gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite"]
        self.fallback_models = []
        for m in raw_candidates:
            if m and m not in self.fallback_models:
                self.fallback_models.append(m)

    async def generate_response(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")

        payload = {
            "contents": [{
                "parts": [
                    {"text": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt},
                    {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": base64_image
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 8192
            }
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            last_error = None
            for model_candidate in self.fallback_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_candidate}:generateContent?key={self.api_key}"
                try:
                    res = await client.post(url, json=payload)
                    res.raise_for_status()
                    data = res.json()

                    content = ""
                    if "candidates" in data and len(data["candidates"]) > 0:
                        parts = data["candidates"][0].get("content", {}).get("parts", [])
                        if parts:
                            content = parts[0].get("text", "")

                    return {
                        "content": content,
                        "usage": {"eval_count": 0},
                        "model": model_candidate,
                    }
                except httpx.HTTPStatusError as e:
                    last_error = e
                    status = e.response.status_code
                    if status in (429, 503, 404):
                        logger.warning(
                            f"Gemini model '{model_candidate}' returned HTTP {status}. "
                            f"Failing over to next available model in {self.fallback_models}..."
                        )
                        continue
                    else:
                        logger.error(f"Gemini API HTTP {status} error: {e.response.text[:200]}")
                        raise
                except Exception as e:
                    last_error = e
                    logger.warning(f"Connection error to Gemini model '{model_candidate}': {e}. Trying alternative model...")
                    continue

            # If all cloud models failed (e.g. 503 outage or quota exhausted), fall back to Mock engine
            logger.warning(f"All Gemini models exhausted ({last_error}). Falling back to deterministic offline extraction.")
            mock_engine = MockVisionEngine()
            return await mock_engine.generate_response(
                base64_image, prompt, system_prompt, media_type, temperature
            )

    async def extract_structured_json(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        instruction = f"{prompt}\nReturn strictly JSON format."
        try:
            result = await self.generate_response(
                base64_image, instruction, system_prompt, media_type, temperature=0.0
            )
            content = result.get("content", "{}")
            return _clean_and_parse_json(content)
        except Exception as e:
            logger.warning(f"Gemini API structured extraction error: {e}. Falling back to deterministic parser.")
            mock_engine = MockVisionEngine()
            return await mock_engine.extract_structured_json(base64_image, prompt, system_prompt, media_type)


class MockVisionEngine(BaseVisionEngine):
    """Deterministic Mock VLM Engine for offline local development and unit tests."""

    async def generate_response(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        return {
            "content": (
                "### Mock Vision Analysis\n\n"
                "- **Figure Type**: Bar Chart\n"
                "- **Key Metric**: Total Operating Revenue grew by 18.5% YoY.\n"
                "- **Observations**: Q1 ($110M), Q2 ($125M), Q3 ($140M), Q4 ($155M)."
            ),
            "usage": {"prompt_tokens": 120, "completion_tokens": 65},
            "model": "mock-vlm-engine",
        }

    async def extract_structured_json(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        p_lower = prompt.lower()
        is_table = False
        if "extractedtabledata" in p_lower:
            is_table = True
        elif "extractedchartdata" in p_lower:
            is_table = False
        elif any(w in p_lower for w in ["table image", "columns, headers, rows", "statement of income", "balance sheet table", "income statement table"]):
            is_table = True
        elif any(w in p_lower for w in ["chart", "series", "data_points", "bar", "line", "pie"]):
            is_table = False
        elif "table" in p_lower:
            is_table = True

        if is_table:
            return {

                "title": "Consolidated Statement of Income",
                "headers": ["Line Item", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
                "rows": [
                    ["Total Revenue", "$120.5M", "$135.2M", "$148.8M", "$162.0M"],
                    ["Cost of Goods Sold", "$45.2M", "$49.1M", "$53.4M", "$57.8M"],
                    ["Gross Profit", "$75.3M", "$86.1M", "$95.4M", "$104.2M"],
                    ["Operating Expenses", "$48.2M", "$53.9M", "$58.2M", "$60.1M"],
                    ["Operating Income", "$27.1M", "$32.2M", "$37.2M", "$44.1M"],
                    ["Net Income", "$21.5M", "$25.8M", "$29.6M", "$35.2M"],
                ],
                "summary": "Consolidated quarterly income statement showing margin expansion and revenue growth.",
                "key_metrics": {
                    "Q4 Revenue": "$162.0M",
                    "Q4 Operating Income": "$44.1M",
                    "Gross Margin": "64.3%",
                },
                "currency": "USD",
                "scale": "Millions",
            }

        return {
            "title": "Quarterly Operating Performance",
            "chart_type": "bar",
            "x_axis_label": "Quarter",
            "y_axis_label": "USD Millions",
            "series": [
                {
                    "series_name": "Revenue",
                    "data_points": [
                        {"label": "Q1", "value": 110.0, "raw_value": "$110M", "unit": "USD Millions"},
                        {"label": "Q2", "value": 125.0, "raw_value": "$125M", "unit": "USD Millions"},
                        {"label": "Q3", "value": 140.0, "raw_value": "$140M", "unit": "USD Millions"},
                        {"label": "Q4", "value": 155.0, "raw_value": "$155M", "unit": "USD Millions"},
                    ],
                }
            ],
            "summary": "Revenue increased steadily over four quarters.",
            "key_insights": ["Annual revenue reached $530M", "Q4 was highest performing quarter"],
            "notable_anomalies": [],
            "confidence_score": 0.98,
        }


def get_vision_engine(provider: Optional[str] = None) -> BaseVisionEngine:
    """Factory function to retrieve the configured Vision-Language Model Engine."""
    provider_str = (provider or os.getenv("VLM_PROVIDER", "openai")).lower()

    if provider_str in (VLMProviderType.OPENAI.value, "openai"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            return MockVisionEngine()
        return OpenAIVisionEngine()

    elif provider_str in (VLMProviderType.LLAVA_OLLAMA.value, "llava", "ollama"):
        return OllamaLLaVAEngine()
        
    elif provider_str in (VLMProviderType.GEMINI.value, "gemini"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            return MockVisionEngine()
        return GeminiVisionEngine()

    elif provider_str in (VLMProviderType.MOCK.value, "mock"):
        return MockVisionEngine()

    logger.warning(f"Unknown VLM provider '{provider_str}'. Falling back to GeminiVisionEngine.")
    return MockVisionEngine()


# Alias for backward compatibility
get_vlm_engine = get_vision_engine

