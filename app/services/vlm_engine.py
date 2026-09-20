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
    """Google Gemini REST API implementation (gemini-3.5-flash-lite)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_VISION_MODEL", model)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

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

        # Strict 5-second pacing gap
        await asyncio.sleep(5)

        async with httpx.AsyncClient(timeout=60.0) as client:
            max_retries = 5
            retry_delay = 4.0
            
            for attempt in range(max_retries + 1):
                try:
                    res = await client.post(f"{self.base_url}?key={self.api_key}", json=payload)
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
                        "model": self.model,
                    }
                except httpx.HTTPStatusError as e:
                    if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < max_retries:
                        logger.warning(f"Gemini API transient error ({e.response.status_code}). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    logger.error(f"Gemini API inference error: {str(e)}")
                    raise
                except Exception as e:
                    logger.error(f"Gemini API inference error: {str(e)}")
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





def get_vision_engine(provider: Optional[str] = None) -> BaseVisionEngine:
    """Factory function to retrieve the configured Vision-Language Model Engine."""
    provider_str = (provider or os.getenv("VLM_PROVIDER", "openai")).lower()

    if provider_str in (VLMProviderType.OPENAI.value, "openai"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")
        return OpenAIVisionEngine()

    elif provider_str in (VLMProviderType.LLAVA_OLLAMA.value, "llava", "ollama"):
        return OllamaLLaVAEngine()
        
    elif provider_str in (VLMProviderType.GEMINI.value, "gemini"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")
        return GeminiVisionEngine()

    logger.warning(f"Unknown VLM provider '{provider_str}'. Falling back to GeminiVisionEngine.")
    return GeminiVisionEngine()
