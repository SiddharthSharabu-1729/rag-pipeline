"""LLM client abstractions supporting synchronous generation and token streaming."""

from abc import ABC, abstractmethod
import asyncio
import json
import re
from typing import AsyncIterator
import httpx
from rag.config import settings
from rag.domain.exceptions import GenerationError


class BaseLLMClient(ABC):
    """Abstract interface for LLM synthesis."""

    @abstractmethod
    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate complete LLM response text."""
        pass

    @abstractmethod
    async def astream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        """Stream response tokens as an asynchronous generator."""
        pass


class MockLLMClient(BaseLLMClient):
    """Hermetic deterministic LLM client for testing, CI/CD, and local execution.

    Synthesizes grounded answers directly from context blocks with citations.
    """

    def __init__(self, model_name: str = "mock-llm-v1") -> None:
        self.model_name = model_name

    def _synthesize(self, user_prompt: str) -> str:
        # Extract question
        q_match = re.search(r"User Question:\s*(.+?)(?:\nGrounded Answer:|$)", user_prompt, flags=re.S)
        query = q_match.group(1).strip() if q_match else ""

        # Extract references
        ref_blocks = re.findall(r"---\s*\[(Ref-\d+)[^\]]*\]\s*---\n(.*?)(?=(?:---\s*\[Ref|\Z))", user_prompt, flags=re.S)

        if not ref_blocks:
            return "Based on the provided documentation, I cannot verify this information."

        q_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]

        grounded_claims: list[str] = []
        for ref_id, content in ref_blocks:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
            for s in sentences:
                s_lower = s.lower()
                matches = sum(1 for t in q_terms if t in s_lower)
                if matches >= 1:
                    grounded_claims.append(f"{s} [{ref_id}]")
                if len(grounded_claims) >= 3:
                    break
            if len(grounded_claims) >= 3:
                break

        if not grounded_claims:
            # Fallback to first sentence of first reference
            first_ref = ref_blocks[0]
            first_sentence = first_ref[1].strip().split("\n")[0][:150]
            return f"According to the documentation, {first_sentence} [{first_ref[0]}]."

        return " ".join(grounded_claims)

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        await asyncio.sleep(0.02)  # Simulate minimal processing
        return self._synthesize(user_prompt)

    async def astream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        full_text = self._synthesize(user_prompt)
        words = full_text.split(" ")
        for word in words:
            await asyncio.sleep(0.01)  # Simulate token streaming delay
            yield word + " "


class OpenAILLMClient(BaseLLMClient):
    """Production OpenAI-compatible streaming LLM client."""

    def __init__(
        self,
        api_key: str | None = settings.openai_api_key,
        model: str = settings.llm_model,
        base_url: str | None = settings.openai_base_url,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise GenerationError("OpenAI API key missing. Set OPENAI_API_KEY or use 'mock' provider.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                raise GenerationError(f"OpenAI generation failed: {e}") from e

    async def astream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        if not self.api_key:
            raise GenerationError("OpenAI API key missing. Set OPENAI_API_KEY or use 'mock' provider.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
            except Exception as e:
                raise GenerationError(f"OpenAI stream failed: {e}") from e


class LLMFactory:
    """Factory to retrieve configured LLM client."""

    @classmethod
    def get_client(cls) -> BaseLLMClient:
        if settings.llm_provider == "openai" and settings.openai_api_key:
            return OpenAILLMClient()
        return MockLLMClient()

