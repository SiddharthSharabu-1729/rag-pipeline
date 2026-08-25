"""Dense embedding clients for semantic representation."""

import hashlib
import math
from typing import Any
import httpx
from rag.config import settings
from rag.domain.exceptions import EmbeddingError
from rag.storage.base import BaseEmbeddingClient


class MockEmbeddingClient(BaseEmbeddingClient):
    """Deterministic, local pseudo-embedding client for offline testing and CI.

    Generates L2-normalized vectors based on token/character n-gram hashing such that
    semantically overlapping texts yield meaningful cosine similarities.
    """

    def __init__(self, dimension: int = settings.embedding_dim) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _generate_vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Hash word into dimension indices
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            for i in range(4):  # distribute across a few bins
                idx = (h >> (i * 8)) % self._dim
                sign = 1.0 if ((h >> (i * 8 + 4)) & 1) else -1.0
                vec[idx] += sign

        # Compute L2 norm
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    async def aembed_query(self, text: str) -> list[float]:
        return self._generate_vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """Production OpenAI-compatible embedding client."""

    def __init__(
        self,
        api_key: str | None = settings.openai_api_key,
        model: str = settings.embedding_model,
        base_url: str | None = settings.openai_base_url,
        dimension: int = settings.embedding_dim,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    async def aembed_query(self, text: str) -> list[float]:
        results = await self.aembed_documents([text])
        return results[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EmbeddingError("OpenAI API key is missing. Set OPENAI_API_KEY or use 'mock' provider.")

        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
            except Exception as e:
                raise EmbeddingError(f"OpenAI embedding request failed: {e}") from e


class EmbeddingFactory:
    """Factory to instantiate configured embedding client."""

    @classmethod
    def get_client(cls) -> BaseEmbeddingClient:
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            return OpenAIEmbeddingClient()
        return MockEmbeddingClient(dimension=settings.embedding_dim)

