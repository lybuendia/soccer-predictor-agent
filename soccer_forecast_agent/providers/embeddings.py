"""Embedding provider abstractions and concrete adapters."""

from collections.abc import Sequence
from typing import Protocol

import openai
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    """Contract for components that convert text batches into embedding vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class OpenAIEmbeddingProvider:
    """Embedding provider backed by the OpenAI embeddings API."""

    def __init__(self, client: openai.OpenAI, model: str = "text-embedding-3-small") -> None:
        """Initialise with an injected OpenAI client and embedding model name."""
        self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts from OpenAI."""
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbeddingProvider:
    """Embedding provider backed by a local Hugging Face sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialise and load the local sentence-transformers model."""
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts from the local model."""
        if not texts:
            return []
        embeddings: Sequence[Sequence[float]] = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [vector.tolist() for vector in embeddings]
