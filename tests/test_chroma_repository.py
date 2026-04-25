from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.models.evidence import ArticleChunk


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_payload = None
        self.query_payload = None

    def upsert(self, **kwargs) -> None:
        self.upsert_payload = kwargs

    def query(self, **kwargs):
        self.query_payload = kwargs
        return {
            "ids": [["chunk-1"]],
            "documents": [["Arsenal squad update and injury news"]],
            "metadatas": [[{
                "source": "https://example.com/story",
                "url": "https://example.com/story",
                "published_at": "2026-04-25T12:00:00+00:00",
                "teams": "Arsenal|Chelsea",
            }]],
        }


class FakeChromaClient:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def get_or_create_collection(self, _name: str) -> FakeCollection:
        return self._collection


def test_chroma_repository_uses_embedding_provider_for_upsert_and_query():
    collection = FakeCollection()
    embeddings = FakeEmbeddingProvider()
    repo = ChromaVectorRepository(FakeChromaClient(collection), embeddings)

    repo.upsert([
        ArticleChunk(
            chunk_id="chunk-1",
            content="Arsenal squad update and injury news",
            source="https://example.com/story",
            url="https://example.com/story",
            published_at="2026-04-25T12:00:00+00:00",
            teams=["Arsenal", "Chelsea"],
        )
    ])

    results = repo.search("Arsenal injuries", teams=["Arsenal"], top_k=3)

    assert embeddings.calls[0] == ["Arsenal squad update and injury news"]
    assert embeddings.calls[1] == ["Arsenal injuries"]
    assert collection.upsert_payload is not None
    assert collection.upsert_payload["embeddings"] == [[36.0, 1.0]]
    assert collection.query_payload["query_embeddings"] == [[16.0, 1.0]]
    assert results[0].teams == ["Arsenal", "Chelsea"]
