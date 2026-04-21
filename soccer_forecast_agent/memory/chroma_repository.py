"""ChromaDB implementation of VectorRepository for news and evidence content."""

from collections.abc import Callable
import re

from soccer_forecast_agent.models.evidence import ArticleChunk


class ChromaVectorRepository:
    """Stores and retrieves article chunks using ChromaDB with OpenAI embeddings."""

    COLLECTION_NAME = "soccer_news"

    def __init__(self, client, embedding_fn: Callable[[list[str]], list[list[float]]]) -> None:
        """Initialise with an injected chromadb.Client and an embedding callable."""
        self._collection = client.get_or_create_collection(self.COLLECTION_NAME)
        self._embed = embedding_fn

    def upsert(self, chunks: list[ArticleChunk]) -> None:
        """Embed and upsert article chunks; existing chunk_ids are overwritten."""
        if not chunks:
            return
        documents = [c.content for c in chunks]
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=documents,
            embeddings=self._embed(documents),
            metadatas=[self._metadata_for(c) for c in chunks],
        )

    def search(self, query: str, teams: list[str], top_k: int = 5) -> list[ArticleChunk]:
        """Return the top_k chunks most semantically relevant to query, filtered by team membership."""
        where = self._team_filter(teams)
        results = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=top_k,
            **({"where": where} if where else {}),
        )
        chunks = []
        documents = results.get("documents", [[]])[0]
        for i, doc in enumerate(documents):
            meta = results["metadatas"][0][i]
            chunks.append(
                ArticleChunk(
                    chunk_id=results["ids"][0][i],
                    content=doc,
                    source=meta["source"],
                    url=meta["url"],
                    published_at=meta["published_at"],
                    teams=meta["teams"].split("|") if meta["teams"] else [],
                )
            )
        return chunks

    def _metadata_for(self, chunk: ArticleChunk) -> dict[str, str | bool]:
        """Build Chroma-compatible scalar metadata, including boolean team filter flags."""
        metadata: dict[str, str | bool] = {
            "source": chunk.source,
            "url": chunk.url,
            "published_at": chunk.published_at,
            "teams": "|".join(chunk.teams),
        }
        for team in chunk.teams:
            metadata[f"team_{self._team_key(team)}"] = True
        return metadata

    def _team_filter(self, teams: list[str]) -> dict | None:
        """Build a Chroma where filter that matches any requested team."""
        clauses = [{f"team_{self._team_key(team)}": True} for team in teams if team.strip()]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$or": clauses}

    def _team_key(self, team: str) -> str:
        """Return a metadata-safe key fragment for a team name."""
        return re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")
