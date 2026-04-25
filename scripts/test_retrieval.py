"""Run manual retrieval queries against the local Chroma fixture set."""

import chromadb
from dotenv import load_dotenv

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.main import build_embedding_provider
from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository


QUERIES = [
    ("Arsenal injuries", ["Arsenal"]),
    ("Chelsea poor form", ["Chelsea"]),
    ("Manchester United Champions League race", ["Manchester United"]),
    ("Tottenham relegation danger", ["Tottenham"]),
    ("Manchester City Arsenal title clash", ["Manchester City", "Arsenal"]),
]


def main() -> None:
    """Print top retrieval hits for a small set of evaluation queries."""
    load_dotenv()
    config = Config.from_env()

    client = chromadb.PersistentClient(path=config.chroma_path)
    repo = ChromaVectorRepository(client=client, embedding_provider=build_embedding_provider(config))

    for query, teams in QUERIES:
        print(f"\nQUERY: {query} | teams={teams}")
        results = repo.search(query=query, teams=teams, top_k=3)
        if not results:
            print("  No results found.")
            continue
        for index, item in enumerate(results, start=1):
            preview = item.content[:180].replace("\n", " ")
            print(f"  {index}. source={item.source} teams={item.teams}")
            print(f"     url={item.url}")
            print(f"     preview={preview}")


if __name__ == "__main__":
    main()
