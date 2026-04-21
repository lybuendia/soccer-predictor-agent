"""CLI entry point — wires all dependencies and runs one forecast scan."""

import sqlite3
import openai
import chromadb
from dotenv import load_dotenv
load_dotenv()  # loads .env before Config.from_env() reads os.environ

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.providers.openai_provider import OpenAIProvider
from soccer_forecast_agent.providers.claude_provider import ClaudeProvider
from soccer_forecast_agent.providers.llm import LLMProvider
from soccer_forecast_agent.analytics.baseline import SimpleBaselineStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository, init_db
from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.tools.fixtures import FixtureFetcher
from soccer_forecast_agent.tools.odds import OddsFetcher
from soccer_forecast_agent.tools.search import WebSearchTool
from soccer_forecast_agent.tools.ingester import ArticleIngester
from soccer_forecast_agent.tools.email_sender import EmailAlertChannel
from soccer_forecast_agent.guardrails.alert_guard import AlertGuard
from soccer_forecast_agent.agents.stats_market import StatsMarketAgent
from soccer_forecast_agent.agents.news_context import NewsContextAgent, ToolDispatcher
from soccer_forecast_agent.agents.synthesis_alert import SynthesisAlertAgent
from soccer_forecast_agent.agents.supervisor import SupervisorAgent


def build_llm_provider(config: Config) -> LLMProvider:
    """Instantiate the configured LLM provider based on LLM_PROVIDER env var."""
    if config.llm_provider == "openai":
        return OpenAIProvider(openai.OpenAI(api_key=config.openai_api_key))
    if config.llm_provider == "claude":
        import anthropic
        return ClaudeProvider(anthropic.Anthropic(api_key=config.anthropic_api_key))
    raise ValueError(f"Unknown LLM_PROVIDER: {config.llm_provider}")


def main() -> None:
    """Wire all dependencies and run one forecast scan."""
    config = Config.from_env()

    db_conn = sqlite3.connect(config.db_path)
    init_db(db_conn)
    seed_source_reliability(db_conn)

    chroma_client = chromadb.PersistentClient(path=config.chroma_path)
    oai_client = openai.OpenAI(api_key=config.openai_api_key)

    sql_repo = SQLiteRepository(db_conn)
    vector_repo = ChromaVectorRepository(
        client=chroma_client,
        embedding_fn=lambda texts: [
            item.embedding
            for item in oai_client.embeddings.create(input=texts, model="text-embedding-3-small").data
        ],
    )

    llm = build_llm_provider(config)
    fixture_fetcher = FixtureFetcher(config.football_data_api_key)
    odds_fetcher = OddsFetcher(config.odds_api_key)
    search_tool = WebSearchTool(config.search_api_key)
    ingester = ArticleIngester(search_tool, vector_repo)
    alert_channel = EmailAlertChannel(
        config.smtp_host, config.smtp_port, config.smtp_user, config.smtp_password, config.alert_email
    )
    guard = AlertGuard(
        min_edge_threshold=config.min_edge_threshold,
        min_confidence_threshold=config.min_confidence_threshold,
        spam_window_hours=config.spam_window_hours,
    )

    stats_agent = StatsMarketAgent(fixture_fetcher, odds_fetcher, SimpleBaselineStrategy(), FeatureExtractor(), sql_repo)
    news_agent = NewsContextAgent(llm, ToolDispatcher(None), vector_repo, sql_repo, sql_repo, config.react_max_steps)
    synthesis_agent = SynthesisAlertAgent(llm, alert_channel, sql_repo, guard, config.base_sensitivity)

    supervisor = SupervisorAgent(stats_agent, news_agent, synthesis_agent)
    supervisor.run(competition="PL", days_ahead=7)


if __name__ == "__main__":
    main()
