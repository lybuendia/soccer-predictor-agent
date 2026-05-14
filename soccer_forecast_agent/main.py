"""CLI entry point — wires all dependencies and runs one forecast scan."""

import logging
import sqlite3
import openai
import chromadb
from dotenv import load_dotenv
load_dotenv()  # loads .env before Config.from_env() reads os.environ

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.providers.openai_provider import OpenAIProvider
from soccer_forecast_agent.providers.claude_provider import ClaudeProvider
from soccer_forecast_agent.providers.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from soccer_forecast_agent.providers.llm import LLMProvider
from soccer_forecast_agent.analytics.baseline import EnhancedBaselineStrategy
from soccer_forecast_agent.analytics.dixon_coles import DixonColesStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository, init_db
from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.tools.fixtures import FixtureFetcher
from soccer_forecast_agent.tools.odds import OddsFetcher
from soccer_forecast_agent.tools.search import WebSearchTool
from soccer_forecast_agent.tools.ingester import ArticleIngester
from soccer_forecast_agent.tools.email_sender import ConsoleAlertChannel, EmailAlertChannel
from soccer_forecast_agent.guardrails.alert_guard import AlertGuard, AlertGuardConfig
from soccer_forecast_agent.agents.stats_market import StatsMarketAgent
from soccer_forecast_agent.agents.news_context import NewsContextAgent, ToolDispatcher
from soccer_forecast_agent.agents.synthesis_alert import SynthesisAlertAgent, SynthesisTuning
from soccer_forecast_agent.agents.supervisor import SupervisorAgent


LOGGER = logging.getLogger(__name__)


class LocalMCPToolClient:
    """Expose local tool instances behind a minimal MCP-like call_tool interface."""

    def __init__(self, fixture_fetcher: FixtureFetcher, odds_fetcher: OddsFetcher, search_tool: WebSearchTool) -> None:
        """Initialise with injected local tool instances."""
        self._fixtures = fixture_fetcher
        self._odds = odds_fetcher
        self._search = search_tool

    def call_tool(self, tool_name: str, arguments: dict) -> object:
        """Dispatch a tool call to the appropriate local tool implementation."""
        if tool_name == "get_fixtures":
            return self._fixtures.fetch_upcoming(competition="PL", days_ahead=arguments.get("days_ahead", 7))
        if tool_name == "get_odds":
            return self._odds.fetch_odds(arguments["home_team"], arguments["away_team"])
        if tool_name == "search_news":
            return self._search.search(arguments["query"], arguments.get("max_results", 5))
        raise ValueError(f"Unknown tool: {tool_name}")


def build_llm_provider(config: Config) -> LLMProvider:
    """Instantiate the configured LLM provider based on LLM_PROVIDER env var."""
    if config.llm_provider == "openai":
        return OpenAIProvider(openai.OpenAI(api_key=config.openai_api_key), model=config.llm_model)
    if config.llm_provider == "claude":
        import anthropic
        return ClaudeProvider(anthropic.Anthropic(api_key=config.anthropic_api_key), model=config.llm_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {config.llm_provider}")


def _build_baseline(sql_repo: SQLiteRepository):
    """Fit Dixon-Coles on all resolved matches; fall back to EnhancedBaseline if too few."""
    resolved = sql_repo.get_all_finished("PL")
    if len(resolved) >= 20:
        dc = DixonColesStrategy()
        dc.fit(resolved)
        LOGGER.info("Dixon-Coles fitted on %d matches.", len(resolved))
        return dc
    LOGGER.warning("Fewer than 20 resolved matches — using EnhancedBaselineStrategy.")
    return EnhancedBaselineStrategy()


def build_embedding_provider(config: Config) -> EmbeddingProvider:
    """Instantiate the configured embedding provider."""
    if config.embedding_provider == "huggingface":
        return SentenceTransformerEmbeddingProvider(model_name=config.embedding_model)
    if config.embedding_provider == "openai":
        client = openai.OpenAI(api_key=config.openai_api_key)
        return OpenAIEmbeddingProvider(client=client, model=config.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {config.embedding_provider}")


def configure_logging(config: Config) -> None:
    """Configure runtime logging, enabling verbose LLM tracing when requested."""
    level = logging.DEBUG if config.debug_llm else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    """Wire all dependencies and run one forecast scan."""
    config = Config.from_env()
    configure_logging(config)
    if config.debug_llm:
        LOGGER.info("DEBUG_LLM is enabled; verbose model interaction logs will be emitted.")

    db_conn = sqlite3.connect(config.db_path)
    init_db(db_conn)
    seed_source_reliability(db_conn)

    chroma_client = chromadb.PersistentClient(path=config.chroma_path)

    sql_repo = SQLiteRepository(db_conn)
    embedding_provider = build_embedding_provider(config)
    vector_repo = ChromaVectorRepository(
        client=chroma_client,
        embedding_provider=embedding_provider,
    )

    llm = build_llm_provider(config)
    fixture_fetcher = FixtureFetcher(config.football_data_api_key)
    odds_fetcher = OddsFetcher(config.odds_api_key)
    search_tool = WebSearchTool(config.search_api_key)
    local_mcp_client = LocalMCPToolClient(fixture_fetcher, odds_fetcher, search_tool)
    ingester = ArticleIngester(search_tool, vector_repo)
    smtp_configured = bool(config.smtp_user and config.smtp_password and config.alert_email)
    alert_channel = (
        EmailAlertChannel(config.smtp_host, config.smtp_port, config.smtp_user, config.smtp_password, config.alert_email)
        if smtp_configured
        else ConsoleAlertChannel()
    )
    guard = AlertGuard(
        AlertGuardConfig(
            min_edge_threshold=config.min_edge_threshold,
            min_confidence_threshold=config.min_confidence_threshold,
            spam_window_hours=config.spam_window_hours,
        )
    )

    baseline = _build_baseline(sql_repo)
    stats_agent = StatsMarketAgent(fixture_fetcher, odds_fetcher, baseline, FeatureExtractor(), sql_repo)
    news_agent = NewsContextAgent(
        llm,
        ToolDispatcher(local_mcp_client),
        vector_repo,
        sql_repo,
        sql_repo,
        config.react_max_steps,
    )
    synthesis_agent = SynthesisAlertAgent(
        llm, alert_channel, sql_repo, sql_repo, guard,
        tuning=SynthesisTuning(base_sensitivity=config.base_sensitivity),
    )

    supervisor = SupervisorAgent(
        stats_agent, news_agent, synthesis_agent,
        min_edge_threshold=config.min_edge_threshold,
    )
    supervisor.run(competition="PL", days_ahead=7)


if __name__ == "__main__":
    main()
