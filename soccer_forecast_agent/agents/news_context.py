"""News and Context agent — ReAct loop with dual retrieval (vector store + live web search)."""

from soccer_forecast_agent.agents.supervisor import GraphState
from soccer_forecast_agent.memory.repository import VectorRepository, EvidenceRepository, SourceReliabilityRepository
from soccer_forecast_agent.providers.llm import LLMProvider


class ToolDispatcher:
    """Routes LLM tool call requests to the appropriate MCP client method."""

    def __init__(self, mcp_client) -> None:
        """Initialise with an injected MCP client."""
        self._mcp = mcp_client

    def dispatch(self, tool_name: str, arguments: dict) -> str:
        """Call the named MCP tool and return the result as a string."""
        raise NotImplementedError


class NewsContextAgent:
    """Runs a ReAct research loop using live web search and local vector retrieval to gather evidence."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_dispatcher: ToolDispatcher,
        vector_repo: VectorRepository,
        evidence_repo: EvidenceRepository,
        source_reliability_repo: SourceReliabilityRepository,
        max_steps: int = 8,
        min_evidence_count: int = 3,
        min_avg_reliability: float = 0.5,
    ) -> None:
        """Initialise with injected LLM, tools, and repositories."""
        self._llm = llm
        self._dispatcher = tool_dispatcher
        self._vector = vector_repo
        self._evidence_repo = evidence_repo
        self._source_reliability = source_reliability_repo
        self._max_steps = max_steps
        self._min_evidence_count = min_evidence_count
        self._min_avg_reliability = min_avg_reliability

    def run(self, state: GraphState) -> GraphState:
        """Seed context from the vector store, then run the ReAct loop to gather evidence."""
        raise NotImplementedError

    def _seed_context(self, home_team: str, away_team: str) -> list[str]:
        """Query the vector store for prior context on both teams before the loop starts."""
        raise NotImplementedError

    def _react_step(self, messages: list[dict], step: int) -> tuple[str, str | None, dict | None]:
        """Run one ReAct step. Returns (thought, tool_name | None, tool_args | None)."""
        raise NotImplementedError

    def _should_stop(self, evidence: list, step: int) -> bool:
        """Return True if stopping conditions are met: budget exhausted or sufficient quality evidence collected."""
        raise NotImplementedError
