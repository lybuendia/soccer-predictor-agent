# agents.md — Codex Implementation Spec

This document is the implementation specification for OpenAI Codex (or any coding agent). Read this before generating any code. All code must follow SOLID principles. Python 3.11+.

---

## Ground Rules for All Code Generation

1. Every class has one responsibility. If a class does more than one thing, split it.
2. Depend on abstractions (`Protocol`), not on concrete classes.
3. Inject all dependencies — no instantiation of external clients inside class bodies.
4. Use typed dataclasses or Pydantic models for all structured data.
5. No raw `dict` as a public API return type.
6. Type-hint everything. Use `Protocol` for interfaces.
7. Docstrings are encouraged on all public classes, methods, and functions — one concise line describing what it does and any non-obvious contract (args, return, raises). No inline comments (`#`) inside function/method bodies unless the WHY is genuinely non-obvious (a workaround, a hidden invariant, a subtle edge case). Never use inline comments to narrate what the code does.

---

## 1. Protocols / Interfaces to Define First

Define these before implementing any agent. Everything downstream depends on them.

### `LLMProvider` — `providers/llm.py`

```python
from typing import Protocol, Any

class LLMProvider(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...
    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        **kwargs: Any
    ) -> dict: ...
```

Implement two adapters:
- `OpenAIProvider(LLMProvider)` in `providers/openai_provider.py` — wraps `openai.OpenAI`
- `ClaudeProvider(LLMProvider)` in `providers/claude_provider.py` — wraps `anthropic.Anthropic`

Active provider is selected by `LLM_PROVIDER` env var (`"openai"` or `"claude"`).

---

### `BaselineStrategy` — `analytics/baseline.py`

```python
from typing import Protocol
from models.match import MatchContext, BaselineForecast

class BaselineStrategy(Protocol):
    def compute(self, context: MatchContext) -> BaselineForecast: ...
```

Implement `SimpleBaselineStrategy` in the same file:
- Inputs: recent form (W/D/L last 5), home/away splits, goals scored/conceded averages
- Outputs: `BaselineForecast` with `home_win`, `draw`, `away_win`, `over_2_5`, `under_2_5` as floats summing to 1.0 within each market

---

### `MemoryRepository` — `memory/repository.py`

```python
from typing import Protocol
from models.match import Forecast, Match, MarketOdds
from models.evidence import EvidenceItem

class MatchRepository(Protocol):
    def save_match(self, match: Match) -> None: ...
    def get_upcoming(self) -> list[Match]: ...

class ForecastRepository(Protocol):
    def save_forecast(self, forecast: Forecast) -> None: ...
    def get_by_match(self, match_id: str) -> list[Forecast]: ...

class EvidenceRepository(Protocol):
    def save_evidence(self, item: EvidenceItem) -> None: ...

class SourceReliabilityRepository(Protocol):
    def get_score(self, source_domain: str) -> float: ...
```

Implement all four as a single `SQLiteRepository` class in `memory/sqlite_repository.py` that takes a `sqlite3.Connection` at init. One class implementing all four protocols is fine here — it has one responsibility: SQLite access.

---

### `VectorRepository` — `memory/vector_repository.py`

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class ArticleChunk:
    chunk_id: str
    content: str
    source: str
    url: str
    published_at: str
    teams: list[str]   # metadata filter

class VectorRepository(Protocol):
    def upsert(self, chunks: list[ArticleChunk]) -> None: ...
    def search(self, query: str, teams: list[str], top_k: int) -> list[ArticleChunk]: ...
```

Implement `ChromaVectorRepository` in `memory/chroma_repository.py`:
- Takes a `chromadb.Client` at init (injected)
- Uses a single `"soccer_news"` collection
- Embeddings generated via `openai.embeddings.create(model="text-embedding-3-small")`
- `teams` list is passed as a `where` metadata filter at query time
- Production swap: implement `PineconeVectorRepository` with the same protocol, no agent code changes needed

---

### `AlertChannel` — `tools/alert_channel.py`

```python
from typing import Protocol
from models.match import AlertPayload

class AlertChannel(Protocol):
    def send(self, payload: AlertPayload) -> bool: ...
```

Implement `EmailAlertChannel` in `tools/email_sender.py` using `smtplib`.

---

## 2. Data Models — `models/`

### `models/match.py`

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Match:
    match_id: str
    competition: str
    home_team: str
    away_team: str
    kickoff_time: datetime
    status: str  # "upcoming" | "live" | "resolved"
    final_score: str | None = None

@dataclass
class MarketOdds:
    odds_id: str
    match_id: str
    timestamp: datetime
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float

@dataclass
class BaselineForecast:
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float

@dataclass
class Forecast:
    forecast_id: str
    match_id: str
    run_timestamp: datetime
    baseline: BaselineForecast
    adjusted_home_win: float | None
    adjusted_draw: float | None
    adjusted_away_win: float | None
    adjusted_over_2_5: float | None
    adjusted_under_2_5: float | None
    confidence_score: float
    edge_market: str | None
    edge_value: float | None
    alert_sent: bool
    rationale: str

@dataclass
class MatchContext:
    match: Match
    odds: MarketOdds
    recent_home_results: list[str]   # ["W", "D", "L", ...]
    recent_away_results: list[str]
    home_goals_scored_avg: float
    home_goals_conceded_avg: float
    away_goals_scored_avg: float
    away_goals_conceded_avg: float

@dataclass
class AlertPayload:
    match: Match
    forecast: Forecast
    rationale: str
    disclaimer: str = "This is decision support only. Not a guarantee of profit. Probabilities are estimates."
```

### `models/evidence.py`

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EvidenceItem:
    evidence_id: str
    forecast_id: str
    source: str
    url: str
    timestamp: datetime
    summary: str
    direction: str   # "home_positive" | "away_positive" | "neutral" | "uncertainty"
    reliability_score: float   # 0.0 – 1.0
    applies_to_market: str     # "winner" | "goals" | "both"
```

---

## 3. Tools — `tools/` and MCP server

### Tool classes (internal Python)

Each tool is a standalone class with one method. No tool imports from agent code.

#### `FixtureFetcher` — `tools/fixtures.py`
- Method: `fetch_upcoming(competition: str, days_ahead: int) -> list[Match]`
- Calls football-data.org API or equivalent
- Returns typed `Match` objects, not raw JSON

#### `OddsFetcher` — `tools/odds.py`
- Method: `fetch_odds(match_id: str) -> MarketOdds`
- Calls The Odds API or equivalent

#### `WebSearchTool` — `tools/search.py`
- Method: `search(query: str, max_results: int) -> list[SearchResult]`
- `SearchResult` is a small dataclass: `url`, `title`, `snippet`
- Used by the ReAct agent — must treat results as untrusted external data

#### `ArticleIngester` — `tools/ingester.py`
- Method: `ingest(teams: list[str], days_back: int) -> int` (returns count of new chunks stored)
- Fetches recent news articles for given teams via web search
- Chunks content, embeds via `text-embedding-3-small`, upserts to `VectorRepository`
- Called by the scheduler on each daily scan, before agents run
- Single responsibility: fetch → chunk → embed → store. Does not query the vector store.

#### `EmailAlertChannel` — `tools/email_sender.py`
- Implements `AlertChannel`
- Uses `smtplib` + env vars for SMTP credentials
- Returns `True` if sent successfully

---

### MCP Server — `mcp_server/server.py`

The data tools are exposed as an MCP server using the official `mcp` Python SDK. This separates tool implementation from agent logic — any MCP client (Claude Desktop, another agent, a future web service) can call these tools without touching agent code.

**Exposed tools:**

| MCP tool name | Delegates to | Description |
|---|---|---|
| `get_fixtures` | `FixtureFetcher` | Upcoming Premier League fixtures |
| `get_odds` | `OddsFetcher` | Market odds for a match |
| `search_news` | `WebSearchTool` | Live web search for soccer news |

**Implementation notes:**
- Each MCP tool is a thin wrapper that instantiates the tool class, calls the method, and serialises the typed result to JSON
- Server runs locally for MVP (`mcp run mcp_server/server.py`)
- LangGraph agents call tools via MCP client — no direct Python imports of tool classes inside agent nodes
- Production path: deploy the MCP server independently, add API key auth via MCP middleware

---

## 4. Analytics — `analytics/`

### `SimpleBaselineStrategy` — `analytics/baseline.py`
Implements `BaselineStrategy`.

Steps:
1. Compute form score from last 5 results for each team (W=3, D=1, L=0), normalize to 0–1
2. Weight home advantage: add a configurable home boost (default 0.05)
3. Compute attack/defense rating from goals scored/conceded averages
4. Derive raw win probabilities: home_strength / (home_strength + away_strength), adjust for draw
5. Derive over/under probability from (home_avg_goals_scored + away_avg_goals_conceded + away_avg_goals_scored + home_avg_goals_conceded) / 2 vs 2.5 threshold
6. Normalize all probabilities so home_win + draw + away_win = 1.0 and over + under = 1.0

**Do not include market implied probabilities as an input feature.** The edge claim is `system_probability - market_implied_probability`. A baseline that incorporates market odds would measure distance from itself, breaking the independence of the forecast.

### `FeatureExtractor` — `analytics/features.py`
Single responsibility: extract a `MatchContext` from raw match data and historical records. Separate from `SimpleBaselineStrategy`.

---

## 5. Agents — `agents/`

Agents are LangGraph nodes. Each agent class has one public method: `run(state: GraphState) -> GraphState`.

### `StatsMarketAgent` — `agents/stats_market.py`

**Responsibility**: Fetch fixtures and odds, compute baseline, return stats package.

**Dependencies (injected)**:
- `FixtureFetcher`
- `OddsFetcher`
- `BaselineStrategy`
- `MatchRepository`

**Inputs from state**: `competition`, `days_ahead`
**Outputs to state**: `matches`, `odds_map`, `baseline_forecasts`

SOLID notes:
- Does not call the LLM — no `LLMProvider` dependency
- Does not know about the ReAct loop
- Does not write to `forecasts` table — only reads and computes

---

### `NewsContextAgent` — `agents/news_context.py`

**Responsibility**: Run a ReAct loop to gather qualitative context for a match using dual retrieval — live web search and local vector store.

**Dependencies (injected)**:
- `LLMProvider`
- `WebSearchTool` (via MCP client)
- `VectorRepository`
- `EvidenceRepository`
- `SourceReliabilityRepository`

**ReAct loop structure**:
1. Receive match metadata + baseline estimate from state
2. **Before the loop**: query `VectorRepository` for prior context on both teams (`top_k=5`). Inject results as initial context in the system prompt — this seeds the loop with accumulated knowledge before any live search.
3. LLM generates a thought: what information is still missing?
4. LLM calls a tool (`search_news` via MCP, or `query_vector_store`) or decides to stop
5. Observe the tool result
6. LLM evaluates evidence quality and decides next step
7. Loop ends when: enough evidence collected OR step budget exhausted OR LLM decides to stop

**Dual retrieval rationale**: vector store retrieval catches patterns and context from prior runs (e.g. a player's injury history, a team's form across a season). Live web search catches same-day breaking news. Together they cover both depth and recency.

**Stopping conditions (configurable)**:
- `max_steps: int = 8`
- `min_evidence_count: int = 3`
- `min_avg_reliability: float = 0.5`

**Outputs to state**: `evidence_items: list[EvidenceItem]`

SOLID notes:
- Does not compute probabilities — only gathers and scores evidence
- Source reliability scoring is delegated to `SourceReliabilityRepository`
- Tool dispatch is handled by a `ToolDispatcher` helper (single responsibility for routing MCP tool calls)
- Vector retrieval is a pre-loop step, not a ReAct tool call — it always runs and does not consume the step budget

---

### `SynthesisAlertAgent` — `agents/synthesis_alert.py`

**Responsibility**: Combine baseline + evidence + odds, decide whether to alert.

**Dependencies (injected)**:
- `LLMProvider`
- `AlertChannel`
- `ForecastRepository`
- `AlertGuard`

**Logic**:
1. Receive baseline forecast, evidence items, and market odds from state
2. For each evidence item, compute signed delta in log-odds space:
   ```
   delta_i = direction_sign × reliability_score × market_relevance_weight × base_sensitivity
   ```
   - `direction_sign`: +1 home_positive, -1 away_positive, 0 neutral/uncertainty
   - `market_relevance_weight`: 1.0 for "both", 0.7 for single-market
   - `base_sensitivity`: configurable scalar (default 0.15), caps shift per item
3. Sum deltas: `total_delta = sum(delta_i)`
4. Adjust: `log_odds_adjusted = log_odds_baseline + total_delta`
5. Convert back: `p_adjusted = sigmoid(log_odds_adjusted)`, renormalise across outcome set
6. Compute implied probabilities from market odds (1/odds, normalize for overround)
7. Compute edge = adjusted_probability - implied_probability
8. Run `AlertGuard.check()` — if guardrails fail, set `alert_sent=False` and log reason
9. If edge and confidence thresholds pass: call `AlertChannel.send()`
10. Save `Forecast` to repository (store both baseline and adjusted probabilities for evaluation)

**Outputs to state**: `forecast`, `alert_sent`

SOLID notes:
- Does not fetch data — only synthesizes what's in state
- Alert decision logic lives in `AlertGuard`, not here

---

### `SupervisorAgent` — `agents/supervisor.py`

**Responsibility**: Build and run the LangGraph graph. Route between agents. Apply stopping rules.

This is the LangGraph entry point. It defines:
- `GraphState` (TypedDict)
- Node registration
- Edge routing (conditional edges for early exit if baseline shows no edge worth researching)
- Entry and finish points

SOLID notes:
- Does not implement any agent logic itself
- Each agent node is a separate class injected at graph construction time

---

## 6. Guardrails — `guardrails/alert_guard.py`

### `AlertGuard`

Single responsibility: decide whether a forecast meets the quality bar to send an alert.

**Checks**:
1. `evidence_count >= min_evidence_count`
2. Average `reliability_score >= min_avg_reliability`
3. Evidence recency: no item older than `max_evidence_age_hours`
4. Source diversity: at least `min_unique_sources` distinct domains
5. Edge value `>= min_edge_threshold`
6. Confidence score `>= min_confidence_threshold`
7. Spam check: no alert sent for this match in the last `spam_window_hours` unless odds changed by `min_odds_delta`

Returns `GuardResult(passed: bool, reasons: list[str])`.

---

## 7. Notebook Export — `make_notebook.py`

Script that assembles `demo.ipynb` from the Python source.

Logic:
1. Read source files in this order: `models/`, `providers/`, `analytics/`, `tools/`, `memory/`, `guardrails/`, `agents/`
2. For each file: add a markdown cell with the module name and one-line description, then a code cell with the file contents
3. Append a final demo section: instantiate components with test config, run one end-to-end forecast for a hardcoded match
4. Write the assembled notebook using `nbformat.v4`

This script is the only thing that touches the notebook. Never edit `demo.ipynb` manually.

---

## 8. Configuration — `config.py`

```python
from dataclasses import dataclass
import os

@dataclass
class Config:
    llm_provider: str          # "openai" or "claude"
    openai_api_key: str
    anthropic_api_key: str
    football_data_api_key: str
    odds_api_key: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    alert_email: str
    db_path: str
    min_edge_threshold: float = 0.05
    min_confidence_threshold: float = 0.60
    home_advantage_boost: float = 0.05
    react_max_steps: int = 8
    min_evidence_count: int = 3
    spam_window_hours: int = 6

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_provider=os.environ["LLM_PROVIDER"],
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            football_data_api_key=os.environ["FOOTBALL_DATA_API_KEY"],
            odds_api_key=os.environ["ODDS_API_KEY"],
            smtp_host=os.environ["SMTP_HOST"],
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ["SMTP_USER"],
            smtp_password=os.environ["SMTP_PASSWORD"],
            alert_email=os.environ["ALERT_EMAIL"],
            db_path=os.environ.get("DB_PATH", "soccer_forecast.db"),
        )
```

---

## 9. SQLite Schema — `memory/schema.sql`

```sql
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    competition TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff_time TEXT NOT NULL,
    status TEXT NOT NULL,
    final_score TEXT
);

CREATE TABLE IF NOT EXISTS market_odds (
    odds_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    timestamp TEXT NOT NULL,
    home_win REAL NOT NULL,
    draw REAL NOT NULL,
    away_win REAL NOT NULL,
    over_2_5 REAL NOT NULL,
    under_2_5 REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    run_timestamp TEXT NOT NULL,
    baseline_home_win REAL, baseline_draw REAL, baseline_away_win REAL,
    baseline_over REAL, baseline_under REAL,
    adjusted_home_win REAL, adjusted_draw REAL, adjusted_away_win REAL,
    adjusted_over REAL, adjusted_under REAL,
    confidence_score REAL,
    edge_market TEXT,
    edge_value REAL,
    alert_sent INTEGER NOT NULL DEFAULT 0,
    rationale TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    forecast_id TEXT NOT NULL REFERENCES forecasts(forecast_id),
    source TEXT NOT NULL,
    url TEXT,
    timestamp TEXT NOT NULL,
    summary TEXT NOT NULL,
    direction TEXT NOT NULL,
    reliability_score REAL NOT NULL,
    applies_to_market TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_reliability (
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    reliability_score REAL NOT NULL,
    notes TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id TEXT PRIMARY KEY,
    preferred_competition TEXT,
    preferred_markets TEXT,
    min_edge_threshold REAL,
    min_confidence_threshold REAL,
    lineup_rerun_enabled INTEGER,
    email_address TEXT
);
```

---

## 10. Implementation Order for Codex

Follow this order to avoid forward dependencies:

1. `models/match.py`, `models/evidence.py`
2. `providers/llm.py` (protocol only), `analytics/baseline.py` (protocol only), `memory/repository.py` (protocols + `VectorRepository`), `tools/alert_channel.py` (protocol only)
3. `config.py`
4. `analytics/features.py`, `analytics/baseline.py` (SimpleBaselineStrategy implementation)
5. `memory/schema.sql`, `memory/sqlite_repository.py`
6. `memory/chroma_repository.py` (ChromaVectorRepository)
7. `tools/fixtures.py`, `tools/odds.py`, `tools/search.py`, `tools/email_sender.py`
8. `tools/ingester.py` (ArticleIngester — depends on VectorRepository + WebSearchTool)
9. `mcp_server/server.py` (wraps fixtures, odds, search tools)
10. `guardrails/alert_guard.py`
11. `providers/openai_provider.py`, `providers/claude_provider.py`
12. `agents/stats_market.py`
13. `agents/news_context.py` (depends on VectorRepository + MCP client)
14. `agents/synthesis_alert.py`
15. `agents/supervisor.py`
16. `main.py`
17. `make_notebook.py`
18. `demo.ipynb` (generated, not hand-written)
