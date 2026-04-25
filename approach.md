# Approach v3: Agentic Soccer Market Intelligence Assistant

## 1. Project Description

This project proposes a multi-agent AI system that monitors Premier League matches, gathers public information and structured match data, estimates probabilities for selected soccer betting markets, and sends email alerts when the system detects a meaningful edge relative to market odds. The system is designed as an agentic decision-support assistant, not an autonomous betting bot.

The architecture combines a statistical baseline, a ReAct-style research workflow, specialized agents, persistent memory, Python-based tool use, and explicit guardrails around evidence quality, uncertainty, and financial-risk framing. The goal is to satisfy the course requirements through deliberate prompting, orchestration of multiple agents, memory across runs, tool calling, and safeguards.

## 2. Finalized MVP Scope

### Domain
- Soccer

### Competition
- Premier League

### Markets
- Match winner (home / draw / away)
- Over/under 2.5 goals

### Core behavior
- Monitor upcoming Premier League fixtures
- Pull market odds for selected markets
- Run a lightweight statistical baseline on historical match data
- Research qualitative context through a ReAct agent
- Compare the system estimate to market odds
- Send an email alert only if edge and confidence thresholds are met

### Product type
- Edge detector
- Not a general “predict every match” system
- Not an autonomous betting system

### Notification channel
- Email

### Storage / memory
- SQLite for MVP

## 3. Architecture Decision

The biggest architectural decision is how probabilities are produced.

A purely LLM-based forecasting system would be easy to build, but weak analytically. LLMs can reason about context, summarize evidence, and explain directional effects well, but they are not the best foundation for calibrated probability generation in a forecasting system.

For that reason, the project uses this principle:

**Use a statistical baseline for probabilities, and use the LLM to research and interpret qualitative context.**

That keeps the system more evaluable, more explainable, and more honest about what each component is good at.

## 4. Practical MVP vs Full Vision

Because implementation time is limited, the project will be built in phases.

### MVP implementation
The MVP will use:
- a lightweight baseline probability approach
- public match data and market odds
- a ReAct research agent for qualitative context
- an email alert workflow
- SQLite memory and evaluation logging

### Full vision
The full version can later replace the lightweight baseline with a stronger soccer forecasting model such as Dixon-Coles and evaluate whether qualitative adjustments improve predictive performance.

This keeps the project realistic while still showing a strong long-term design.

## 5. Agent Architecture

The system will use a supervisor-based multi-agent architecture with four core agents.

### 5.1 Supervisor Agent
Responsible for:
- orchestrating the workflow
- deciding which agent runs next
- managing shared state
- applying stopping rules
- coordinating memory updates

### 5.2 Stats and Market Agent
Responsible for:
- fetching upcoming fixtures
- retrieving market odds
- computing a structured baseline view from match data
- producing a stats package for downstream agents

This agent uses Python tools for actual analysis rather than only text reasoning.

### 5.3 News and Context Agent
Responsible for:
- using a ReAct loop to search for injuries, suspensions, lineup uncertainty, congestion, motivation, and other relevant context
- evaluating source quality and recency
- summarizing evidence with direction and relevance

This is the main ReAct research agent and the clearest demonstration of agentic behavior in the project.

### 5.4 Synthesis and Alert Agent
Responsible for:
- combining the baseline view, qualitative evidence, and market odds
- applying a log-odds adjustment from evidence (see section 5.5)
- deciding whether a meaningful edge exists
- assigning a confidence score
- generating the final rationale
- deciding whether to send an email alert

This keeps the architecture simpler than splitting forecasting and alerting into separate agents.

### 5.5 Qualitative Adjustment Mechanism

The synthesis agent adjusts baseline probabilities in log-odds space. For each evidence item, a signed delta is computed as:

```
delta_i = direction_sign × reliability_score × market_relevance_weight × base_sensitivity
```

Where:
- `direction_sign` is +1 (favours home), -1 (favours away), or 0 (neutral / uncertainty)
- `reliability_score` is the source reliability value (0.0 – 1.0)
- `market_relevance_weight` is 1.0 for "both", 0.7 for single-market relevance
- `base_sensitivity` is a configurable scalar (default 0.15) that caps the maximum shift per item

Total adjustment: `total_delta = sum(delta_i)` across all evidence items.

Adjusted log-odds: `log_odds_adjusted = log_odds_baseline + total_delta`

Convert back: `p_adjusted = sigmoid(log_odds_adjusted)`, then renormalise across the outcome set (home_win + draw + away_win = 1.0).

This mechanism is fully traceable — every probability shift is attributable to specific evidence items — and avoids the boundary violations that additive adjustments on raw probabilities produce.

## 6. Why ReAct Fits This Problem

ReAct is a strong fit because soccer forecasting is not a one-shot question-answering task. A useful estimate often requires iterative investigation:

1. inspect the match and market
2. identify what information is missing
3. search for relevant evidence
4. interpret findings
5. compare conflicting signals
6. decide whether more research is needed
7. stop only when evidence is sufficient

This think -> act -> observe -> revise pattern maps directly to the News and Context Agent.

## 7. ReAct Loop Design

### 7.1 Inputs
Each run starts with:
- match metadata
- market odds
- current time
- relevant team history from memory
- user thresholds and preferences
- system guardrails

### 7.2 What the ReAct agent investigates
- key player availability
- confirmed or expected lineup changes
- schedule congestion
- motivation and context
- recent developments not yet reflected in data
- whether the evidence is strong enough to justify an alert

### 7.3 Tools available
The ReAct agent may:
- search recent soccer news (live web)
- search the general web
- retrieve official club or league information
- query the local vector store for semantically similar prior articles and team context
- retrieve prior notes from memory
- store evidence items

### 7.4 Structured evidence output
Each evidence item includes:
- source
- timestamp / recency
- evidence summary
- direction
- reliability score
- market relevance

### 7.5 Stopping conditions
The loop ends when:
- enough evidence has been collected
- evidence is too weak or conflicting
- the step budget is exhausted
- a guardrail forces abstention

## 8. Memory Design

Memory should be explicit because it is one of the course requirements.

### 8.1 User preference memory
Stores:
- preferred competition
- preferred markets
- minimum edge threshold
- minimum confidence threshold
- whether lineup re-checks are enabled
- notification preferences

### 8.2 Forecast history memory
Stores:
- match
- timestamp
- baseline estimate
- final estimate
- market odds
- confidence
- whether an alert was sent
- final outcome

### 8.3 Source reliability memory
Stores:
- source name or domain
- category
- rule-based reliability score
- notes

For MVP, source reliability will be rule-based rather than learned automatically.

### 8.4 Context memory
Stores:
- recent reasoning summaries for the same teams
- unresolved uncertainties from earlier scans
- lineup or injury notes worth re-checking

### 8.5 Vector memory (RAG layer)
Structured data (fixtures, odds, forecasts, preferences) is stored in SQLite — exact lookup and relational queries are the right tool there. RAG is used specifically for the unstructured news and evidence layer, where fuzzy semantic retrieval adds genuine value.

A local ChromaDB vector store holds ingested articles, injury reports, match previews, and prior reasoning summaries. The `ArticleIngester` runs on the daily scan schedule, fetches new soccer news, embeds content using an injectable embedding provider (local Hugging Face by default, OpenAI as a swap path), and stores it in Chroma with team and match metadata as filters.

At research time, the News/Context Agent queries Chroma alongside live web search:
- **Live web search**: catches breaking news, same-day updates
- **Vector retrieval**: surfaces relevant prior context about the same teams, players, or conditions from the accumulated knowledge base

This dual retrieval strategy means the system gets more capable over time as the corpus grows — a property that is directly valuable if the system runs in production.

**Production path**: swap ChromaDB for Pinecone or Weaviate without changing the retrieval interface.

## 9. Proposed SQLite Schema

A lightweight MVP schema could include:

### `matches`
- match_id
- competition
- home_team
- away_team
- kickoff_time
- status
- final_score

### `market_odds`
- odds_id
- match_id
- timestamp
- home_win_odds
- draw_odds
- away_win_odds
- over_odds
- under_odds
- market_line

### `forecasts`
- forecast_id
- match_id
- run_timestamp
- baseline_home_win
- baseline_draw
- baseline_away_win
- baseline_over
- baseline_under
- adjusted_home_win
- adjusted_draw
- adjusted_away_win
- adjusted_over
- adjusted_under
- confidence_score
- edge_market
- edge_value
- alert_sent
- rationale

### `evidence`
- evidence_id
- forecast_id
- source
- url
- timestamp
- summary
- direction
- reliability_score
- applies_to_market

### `user_preferences`
- preference_id
- preferred_competition
- preferred_markets
- min_edge_threshold
- min_confidence_threshold
- lineup_rerun_enabled
- email_address

### `source_reliability`
- source_id
- source_name
- category
- reliability_score
- notes
- updated_at

## 10. Guardrails and Safeguards

### 10.1 No autonomous betting
The system never places bets. It only sends research-backed alerts.

### 10.2 Evidence quality guardrails
No alert unless:
- enough evidence items exist
- evidence is recent enough
- evidence comes from sufficiently reliable sources
- source diversity is adequate

### 10.3 Lineup uncertainty guardrail
If a match appears highly sensitive to lineup uncertainty and lineups are not confirmed:
- lower confidence
- delay alert
- or abstain

### 10.4 Rule-based source reliability
For MVP, source trust starts with explicit rules rather than learned trust. For example:
- official club or league sources: high reliability
- major sports outlets: medium-high reliability
- secondary blogs: low-medium reliability
- rumor sites or anonymous social posts: low reliability

This keeps the system explainable and easier to debug.

### 10.5 Alert spam suppression
The system should not repeatedly alert on the same match unless:
- odds changed materially
- new lineup information changed the assessment
- the edge crossed a threshold

### 10.6 Prompt injection defense
External web content must be treated as untrusted data, not as instructions.

### 10.7 Conservative framing
Every alert should clearly state:
- this is decision support
- not guaranteed profit
- probabilities are estimates, not certainties

## 11. Framework Selection

The runtime system will use a LangGraph-first architecture.

### 11.1 LangGraph
LangGraph is the main orchestration layer for:
- multi-agent routing
- stateful workflow execution
- stopping conditions
- retries
- explicit control over the graph

### 11.2 Direct model integration
The project does not need LangChain as a required abstraction layer for MVP. Model calls can be made directly from agent nodes unless tool wrapping convenience is needed later.

### 11.3 Python
Python is the implementation language for:
- agent logic
- statistical computations
- data handling
- SQLite access
- scheduling
- email workflow

### 11.4 SQLite
SQLite is the structured memory and persistence layer for MVP: fixtures, odds, forecasts, evidence metadata, source reliability, and user preferences. All relational and exact-lookup data lives here.

### 11.5 ChromaDB (RAG layer)
ChromaDB is the vector store for unstructured news and evidence content. Embeddings are generated through an injectable provider interface so the project can use a local Hugging Face model for low-cost development and swap to OpenAI later if validation shows it improves retrieval quality. Collections are filtered by team and date at retrieval time. Production replacement is Pinecone or Weaviate via the same `VectorRepository` interface.

### 11.6 MCP — Tool Layer
The data tools (fixture fetcher, odds fetcher, web search) are exposed as an **MCP server** using the official `mcp` Python SDK. The LangGraph agents are MCP clients. This separates tool implementation from agent logic and makes the tool layer independently reusable — a web dashboard, a mobile app, or Claude Desktop can connect to the same MCP server without any agent code.

For MVP the MCP server runs locally alongside the agent process. Production path: deploy the MCP server independently with auth.

### 11.7 APScheduler
APScheduler handles:
- daily scans
- pre-match deep analysis
- optional lineup re-checks

### 11.8 No UI for MVP
The project does not require a UI. Email alerts will serve as visible proof of end-to-end system behavior.

## 12. Development Workflow Acceleration

Claude Code and Codex can accelerate development, but they are not part of the runtime architecture.

### Claude Code can help with:
- scaffolding the codebase
- generating agent modules
- refactoring shared schemas
- building tests
- creating reusable dev workflows

### Codex can help with:
- parallel implementation tasks
- drafting evaluation scripts
- reviewing structure
- speeding up larger code changes

This keeps the submitted system academically clean while still using modern coding-agent workflows during development.

## 13. Baseline Analytics Strategy

Because the full statistical model may take more time than the MVP allows, the project will use a staged analytics strategy.

### MVP baseline
The first version can use a simpler baseline built from:
- recent form
- home/away splits
- goals scored
- goals conceded

Market implied probabilities are intentionally excluded from the baseline. Including them would create circularity: the edge claim is `system_probability - market_implied_probability`, so a baseline that incorporates market odds would be measuring distance from itself. The baseline must remain independent of market prices.

This is enough to demonstrate:
- Python tool calling
- structured quantitative analysis
- a non-LLM baseline component
- comparison against market odds

### Advanced baseline
A more rigorous soccer forecasting model, such as Dixon-Coles, can be added later as an advanced phase.

That model would:
- estimate attack and defense strengths
- model goals as a Poisson process
- produce scoreline probabilities
- derive winner and over/under probabilities
- provide a stronger basis for calibration analysis

## 14. Evaluation Plan

### Primary success metric
**Brier score** is the north star metric. It directly measures probabilistic calibration — penalising both overconfidence and underconfidence — and is well understood in forecasting literature. All other metrics below are secondary supporting evidence.

### 14.1 Forecast quality
- Brier score (primary)
- calibration over resolved matches
- comparison against raw market implied probabilities
- optional comparison between simple baseline and later advanced baseline

### 14.2 Adjustment evaluation
- compare baseline-only estimates vs final estimates after qualitative synthesis
- evaluate whether the research agent helped or hurt forecast quality

### 14.3 Alert usefulness
- number of alerts generated
- average edge on alerted matches
- abstention rate
- low-value alert rate

### 14.4 Agent behavior
- number of tool calls
- number of ReAct steps
- evidence diversity
- percentage of no-alert decisions

## 15. Data Flow

1. Scheduler triggers a scan
2. Stats and Market Agent fetches fixtures and market odds
3. Stats and Market Agent computes a baseline estimate in Python
4. News and Context Agent runs a ReAct research loop
5. Synthesis and Alert Agent combines baseline, evidence, and market odds
6. If thresholds are met, send email alert
7. Store forecast, evidence, and alert state in SQLite
8. After the match resolves, store outcomes for evaluation

## 16. Execution Schedule

### Daily scan
- identify upcoming fixtures
- run the baseline estimate
- select matches worth deeper analysis

### Deep analysis pass
- run 6 to 12 hours before kickoff

### Optional lineup re-check
- run 60 to 90 minutes before kickoff if the match was flagged as lineup-sensitive

## 17. Roadmap

---

### Phase 0 — Final Design

- [x] Confirm data sources (football-data.org, The Odds API, Tavily)
- [x] Define SQLite schema
- [x] Define vector store design (ChromaDB)
- [x] Define MCP server scope
- [x] Finalize agent architecture (Supervisor + 3 agents)
- [x] Define log-odds adjustment mechanism
- [x] Set guardrail thresholds
- [x] Define primary success metric (Brier score)
- [x] Write CLAUDE.md and agents.md for Codex
- [x] Write course proposal paragraph

---

### Phase 1 — Core Infrastructure

- [x] Python package structure and `__init__.py` files
- [x] All data models (`Match`, `Odds`, `Forecast`, `Evidence`) — `models/`
- [x] All abstract protocols (`LLMProvider`, `BaselineStrategy`, `MemoryRepository`, `VectorRepository`, `AlertChannel`)
- [x] OpenAI and Claude provider adapters — `providers/`
- [x] Config dataclass with `from_env()` — `config.py`
- [x] SQLite schema — `memory/schema.sql`
- [x] `init_db()` — runs schema on first startup
- [x] SQLite repository — all CRUD methods — `memory/sqlite_repository.py`
- [x] Source reliability seed data — `memory/seed_sources.py`
- [x] ChromaDB vector repository — `memory/chroma_repository.py`
- [x] `FixtureFetcher` — football-data.org v4 — `tools/fixtures.py`
- [x] `OddsFetcher` — The Odds API v4 with bookmaker ranking — `tools/odds.py`
- [x] `WebSearchTool` — Tavily REST — `tools/search.py`
- [x] `EmailAlertChannel` — SMTP STARTTLS — `tools/email_sender.py`
- [x] MCP server skeleton — `mcp_server/server.py`
- [x] `AlertGuard` — all guardrail checks — `guardrails/alert_guard.py`
- [x] `content_guard.wrap_external()` — prompt injection defence — `guardrails/content_guard.py`
- [x] `main.py` — full dependency wiring skeleton
- [x] `.env.example`, `.gitignore`, `python-dotenv` wired in `main.py`
- [x] `requirements.txt`
- [x] `ArticleIngester.ingest()` and `_chunk()` — `tools/ingester.py`

**Phase 1 completion notes**

The first implementation pass now includes the full core infrastructure scaffold: typed dataclasses, repository protocols, SQLite schema and repository, source reliability seed data, OpenAI and Claude provider adapters, football-data.org fixtures, The Odds API odds fetching, Tavily search, SMTP alerts, MCP tool exposure, Chroma vector storage, local/Hugging Face embedding support with an OpenAI swap path, and environment-driven dependency wiring.

Key infrastructure refinements completed during validation:
- `ArticleIngester` now searches recent team news, chunks result text with overlap, creates stable chunk IDs, and upserts typed `ArticleChunk` records.
- `ChromaVectorRepository` now uses an injected embedding provider for both upserts and queries, stores scalar team metadata that Chroma can filter reliably, and can switch between local Hugging Face and OpenAI embeddings without changing repository code.
- The MCP server now returns JSON rather than Python repr strings, and `get_odds` accepts `home_team` and `away_team` to match `OddsFetcher`.
- `AlertGuard` now normalizes timestamps to UTC before recency and spam-window checks.
- `EmailAlertChannel` handles missing edge values defensively when rendering subjects and bodies.
- `main.py` now keeps provider selection behind `LLMProvider` and `EmbeddingProvider`.

Current validation status:
- `python -m compileall soccer_forecast_agent tests scripts` passes in the Python 3.11 virtual environment.
- Core infrastructure tests pass for baseline analytics, feature extraction, alert guardrails, SQLite repository behavior, article ingestion, config defaults, and Chroma embedding-provider injection.
- A local retrieval sanity check with six web-sourced sample article summaries and a Hugging Face sentence-transformers model produced relevant top results for team/topic queries such as Arsenal injuries, Chelsea poor form, Tottenham relegation danger, and Manchester City vs Arsenal title-clash context.
- External-service production validation is still pending for live fixture/odds/search/email workflows.

---

### Phase 2 — Baseline Analytics

- [x] `SimpleBaselineStrategy.compute()` — form + goals baseline — `analytics/baseline.py`
- [x] `FeatureExtractor.extract()` — builds `MatchContext` — `analytics/features.py`
- [x] `StatsMarketAgent.run()` — fetch fixtures + odds + compute baseline — `agents/stats_market.py`
- [x] Historical match data ingestion (for form and goals averages)
- [x] Unit tests for baseline strategy — `tests/test_baseline.py`
- [x] Unit tests for feature extractor — `tests/test_features.py`

**Phase 2 implementation notes**

- Historical finished-match ingestion now runs through `FixtureFetcher.fetch_finished()` and `scripts/ingest_historical_matches.py`, storing resolved Premier League matches in SQLite for later feature extraction.
- `SQLiteRepository.get_recent_finished()` now supplies recent team history directly to `StatsMarketAgent`.
- `StatsMarketAgent.run()` now derives recent `W/D/L`, goals scored, and goals conceded from real stored match history instead of placeholder defaults.
- A YAML-backed team-name normalization layer now canonicalizes variations such as `Arsenal FC`, `Arsenal`, `Man City`, and `Tottenham Hotspur` so fixture ingestion, odds matching, and manual inspection use consistent team identities across APIs.
- Manual baseline validation using the historical SQLite store confirmed that teams with clearly different recent form produce different derived features and materially different baseline forecasts.
- `SimpleBaselineStrategy` was refined with a conservative win-probability floor and renormalization so the 1X2 market avoids unrealistic zero-probability outcomes in lopsided recent-form cases.

---

### Phase 3 — ReAct Research Agent

- [ ] `ToolDispatcher.dispatch()` — MCP client routing — `agents/news_context.py`
- [ ] `NewsContextAgent._seed_context()` — vector pre-load before loop
- [ ] `NewsContextAgent._react_step()` — single ReAct iteration
- [ ] `NewsContextAgent._should_stop()` — stopping conditions
- [ ] `NewsContextAgent.run()` — full loop wired to state
- [ ] System prompt design for ReAct agent
- [ ] Structured evidence extraction from LLM output
- [ ] Unit tests for stopping conditions — `tests/test_news_context.py`

---

### Phase 4 — Synthesis and Alerts

- [x] `SynthesisAlertAgent._adjust_probability()` — log-odds math
- [x] `SynthesisAlertAgent._implied_probability()` — market conversion
- [x] `EmailAlertChannel.send()` and `_render_body()` — `tools/email_sender.py`
- [ ] `SynthesisAlertAgent.run()` — full synthesis + guard + alert
- [ ] System prompt for synthesis agent
- [ ] End-to-end alert flow test — `tests/test_alert_flow.py`

---

### Phase 5 — Orchestration and Evaluation

- [x] `GraphState` TypedDict — `agents/supervisor.py`
- [ ] `SupervisorAgent.build_graph()` — LangGraph nodes + edges
- [ ] `SupervisorAgent.run()` — execute graph, return forecasts
- [ ] Conditional edge: skip research if no baseline edge
- [ ] `ArticleIngester` called on each daily scan in `main.py`
- [ ] Outcome update after match resolves
- [ ] Brier score evaluation pipeline — `analytics/evaluation.py`
- [ ] Basic reporting: alert count, avg edge, abstention rate

---

### Phase 6 — Notebook Export

- [ ] `make_notebook.py` — assembles `demo.ipynb` from source
- [ ] `demo.ipynb` — end-to-end executed demo
- [ ] Verify notebook executes top-to-bottom with `nbconvert`

---

### Phase 7 — Advanced Analytics (Stretch)

- [ ] Dixon-Coles model — `analytics/dixon_coles.py`
- [ ] Compare Dixon-Coles vs simple baseline on Brier score
- [ ] Calibration plot and final comparison report

## 18. Final Recommendation

The strongest realistic version of this project is a multi-agent soccer market intelligence assistant for Premier League matches that combines:
- a practical baseline analytics layer
- a ReAct research agent for qualitative context
- SQLite memory
- email alerts
- strong guardrails around evidence quality and uncertainty

The advanced statistical forecasting model should remain part of the roadmap, but as a later phase rather than a hard requirement for the first working version.
