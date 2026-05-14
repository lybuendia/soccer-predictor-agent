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
- `python -m compileall soccer_forecast_agent tests scripts` passes in the project Python 3.11 virtual environment.
- Core infrastructure tests pass for baseline analytics, feature extraction, alert guardrails, SQLite repository behavior, article ingestion, config defaults, and Chroma embedding-provider injection.
- A local retrieval sanity check with six web-sourced sample article summaries and a Hugging Face sentence-transformers model produced relevant top results for team/topic queries such as Arsenal injuries, Chelsea poor form, Tottenham relegation danger, and Manchester City vs Arsenal title-clash context.
- In the local `.venv`, the offline automated suite currently passes with `37 passed, 3 deselected` when live-network LLM smoke tests are excluded.
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

- [x] `ToolDispatcher.dispatch()` — MCP client routing — `agents/news_context.py`
- [x] `NewsContextAgent._seed_context()` — vector pre-load before loop
- [x] `NewsContextAgent._react_step()` — single ReAct iteration
- [x] `NewsContextAgent._should_stop()` — stopping conditions
- [x] `NewsContextAgent.run()` — full loop wired to state
- [x] System prompt design for ReAct agent
- [x] Structured evidence extraction from LLM output
- [x] YAML-backed prompt repository for Phase 3 prompts — `prompts/news_context.yaml`, `prompts/loader.py`
- [x] Unit tests for stopping conditions and loop behavior — `tests/test_news_context.py`
- [x] Live provider smoke test for `chat()` and `chat_with_tools()` — `tests/test_live_llm_provider.py`
- [x] Live end-to-end smoke test for `NewsContextAgent.run()` with controlled retrieval/tool inputs — `tests/test_live_news_context.py`
- [ ] Small manual quality review over 5–10 representative matches: summary quality, direction labeling, and search query quality

**Phase 3 implementation notes**

- The ReAct agent now performs dual retrieval in the intended order: vector-store seeding first, then live search only when additional evidence is needed.
- The local runtime path now wires `NewsContextAgent` through a small MCP-compatible adapter in `main.py`, so tool calls from the ReAct loop can reach the real local `WebSearchTool` instead of failing on `ToolDispatcher(None)`.
- Prompt text for the research loop and evidence extraction has been moved out of agent code into a YAML-backed prompt module so prompt tuning can happen without editing orchestration logic.
- The live smoke tests now validate both provider-level tool calling and a real-model `NewsContextAgent.run()` path with controlled seeded context and controlled MCP search results.
- Based on current validation, the main remaining Phase 3 risk is no longer wiring. It is output quality: whether evidence summaries, direction labels, and model-generated search queries remain sensible across a small set of representative matches.
- A lightweight manual review is enough for Phase 3. Full prompt optimization, larger-scale prompt evaluation, and stricter query-style tuning can wait until Phase 4 and Phase 5, when this evidence is actually influencing forecast adjustments and alert decisions.

---

### Phase 4 — Synthesis and Alerts

- [x] `SynthesisAlertAgent._adjust_probability()` — log-odds math
- [x] `SynthesisAlertAgent._implied_probability()` — market conversion
- [x] `EmailAlertChannel.send()` and `_render_body()` — `tools/email_sender.py`
- [x] `SynthesisAlertAgent.run()` — full synthesis + guard + alert
- [x] System prompt for synthesis agent — `prompts/synthesis.yaml`
- [x] Offline synthesis/alert flow coverage — `tests/test_synthesis_alert.py`
- [x] `InterpretedEvidence` dataclass — synthesis-only semantic layer on top of stored `EvidenceItem`
- [x] `SynthesisResult` typed intermediate — holds adjusted forecast, confidence, edge market, edge value
- [x] Per-market direction fields (`winner_direction`, `goals_direction`) replacing single `direction`
- [x] Market weight field (1.0 for "both", 0.7 for single-market evidence)
- [x] Market scope enforcement in `_extract_evidence` — prevents LLM output from moving the wrong market
- [x] `draw_positive` heuristic — reduces home and away log-odds by 0.5 so draw share rises after renormalization
- [x] `dataclasses.replace` pattern for immutable `Forecast` construction — corrected rationale and alert_sent set before persist
- [x] `SynthesisDecision` structured LLM contract — synthesis LLM now recommends market, adjustment strength, conviction, and alert-worthiness
- [x] Bounded LLM-guided adjustment path — baseline treated as a prior; code-owned caps translate synthesis labels into probability moves
- [x] Deterministic fallback decision when synthesis LLM is unavailable
- [x] Recommended-market edge check — alerts only pass when the LLM's chosen market still clears deterministic edge and quality guardrails
- [x] Human-readable synthesis output — final summaries now separate the market category, recommended bet, market source, confidence, and the synthesis LLM's judgment

**Phase 4 implementation notes**

- `SynthesisAlertAgent.run()` now treats the statistical baseline as a prior and asks the synthesis LLM for a structured `SynthesisDecision`: market category, recommended market, bounded winner/goals adjustment strength, LLM conviction score, alert-worthiness judgment, rationale points, and risk points.
- The synthesis LLM no longer acts only as a rationale writer. Its structured decision now directly influences the bounded probability movement and whether the signal should even be considered alert-worthy before hard guardrails are applied.
- Final numeric adjustments remain code-owned and bounded: the LLM recommends direction and strength, while deterministic log-odds caps, normalization, edge math, persistence, and anti-spam checks remain outside model control.
- The synthesis LLM's recommendation is advisory but substantive: it chooses the market and bounded adjustment strength, while the code still refuses alerts when that recommended market does not clear the odds-based edge threshold or other hard guardrails.
- A deterministic fallback synthesis decision is generated from aggregated evidence directions when the synthesis LLM is unavailable, so the forecast pipeline remains usable during partial outages or in restricted environments.
- A dedicated `InterpretedEvidence` dataclass was introduced to separate synthesis semantics (market directions, market weight) from the stored `EvidenceItem` record. Evidence items are stored in SQLite; interpreted evidence is the synthesis-only view created at extraction time by the news agent.
- Market scope enforcement ensures that evidence flagged as `applies_to_market="winner"` cannot move the goals market, and vice versa — this prevents malformed LLM output from contaminating the wrong market's probability.
- The `draw_positive` treatment is a documented heuristic: because log-odds space has no direct draw axis, draw probability rises indirectly by reducing home and away, keeping raw draw constant, and relying on renormalization to increase draw's share.
- User-facing summaries now expose both layers of the decision: the quantitative edge view and the synthesis LLM's recommendation in plain language, including market category, recommended bet, confidence, and market source provenance.
- Offline tests cover the happy path, the guard-failure path (including LLM-declared non-alert-worthy decisions), bounded draw adjustments, and confidence sensitivity to LLM conviction.

---

### Phase 5 — Orchestration and Evaluation

- [x] `GraphState` TypedDict — `agents/supervisor.py`
- [x] `SupervisorAgent.build_graph()` — LangGraph nodes + conditional edges
- [x] `SupervisorAgent.run()` — execute graph, return all forecasts
- [x] Match-loop pattern: `setup_next_match` node pops from `pending_match_ids`, resets per-match state
- [x] `pending_match_ids` and `all_forecasts` added to `GraphState`
- [x] `StatsMarketAgent` populates `pending_match_ids` from computed baselines
- [x] `SynthesisAlertAgent` appends each forecast to `all_forecasts`
- [x] Fixed ReAct message loop: `tool_calls` preserved in assistant history, `tool_call_id` passed with tool responses
- [x] `format_assistant_turn` and `format_tool_result` added to `LLMProvider` protocol and both adapters
- [x] Team name matching bug fixed in `OddsFetcher`: bookmaker outcome lookup now uses odds-API names, not football-data names
- [x] Full `team_aliases.yaml` — added Brentford, Crystal Palace, Fulham, Bournemouth, Sunderland, fixed Forest → Nottingham Forest
- [x] `ConsoleAlertChannel` — prints alerts to stdout when SMTP is not configured, used in demo and CI
- [x] Full end-to-end pipeline validated: 10/10 PL fixtures processed, 10 alerts generated with rationales
- [x] Conditional edge skip: three-way routing in `SupervisorAgent` — research / synthesise / done
- [x] `_has_baseline_edge()` — computes overround-normalised implied probs, checks max edge vs threshold
- [x] `min_edge_threshold` injected into `SupervisorAgent` from config; `[EDGE]` / `[SKIP]` log per match
- [x] `analytics/evaluation.py` — `MatchOutcome`, `brier_score_winner`, `brier_score_goals`, `aggregate`
- [x] `get_recent_finished()` — added `before: datetime` parameter for point-in-time backtesting
- [x] `validation/backtest.py` — runs SimpleBaselineStrategy on 345 resolved PL matches, prints Brier score report
- [x] `scripts/weekly_refresh.py` — cron-ready: ingest resolved outcomes → run forecast pipeline → print summary
- [x] `OddsFetcher` — added `winner_market_source`, `goals_market_source`, `market_sources_seen` provenance fields
- [x] `EmailAlertChannel` / `ConsoleAlertChannel` — human-readable market labels, plain-English edge explanation
- [x] Basic reporting: alert count, avg edge, abstention rate — `weekly_refresh.py` `print_summary()` prints per-match edge table plus aggregate counts after every run
- [x] Outcome update after match resolves — `weekly_refresh.py` step 1 calls `FixtureFetcher.fetch_finished()` and upserts resolved matches into SQLite on every scheduled run; no separate script needed
- [x] `ArticleIngester` in pipeline — `ArticleIngester` is fully implemented; by design it is invoked via `scripts/populate_rag.py` (pre-population) rather than inside `main.py` startup, so the forecast pipeline stays fast for manual runs
- [x] `scripts/populate_rag.py` — standalone RAG pre-population script (separate from forecast pipeline)

**Phase 5 implementation notes**

- The LangGraph graph uses a match-loop pattern rather than fan-out: `stats_market → setup_next_match → news_context → synthesis → setup_next_match` (loop), exiting to END when `pending_match_ids` is empty. This gives sequential per-match processing with clean per-match state resets between iterations.
- The ReAct message loop had a correctness bug: when the model made a tool call, the assistant message was reconstructed from extracted text, losing the `tool_calls` field, and tool responses lacked `tool_call_id`. The OpenAI API rejects both. The fix extended `LLMProvider` with `format_assistant_turn` and `format_tool_result`, implemented by each adapter in provider-specific formats (OpenAI uses `tool_call_id`; Anthropic uses `tool_result` inside a user message).
- Team name matching between football-data.org and the-odds-api.com required two fixes: (1) `fetch_odds` now passes `event["home_team"]` / `event["away_team"]` to `_parse_odds` so bookmaker outcome lookup uses the API's own team name strings; (2) several PL clubs were missing from `team_aliases.yaml`.
- `ConsoleAlertChannel` was added as a zero-config fallback for demo use when SMTP credentials are absent. `main.py` auto-selects it when `smtp_user` or `smtp_password` are empty.
- Conditional edge skip: `_route_after_setup` replaces the old two-way `_route_next`. Matches with max baseline edge below `min_edge_threshold` bypass the ReAct loop entirely and go straight to synthesis with no news evidence — the guard still applies, confidence is lower (0.30 floor), adjusted probabilities equal baseline. This prevents unnecessary Tavily and LLM calls on flat matches.
- Backtesting on 345 resolved PL matches: winner market Brier score 0.2126 (beats random 0.222); goals market 0.2689 (above random 0.250, expected since the baseline has no xG signal). Most-likely outcome correct 45.8%. Both scores are the right reference point for comparing after qualitative adjustment is added.
- RAG pre-population is kept as a separate `scripts/populate_rag.py` concern rather than wired into `main.py` startup, so the forecast pipeline stays fast for manual runs. `ArticleIngester` is fully implemented — the script is the only missing piece.

---

### Phase 6 — Notebook Export

- [ ] `make_notebook.py` — assembles `demo.ipynb` from source
- [ ] `demo.ipynb` — end-to-end executed demo
- [ ] Verify notebook executes top-to-bottom with `nbconvert`

---

### Phase 7 — Advanced Analytics (Stretch)

- [x] Dixon-Coles model — `analytics/dixon_coles.py`
- [x] Compare Dixon-Coles vs simple baseline on Brier score
- [x] `get_all_finished()` added to `SQLiteRepository` — DC fitting entry point
- [x] DC wired as default baseline in `main.py` and `weekly_refresh.py` with Enhanced fallback
- [ ] Calibration plot and final comparison report (deferred to notebook)

---

### Phase 8 — Scheduled Pipelines and Productionization

- [ ] Refactor manual `scripts/` utilities into dedicated ingestion/workflow modules for runtime use
- [ ] Separate developer-only inspection/debug scripts from scheduled data pipelines
- [ ] Add scheduled refresh pipeline for upcoming fixtures and historical finished matches into SQLite
- [ ] Add scheduled refresh pipeline for market odds updates
- [ ] Add scheduled news ingestion pipeline for search -> chunk -> embed -> Chroma upsert
- [ ] Add APScheduler or cron entry points for recurring ingestion and refresh jobs
- [ ] Add structured logging and error handling for scheduled jobs
- [ ] Document the production path for local jobs vs deployed scheduler execution

---

### Phase 9 — Production Upgrade (Post-Submission Roadmap)

This phase describes what a genuinely production-grade version looks like — the changes that move it from a well-engineered academic project to something with real forecasting edge and operational reliability.

---

#### 9.1 xG-Powered Dixon-Coles (Biggest Single Improvement)

The current DC model uses actual goals, which are noisy. A team that hits the post three times and loses 0-1 looks identical to a team that was dominated and lucky. Expected Goals (xG) measures the quality of chances created and conceded — it is a far better signal of true team strength.

- Ingest per-match xG data from Understat, FBref, or StatsBomb Open Data
- Replace `home_goals` / `away_goals` in the DC log-likelihood with `home_xG` / `away_xG`
- DC parameters now reflect genuine attacking and defensive quality, not shot-stopping luck

**Expected impact:** Winner Brier below 0.200, Goals Brier below 0.230. This single change closes roughly half the gap between the current model and professional closing-line models. It is the highest-leverage improvement available.

---

#### 9.2 Multi-Season Historical Data

One season (~385 matches, 33 teams) is not enough for stable DC parameter estimates, especially for promoted clubs. Three seasons would give ~1,100 training matches and dramatically tighter team strength estimates.

- Ingest 3–5 seasons via football-data.org or StatsBomb Open
- Use steeper time-decay (ξ ≈ 0.003–0.005) so older seasons inform parameters without dominating recent form
- Automatic gameweek re-fitting: DC parameters refresh after each round of fixtures
- Promoted team priors: seed from their Championship xG stats rather than defaulting to league average

---

#### 9.3 Lineup-Aware Strength Adjustment

DC captures long-run team strength but cannot see that Haaland is suspended or that Arsenal's first-choice XI is unavailable. Lineup data closes this gap and is available 60 minutes before kickoff.

- Integrate API-Football or RapidAPI Football for confirmed lineups
- Build a `LineupAdjuster` that modifies λ_h and λ_a from DC based on player importance scores:
  - First-choice striker absent → multiply home attack lambda by 0.82
  - First-choice keeper absent → multiply opponent attack lambda by 1.08
- Player importance derived from season xG contribution or Transfermarkt valuations
- Trigger a synthesis re-run when confirmed lineups deviate from expected

**Why this matters:** Lineup confirmation is where most of the remaining edge lives. Statistical models see the long run; lineups see the day. Combining both captures both.

---

#### 9.4 Sharp Money Signal via Pinnacle Line Movement

Pinnacle is the sharpest bookmaker — sharp bettors can bet there and Pinnacle accepts them. When Pinnacle's line moves significantly between open and close, institutional money has come in and the market has updated.

- Track opening and closing Pinnacle odds per match via The Odds API
- Compute `line_movement = closing_implied_prob - opening_implied_prob` per market
- If Pinnacle moves in the same direction as the system's edge → boost confidence score
- If Pinnacle moves against the system's edge → apply confidence penalty and consider suppressing the alert
- Log every line movement against every alert decision for retrospective calibration

**Why this matters:** The DC model cannot see information that sharp bettors have. Pinnacle's closing line is the best publicly available proxy for true probability. Agreement with line movement validates the edge; disagreement is a red flag.

---

#### 9.5 Weather Integration for Goals Market

Rain, wind, and cold measurably suppress goals in outdoor football. This is a known, consistently exploitable signal the current goals model ignores entirely.

- Free tier: Open-Meteo or WeatherAPI — forecast weather at each stadium at kickoff
- Build a `WeatherAdjuster` that modifies expected total goals:
  - Wind > 30 km/h → multiply expected total goals by 0.92
  - Heavy rain → 0.94
  - Temperature < 2°C → 0.96
- Adjustments applied in log-odds space before DC output is finalized
- Strongest effect: outdoor northern English grounds (Newcastle, Everton, Burnley) in November–February

---

#### 9.6 Multi-Competition Expansion

The entire architecture is competition-agnostic. Extending to La Liga, Bundesliga, and Champions League is a configuration change, not a rebuild.

- Add competition configs (API keys, team alias YAMLs, league-average goals)
- Fit separate DC models per competition
- Route alerts by user competition preferences
- **High-value targets:** Champions League (higher liquidity, more news, bigger stakes) and La Liga (Understat xG data is excellent for Spanish clubs)

---

#### 9.7 Real-Time Alerts via Telegram Bot

Email has latency and lands in promotions folders. Lineup confirmation happens 60–90 minutes before kickoff and the window to act closes fast.

- Replace or supplement `EmailAlertChannel` with `TelegramAlertChannel` (python-telegram-bot)
- Alerts formatted as rich messages with inline buttons: *View full rationale* / *Suppress this match*
- Bot commands: `/forecasts` (today's edges), `/calibration` (30-day Brier), `/suppress Arsenal`
- Lineup re-check alerts become urgent pushes the moment confirmed XIs deviate from expected

---

#### 9.8 Automated Rolling Calibration and Self-Tuning

`base_sensitivity` (the log-odds adjustment scalar, currently 0.15) is static. If the ReAct agent's qualitative adjustments are systematically over- or under-confident, the system never corrects.

- After each match resolves, compute whether the qualitative adjustment improved or hurt the Brier score
- Maintain a rolling 60-match window of `adjustment_brier_delta`
- If adjustments are consistently hurting: decay `base_sensitivity` by 5%
- If adjustments are consistently helping: increase by 3% up to a cap of 0.25
- Source reliability scores updated the same way: sources whose evidence led to correct adjustments see their score bumped; misleading sources get penalised
- This closes the feedback loop — the system measurably improves over a season rather than staying static

---

#### 9.9 Cloud Deployment on a $15/Month Stack

The full pipeline can run on a $5/month VPS (Hetzner CX11, DigitalOcean Droplet) with zero infrastructure overhead.

- Dockerize: `Dockerfile` + `docker-compose.yml` (app + ChromaDB volume)
- Cron on the VPS:
  - `06:00` daily: `populate_rag.py` (fresh team news)
  - `08:00` daily: `weekly_refresh.py --days-ahead 7` (forecast scan)
  - Match day −90 min: lineup re-check trigger (APScheduler)
- PostgreSQL upgrade path: swap SQLite with zero code change via a SQLAlchemy abstraction layer on `SQLiteRepository`
- Secrets via `.env` on the VPS or GitHub Actions secrets for CI/CD
- **Total cost:** VPS $5 + The Odds API free + football-data.org free + Tavily $5 free credits + OpenAI ~$3/week ≈ **under $15/month for a fully automated full-season system**

---

#### 9.10 Live Calibration Dashboard

`validation/backtest.py` produces a printed report. A production system needs a live view so drift is visible the moment it starts.

- FastAPI endpoint serving calibration metrics as JSON
- Streamlit dashboard:
  - Rolling 30-match Brier scores (DC vs Enhanced, winner vs goals)
  - Alert history: edge size, market, confidence, outcome
  - Calibration curve: predicted probability bucket vs actual frequency
- Evidence source hit rate: which sources consistently led to correct directional calls
- Line movement agreement rate: how often does the system agree with Pinnacle's closing direction?
- Auto-refreshes after each `weekly_refresh.py` run

---

#### 9.11 Forecast Engine V2: Stacked Ensemble + Closing-Line Calibration

The current architecture is strong because it separates a statistical prior from LLM-guided qualitative adjustment. The next realistic leap is to stop treating that as a single-path forecast and instead turn it into an ensemble whose final probabilities are learned from historical performance.

- Build three forecast components per match:
  - `StatsPrior`: Dixon-Coles / xG-driven pre-news probabilities
  - `ContextPrior`: lineup, injury, weather, and qualitative synthesis signals converted into structured numeric features
  - `MarketPrior`: opening odds, current odds, line movement, and bookmaker disagreement metrics
- Train a lightweight meta-model on historical resolved matches:
  - multinomial logistic regression or gradient-boosted trees for `home/draw/away`
  - binary calibrated model for `over/under`
  - inputs are the three priors plus confidence, source diversity, evidence recency, and line movement features
- Use time-split validation only:
  - train on older match windows, validate on later windows
  - never random-split football matches across time or the closing-line features will leak future information
- Add explicit probability calibration:
  - isotonic regression or Platt scaling on top of the ensemble outputs
  - separate calibration per market family (`winner`, `goals`)
- Teach the model abstention:
  - train a second-stage classifier for `alert_worthy`
  - target is whether a forecast historically beat the closing line by enough margin after vig
  - this becomes the production version of today's heuristic `AlertGuard` thresholds
- Log feature attribution for every forecast:
  - which part of the ensemble moved the probability most
  - whether the edge came from stats, injuries, line movement, or disagreement with the market
- Keep the LLM bounded and legible:
  - the LLM still does evidence synthesis and structured adjudication
  - but the final probability layer is learned from history rather than hand-tuned weights alone

**Why this matters:** this is the point where the project stops being "a good model plus smart glue" and becomes a true forecasting system. The market remains the strongest benchmark in the world; the ensemble learns when to trust the stats, when to trust the news, and when to simply stand down.

**Most realistic production target:** beat the current baseline and reduce false-positive alerts, not "beat Pinnacle every week." The win condition is sharper calibration, cleaner abstention, and fewer emotionally convincing but negative-EV alerts.

---

## 18. Final Recommendation

The strongest realistic version of this project is a multi-agent soccer market intelligence assistant for Premier League matches that combines:
- a practical baseline analytics layer
- a ReAct research agent for qualitative context
- SQLite memory
- email alerts
- strong guardrails around evidence quality and uncertainty

The Phase 9 roadmap above is the path from a well-calibrated academic system to a genuinely production-grade forecasting pipeline.

**Highest-leverage single change: xG-powered Dixon-Coles (9.1).** It addresses the root cause of the Goals Brier being above random and would likely push Winner Brier below 0.200.

**Highest-leverage operational change: Telegram alerts + lineup integration (9.3 + 9.7).** The window between lineup confirmation and kickoff is where the real edge lives, and email is too slow to capture it.

**Highest-leverage production upgrade: Forecast Engine V2 (9.11).** A stacked ensemble with closing-line-aware calibration is the most realistic way to turn today's strong architecture into a system that knows when to trust itself and when to abstain.
