# Learning Guide: SoccerForecastAgent

This document is a study companion for the project. The goal is not just to describe what files exist, but to help you understand:

- what has already been implemented
- why it was implemented that way
- which core AI and agentic AI concepts the project is demonstrating
- which software engineering concepts are being practiced
- how the pieces will fit together as the next phases are built

Use this file as a running explanation while you implement the rest of the system.

## 1. Big Picture

This project is a multi-agent soccer forecasting assistant focused on Premier League matches.

At a high level, the system is supposed to:

1. fetch upcoming fixtures
2. fetch market odds
3. compute a baseline probability estimate using simple soccer statistics
4. research qualitative context like injuries, lineups, and motivation
5. combine the quantitative baseline with qualitative evidence
6. decide whether there is enough signal to send an alert

The most important architectural decision was this:

**Do not let the LLM invent the probabilities from scratch.**

Instead:

- the statistical baseline produces the first probability estimate
- the LLM is used for research, interpretation, and explanation

That decision matters because it keeps the system more:

- explainable
- testable
- honest about uncertainty
- suitable for evaluation later

## 2. Core Agentic AI Concepts

Because this is an agentic AI course, it is important to understand not just the code, but the AI ideas the code is trying to demonstrate.

### 2.1 What Makes a System "Agentic"

A system is usually called agentic when it does more than answer one prompt once.

An agentic system typically:

- has a goal
- can decide what action to take next
- can use tools
- can keep or update state
- can react to observations
- can stop when a condition is met

A normal chatbot interaction is often:

- user asks one thing
- model answers once

An agentic workflow is more like:

1. inspect the current state
2. decide what is missing
3. call a tool
4. observe the result
5. revise the plan
6. continue or stop

This project is agentic because the final system is meant to:

- inspect a match
- fetch structured data
- research context
- synthesize findings
- decide whether to alert

That is not one-shot generation. It is a stateful decision process.

### 2.2 Agents vs Tools

One of the most important course concepts is the difference between an agent and a tool.

A tool:

- performs a specific action
- usually has a narrow input/output contract
- does not decide the broader workflow

Examples in this project:

- `FixtureFetcher`
- `OddsFetcher`
- `WebSearchTool`
- `EmailAlertChannel`

An agent:

- receives a broader goal
- decides what information matters
- determines what should happen next
- updates shared state

Examples in this project:

- `StatsMarketAgent`
- `NewsContextAgent`
- `SynthesisAlertAgent`
- `SupervisorAgent`

This distinction matters because a lot of beginner projects call everything an "agent" when some pieces are really just tools.

### 2.3 Multi-Agent Architecture

This project uses a multi-agent design.

That means different agents have specialized responsibilities instead of one giant agent trying to do everything.

Why split agents?

- better separation of concerns
- easier testing
- clearer prompts
- easier debugging
- lower cognitive load per agent

In this project:

- the stats agent handles structured quantitative work
- the news agent handles research
- the synthesis agent handles decision-making
- the supervisor handles routing and orchestration

This is a core agent design lesson:

**specialized agents are usually easier to control than one monolithic agent.**

### 2.4 ReAct

ReAct stands for a pattern that combines reasoning and acting.

The idea is:

- think about what is needed
- act by calling a tool
- observe the result
- think again

This matters because many real tasks are not answerable from memory alone.

In this project, ReAct is a natural fit because the system may need to:

- search for injury news
- inspect multiple sources
- compare conflicting reports
- decide whether evidence is sufficient

So instead of asking the model to guess all of that in one shot, the system will later let it work step by step.

That is a key idea in agentic AI:

**reasoning improves when the model can interact with the world instead of hallucinating missing information.**

### 2.5 Tool Use

Tool use means the model is not limited to its internal training knowledge.

It can call systems that provide fresh or precise information.

Examples in this project:

- fixture API for upcoming matches
- odds API for current market data
- web search for recent news
- vector retrieval for prior context

Why tool use matters:

- it reduces hallucination
- it adds access to current information
- it makes the system more grounded
- it allows workflows that pure text generation cannot do reliably

This is one of the strongest indicators that a project is truly agentic rather than just prompt-based.

### 2.6 Memory

Memory is another major course concept.

An agentic system becomes much more useful when it can carry information across steps or across runs.

This project uses multiple kinds of memory:

- short-lived workflow state in the graph
- structured memory in SQLite
- semantic memory in Chroma

It helps to separate these mentally:

#### Workflow state

This is the temporary state of one execution.

Examples:

- current match
- current evidence list
- baseline forecast
- current errors

This lives in the graph state and is passed between agents.

#### Persistent structured memory

This is durable information the system stores exactly.

Examples:

- saved matches
- saved forecasts
- source reliability
- alert history

This supports traceability and evaluation.

#### Persistent semantic memory

This is fuzzy text recall through embeddings.

Examples:

- prior article chunks
- context from earlier research
- team-related background information

This supports retrieval, not exact bookkeeping.

A big lesson here:

**memory is not one thing. Different memory types solve different problems.**

### 2.7 Retrieval-Augmented Generation (RAG)

RAG means the model is given retrieved external context before or during generation.

In this project, the news agent will use:

- live web search for recency
- vector retrieval for prior context

That combination is useful because each retrieval type solves a different problem:

- web search finds breaking updates
- vector search finds semantically related stored knowledge

This is sometimes called dual retrieval in the project documents.

The important concept is that the model is not expected to remember every relevant fact by itself. Retrieval extends the model with external context.

### 2.8 Orchestration

Orchestration is the logic that determines:

- which agent runs
- in what order
- with what shared state
- under what stopping conditions

In this project, that role belongs to the supervisor and the LangGraph workflow.

This is a very important agentic concept because agent systems are not just prompts. They are controlled processes.

Without orchestration:

- tools may be called at the wrong time
- state may become inconsistent
- agents may duplicate work
- stopping conditions may be unclear

You can think of orchestration as the control plane of the agent system.

### 2.9 Guardrails

Guardrails are rules or checks that keep the system from behaving recklessly.

In this project, guardrails include:

- minimum evidence count
- minimum reliability
- evidence recency
- source diversity
- confidence threshold
- edge threshold
- spam suppression
- prompt injection defense

This matters because agentic systems can feel competent while still making low-quality decisions.

Guardrails are how you prevent the system from acting on weak or unsafe outputs.

This is one of the most important ideas in applied AI:

**capability without control is not enough.**

### 2.10 Evaluation and Calibration

Another core AI concept in this project is evaluation.

The system is not just supposed to produce answers. It is supposed to produce probabilities that can later be judged.

That is why the project uses Brier score as the primary success metric.

Brier score matters because it evaluates probabilistic quality, not just whether one prediction was "right."

This is important in forecasting systems because:

- a 0.55 prediction and a 0.95 prediction should not be judged the same way
- overconfidence should be penalized
- underconfidence should also be penalized

This is a central lesson for AI systems that generate probabilities:

**good forecasting is about calibration, not just occasional correctness.**

### 2.11 Why This Project Counts as Agentic AI

If you ever need to explain why this project belongs in an agentic AI course, here is the concise answer:

It includes:

- multiple specialized agents
- tool calling
- persistent memory
- retrieval
- multi-step reasoning
- orchestration
- stopping rules
- guardrails
- evaluation

That combination is what makes it more than a normal LLM app.

## 3. What Has Been Implemented So Far

As of the current state of the project, Phases 0 through 5 are complete and the full end-to-end pipeline has been validated on live data.

That means the project now has:

- the design documents
- the package structure
- the core data models
- the abstraction layer through protocols
- configuration loading
- SQLite persistence
- Chroma vector storage
- source reliability seeding
- OpenAI and Claude adapters
- external tools for fixtures, odds, web search, email
- an MCP server skeleton
- guardrails
- article ingestion into the vector store
- the baseline analytics pipeline
- the `StatsMarketAgent.run()` workflow
- the `NewsContextAgent` ReAct research loop with dual retrieval
- YAML-backed prompts for the research agent
- `SynthesisAlertAgent.run()` — full log-odds adjustment, edge calculation, confidence scoring, rationale generation, guard check, and alert delivery
- `InterpretedEvidence` — the synthesis-layer view of evidence with per-market directions and market weight
- `SupervisorAgent.build_graph()` and `run()` — full LangGraph orchestration across all three agents
- unit and live tests for all major agent workflows
- `ConsoleAlertChannel` as a zero-config demo fallback

The full pipeline has been validated end-to-end on real live data: 10 upcoming Premier League fixtures, 10/10 matching odds retrieved, all matches researched, all alerts generated with rationales.

What remains are stretch and evaluation pieces:

- Brier score evaluation pipeline after matches resolve
- outcome tracking
- notebook export (`demo.ipynb`)
- advanced Dixon-Coles baseline (stretch)

## 4. Why the Project Was Built in Phases

This project was intentionally divided into phases because the architecture has dependencies.

For example:

- agents depend on tools
- tools depend on models
- repositories depend on schema and models
- orchestration depends on agents already existing

If you tried to build the agent logic first, you would constantly be blocked by missing infrastructure.

So the current implementation follows a sensible order:

1. define data structures
2. define interfaces
3. build persistence and tools
4. wire dependencies
5. implement the agents
6. orchestrate them
7. add evaluation and notebook export

This is a core engineering idea: **build from stable foundations upward**.

## 6. How Phase 3 Was Implemented

Phase 3 is the research layer of the system.

Its job is not to predict the match directly. Its job is to gather qualitative evidence that can later adjust the statistical baseline.

The main class is:

- `soccer_forecast_agent/agents/news_context.py`

The prompt source lives in:

- `soccer_forecast_agent/prompts/news_context.yaml`
- `soccer_forecast_agent/prompts/loader.py`

### 6.1 The goal of `NewsContextAgent`

`NewsContextAgent` is responsible for answering this question:

**What non-statistical context matters for this match, and how should that context be stored as structured evidence?**

Examples of that context:

- injuries
- lineup news
- rotation
- congestion
- motivation
- uncertainty

The important design choice is that this agent does not compute probabilities.

It only:

1. gathers context
2. structures it
3. saves it for downstream use

That is a very good example of single responsibility.

### 6.2 What happens inside `run()`

The `run()` method is the full Phase 3 workflow.

It goes in this order:

1. read `current_match_id` from shared state
2. find the `Match` object for that id
3. load the baseline forecast for that match
4. seed prior context from the vector store
5. extract structured evidence from the seeded context
6. build the initial research prompt
7. enter the ReAct loop
8. if the model requests a tool, dispatch it
9. convert tool output into structured evidence
10. save the evidence
11. stop when the evidence quality bar or step budget says to stop
12. return updated state

That means `run()` is coordinating a loop, not just making one LLM call.

### 6.3 Step 1: seed context before live search

Before the LLM starts deciding what to search for, the agent calls `_seed_context()`.

This method:

- builds a query from the home team and away team
- searches the vector store with both teams as metadata filters
- returns formatted text chunks

Why this matters:

- the agent should not start from zero every time
- older but still relevant context can be useful
- the system can avoid unnecessary searches if stored context is already enough

This is the “retrieval before action” part of the design.

### 6.4 Step 2: build the research prompt

The research prompt is now stored in YAML instead of inline strings.

The agent calls:

- `render_news_context_research_messages(...)`

That renderer creates:

- a `system` message
- a `user` message

The user message includes:

- match metadata
- baseline probabilities
- seeded vector context
- a short preview of evidence already collected

This is useful because the model is not just asked “find news.”

It is shown:

- what match it is working on
- what the current baseline says
- what evidence already exists

So the model can reason about what information is still missing.

### 6.5 Step 3: the ReAct loop itself

The core ReAct step is implemented in `_react_step()`.

ReAct means:

1. reason about what is missing
2. act by calling a tool if needed
3. observe the result
4. revise the next step

In code, `_react_step()` does this:

1. send the current messages plus tool definitions to `llm.chat_with_tools(...)`
2. read the assistant text
3. extract the first tool call if one exists
4. return:
   - `thought`
   - `tool_name`
   - `tool_args`

So each loop iteration can result in one of two things:

- the model decides to stop and only returns text
- the model requests a tool call

That is the key ReAct pattern:

- **thought -> action request -> observation -> next thought**

### 6.6 Step 4: tool dispatch

If the model requests a tool, the agent uses `ToolDispatcher.dispatch()`.

This dispatcher:

- validates the tool name
- calls the injected MCP client
- serializes the result into JSON text

Right now the important live research tool is:

- `search_news`

In the current local runtime path, `main.py` injects a small MCP-compatible adapter rather than a real remote MCP client.

That adapter exposes `call_tool(...)` and forwards:

- `search_news` to `WebSearchTool`
- `get_fixtures` to `FixtureFetcher`
- `get_odds` to `OddsFetcher`

This matters because the MCP server object itself is not the same thing as an MCP client.

So the practical MVP wiring is:

- local tools
- wrapped behind a client-like adapter
- routed through `ToolDispatcher`

This is useful because the LLM never directly reaches into tool code.

Instead:

- the agent gets a tool request
- the dispatcher routes it
- the tool result comes back as text

That separation is cleaner and easier to test.

### 6.7 Step 5: turn raw text into `EvidenceItem`

This is one of the most important parts of Phase 3.

The tool output itself is not yet structured evidence.

It is just raw text or JSON-like content.

So the agent calls `_extract_evidence()`.

That method:

1. uses a second prompt flow
2. asks the model to return strict JSON only
3. parses the JSON array
4. validates allowed direction and market values
5. looks up source reliability
6. creates `EvidenceItem` dataclasses

This is the point where unstructured retrieval becomes structured agent memory.

That is a major concept in agent systems:

- retrieval alone is not enough
- the system must transform observations into a stable internal representation

### 6.8 Step 6: stopping conditions

The loop stops in `_should_stop()`.

Right now the stopping logic is simple and intentionally conservative.

It stops when:

- `step >= max_steps`

or when:

- `len(evidence) >= min_evidence_count`
- average reliability is at least `min_avg_reliability`

This matters because agent loops need explicit stopping rules.

Without stopping rules, the model can:

- over-search
- waste tokens
- chase low-value evidence

So the loop is not “let the LLM decide forever.”

It is “let the LLM explore, but inside clear limits.”

### 6.9 Step 7: persistence

Every extracted evidence item is saved through `EvidenceRepository`.

That means the output of the research loop is not just temporary text in memory.

It becomes part of the system’s stored record.

This is important for:

- later synthesis
- later evaluation
- debugging
- auditability

### 6.10 Why there are two different prompts

Phase 3 uses two prompt flows for a reason.

The research prompt and the extraction prompt are not the same task.

The research prompt is for:

- deciding what is missing
- deciding whether to use a tool
- summarizing what matters

The extraction prompt is for:

- converting raw material into strict JSON evidence objects

Keeping those tasks separate is cleaner because the model is being asked to do two different kinds of work.

### 6.11 Why the prompts were moved to YAML

Originally the prompts were inline in `news_context.py`.

They were moved to:

- `prompts/news_context.yaml`

and rendered through:

- `prompts/loader.py`

That change matters because it:

- separates prompt text from control flow
- makes prompt iteration easier
- makes prompt testing easier
- keeps the agent code focused on orchestration logic

This is a good example of refactoring for maintainability, not just correctness.

### 6.12 How Phase 3 was validated

Phase 3 was not validated with only one kind of test.

It was validated in layers.

#### Unit and fake-based tests

`tests/test_news_context.py` checks:

- tool dispatch
- vector seeding
- stopping rules
- full fake-driven `run()` behavior
- malformed JSON handling
- unknown tool handling
- evidence save failure handling

These tests answer:

- does the code behave correctly?

#### Live provider smoke tests

`tests/test_live_llm_provider.py` checks:

- real `chat()` behavior
- real `chat_with_tools()` behavior

These tests answer:

- can the provider really talk to the selected model?

#### Live `NewsContextAgent` smoke test

`tests/test_live_news_context.py` checks:

- real model in the loop
- controlled vector input
- controlled tool result
- structured evidence output

This answers:

- can the real model actually complete the research-agent workflow?

### 6.13 What is still left after Phase 3

Phase 3 proves that the research loop works.

It does not yet prove that the whole product is finished.

The next stages still need:

- synthesis of baseline plus evidence
- edge calculation
- alert decisions
- orchestration across agents
- evaluation after matches resolve

So the right mental model is:

- Phase 2 gave the system its first estimate
- Phase 3 gave the system the ability to research context
- Phase 4 will combine those two things into a forecast decision

## 5. Core Architectural Concepts

### 4.1 SOLID

This codebase is trying to follow SOLID principles. You will see that especially in how the files are separated.

#### Single Responsibility Principle

Each class is supposed to do one thing.

Examples:

- `FixtureFetcher` only fetches fixtures
- `OddsFetcher` only fetches odds
- `EmailAlertChannel` only sends alerts
- `SQLiteRepository` only handles SQLite persistence
- `FeatureExtractor` only builds `MatchContext`

That makes the system easier to test and easier to change.

#### Dependency Inversion

High-level code should depend on abstractions, not on concrete SDKs.

That is why the project defines protocols like:

- `LLMProvider`
- `BaselineStrategy`
- repository protocols
- `AlertChannel`

The agents should later depend on those protocols, not directly on `openai`, `anthropic`, `sqlite3`, or `smtplib`.

This gives you swapability. For example:

- OpenAI can be replaced by Claude
- Chroma can be replaced later by Pinecone
- email could be replaced by Slack or SMS

### 4.2 Typed Models Instead of Raw Dictionaries

The project uses dataclasses for structured data.

Examples:

- [models/match.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/models/match.py)
- [models/evidence.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/models/evidence.py)

Why this matters:

- fields are explicit
- data shape is easier to reason about
- type hints improve safety
- tools and repositories have clearer contracts

Instead of returning vague JSON-like objects everywhere, the project passes around well-defined domain objects such as:

- `Match`
- `MarketOdds`
- `BaselineForecast`
- `Forecast`
- `EvidenceItem`
- `ArticleChunk`

This is good software design and also very helpful for debugging.

### 4.3 Protocols as Interfaces

Python does not force interfaces the way Java does, but `Protocol` gives you a lightweight version of that idea.

Look at:

- [providers/llm.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/providers/llm.py)
- [memory/repository.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/memory/repository.py)
- [tools/alert_channel.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/tools/alert_channel.py)
- [analytics/baseline.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/analytics/baseline.py)

These files define the expected behavior of components before the concrete implementations are written.

That gives you two benefits:

1. the design is clearer
2. the code becomes easier to swap and test

## 6. The Domain Model

The domain model is the vocabulary of the system.

### 5.1 Match and Market Data

In [models/match.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/models/match.py), the project defines:

- `Match`
- `MarketOdds`
- `BaselineForecast`
- `Forecast`
- `MatchContext`
- `AlertPayload`

These are important because they separate different layers of meaning:

- `Match` is the real-world event
- `MarketOdds` is what the betting market says
- `BaselineForecast` is what the statistical model says before qualitative adjustment
- `Forecast` is the stored record of a complete run
- `MatchContext` is the input package for the baseline strategy
- `AlertPayload` is the message-ready package for notifications

This separation prevents conceptual mixing.

For example, a very common mistake would be to merge market odds and forecast probabilities into one object. This project avoids that, which is the right call.

### 5.2 Evidence and Vector Chunks

In [models/evidence.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/models/evidence.py), the project defines:

- `EvidenceItem`
- `ArticleChunk`

These represent two different things:

- `EvidenceItem` is curated reasoning-ready evidence used by the agent
- `ArticleChunk` is raw stored text used for semantic retrieval

That distinction is subtle and important.

A chunk is not automatically evidence.

The vector store contains potentially useful text. The agent later decides what actually becomes evidence.

## 7. Persistence Design

The system uses two types of memory:

- structured memory in SQLite
- unstructured semantic memory in Chroma

This is one of the strongest design decisions in the project.

### 6.1 Why SQLite

SQLite is used for:

- matches
- forecasts
- evidence metadata
- source reliability
- user preferences

These are structured records with exact fields. You want reliable queries, not fuzzy retrieval.

Relevant files:

- [memory/schema.sql](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/memory/schema.sql)
- [memory/sqlite_repository.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/memory/sqlite_repository.py)

### 6.2 Why Chroma

Chroma is used for the news and context layer.

Why?

Because articles and reasoning notes are unstructured text. You do not usually know the exact words you want to search for later. That is where embeddings and semantic search help.

Relevant file:

- [memory/chroma_repository.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/memory/chroma_repository.py)

### 6.3 One Important Design Lesson

Not all memory is the same.

Use:

- relational storage for exact structured facts
- vector storage for fuzzy semantic recall

That is a core AI systems design pattern.

## 8. The SQLite Repository

`SQLiteRepository` is one class implementing several repository protocols.

At first that might look like a violation of Single Responsibility, but it is acceptable here because the responsibility is still one thing:

**SQLite access.**

Inside it, you can see several categories of methods:

- match persistence
- forecast persistence
- evidence persistence
- source reliability lookup

This class is important because it keeps SQL out of the agent logic.

That is exactly what the repository pattern is for.

Instead of an agent doing this:

```python
cursor.execute("SELECT * FROM forecasts ...")
```

the agent should later do this:

```python
forecast_repo.get_by_match(match_id)
```

That separation keeps business logic from becoming tangled with storage logic.

## 9. Source Reliability

The project seeds a `source_reliability` table in:

- [memory/seed_sources.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/memory/seed_sources.py)

This is a very practical MVP decision.

Instead of trying to build an automatic trust-learning system, the project starts with rule-based reliability scores:

- official sites are high reliability
- major outlets are medium-high
- social sources are low

Why this is a good decision:

- it is explainable
- it is easy to debug
- it satisfies the guardrail requirements
- it avoids pretending the system has learned trust when it has not

This is a broader lesson in MVP design:

**Prefer explicit simple rules when they are enough to support the behavior you need.**

## 10. LLM Provider Abstraction

The project has:

- [providers/llm.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/providers/llm.py)
- [providers/openai_provider.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/providers/openai_provider.py)
- [providers/claude_provider.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/providers/claude_provider.py)

This is the abstraction layer for model calls.

Why it exists:

- the rest of the application should not care which model vendor is active
- only the provider adapter should know the SDK details

This is a classic adapter pattern.

The application expects:

- `chat(...)`
- `chat_with_tools(...)`

Each provider translates that expectation into the vendor-specific API call.

This is one of the cleanest decisions in the codebase.

## 11. Configuration and Environment Variables

The configuration is centralized in:

- [config.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/config.py)

This file defines a `Config` dataclass and a `from_env()` constructor.

That gives you:

- one place to see runtime settings
- a predictable source of truth
- cleaner dependency wiring in `main.py`

This is better than reading environment variables all over the codebase.

The general rule is:

**read environment once, then pass structured configuration through the system.**

## 12. External Tools Layer

The project has several tool classes under `tools/`.

### 11.1 FixtureFetcher

File:

- [tools/fixtures.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/tools/fixtures.py)

What it teaches:

- wrap external APIs behind your own typed interface
- return domain objects, not raw API responses

### 11.2 OddsFetcher

File:

- [tools/odds.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/tools/odds.py)

Important design choice:

The fetcher includes bookmaker ranking logic instead of blindly taking the first bookmaker returned.

That matters because market data often comes from several sources, and not all sources are equally useful or consistent.

This is a good example of putting small domain knowledge close to the tool layer.

### 11.3 WebSearchTool

File:

- [tools/search.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/tools/search.py)

Key concept:

External search results are useful, but untrusted.

That is why the codebase also includes:

- [guardrails/content_guard.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/guardrails/content_guard.py)

This is about prompt injection defense.

The LLM should treat web results as data to analyze, not instructions to obey.

That is a very important concept in agentic systems.

### 11.4 ArticleIngester

File:

- [tools/ingester.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/tools/ingester.py)

This class was one of the last pieces completed in Phase 1.

It does four things:

1. searches for team-related news
2. combines title and snippet into text
3. chunks the text into overlapping segments
4. stores the chunks in the vector repository

Why chunking matters:

- embeddings have practical context-size limits
- smaller chunks retrieve better than giant blobs
- overlap helps preserve continuity across chunk boundaries

The stable chunk ID logic is also important because it supports repeat ingestion without creating uncontrolled duplication.

### 11.5 EmailAlertChannel

File:

- [tools/email_sender.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/tools/email_sender.py)

This class turns forecast data into a human-readable alert message.

The important lesson here is that notification formatting is a separate responsibility from forecast generation.

The forecast engine decides *what* to send.
The alert channel decides *how* to send it.

That is good separation of concerns.

## 13. MCP Server

File:

- [mcp_server/server.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/mcp_server/server.py)

This server exposes tools like:

- `get_fixtures`
- `get_odds`
- `search_news`

Why this matters:

The tools are being separated from the agents.

Instead of tightly coupling an agent to local Python imports, the project is moving toward a service-style tool layer.

That gives future flexibility:

- another MCP client could use the same tools
- the tool layer can be deployed separately later
- the agents become cleaner and more portable

This is an advanced architecture choice for a student project and a good one.

## 14. Vector Repository Design

One of the important fixes made during implementation was in the Chroma layer.

Originally, it is easy to accidentally build a vector repository that stores documents without actually using the intended embedding function properly.

The current implementation now:

- uses the injected embedding function during upsert
- uses the same embedding function during search
- stores team metadata in a scalar-filter-friendly way

This is a useful lesson:

**RAG systems fail quietly if metadata and embedding flow are not designed carefully.**

You can have code that looks right, but retrieval will be poor or misleading if:

- the query path and ingest path embed differently
- metadata filters are malformed
- chunks are too coarse

## 15. Guardrails

The project takes guardrails seriously.

Relevant files:

- [guardrails/alert_guard.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/guardrails/alert_guard.py)
- [guardrails/content_guard.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/guardrails/content_guard.py)

### 14.1 AlertGuard

`AlertGuard` enforces conditions before an alert is allowed.

It checks things like:

- enough evidence items
- average reliability
- evidence recency
- source diversity
- edge threshold
- confidence threshold
- spam suppression

Why this matters:

Without guardrails, an agentic system often becomes noisy and reckless.

This project is explicitly trying to avoid that.

### 14.2 UTC Normalization

A subtle but real bug risk in time-based systems is mixing:

- naive datetimes
- timezone-aware datetimes

That is why `AlertGuard` was updated to normalize timestamps to UTC before comparisons.

This is a classic practical engineering lesson:

**time handling is rarely trivial, even in small systems.**

## 16. Analytics Layer

The analytics code is in:

- [analytics/baseline.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/analytics/baseline.py)
- [analytics/features.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/analytics/features.py)

### 15.1 FeatureExtractor

`FeatureExtractor` builds a `MatchContext`.

This is a useful design because it creates a clean handoff:

- raw data comes in
- structured model-ready context comes out

That keeps baseline logic simpler.

### 15.2 SimpleBaselineStrategy

`SimpleBaselineStrategy` is the first forecasting method.

It uses:

- recent form
- goals scored averages
- goals conceded averages
- a small home advantage boost

One especially important decision:

**it does not use market odds as an input feature.**

Why?

Because later the project wants to compare:

`system_probability - market_implied_probability`

If the system probability already depends on market odds, that comparison becomes circular and much less meaningful.

That is one of the smartest design decisions in the whole project.

## 17. Main Dependency Wiring

The current wiring lives in:

- [main.py](/Users/lizethbuendia/Desktop/SoccerForecastAgent/soccer_forecast_agent/main.py)

This file is not the final business logic. It is the composition root.

A composition root is the place where the application assembles its concrete dependencies.

That means:

- config is loaded
- the database connection is created
- repositories are instantiated
- the vector store is configured
- tools are instantiated
- provider adapters are chosen
- agents are assembled

Even though the agent methods are not fully implemented yet, the existence of this wiring is important because it proves the architecture has a coherent shape.

## 18. All Agent Files Are Now Complete

All four agent files are now fully implemented:

- `agents/stats_market.py` — fixtures, odds, baseline, populates `pending_match_ids`
- `agents/news_context.py` — ReAct research loop, dual retrieval, structured evidence extraction
- `agents/synthesis_alert.py` — log-odds adjustment, edge calculation, confidence scoring, alert decision
- `agents/supervisor.py` — LangGraph graph, match-loop routing, full pipeline orchestration

The project has been validated end-to-end on live Premier League data.

The phases that built toward this:

1. defined the data structures and interfaces so agent code had something stable to depend on
2. implemented tools and repositories before agent logic so agents were never blocked by missing infrastructure
3. implemented agents bottom-up: stats first, research second, synthesis third, orchestration last

This bottom-up order is the right way to build systems with dependencies.

## 19. How Phase 4 Was Implemented

Phase 4 is the decision layer of the system.

Its job is to combine the statistical baseline with qualitative evidence, decide whether a meaningful edge exists, and either send an alert or withhold it with an explanation.

### The `InterpretedEvidence` split

The most important Phase 4 design decision was separating two distinct concepts that were initially one.

`EvidenceItem` is the stored record. It lives in SQLite. It describes what was found and where.

`InterpretedEvidence` is the synthesis-layer view. It has `winner_direction`, `goals_direction`, and `market_weight`. It is produced by the news agent at evidence extraction time and lives only in workflow state — not in the database.

Why this split matters:

- the synthesis agent only needs to know direction and weight, not the full provenance details
- storing synthesis semantics in the evidence record would mix two layers of meaning
- each layer has a single, clear job

### Market-specific direction fields

The old design used a single `direction` field ("home_positive", "away_positive", "neutral", "uncertainty") for all markets.

That was a bug waiting to happen.

An injury to the away team's striker is `away_positive` for the winner market (home more likely to win) but it might also push toward `under_positive` for the goals market (fewer away goals expected).

The fix was to add:

- `winner_direction` — which winner outcome does this evidence favor?
- `goals_direction` — does this evidence push toward more or fewer goals?

That change made it possible to apply evidence correctly to each market independently.

### Market scope enforcement

The news agent now enforces that evidence flagged as `applies_to_market="winner"` cannot move the goals market.

This is important because the LLM might return `applies_to_market="winner"` but also populate `goals_direction="over_positive"`. Without enforcement, that would move both markets even though the evidence was only meant for one.

The fix is simple: if `market == "winner"`, force `goals_direction = "neutral"`. If `market == "goals"`, force `winner_direction = "neutral"`.

### Market weight

Evidence that explicitly applies to both markets is weighted at 1.0. Single-market evidence is discounted to 0.7.

Why:

- "both" evidence (like a red card or high-profile injury) is directly relevant to every market
- single-market evidence is less universally informative
- discounting prevents single-market observations from having the same influence as broader signals

### The `draw_positive` heuristic

The log-odds adjustment formula operates on individual markets independently — home win, away win, over, under.

There is no "draw log-odds axis."

To increase the draw probability, the system reduces both home and away win log-odds by 0.5 while keeping the raw draw probability fixed. After renormalization, draw's share of the winner market rises.

This is explicitly documented as a heuristic in the code, not a principled probability update. A future improvement would be a direct draw-log-odds axis.

### Immutable forecast construction

The `Forecast` dataclass is frozen. It cannot be mutated after creation.

But in synthesis, there are two moments when the forecast changes:

1. the rationale is updated to include guard failure reasons
2. `alert_sent` is set after the alert attempt

The fix was to use `dataclasses.replace()` twice:

- first to build a corrected-rationale forecast before sending
- second to attach the `alert_sent` result before persisting

This ensures the persisted record and the sent payload are always consistent, and frozen-ness is never bypassed.

---

## 20. How Phase 5 Was Implemented

Phase 5 is the orchestration layer.

Its job is to connect the three agents into a repeatable workflow that processes every fixture in one run.

### The LangGraph graph shape

The graph has four nodes:

1. `stats_market` — fetches all fixtures, odds, and baselines for the competition; sets `pending_match_ids`
2. `setup_next_match` — pops the next match ID from `pending_match_ids`, resets per-match state
3. `news_context` — runs the ReAct loop for the current match
4. `synthesis` — runs synthesis and alert for the current match, appends to `all_forecasts`

The routing logic is simple:

- after `setup_next_match`, if `current_match_id` is set → go to `news_context`
- if `pending_match_ids` is empty → go to `END`
- after `synthesis` → always go back to `setup_next_match`

This creates a match-processing loop. Each iteration handles one fixture completely before moving to the next.

### Why per-match state must be reset

The graph state is shared across all iterations.

If `evidence_items`, `interpreted_evidence`, and `forecast` were carried forward, each new match would start with the previous match's evidence. The synthesis agent would combine evidence from two different fixtures.

`setup_next_match` resets these fields to `[]` and `None` before each iteration.

### Fixing the ReAct message loop

When the OpenAI API is used for tool calling, the message history must follow a strict structure:

1. the assistant message must include the `tool_calls` array
2. the tool response must include a `tool_call_id` matching the call

The original code was reconstructing the assistant message from only the text content, throwing away `tool_calls`. It was also adding a `role: "tool"` message without a `tool_call_id`. The API rejected this with a `BadRequestError`.

The fix required the provider to own message formatting, since the two APIs (OpenAI and Anthropic) need different shapes:

- `format_assistant_turn(response)` — returns the raw response dict for OpenAI; wraps content list as an assistant message for Anthropic
- `format_tool_result(tool_call_id, content)` — returns `role: "tool"` with `tool_call_id` for OpenAI; returns `role: "user"` with `type: "tool_result"` for Anthropic

These two methods were added to the `LLMProvider` protocol and implemented in both adapters.

This is a good example of the Dependency Inversion principle: the `NewsContextAgent` does not know or care which provider is active. It calls the protocol methods and the adapters handle the provider-specific formatting.

### Team name matching between APIs

football-data.org and the-odds-api.com use different team name strings for the same clubs.

For example:
- "Brentford FC" (football-data) vs "Brentford" (odds-api)
- "AFC Bournemouth" (football-data) vs "Bournemouth" (odds-api)
- "Forest" (football-data) vs "Nottingham Forest" (odds-api)

The `TeamNameNormalizer` handles this by canonicalizing both names before comparison. But there was a second bug: `_best_h2h` was using football-data names to look up bookmaker outcome prices, while the outcome keys use odds-API names.

The fix: `fetch_odds` now passes `event["home_team"]` and `event["away_team"]` to `_parse_odds` instead of the football-data names. This ensures outcome lookup always uses the name the API itself provided.

---

Here are the most important concrete decisions made so far.

### Decision 1: Use SQLite plus Chroma, not one storage system for everything

Reason:

- structured and unstructured data need different retrieval strategies

### Decision 2: Use provider adapters

Reason:

- avoid locking the system to one LLM vendor

### Decision 3: Use a statistical baseline, not LLM-only forecasting

Reason:

- better evaluation and cleaner reasoning about edge

### Decision 4: Keep market odds out of the baseline model

Reason:

- avoid circular edge calculation

### Decision 5: Use MCP as the tool boundary

Reason:

- keep tools reusable and decoupled from agent code

### Decision 6: Seed source reliability explicitly

Reason:

- easier to explain and debug than fake learned trust

### Decision 7: Add prompt-injection defense early

Reason:

- web search results are untrusted from the start

### Decision 8: Normalize time handling in guardrails

Reason:

- prevent subtle runtime bugs in recency and spam checks

## 21. What to Focus on for the Remaining Work

The core pipeline is done. Three areas remain before the project is fully complete.

### Evaluation pipeline

After matches resolve, the system needs to:

- store final outcomes in the `matches` table
- compare baseline and adjusted forecast probabilities to actual outcomes
- compute Brier score across the resolved match set
- compare baseline-only vs adjusted-only Brier score to measure whether the research agent improved calibration

This is the only way to know if the system's adjustments were useful, not just plausible-sounding.

### Notebook export

The professor requires a single executed Jupyter notebook.

`make_notebook.py` assembles `demo.ipynb` from the Python module source.

The notebook should demonstrate the full end-to-end flow in a way that is readable without running the code.

### Output quality review

The pipeline is working, but working correctly is not the same as working well.

A manual review of 5–10 matches is worth doing:

- are the evidence summaries accurate?
- are the direction labels (`home_positive`, `over_positive`, etc.) sensible?
- are the search queries the model generates specific to the match, or generic?
- are the rationales coherent and match-specific?

This review does not require code changes. It just requires reading the output carefully and deciding whether to tune prompts.

## 21. A Good Mental Model for the Whole System

If you want one simple way to think about the architecture, use this:

- `models/` define the language of the system
- `protocols/` define the contracts
- `tools/` gather outside information
- `memory/` stores what the system knows
- `analytics/` computes the first estimate
- `agents/` reason over the information
- `guardrails/` prevent low-quality behavior
- `main.py` assembles the pieces

That is the shape of the project.

## 22. Suggested Way to Use This Document

As you continue implementing, update this file in small sections:

1. what was added
2. why it matters
3. what concept it teaches
4. what tradeoff was chosen

That way, by the end of the project, this file becomes both:

- a learning journal
- a technical explanation of the system

## 23. Current Summary

The project is fully operational.

The end-to-end pipeline runs on live Premier League data:

1. `StatsMarketAgent` fetches 10 upcoming fixtures, retrieves market odds for all 10 via `OddsFetcher`, and computes a form/goals baseline for each match
2. `NewsContextAgent` runs a ReAct research loop per match, seeding from the vector store and then searching Tavily for recent news
3. `SynthesisAlertAgent` applies log-odds adjustment from the interpreted evidence, calculates the best-edge market, scores confidence, generates a rationale via the LLM, and passes the result to `AlertGuard`
4. `SupervisorAgent` loops through all fixtures in a single LangGraph run and collects all forecasts

The output is a list of `Forecast` objects — one per fixture — stored in SQLite with full provenance: baseline probabilities, adjusted probabilities, evidence items, rationale, edge market, edge value, confidence score, and whether an alert was sent.

What remains:

- evaluation after matches resolve (Brier score)
- notebook export for submission
- optional output quality review across representative matches
