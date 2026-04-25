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

As of the current state of the project:

- Phase 0 is complete
- Phase 1 is complete
- parts of Phase 2 and Phase 4 are started

That means the project already has:

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
- the basic baseline analytics functions

What is not implemented yet are the agent workflows themselves, especially:

- `StatsMarketAgent.run()`
- the ReAct loop in `NewsContextAgent`
- `SynthesisAlertAgent.run()`
- `SupervisorAgent.build_graph()` and `run()`

That is normal. The foundation is supposed to come first.

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

## 18. Why Some Agent Files Are Still Incomplete

You will notice there are still `NotImplementedError` markers in:

- `agents/stats_market.py`
- `agents/news_context.py`
- `agents/supervisor.py`
- `agents/synthesis_alert.py`

That does not mean the project is broken.

It means the implementation is following the roadmap:

- first the infrastructure
- then the orchestration and decision logic

This is actually a healthy way to build the system.

It is easier to implement agent behavior once:

- tools already work
- repositories already work
- models already exist
- configuration is stable

## 19. What Decisions Were Made During Implementation

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

## 20. What You Should Learn Next While Building

The next implementation phases are a good chance to focus on these concepts.

### Next concept: stateful orchestration

When you implement `SupervisorAgent`, pay attention to:

- shared state
- node responsibilities
- routing
- stopping rules

### Next concept: ReAct loops

When you implement `NewsContextAgent`, focus on:

- how the model decides what information is missing
- how tool results are fed back in
- when the loop should stop
- how evidence is structured

### Next concept: probability adjustment

When you implement `SynthesisAlertAgent.run()`, focus on:

- log-odds adjustment
- renormalization
- implied probability
- edge calculation
- guardrail enforcement

### Next concept: evaluation

When you later add Brier score and resolved-match tracking, focus on:

- calibration
- outcome logging
- baseline versus adjusted forecast comparison

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

Right now, the project has a strong foundation.

The most valuable thing completed so far is not a flashy demo. It is the structure:

- clear data models
- clean abstractions
- persistence layers
- tool layer
- vector memory
- guardrails
- dependency wiring

That structure is what will make the later agent implementation easier, cleaner, and more defensible in your final submission.
