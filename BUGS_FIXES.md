# Bugs And Fixes Backlog

## High Priority

- Fix team-name normalization everywhere historical and live data meet.
  Status: fixed in `SQLiteRepository` read/query path and match persistence.
  Follow-up: backfill canonical team names in existing SQLite rows if we want the database itself to be clean, not just normalized on read.

- Prevent The Odds API quota burn from repeated board fetches.
  Status: fixed with in-process caching in `OddsFetcher`.
  Follow-up: add persistent snapshot caching for debug/demo runs and a graceful fallback when quota is exhausted.

- Add reproducible debug mode for multi-run comparisons.
  Problem: live Tavily results and LLM synthesis vary between runs, which makes debugging difficult.
  Suggested fix:
  - `REPRO_MODE=1`
  - cache `search_news` responses per match/query
  - persist the exact evidence bundle used for synthesis
  - optionally replay synthesis from stored evidence only

## Modeling / Forecast Quality

- Tighten `SynthesisAlertAgent` so the LLM cannot flip the recommended market too freely when evidence is mixed.
  Suggested fix:
  - require primary market plus runner-up market
  - require explicit reason when moving away from the baseline favorite
  - add a code-side rule that weak evidence cannot justify a market flip without a higher edge threshold

- Separate `llm_lean_market` from `value_market` and `alert_market`.
  Problem: the LLM may lean toward a side even when the market edge is negative, which can sound like a recommendation.
  Suggested fix:
  - keep all three fields distinct
  - only surface `alert_market` as the actual bet recommendation

- Add evaluation logging for every forecast stage.
  Suggested trace fields:
  - canonical teams and match id
  - baseline forecast
  - seeded RAG chunks
  - live search results
  - extracted evidence
  - synthesis decision JSON
  - adjusted probabilities
  - market-implied probabilities
  - final guard result

## Product / UX

- Improve human-facing wording when no value edge exists.
  Suggested fix:
  - use `LLM lean` when the qualitative model prefers a side
  - use `Value signal: none` when edge is non-positive
  - reserve `Recommended bet` for positive-EV alerts only

- Add a plain-language `Synthesis LLM view` section to the alert output.
  Goal: make the second LLM pass clearly visible to the user instead of blending it into the quantitative output.

## Infrastructure / Demo

- Add persistent odds snapshot caching for offline/demo use.
  Suggested fix:
  - save the latest successful EPL odds board to disk
  - if live fetch fails, optionally reuse the snapshot
  - enable with something like `USE_CACHED_ODDS=1`

- Add notebook-safe demo mode.
  Problem: live API quota and internet issues can break the course demo.
  Suggested fix:
  - make live cells optional
  - provide cached outputs and clear fallback messaging
