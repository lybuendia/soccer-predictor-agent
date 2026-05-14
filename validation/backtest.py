"""
Backtesting evaluation — compares Simple, Enhanced, and Dixon-Coles baselines
on resolved PL matches using a chronological train/test split.

Simple and Enhanced use rolling point-in-time features (last 5 games).
Dixon-Coles is fitted on the training half then evaluated on the test half.

Usage:
    source .venv/bin/activate
    python validation/backtest.py [--train-pct 0.70]
"""

import argparse
import sqlite3
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository, init_db
from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.analytics.baseline import SimpleBaselineStrategy, EnhancedBaselineStrategy
from soccer_forecast_agent.analytics.dixon_coles import DixonColesStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor
from soccer_forecast_agent.analytics.evaluation import (
    outcome_from_score,
    brier_score_winner,
    brier_score_goals,
    aggregate,
)
from soccer_forecast_agent.models.match import MarketOdds, Match


MIN_PRIOR_MATCHES = 3  # skip a test match if either team has fewer than this


def _dummy_odds(match_id: str) -> MarketOdds:
    """Return placeholder odds — baseline strategy does not use odds, but FeatureExtractor requires them."""
    return MarketOdds(
        odds_id="backtest",
        match_id=match_id,
        timestamp=datetime.now(timezone.utc),
        home_win=2.5,
        draw=3.2,
        away_win=2.8,
        over_2_5=1.9,
        under_2_5=1.9,
    )


def _result_for_team(team: str, match: Match) -> str:
    """Return W, D, or L for a team from a resolved match."""
    if match.final_score is None:
        raise ValueError(f"Match {match.match_id} has no final score")
    home_g, away_g = map(int, match.final_score.split("-", 1))
    goals_for = home_g if match.home_team == team else away_g
    goals_against = away_g if match.home_team == team else home_g
    if goals_for > goals_against:
        return "W"
    if goals_for < goals_against:
        return "L"
    return "D"


def _goals_for_team(team: str, match: Match) -> tuple[float, float]:
    """Return (scored, conceded) for a team from a resolved match."""
    if match.final_score is None:
        raise ValueError(f"Match {match.match_id} has no final score")
    home_g, away_g = map(int, match.final_score.split("-", 1))
    if match.home_team == team:
        return float(home_g), float(away_g)
    return float(away_g), float(home_g)


def _build_context(
    match: Match,
    repo: SQLiteRepository,
    competition: str,
    extractor: FeatureExtractor,
) -> tuple | None:
    """Extract point-in-time features for a match. Returns (context, outcome) or None if skipped."""
    home_history = repo.get_recent_finished(
        match.home_team, competition, limit=5, before=match.kickoff_time
    )
    away_history = repo.get_recent_finished(
        match.away_team, competition, limit=5, before=match.kickoff_time
    )
    if len(home_history) < MIN_PRIOR_MATCHES or len(away_history) < MIN_PRIOR_MATCHES:
        return None

    context = extractor.extract(
        match=match,
        odds=_dummy_odds(match.match_id),
        home_results=[_result_for_team(match.home_team, m) for m in home_history],
        away_results=[_result_for_team(match.away_team, m) for m in away_history],
        home_goals_scored=[_goals_for_team(match.home_team, m)[0] for m in home_history],
        home_goals_conceded=[_goals_for_team(match.home_team, m)[1] for m in home_history],
        away_goals_scored=[_goals_for_team(match.away_team, m)[0] for m in away_history],
        away_goals_conceded=[_goals_for_team(match.away_team, m)[1] for m in away_history],
    )
    return context, outcome_from_score(match.final_score)


def run_backtest(repo: SQLiteRepository, competition: str = "PL", train_pct: float = 0.70) -> None:
    """Compare Simple, Enhanced, and Dixon-Coles baselines on a chronological train/test split.

    Simple and Enhanced use rolling point-in-time history throughout.
    Dixon-Coles is fitted on the first train_pct of matches, then evaluated
    on the test portion — giving a genuine out-of-sample comparison.
    """
    simple = SimpleBaselineStrategy()
    enhanced = EnhancedBaselineStrategy()
    extractor = FeatureExtractor()

    rows = repo._conn.execute(
        "SELECT * FROM matches WHERE status = 'resolved' AND final_score IS NOT NULL "
        "AND competition = ? ORDER BY kickoff_time ASC",
        (competition,),
    ).fetchall()
    all_matches = [repo._row_to_match(r) for r in rows]

    # Train/test split
    cutoff_idx = max(int(len(all_matches) * train_pct), 20)
    train_matches = all_matches[:cutoff_idx]
    test_matches = all_matches[cutoff_idx:]
    cutoff_date = test_matches[0].kickoff_time.strftime("%Y-%m-%d") if test_matches else "N/A"

    print(f"\n  Fitting Dixon-Coles on {len(train_matches)} matches "
          f"(before {cutoff_date}), testing on {len(test_matches)} ...")
    dc = DixonColesStrategy().fit(train_matches)
    print(f"  DC fitted: {len(dc.params.teams)} teams, "  # type: ignore[union-attr]
          f"home_adv={dc.params.home_adv:.3f}, rho={dc.params.rho:.3f}\n")  # type: ignore[union-attr]

    simple_winner: list[float] = []
    simple_goals: list[float] = []
    enh_winner: list[float] = []
    enh_goals: list[float] = []
    dc_winner: list[float] = []
    dc_goals: list[float] = []
    skipped = 0
    results_table: list[tuple] = []

    for match in test_matches:
        built = _build_context(match, repo, competition, extractor)
        if built is None:
            skipped += 1
            continue
        context, outcome = built

        sf = simple.compute(context)
        ef = enhanced.compute(context)
        df = dc.compute(context)

        sw = brier_score_winner(sf, outcome)
        sg = brier_score_goals(sf, outcome)
        ew = brier_score_winner(ef, outcome)
        eg = brier_score_goals(ef, outcome)
        dw = brier_score_winner(df, outcome)
        dg = brier_score_goals(df, outcome)

        simple_winner.append(sw)
        simple_goals.append(sg)
        enh_winner.append(ew)
        enh_goals.append(eg)
        dc_winner.append(dw)
        dc_goals.append(dg)

        actual = "H" if outcome.home_win else ("D" if outcome.draw else "A")
        results_table.append((
            match.kickoff_time.strftime("%Y-%m-%d"),
            match.home_team[:18],
            match.away_team[:18],
            match.final_score,
            actual,
            f"{sf.home_win:.2f}", f"{sf.draw:.2f}", f"{sf.away_win:.2f}", f"{sf.over_2_5:.2f}",
            f"{sw:.4f}", f"{sg:.4f}",
            f"{ef.home_win:.2f}", f"{ef.draw:.2f}", f"{ef.away_win:.2f}", f"{ef.over_2_5:.2f}",
            f"{ew:.4f}", f"{eg:.4f}",
            f"{df.home_win:.2f}", f"{df.draw:.2f}", f"{df.away_win:.2f}", f"{df.over_2_5:.2f}",
            f"{dw:.4f}", f"{dg:.4f}",
        ))

    simple_scores = aggregate(simple_winner, simple_goals)
    enh_scores = aggregate(enh_winner, enh_goals)
    dc_scores = aggregate(dc_winner, dc_goals)
    _print_report(results_table, simple_scores, enh_scores, dc_scores, skipped, cutoff_date, train_pct)


def _print_report(
    rows: list[tuple],
    simple_scores,
    enh_scores,
    dc_scores,
    skipped: int,
    cutoff_date: str,
    train_pct: float,
) -> None:
    """Print the full backtesting report with three-way strategy comparison."""
    hdr = (
        f"{'Date':<12}{'Home':<20}{'Away':<20}{'Score':<7}{'Res':<5}"
        f"{'S:P(H)':<7}{'S:P(O)':<7}{'S:BS_W':<7}{'S:BS_G':<8}"
        f"{'E:P(H)':<7}{'E:P(O)':<7}{'E:BS_W':<7}{'E:BS_G':<8}"
        f"{'DC:P(H)':<8}{'DC:P(O)':<8}{'DC:BS_W':<8}{'DC:BS_G':<7}"
    )
    div = "-" * len(hdr)

    print()
    print("=" * len(hdr))
    print(f"  BACKTEST — test set from {cutoff_date} ({100*(1-train_pct):.0f}% of data)")
    print(f"  S = Simple (rolling 5)  |  E = Enhanced (rolling 5)  |  DC = Dixon-Coles (MLE on {train_pct:.0%} train)")
    print("=" * len(hdr))
    print()
    print(hdr)
    print(div)

    s_correct = e_correct = d_correct = 0
    for row in rows:
        (date, home, away, score, actual,
         sph, spd, spa, spo, sbsw, sbsg,
         eph, epd, epa, epo, ebsw, ebsg,
         dph, dpd, dpa, dpo, dbsw, dbsg) = row

        def best(h, d, a):
            m = {"H": float(h), "D": float(d), "A": float(a)}
            return max(m, key=m.get)

        s_pred = best(sph, spd, spa)
        e_pred = best(eph, epd, epa)
        d_pred = best(dph, dpd, dpa)
        sm = "✓" if s_pred == actual else " "
        em = "✓" if e_pred == actual else " "
        dm = "✓" if d_pred == actual else " "
        if s_pred == actual: s_correct += 1
        if e_pred == actual: e_correct += 1
        if d_pred == actual: d_correct += 1

        print(
            f"{date:<12}{home:<20}{away:<20}{score:<7}{actual:<5}"
            f"{sph:<7}{spo:<7}{sbsw:<7}{sbsg:<7}{sm} "
            f"{eph:<7}{epo:<7}{ebsw:<7}{ebsg:<7}{em} "
            f"{dph:<8}{dpo:<8}{dbsw:<8}{dbsg:<6}{dm}"
        )

    print(div)
    n = simple_scores.n_matches

    def delta(new, old):
        d = old - new
        return f"{'↓' if d > 0 else '↑'}{abs(d):.4f}"

    print()
    print(f"  Test matches evaluated : {n}  |  Skipped : {skipped}  (insufficient history)")
    print()
    print(f"  {'Metric':<34} {'Simple':>8}  {'Enhanced':>8}  {'Vs S':>8}  {'DC':>8}  {'Vs S':>8}")
    print(f"  {'-'*34} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(
        f"  {'Winner Brier  (random ≈ 0.222)':<34} "
        f"{simple_scores.winner_brier:>8.4f}  "
        f"{enh_scores.winner_brier:>8.4f}  "
        f"{delta(enh_scores.winner_brier, simple_scores.winner_brier):>8}  "
        f"{dc_scores.winner_brier:>8.4f}  "
        f"{delta(dc_scores.winner_brier, simple_scores.winner_brier):>8}"
    )
    print(
        f"  {'Goals  Brier  (random ≈ 0.250)':<34} "
        f"{simple_scores.goals_brier:>8.4f}  "
        f"{enh_scores.goals_brier:>8.4f}  "
        f"{delta(enh_scores.goals_brier, simple_scores.goals_brier):>8}  "
        f"{dc_scores.goals_brier:>8.4f}  "
        f"{delta(dc_scores.goals_brier, simple_scores.goals_brier):>8}"
    )
    s_acc = s_correct / n * 100 if n else 0
    e_acc = e_correct / n * 100 if n else 0
    d_acc = d_correct / n * 100 if n else 0
    print(f"  {'Accuracy (most-likely outcome)':<34} {s_acc:>7.1f}%  {e_acc:>7.1f}%  {'':>8}  {d_acc:>7.1f}%")
    print()
    print("=" * len(hdr))
    print()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Backtest: Simple vs Enhanced vs Dixon-Coles")
    parser.add_argument("--train-pct", type=float, default=0.70,
                        help="Fraction of matches used to train Dixon-Coles (default 0.70)")
    args = parser.parse_args()

    config = Config.from_env()
    conn = sqlite3.connect(config.db_path)
    init_db(conn)
    seed_source_reliability(conn)
    repo = SQLiteRepository(conn)
    run_backtest(repo, train_pct=args.train_pct)
    conn.close()
