"""Dixon-Coles football forecasting model with maximum-likelihood parameter estimation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from soccer_forecast_agent.models.match import Match, MatchContext, BaselineForecast


class DCParams:
    """Fitted Dixon-Coles model parameters."""

    __slots__ = ("attack", "defense", "home_adv", "rho", "teams", "n_matches")

    def __init__(
        self,
        attack: dict[str, float],
        defense: dict[str, float],
        home_adv: float,
        rho: float,
        teams: list[str],
        n_matches: int,
    ) -> None:
        self.attack = attack
        self.defense = defense
        self.home_adv = home_adv
        self.rho = rho
        self.teams = teams
        self.n_matches = n_matches


class DixonColesStrategy:
    """Dixon-Coles Poisson model (1997) for football match outcome probabilities.

    Each team has attack (α) and defence (δ) strengths estimated jointly via
    time-weighted maximum-likelihood on historical scorelines.  A home
    advantage multiplier (γ) and a low-score correction parameter (ρ) are
    estimated alongside the team parameters.

    Expected goals:
        λ (home) = α_home · δ_away · γ
        μ (away) = α_away · δ_home

    Parameterised in log-space so all strengths stay positive.  The first
    team's attack is fixed at 0 for identifiability.

    Usage:
        strategy = DixonColesStrategy()
        strategy.fit(resolved_matches)
        forecast = strategy.compute(match_context)
    """

    MAX_GOALS: int = 10

    def __init__(self, xi: float = 0.002) -> None:
        """xi: exponential time-decay per day (default 0.002 ≈ half-weight after ~1 year)."""
        self._xi = xi
        self._params: DCParams | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once fit() has completed successfully."""
        return self._params is not None

    @property
    def params(self) -> DCParams | None:
        """Read-only access to fitted parameters for inspection."""
        return self._params

    def fit(self, matches: list[Match]) -> "DixonColesStrategy":
        """Estimate all parameters from resolved match records. Returns self for chaining."""
        scored = [
            m for m in matches
            if m.final_score and "-" in m.final_score and m.status == "resolved"
        ]
        if len(scored) < 20:
            raise ValueError(
                f"Too few resolved matches for stable fitting "
                f"({len(scored)} provided, need ≥20)."
            )

        teams = sorted({m.home_team for m in scored} | {m.away_team for m in scored})
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        ref_date = max(m.kickoff_time for m in scored)
        weights = np.array(
            [np.exp(-self._xi * max((ref_date - m.kickoff_time).days, 0)) for m in scored],
            dtype=float,
        )

        home_ids = np.array([idx[m.home_team] for m in scored], dtype=int)
        away_ids = np.array([idx[m.away_team] for m in scored], dtype=int)
        home_goals = np.array([int(m.final_score.split("-")[0]) for m in scored], dtype=int)
        away_goals = np.array([int(m.final_score.split("-")[1]) for m in scored], dtype=int)

        # Parameter layout: [attack_1..attack_{n-1}, defense_0..defense_{n-1}, home_adv, rho]
        # attack_0 = 0 is fixed (identifiability constraint)
        x0 = np.zeros(2 * n + 1, dtype=float)
        x0[2 * n - 1] = 0.25  # warm start: modest log home advantage

        bounds = (
            [(None, None)] * (2 * n - 1)  # attack + defence: unconstrained
            + [(None, None)]               # home_adv: unconstrained
            + [(-0.9, 0.9)]               # rho: bounded to keep tau positive
        )

        result = minimize(
            fun=self._neg_log_likelihood,
            x0=x0,
            args=(n, home_ids, away_ids, home_goals, away_goals, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12},
        )

        attack_arr = np.concatenate([[0.0], result.x[: n - 1]])
        defense_arr = result.x[n - 1 : 2 * n - 1]

        self._params = DCParams(
            attack={t: float(attack_arr[idx[t]]) for t in teams},
            defense={t: float(defense_arr[idx[t]]) for t in teams},
            home_adv=float(result.x[2 * n - 1]),
            rho=float(result.x[2 * n]),
            teams=teams,
            n_matches=len(scored),
        )
        return self

    def compute(self, context: MatchContext) -> BaselineForecast:
        """Return match outcome probabilities from fitted DC parameters.

        Teams absent from the training set receive league-average parameters
        (attack = defence = 0 in log space, i.e. exactly the geometric mean team).
        """
        if self._params is None:
            raise RuntimeError("Must call fit() before compute().")

        p = self._params
        home, away = context.match.home_team, context.match.away_team

        lam = np.exp(p.attack.get(home, 0.0) + p.defense.get(away, 0.0) + p.home_adv)
        mu = np.exp(p.attack.get(away, 0.0) + p.defense.get(home, 0.0))

        prob = self._score_matrix(lam, mu, p.rho)

        # home_win: P(home goals > away goals) = lower triangle (row > col)
        home_win = float(np.tril(prob, -1).sum())
        draw = float(np.diag(prob).sum())
        away_win = float(np.triu(prob, 1).sum())

        # Normalise to absorb truncation error from finite MAX_GOALS
        total = home_win + draw + away_win or 1.0
        home_win /= total
        draw /= total
        away_win /= total

        g = self.MAX_GOALS + 1
        xs, ys = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        over_2_5 = float(prob[(xs + ys) > 2].sum())
        over_2_5 = min(max(over_2_5, 0.05), 0.95)

        return BaselineForecast(
            home_win=round(home_win, 4),
            draw=round(draw, 4),
            away_win=round(away_win, 4),
            over_2_5=round(over_2_5, 4),
            under_2_5=round(1.0 - over_2_5, 4),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _neg_log_likelihood(
        self,
        x: np.ndarray,
        n: int,
        home_ids: np.ndarray,
        away_ids: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Negative weighted log-likelihood used as the objective for scipy.minimize."""
        attack = np.concatenate([[0.0], x[: n - 1]])
        defense = x[n - 1 : 2 * n - 1]
        h = x[2 * n - 1]
        rho = x[2 * n]

        lam = np.exp(attack[home_ids] + defense[away_ids] + h)
        mu = np.exp(attack[away_ids] + defense[home_ids])

        hg = home_goals.astype(float)
        ag = away_goals.astype(float)
        log_ph = -lam + hg * np.log(lam + 1e-15) - gammaln(hg + 1)
        log_pa = -mu + ag * np.log(mu + 1e-15) - gammaln(ag + 1)

        tau = np.ones(len(lam))
        m00 = (home_goals == 0) & (away_goals == 0)
        m10 = (home_goals == 1) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m11 = (home_goals == 1) & (away_goals == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m11] = 1.0 - rho

        if np.any(tau <= 1e-10):
            return 1e12

        return -float(np.sum(weights * (np.log(tau) + log_ph + log_pa)))

    def _score_matrix(self, lam: float, mu: float, rho: float) -> np.ndarray:
        """Return normalised P(home=x, away=y) matrix for x, y in 0..MAX_GOALS."""
        g = self.MAX_GOALS + 1
        xs = np.arange(g, dtype=float)
        log_px = -lam + xs * np.log(max(lam, 1e-15)) - gammaln(xs + 1)
        log_py = -mu + xs * np.log(max(mu, 1e-15)) - gammaln(xs + 1)
        P = np.outer(np.exp(log_px), np.exp(log_py))

        P[0, 0] *= max(1.0 - lam * mu * rho, 1e-10)
        P[1, 0] *= max(1.0 + mu * rho, 1e-10)
        P[0, 1] *= max(1.0 + lam * rho, 1e-10)
        P[1, 1] *= max(1.0 - rho, 1e-10)

        total = P.sum()
        return P / total if total > 0 else P
