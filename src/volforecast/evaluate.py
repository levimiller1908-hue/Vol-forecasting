"""Forecast scoring: loss functions, Diebold-Mariano tests and VaR backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.univariate import StudentsT
from scipy import stats

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
# The realised squared return is a noisy but unbiased proxy for the true
# variance. Patton (2011) shows MSE and QLIKE still rank forecasts correctly
# when scored against such a proxy; QLIKE is less dominated by a few extreme days.


def qlike(r2: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Normalised QLIKE: r2/h - log(r2/h) - 1. Zero for a perfect forecast.

    Undefined when r2 == 0, so callers drop zero-return days first.
    """
    x = r2 / h
    return x - np.log(x) - 1


def mse(r2: np.ndarray, h: np.ndarray) -> np.ndarray:
    return (r2 - h) ** 2


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def newey_west_var(d: np.ndarray, lags: int | None = None) -> float:
    """Long-run variance of a series using Bartlett-weighted autocovariances."""
    d = d - d.mean()
    n = len(d)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = d @ d / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / n
    return v


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """DM test of equal average loss. Negative statistic means A beats B.

    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    stat = d.mean() / np.sqrt(newey_west_var(d) / len(d))
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


# ---------------------------------------------------------------------------
# Value at Risk backtest
# ---------------------------------------------------------------------------


def var_threshold(var: np.ndarray, nu: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """One-day VaR as a positive loss in percent.

    Uses the normal quantile where nu is NaN (Historical, EWMA) and the
    unit-variance Student-t quantile where nu is given (GARCH models).
    """
    sigma = np.sqrt(var)
    q = np.full(len(var), stats.norm.ppf(alpha))
    has_nu = ~np.isnan(nu)
    if has_nu.any():
        t = StudentsT()
        q[has_nu] = [t.ppf(alpha, np.array([v])) for v in nu[has_nu]]
    return -q * sigma


def kupiec(hits: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value)."""
    n, x = len(hits), int(hits.sum())
    pi = x / n
    ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll_alt = (n - x) * np.log(1 - pi) + (x * np.log(pi) if x > 0 else 0.0)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence test: do VaR breaches cluster?

    Returns (LR statistic, p-value).
    """
    h = hits.astype(int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))

    def ll(a: int, b: int, p: float) -> float:
        return (a * np.log(1 - p) if a else 0.0) + (b * np.log(p) if b else 0.0)

    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr = -2 * (ll(n00 + n10, n01 + n11, p) - (ll(n00, n01, p01) + ll(n10, n11, p11)))
    return float(lr), float(stats.chi2.sf(lr, 1))


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def add_losses(forecasts: pd.DataFrame) -> pd.DataFrame:
    df = forecasts.copy()
    r2 = df["ret"].to_numpy() ** 2
    h = df["var"].to_numpy()
    df["mse"] = mse(r2, h)
    with np.errstate(divide="ignore"):
        df["qlike"] = np.where(r2 > 0, qlike(np.where(r2 > 0, r2, 1.0), h), np.nan)
    df["var99"] = var_threshold(h, df["nu"].to_numpy(), 0.01)
    df["hit"] = df["ret"] < -df["var99"]
    return df


def loss_table(df: pd.DataFrame, benchmark: str = "EWMA") -> pd.DataFrame:
    """Mean QLIKE and MSE by ticker and model, plus % change versus the benchmark."""
    g = df.groupby(["ticker", "model"])[["qlike", "mse"]].mean()
    bench = g.xs(benchmark, level="model")
    rel = (g / bench.reindex(g.index, level="ticker") - 1) * 100
    out = g.join(rel.add_suffix("_vs_" + benchmark.lower() + "_pct"))
    return out


def dm_table(df: pd.DataFrame, pairs: list[tuple[str, str]], loss: str = "qlike") -> pd.DataFrame:
    rows = []
    for ticker, sub in df.groupby("ticker"):
        wide = sub.pivot(index="date", columns="model", values=loss).dropna()
        for a, b in pairs:
            stat, p = diebold_mariano(wide[a].to_numpy(), wide[b].to_numpy())
            rows.append({"ticker": ticker, "model_a": a, "model_b": b, "loss": loss,
                         "dm_stat": stat, "p_value": p, "n": len(wide)})
    return pd.DataFrame(rows)


def var_table(df: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    rows = []
    for (ticker, model), sub in df.groupby(["ticker", "model"]):
        hits = sub.sort_values("date")["hit"].to_numpy()
        lr_uc, p_uc = kupiec(hits, alpha)
        lr_ind, p_ind = christoffersen(hits)
        lr_cc = lr_uc + lr_ind
        rows.append({
            "ticker": ticker, "model": model, "days": len(hits), "breaches": int(hits.sum()),
            "breach_rate_pct": 100 * hits.mean(), "expected": alpha * len(hits),
            "kupiec_p": p_uc, "christoffersen_p": p_ind,
            "cond_coverage_p": float(stats.chi2.sf(lr_cc, 2)),
        })
    return pd.DataFrame(rows)


def by_year(df: pd.DataFrame, benchmark: str = "EWMA") -> pd.DataFrame:
    """Mean QLIKE by calendar year, as % change versus the benchmark (all tickers pooled)."""
    d = df.assign(year=df["date"].dt.year)
    g = d.groupby(["year", "model"])["qlike"].mean().unstack("model")
    return (g.div(g[benchmark], axis=0) - 1) * 100
