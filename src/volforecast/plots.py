"""Figures for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colour per model, used in every figure
COLORS = {
    "GJR-GARCH": "#2a78d6",
    "EWMA": "#eb6834",
    "GARCH": "#1baf7a",
    "Historical 21d": "#eda100",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TICKER_ORDER = ["SPY", "XLK", "XLF", "XLE", "XLU", "XLV"]


def _style(ax, title: str, ylabel: str) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, **kw)
    return leg


def covid_forecasts(returns: pd.DataFrame, scored: pd.DataFrame, path: Path) -> None:
    """SPY forecast volatility through the 2020 sell-off against realised volatility."""
    start, end = "2020-02-01", "2020-06-30"
    sub = scored[(scored.ticker == "SPY") & scored.date.between(start, end)]
    wide = sub.pivot(index="date", columns="model", values="var")
    ann = np.sqrt(wide * 252)

    r = returns["SPY"]
    realised = np.sqrt((r ** 2).rolling(5, center=True).mean() * 252).loc[start:end]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ax.plot(realised.index, realised, color=MUTED, linewidth=1.5, linestyle=(0, (3, 2)),
            label="Realised (5-day centred, hindsight)")
    for model in ["EWMA", "GJR-GARCH"]:
        ax.plot(ann.index, ann[model], color=COLORS[model], linewidth=2, label=f"{model} forecast")
    _style(ax, "SPY: one-day-ahead volatility forecasts through the 2020 sell-off",
           "Annualised volatility (%)")
    ax.set_ylim(bottom=0)
    _legend(ax, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _grouped_bars(ax, table: pd.DataFrame, models: list[str], label_fmt: str) -> None:
    n = len(models)
    width = 0.8 / n
    x = np.arange(len(table.index))
    for i, m in enumerate(models):
        vals = table[m].to_numpy()
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.88, color=COLORS[m], label=m, zorder=2)
        for xi, v in zip(pos, vals):
            va = "bottom" if v >= 0 else "top"
            ax.text(xi, v + (0.08 if v >= 0 else -0.08), label_fmt.format(v), ha="center", va=va,
                    fontsize=7, color=INK_2)
    ax.set_xticks(x, table.index)
    ax.tick_params(axis="x", colors=INK_2, labelsize=10)


def loss_vs_ewma(losses: pd.DataFrame, path: Path) -> None:
    """QLIKE loss relative to EWMA, by ETF and model."""
    t = losses["qlike_vs_ewma_pct"].unstack("model").reindex(TICKER_ORDER)
    models = ["Historical 21d", "GARCH", "GJR-GARCH"]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    _grouped_bars(ax, t, models, "{:+.1f}")
    ax.axhline(0, color=INK_2, linewidth=1, zorder=3)
    ax.text(len(t) - 0.45, 0.15, "EWMA = 0", ha="right", va="bottom", fontsize=8.5, color=INK_2)
    _style(ax, "Forecast loss versus EWMA (QLIKE, below zero = more accurate)",
           "% difference in mean QLIKE")
    lo, hi = t[models].min().min(), t[models].max().max()
    ax.set_ylim(lo - 1.2, hi + 1.2)
    _legend(ax, loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def var_breaches(var: pd.DataFrame, path: Path) -> None:
    """99% VaR breach rate by ETF and model against the 1% target."""
    t = var.pivot(index="ticker", columns="model", values="breach_rate_pct").reindex(TICKER_ORDER)
    models = ["Historical 21d", "EWMA", "GARCH", "GJR-GARCH"]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    _grouped_bars(ax, t, models, "{:.1f}")
    ax.axhline(1.0, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3, label="1% target")
    _style(ax, "99% one-day VaR: share of days losses exceeded the VaR estimate",
           "Breach rate (%)")
    ax.set_ylim(0, t[models].max().max() + 0.5)
    _legend(ax, loc="upper right", ncol=5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def yearly(yearly_table: pd.DataFrame, path: Path) -> None:
    """GJR-GARCH loss versus EWMA by calendar year, all ETFs pooled."""
    s = yearly_table["GJR-GARCH"]
    labels = [str(y) for y in s.index]
    labels[-1] += " YTD"
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    ax.bar(labels, s.to_numpy(), width=0.6, color=COLORS["GJR-GARCH"], zorder=2)
    for x, v in zip(labels, s.to_numpy()):
        ax.text(x, v + (0.15 if v >= 0 else -0.15), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=1, zorder=3)
    _style(ax, "GJR-GARCH loss versus EWMA by year (all six ETFs pooled)",
           "% difference in mean QLIKE")
    ax.tick_params(axis="x", colors=INK_2, labelsize=9.5)
    ax.set_ylim(s.min() - 1.2, max(s.max(), 0) + 1.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def make_all(returns, scored, losses, var, yearly_table, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    covid_forecasts(returns, scored, out_dir / "spy_2020_forecasts.png")
    loss_vs_ewma(losses, out_dir / "qlike_vs_ewma.png")
    var_breaches(var, out_dir / "var_breach_rates.png")
    yearly(yearly_table, out_dir / "qlike_by_year.png")
