"""Run the full study: download data, backtest every model, score, save results.

Usage:
    python run.py              # uses cached prices if present
    python run.py --refresh    # re-download prices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from volforecast import backtest, evaluate  # noqa: E402
from volforecast.data import load_prices, log_returns  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "prices.csv"
RESULTS = ROOT / "results"
END = "2026-10-14"  # fixed so results are reproducible; yfinance treats end as exclusive

PAIRS = [("GARCH", "EWMA"), ("GJR-GARCH", "EWMA"), ("GJR-GARCH", "GARCH"), ("EWMA", "Historical 21d")]


def main(refresh: bool = False, make_plots: bool = True) -> None:
    RESULTS.mkdir(exist_ok=True)
    returns = log_returns(load_prices(end=END, cache=DATA, refresh=refresh))

    forecasts, params = backtest.run(returns)
    forecasts.to_csv(RESULTS / "forecasts.csv", index=False, float_format="%.6f")
    params.to_csv(RESULTS / "garch_params.csv", index=False, float_format="%.6f")

    scored = evaluate.add_losses(forecasts)
    losses = evaluate.loss_table(scored)
    dm = evaluate.dm_table(scored, PAIRS)
    var = evaluate.var_table(scored)
    yearly = evaluate.by_year(scored)

    losses.to_csv(RESULTS / "losses.csv", float_format="%.4f")
    dm.to_csv(RESULTS / "diebold_mariano.csv", index=False, float_format="%.4f")
    var.to_csv(RESULTS / "var_backtest.csv", index=False, float_format="%.4f")
    yearly.to_csv(RESULTS / "qlike_by_year.csv", float_format="%.2f")

    pd.set_option("display.width", 140)
    first, last = scored["date"].min().date(), scored["date"].max().date()
    print(f"Out-of-sample: {first} to {last}, {scored.groupby('ticker')['date'].nunique().iloc[0]} days per ticker\n")
    print("QLIKE % vs EWMA (negative = better):")
    print(losses["qlike_vs_ewma_pct"].unstack("model").round(1), "\n")
    print("Diebold-Mariano on QLIKE (negative = first model better):")
    print(dm.pivot(index="ticker", columns=["model_a", "model_b"], values="dm_stat").round(2), "\n")
    print("99% VaR breach rate % (target 1.0):")
    print(var.pivot(index="ticker", columns="model", values="breach_rate_pct").round(2), "\n")
    print("Kupiec p-values:")
    print(var.pivot(index="ticker", columns="model", values="kupiec_p").round(3), "\n")

    if make_plots:
        from volforecast import plots
        plots.make_all(returns, scored, losses, var, yearly, RESULTS / "figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    main(refresh=args.refresh, make_plots=not args.no_plots)
