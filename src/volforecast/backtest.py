"""Run every model on every ticker over the same out-of-sample period."""

from __future__ import annotations

import pandas as pd

from . import models

MODELS = ["Historical 21d", "EWMA", "GARCH", "GJR-GARCH"]


def run(returns: pd.DataFrame, first_target: str = "2015-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast next-day variance for each ticker and model.

    Returns
    -------
    forecasts : long DataFrame with columns
        date, ticker, model, var, nu, ret
        (`ret` is the realised return on the target date; `nu` is NaN for
        models that assume normal errors).
    params : estimated GARCH and GJR-GARCH parameters at each refit.
    """
    frames, param_frames = [], []
    for ticker in returns.columns:
        r = returns[ticker]
        per_model = {
            "Historical 21d": models.historical(r).to_frame("var"),
            "EWMA": models.ewma(r).to_frame("var"),
        }
        for name, o in [("GARCH", 0), ("GJR-GARCH", 1)]:
            fc, params = models.rolling_garch(r, o=o, first_target=first_target)
            per_model[name] = fc
            param_frames.append(params.assign(ticker=ticker, model=name))

        for name, fc in per_model.items():
            fc = fc.loc[fc.index >= pd.Timestamp(first_target)].copy()
            fc["ret"] = r.reindex(fc.index)
            frames.append(fc.assign(ticker=ticker, model=name))

    forecasts = pd.concat(frames).reset_index()
    if "nu" not in forecasts:
        forecasts["nu"] = float("nan")
    forecasts = forecasts[["date", "ticker", "model", "var", "nu", "ret"]]

    # Keep only target dates where every model has a forecast, so all are scored on the same days
    counts = forecasts.groupby(["ticker", "date"])["model"].transform("nunique")
    forecasts = forecasts[counts == len(MODELS)].sort_values(["ticker", "model", "date"])

    params = pd.concat(param_frames).reset_index()
    return forecasts.reset_index(drop=True), params
