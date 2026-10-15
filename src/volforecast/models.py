"""One-day-ahead variance forecasts.

Every function returns forecasts indexed by the date being forecast (the
target date), built only from returns observed before that date.
Returns are daily log returns in percent, so variances are in percent squared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def _to_target_index(origin_values: pd.Series, returns: pd.Series) -> pd.Series:
    """Move a forecast made at the close of day t onto day t+1."""
    dates = returns.index
    pos = dates.get_indexer(origin_values.index)
    keep = pos < len(dates) - 1
    out = pd.Series(origin_values.values[keep], index=dates[pos[keep] + 1])
    out.index.name = "date"
    return out


def historical(returns: pd.Series, window: int = 21) -> pd.Series:
    """Sample variance of the last `window` daily returns."""
    origin = returns.rolling(window).var().dropna()
    return _to_target_index(origin, returns)


def ewma(returns: pd.Series, lam: float = 0.94, init_window: int = 30) -> pd.Series:
    """RiskMetrics EWMA: s2[t+1] = lam * s2[t] + (1 - lam) * r[t]^2."""
    r = returns.to_numpy()
    s2 = np.empty(len(r))
    s2[0] = np.var(r[:init_window], ddof=1)
    for t in range(len(r) - 1):
        s2[t + 1] = lam * s2[t] + (1 - lam) * r[t] ** 2
    # s2[t] is the forecast for day t, made at the close of day t-1
    out = pd.Series(s2, index=returns.index)
    out.index.name = "date"
    return out.iloc[init_window:]


def rolling_garch(
    returns: pd.Series,
    o: int = 0,
    window: int = 1000,
    refit_every: int = 21,
    first_target: str = "2015-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-window GARCH(1,1) or GJR-GARCH(1,1,1) with Student-t errors.

    The model is re-estimated every `refit_every` days on the previous
    `window` returns. Between refits the parameters stay fixed and the
    variance is updated each day with the latest return, so every forecast
    uses only data available at the time.

    Returns
    -------
    forecasts : DataFrame indexed by target date with columns
        `var` (forecast variance) and `nu` (t degrees of freedom).
    params : DataFrame of estimated parameters, one row per refit.
    """
    dates = returns.index
    n = len(dates)
    model = arch_model(
        returns, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist="t", rescale=False
    )

    # Position of the first target date; the first fit uses data up to the day before
    k0 = int(dates.searchsorted(pd.Timestamp(first_target)))
    if k0 < window:
        raise ValueError("Not enough history before first_target for the estimation window")

    rows, param_rows = [], []
    for k in range(k0, n, refit_every):
        res = model.fit(first_obs=k - window, last_obs=k, disp="off", show_warning=False)
        # Forecast origins k-1 ... k+refit_every-2, i.e. targets k ... k+refit_every-1
        last_origin = min(k + refit_every - 2, n - 2)
        fc = res.forecast(horizon=1, start=dates[k - 1], reindex=False)
        var = fc.variance["h.1"].loc[dates[k - 1] : dates[last_origin]]
        targets = dates[k : last_origin + 2]
        rows.append(pd.DataFrame({"var": var.to_numpy(), "nu": res.params["nu"]}, index=targets))
        param_rows.append(res.params.rename(dates[k]))

    forecasts = pd.concat(rows)
    forecasts.index.name = "date"
    params = pd.DataFrame(param_rows)
    params.index.name = "refit_date"
    return forecasts, params
