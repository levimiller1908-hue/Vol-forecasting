import numpy as np
import pandas as pd
import pytest
from scipy import stats

from volforecast import evaluate, models


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    return pd.Series(rng.standard_t(6, size=len(idx)), index=idx)


def test_ewma_matches_recursion(returns):
    lam = 0.94
    fc = models.ewma(returns, lam=lam, init_window=30)
    r = returns.to_numpy()
    s2 = np.var(r[:30], ddof=1)
    for t in range(40):
        s2 = lam * s2 + (1 - lam) * r[t] ** 2
    # s2 is now the forecast for position 40, made after observing r[39]
    assert fc.loc[returns.index[40]] == pytest.approx(s2)


def test_historical_uses_only_past_returns(returns):
    fc = models.historical(returns, window=21)
    target = returns.index[100]
    expected = returns.iloc[79:100].var()
    assert fc.loc[target] == pytest.approx(expected)


def test_garch_has_no_lookahead(returns):
    full, _ = models.rolling_garch(returns, window=500, refit_every=50, first_target="2012-01-02")
    cut = returns.index[1100]
    part, _ = models.rolling_garch(returns.loc[:cut], window=500, refit_every=50, first_target="2012-01-02")
    common = part.index.intersection(full.index)
    assert len(common) > 500
    np.testing.assert_allclose(part.loc[common, "var"], full.loc[common, "var"])


def test_qlike_is_zero_at_true_variance():
    h = np.linspace(0.5, 2, 31)
    loss = evaluate.qlike(np.ones_like(h), h)
    assert loss.min() == pytest.approx(0)
    assert h[loss.argmin()] == pytest.approx(1.0)


def test_diebold_mariano_sign():
    rng = np.random.default_rng(1)
    b = rng.uniform(1, 2, 1000)
    stat, p = evaluate.diebold_mariano(b - 0.1, b + rng.normal(0, 0.05, 1000))
    assert stat < 0 and p < 0.01


def test_kupiec_accepts_exact_rate():
    hits = np.zeros(1000, dtype=bool)
    hits[::100] = True  # exactly 1%
    lr, p = evaluate.kupiec(hits, 0.01)
    assert lr == pytest.approx(0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_var_threshold_normal_quantile():
    v = evaluate.var_threshold(np.array([4.0]), np.array([np.nan]), 0.01)
    assert v[0] == pytest.approx(-stats.norm.ppf(0.01) * 2)
