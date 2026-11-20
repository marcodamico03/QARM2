"""Tests of src/backtest.py with known answers (contract, section 6).

Run from the project folder:
    python notebooks/test_backtest.py
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.backtest as B
from src.config import load_params


def fake_data(months=24, seed=0):
    """Made-up monthly data: two assets and a risk-free rate (some months negative, as in CHF)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    returns = pd.DataFrame(rng.normal(0.005, 0.04, size=(months, 2)), index=dates, columns=["A", "B"])
    rf = pd.Series(rng.normal(0.0005, 0.001, size=months), index=dates)
    return returns, rf


def test_params(window=6, cost_bp=0.0):
    p = copy.deepcopy(load_params())
    p["backtest"]["window_months"], p["backtest"]["transaction_cost_bp"] = window, cost_bp
    return p


def test_constant_weights():
    # Known answer: rebalanced monthly to fixed weights, no costs, the value grows by
    # prod(1 + rf + w'(r - rf)), starting at the first month after the window
    returns, rf = fake_data()
    w = np.array([0.6, 0.3])                                   # 10% cash
    r = B.run_backtest(returns, rf, lambda window, date: w, test_params(), initial_value=100.0)
    later = returns.index[6:]
    expected = 100.0 * np.prod(1 + rf[later] + (returns.loc[later] - rf[later].to_numpy()[:, None]) @ w)
    assert abs(r["value"].iloc[-1] - expected) < 1e-9 and r.index.equals(later)

    # All cash: the value grows at the risk-free rate (negative months included)
    r = B.run_backtest(returns, rf, lambda window, date: np.zeros(2), test_params(), initial_value=100.0)
    assert abs(r["value"].iloc[-1] - 100.0 * np.prod(1 + rf[later])) < 1e-9
    print("constant weights OK")


def test_costs_and_turnover():
    # Zero returns: switching between 100% A and 100% B. First month: from cash to A,
    # turnover 1. Then A -> B -> A: turnover 2 each month. Cost = 10 bp x turnover x value.
    dates = pd.date_range("2000-01-31", periods=10, freq="ME")
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    rf = pd.Series(0.0, index=dates)
    flip = lambda window, date: np.array([1.0, 0.0]) if date.month % 2 else np.array([0.0, 1.0])
    r = B.run_backtest(returns, rf, flip, test_params(window=2, cost_bp=10), initial_value=100.0)
    assert np.allclose(r["turnover"], [1] + [2] * 7)
    value = 100.0
    for turnover in r["turnover"]:
        value *= 1 - 0.001 * turnover
    assert abs(r["value"].iloc[-1] - value) < 1e-12 and np.allclose(r["cost"].iloc[0], 0.1)
    print("costs and turnover OK")


def test_contributions():
    # Zero returns, no costs: the value is the start plus every contribution
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    rf = pd.Series(0.0, index=dates)
    r = B.run_backtest(returns, rf, lambda w, d: np.array([0.5, 0.5]), test_params(window=2),
                       initial_value=1000.0, contributions=100.0)
    assert np.allclose(r["value"], 1000.0 + 100.0 * np.arange(1, 11))
    print("contributions OK")


def test_no_look_ahead():
    # 1. The strategy only ever sees the `window` months strictly before the decision date
    returns, rf = fake_data()
    seen = []
    def spy(window, date):
        seen.append((len(window), window.index.max(), date))
        return np.array([0.5, 0.5])
    B.run_backtest(returns, rf, spy, test_params(window=6))
    assert all(n == 6 and last < date for n, last, date in seen)
    assert all(returns.index.get_loc(last) == returns.index.get_loc(date) - 1 for _, last, date in seen)

    # 2. Changing the future does not change any past decision
    momentum = lambda window, date: np.clip(window.mean().to_numpy() * 50 + 0.5, 0, 1)
    base = B.run_backtest(returns, rf, momentum, test_params())
    changed = returns.copy()
    changed.iloc[15:] = changed.iloc[15:] * -3                    # rewrite month 15 onwards
    other = B.run_backtest(changed, rf, momentum, test_params())
    weights = [c for c in base.columns if c.startswith("w_")]
    first = returns.index[15]
    assert np.allclose(base.loc[:first, weights], other.loc[:first, weights])   # decided before month 16
    print("no look-ahead OK")


def test_errors():
    returns, rf = fake_data()
    for bad in (lambda w, d: np.array([np.nan, 0.5]), lambda w, d: np.array([1.0])):
        try:
            B.run_backtest(returns, rf, bad, test_params())
            raise AssertionError("invalid weights should be refused")
        except ValueError:
            pass
    print("errors OK")


def test_on_saved_data():
    # The real data: an equally weighted portfolio runs from the first month with a full 60-month window
    from src.data import load_returns, load_rf
    returns, rf = load_returns(), load_rf()
    params = load_params()
    n = returns.shape[1]
    r = B.run_backtest(returns, rf, lambda w, d: np.ones(n) / n, params)
    window = params["backtest"]["window_months"]
    assert r.index[0] == returns.index[window] and len(r) == len(returns) - window
    assert r["value"].notna().all() and (r["value"] > 0).all()
    print(f"saved data OK ({r.index[0]:%Y-%m} to {r.index[-1]:%Y-%m}, {len(r)} months; "
          f"equal weights: CHF {params['client']['portfolio_chf']:,.0f} -> CHF {r['value'].iloc[-1]:,.0f})")


if __name__ == "__main__":
    test_constant_weights()
    test_costs_and_turnover()
    test_contributions()
    test_no_look_ahead()
    test_errors()
    test_on_saved_data()
    print("ALL TESTS PASSED")
