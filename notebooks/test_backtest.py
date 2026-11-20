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


def career_data(params, months=36, bond=None, sector=None):
    """Fake excess returns of the career assets. By default each month equals the EXPECTED
    career return, so the career shocks are zero (a known answer)."""
    import src.career as C
    dates = pd.date_range("2005-01-31", periods=months, freq="ME")
    c = params["career"]
    data = pd.DataFrame({"BOND_CHF": c["bond_premium"] / 12, "BANKS_EU": c["sector_premium"] / 12}, index=dates)
    if bond is not None:
        data["BOND_CHF"] = bond
    if sector is not None:
        data["BANKS_EU"] = sector
    return data


def test_client_path():
    import src.career as C
    params = load_params()
    age0, s0, g = params["client"]["age"], params["client"]["salary_chf"], params["career"]["salary_growth"]

    # 1. Returns exactly as expected: no shock, so H is exactly the step 2 model value every month
    path = B.client_path(params, career_data(params))
    assert np.allclose(path["career_shock"], 0, atol=1e-15)
    assert abs(path["H_start"].iloc[0] - C.human_capital(age0, s0, params)) < 1e-6
    for k in (0, 12, 35):
        age, salary = age0 + k / 12, s0 * (1 + g) ** (k / 12)
        assert abs(path["age"].iloc[k] - age) < 1e-12 and abs(path["salary"].iloc[k] - salary) < 1e-6
        assert abs(path["H_start"].iloc[k] - C.human_capital(age, salary, params)) < 1e-6
    assert np.allclose(path["contribution"], params["backtest"]["savings_rate"] * path["salary"] / 12)

    # 2. A bank crash (-30% in month 10) lowers the banker's H from then on, never before
    crash = career_data(params)
    crash.iloc[10, crash.columns.get_loc("BANKS_EU")] = -0.30
    hit = B.client_path(params, crash)
    assert np.allclose(hit["H_start"].iloc[:11], path["H_start"].iloc[:11])     # month 10 decided before the crash
    assert (hit["H_start"].iloc[11:] < path["H_start"].iloc[11:]).all()
    beta_sector = path["beta_sector"].iloc[10]
    assert abs(hit["H_end"].iloc[10] / path["H_end"].iloc[10] - (1 - beta_sector * (0.30 + params["career"]["sector_premium"] / 12))) < 1e-12

    # 3. A civil servant (no sector) does not feel the bank crash
    q = C.params_for_persona(params, "civil_servant")
    calm, crashed = B.client_path(q, career_data(q)), B.client_path(q, crash)
    assert np.allclose(calm["H_start"], crashed["H_start"])

    # 4. Retirement: from 65 on, no career, no savings
    old = copy.deepcopy(params)
    old["client"]["age"] = 64
    late = B.client_path(old, career_data(old, months=24))
    after = late["age"] >= 65
    assert after.any() and (late.loc[after, "H_start"] == 0).all() and (late.loc[after, "contribution"] == 0).all()
    print("client path OK (no shock = step 2 value; a bank crash hits the banker, not the civil servant)")


def test_client_on_saved_data():
    from src.data import load_returns, load_rf
    params = load_params()
    excess = load_returns().sub(load_rf(), axis=0)
    start = excess.index[params["backtest"]["window_months"]]
    path = B.client_path(params, excess, start)
    assert path.index[0] == start and path["H_start"].notna().all() and (path["H_start"] > 0).all()
    h = path["H_end"]
    print(f"client on saved data OK (banker from {start:%Y-%m}: H CHF {path['H_start'].iloc[0]:,.0f}; "
          f"Aug 2008 {h.loc['2008-08'].iloc[0]:,.0f} -> Feb 2009 {h.loc['2009-02'].iloc[0]:,.0f}; "
          f"{path.index[-1]:%Y-%m} {h.iloc[-1]:,.0f} at age {path['age'].iloc[-1]:.1f})")


if __name__ == "__main__":
    test_constant_weights()
    test_costs_and_turnover()
    test_contributions()
    test_no_look_ahead()
    test_errors()
    test_on_saved_data()
    test_client_path()
    test_client_on_saved_data()
    print("ALL TESTS PASSED")
