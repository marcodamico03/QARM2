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


def short_sample(params, extra_months=18):
    """Real data cut to the first window + a few months (fast tests)."""
    from src.data import load_returns, load_rf, load_sector_returns
    n = params["backtest"]["window_months"] + extra_months
    returns = load_returns().join(load_sector_returns()).iloc[:n]
    return returns, load_rf().iloc[:n]


def test_strategies():
    import src.career as C
    import src.optimizer as O
    params = load_params()
    returns, rf = short_sample(params)
    results, path, gamma = B.run_strategies(params, returns, rf)
    assets = O.client_assets(params)
    weights = [f"w_{a}" for a in assets]
    window = params["backtest"]["window_months"]

    # gamma comes from the first window only
    g, _ = O.risk_tolerance(params, returns.sub(rf, axis=0)[assets].iloc[:window])
    assert abs(gamma - g) < 1e-12

    # First month: the client's omega is the step 2 / step 3 balance sheet
    first = path.index[0]
    F = params["client"]["portfolio_chf"] + path["contribution"].iloc[0]
    H = C.human_capital(params["client"]["age"], params["client"]["salary_chf"], params)
    assert abs(path.loc[first, "H_start"] - H) < 1e-6

    # (C) in the first month = Marco's solver on the same inputs, done by hand
    excess = returns.sub(rf, axis=0)[assets]
    cov, _ = O.estimate_covariance(excess.iloc[:window], params)
    mu = O.expected_returns(cov, params)[0].to_numpy()
    P = path.loc[first, "P"]
    W = H + P + F
    sigma_H = H / (H + P) * C.career_covariance(cov, path.loc[first, "beta_bond"], path.loc[first, "beta_sector"], params)
    cons = O.build_constraints(assets, params, params["client"]["deferred_shares_chf"] / F)
    x = O.solve_portfolio(mu, cov.to_numpy(), sigma_H, (H + P) / W, gamma, cons)["x"]
    assert np.allclose(results["C"].loc[first, weights].to_numpy(), x, atol=1e-10)

    # (B): (100 - age)% world equities, the rest CHF bonds, after the lock-up
    b = results["B"]
    lock = params["client"]["deferred_shares_chf"] / F
    assert abs(b.loc[first, "w_EQ_WORLD"] - 0.65 * (1 - lock)) < 1e-12
    assert abs(b.loc[first, "w_BOND_CHF"] - 0.35 * (1 - lock)) < 1e-12 and abs(b.loc[first, "w_EMPLOYER"] - lock) < 1e-12

    # Every month, every strategy: valid weights (long-only, no borrowing, caps, lock-up held)
    caps = {a: c for a, c in params["optimization"]["position_caps"].items() if a in assets}
    for name, r in results.items():
        w = r[weights]
        assert (w.to_numpy() > -1e-9).all() and (w.sum(axis=1) < 1 + 1e-9).all(), name
        assert (r["w_EMPLOYER"] >= params["client"]["deferred_shares_chf"] / (r["value"] + 1e9) - 1e-9).all()
        for a, c in caps.items():
            if a != "EMPLOYER":
                assert (w[f"w_{a}"] <= c + 1e-9).all(), (name, a)
        assert np.allclose(r["W"], r["value"] + r["H"] + r["P"])

    # The hedge never holds MORE of the career's sector than the same client without it
    assert (results["C"]["w_BANKS_EU"] <= results["D"]["w_BANKS_EU"] + 1e-9).all()

    # No look-ahead with the real strategies: rewriting month 70 on changes nothing before it
    changed = returns.copy()
    changed.iloc[70:] = changed.iloc[70:] * -2
    other, _, _ = B.run_strategies(params, changed, rf, strategies=("C",))
    cut = returns.index[70]
    assert np.allclose(results["C"].loc[:cut, weights], other["C"].loc[:cut, weights])
    print(f"strategies OK (gamma {gamma:.3f} from the first window; (C) month 1 = solver by hand; "
          f"valid weights every month; no look-ahead)")


def test_metric_formulas():
    # Certainty equivalent: known mean and variance
    r = pd.Series([0.01, -0.01, 0.03, -0.03])                 # mean 0, variance 0.02 / 3 x ... computed below
    gamma = 0.25
    assert abs(B.certainty_equivalent_return(r, gamma) - (12 * r.mean() - 12 * r.var() / (2 * gamma))) < 1e-15
    assert abs(B.certainty_equivalent_return(pd.Series([0.01] * 5), gamma) - 0.12) < 1e-12   # no risk: CE = mean
    # Maximum drawdown: +10%, -50%, +20% -> peak 1.1, trough 0.55: -50%
    assert abs(B.max_drawdown([0.10, -0.50, 0.20]) + 0.5) < 1e-12 and B.max_drawdown([0.01, 0.02]) == 0

    # Total wealth: with no career and no pension it is the portfolio's excess return
    dates = pd.date_range("2010-01-31", periods=3, freq="ME")
    result = pd.DataFrame({"value": [101.0, 99.0, 102.0], "return": [0.01, -0.0198019801980198, 0.0303030303030303]}, index=dates)
    rf = pd.Series(0.001, index=dates)
    path = pd.DataFrame({"H_start": 0.0, "P": 0.0, "career_excess": 0.05}, index=dates)
    assert np.allclose(B.total_wealth_excess(result, path, rf), result["return"] - rf)
    # Half portfolio, half career: the average of the two excess returns
    path["H_start"] = result["value"] / (1 + result["return"])
    assert np.allclose(B.total_wealth_excess(result, path, rf), 0.5 * (result["return"] - rf) + 0.5 * 0.05)
    print("metric formulas OK")


def test_metrics_on_saved_data():
    params = load_params()
    results, path, gamma = B.run_strategies(params)
    from src.data import load_rf
    table = B.compute_metrics(results, path, load_rf(), gamma)
    crises = B.crisis_table(results)
    assert list(table.index) == ["A", "B", "C", "D"] and table["CE gain vs (A) (CHF per year)"]["A"] == 0
    # The hedge does what it is built for: (C) moves less with the client's sector than (D)
    assert table.loc["C", "correlation portfolio / sector part"] < table.loc["D", "correlation portfolio / sector part"]
    # Every crisis has the four strategies, and W changes include the career
    assert len(crises) == 4 * len(B.CRISES)
    print("metrics on saved data OK")
    return table, crises


def test_significance_formulas():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0.002, 0.02, 240))
    # lags = 0: the plain t-stat (variance with ddof = 0)
    mean, se, t = B.newey_west_tstat(x, lags=0)
    assert abs(se - x.std(ddof=0) / np.sqrt(len(x))) < 1e-15 and abs(t - mean / se) < 1e-12
    # Positively autocorrelated months: Newey-West gives a LARGER standard error than the plain one
    e = rng.normal(0, 0.02, 240)
    ar = np.zeros(240)
    for i in range(1, 240):
        ar[i] = 0.6 * ar[i - 1] + e[i]
    assert B.newey_west_tstat(ar)[1] > B.newey_west_tstat(ar, lags=0)[1]

    # Bootstrap: b = a + 0.1% a month has the same variance, so EVERY history gains exactly 1.2% a year
    a = pd.Series(rng.normal(0.003, 0.03, 120))
    gain, boot = B.block_bootstrap_ce_gain(a, a + 0.001, gamma=0.25, draws=500)
    assert abs(gain - 0.012) < 1e-12 and np.allclose(boot, 0.012) and (boot > 0).all()
    # Same series: no gain in any history (p-value 1)
    _, boot = B.block_bootstrap_ce_gain(a, a, gamma=0.25, draws=200)
    assert np.allclose(boot, 0) and (boot <= 0).mean() == 1.0
    # Same seed, same answer
    assert np.array_equal(B.block_bootstrap_ce_gain(a, a * 1.1, 0.25, draws=100, seed=7)[1],
                          B.block_bootstrap_ce_gain(a, a * 1.1, 0.25, draws=100, seed=7)[1])
    # Same decisions, different wealth: the fair comparison gives exactly 0 gain. (With each
    # strategy's own wealth shares it would not: that bias is what same_wealth_excess removes.)
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    r = pd.Series(rng.normal(0.006, 0.04, 60), index=dates)
    rich = pd.DataFrame({"value": 300.0, "return": r})          # same returns, three times the money
    poor = pd.DataFrame({"value": 100.0, "return": r})
    path = pd.DataFrame({"H_start": 1000.0, "P": 0.0, "career_excess": rng.normal(0.002, 0.01, 60)}, index=dates)
    rf = pd.Series(0.0, index=dates)
    fair = B.same_wealth_excess(rich, poor, path, rf)
    assert np.allclose(fair, B.total_wealth_excess(poor, path, rf))
    own = B.total_wealth_excess(rich, path, rf) - B.total_wealth_excess(poor, path, rf)
    assert np.abs(own).max() > 1e-4                              # the biased version sees a difference
    print("significance formulas OK (Newey-West, block bootstrap, fair same-wealth comparison)")


def test_wrong_beta_theory():
    # Known answer (proposal 4.2): one factor s, career exposure beta (true), portfolio short k
    # of it in (C) and 0 in (D). With wealth shares sF (portfolio) and sH (career), the hedge in
    # career units is h = sF k / sH, and the variance saved is exactly sH^2 h (2 beta - h) var(s)
    # (exact with sample moments). So the CE gain is 12 x saved / (2 gamma): a straight line in
    # beta, zero at beta = h / 2.
    rng = np.random.default_rng(5)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    s = rng.normal(0, 0.05, 120)
    s = s - s.mean()                                           # no mean effect: only risk matters
    F, H, gamma, k = 100.0, 900.0, 0.25, 2.0
    sF, sH = F / (F + H), H / (F + H)
    h = sF * k / sH
    rf = pd.Series(0.0, index=dates)
    d = pd.DataFrame({"value": F, "return": 0.0}, index=dates)          # value constant: F/W constant
    c = pd.DataFrame({"value": F, "return": -k * s}, index=dates)
    gains = []
    for beta in (0.0, h / 2, h, 2 * h):
        path = pd.DataFrame({"H_start": H, "P": 0.0, "career_excess": beta * s}, index=dates)
        a = B.total_wealth_excess(d, path, rf)
        b = B.same_wealth_excess(c, d, path, rf)
        gain = B.certainty_equivalent_return(b, gamma) - B.certainty_equivalent_return(a, gamma)
        theory = 12 * sH ** 2 * h * (2 * beta - h) * np.var(s, ddof=1) / (2 * gamma)
        assert abs(gain - theory) < 1e-12, (beta, gain, theory)
        gains.append(gain)
    assert gains[0] < 0 and abs(gains[1]) < 1e-12 and gains[2] > 0 and gains[3] > gains[2]
    print("wrong-beta theory OK (gain = h(2 beta - h), break-even at beta = h/2)")


def test_wrong_beta_on_saved_data():
    params = load_params()
    curve = B.wrong_beta_curve(params, draws=300)
    gains = curve["CE gain C vs D (CHF per year)"].to_numpy()
    # Almost a straight line in the true beta, as the theory says. Not exactly: a different true
    # beta_sector also changes the career's discount rate, hence its value and wealth share.
    m = curve.index.to_numpy()
    slope, intercept = np.polyfit(m, gains, 1)
    r2 = 1 - np.var(gains - (slope * m + intercept)) / np.var(gains)
    assert r2 > 0.99, r2
    assert slope > 0 and (np.diff(gains) > 0).all()            # the hedge is worth more when the exposure is real
    print(f"wrong-beta on saved data OK (gain CHF {gains[0]:,.0f} at true beta 0 -> CHF {gains[-1]:,.0f} at 2x)")
    return curve


if __name__ == "__main__":
    test_constant_weights()
    test_costs_and_turnover()
    test_contributions()
    test_no_look_ahead()
    test_errors()
    test_on_saved_data()
    test_client_path()
    test_client_on_saved_data()
    test_strategies()
    test_metric_formulas()
    test_metrics_on_saved_data()
    test_significance_formulas()
    test_wrong_beta_theory()
    test_wrong_beta_on_saved_data()
    print("ALL TESTS PASSED")
