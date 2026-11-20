"""Backtest engine (proposal, section 4; contract, section 3).

Built step by step:
    4.1 Walk-forward engine (any strategy, no look-ahead, costs, savings)
    4.2 The client over time (age, salary, career value H, pension, savings)
    4.3 Strategies (A) standard, (B) 100 - age, (C) total wealth, (D) no hedge, run month by month
    4.4 Metrics (certainty equivalent in CHF, correlation with the career, bad months, crises)
    4.5 Robustness (wrong beta, job-loss threshold, sector-hedge fragility)

Timing (no look-ahead): the weights for month t are decided with the data of the
`window_months` months BEFORE t only, then held during month t. Cash earns the CHF
risk-free rate of the month (it can be negative). Everything is monthly.
"""
import inspect

import numpy as np
import pandas as pd

from src.career import career_betas, human_capital


def run_backtest(returns, rf, strategy, params, initial_value=None, contributions=None):
    """Walk-forward backtest of one strategy (contract, section 3).

    returns:   DataFrame of monthly asset returns in CHF (total, not excess), month-end dates
    rf:        Series of the monthly CHF risk-free rate, same dates
    strategy:  function(window_excess, date) or function(window_excess, date, value) -> weights
               of the risky assets (same columns as `returns`); the rest is cash. window_excess
               holds ONLY the months before `date`; value = portfolio value in CHF before trading
               (after the contribution), needed when the decision depends on the client's wealth.
    params:    uses params["backtest"]: window_months, transaction_cost_bp
    initial_value:  portfolio value in CHF at the start (default params client portfolio_chf)
    contributions:  CHF added at the start of each month (a number, a Series by date, or None)

    Each month t (from the first month with a full window):
      1. add the contribution, 2. compute the target weights from the past window,
      3. trade from the drifted holdings to the target and pay costs on the turnover,
      4. hold during month t (assets earn their return, cash earns rf).
    Returns a DataFrame by month: value (end of month, CHF), return (net of costs, on the
    value after the contribution), turnover, cost (CHF), contribution (CHF), and the
    target weights (columns "w_<asset>")."""
    if not returns.index.equals(rf.index):
        raise ValueError("returns and rf must have the same dates")
    if returns.isna().any().any() or rf.isna().any():
        raise ValueError("the backtest data must have no missing values")
    window = int(params["backtest"]["window_months"])
    cost_rate = params["backtest"]["transaction_cost_bp"] / 10_000
    if len(returns) <= window:
        raise ValueError(f"need more than {window} months of data, got {len(returns)}")
    value = float(params["client"]["portfolio_chf"] if initial_value is None else initial_value)

    assets = list(returns.columns)
    excess = returns.sub(rf, axis=0)
    holdings = np.zeros(len(assets))          # CHF in each risky asset
    cash = value
    pass_value = len(inspect.signature(strategy).parameters) >= 3
    rows = []
    for t in range(window, len(returns)):
        date = returns.index[t]
        if contributions is None:
            added = 0.0
        elif np.isscalar(contributions):
            added = float(contributions)
        else:
            added = float(contributions.get(date, 0.0))
        cash += added
        start_value = holdings.sum() + cash

        past = excess.iloc[t - window:t]
        weights = np.asarray(strategy(past, date, start_value) if pass_value else strategy(past, date), dtype=float)
        if weights.shape != (len(assets),) or not np.isfinite(weights).all():
            raise ValueError(f"the strategy returned invalid weights on {date.date()}: {weights}")

        drifted = holdings / start_value if start_value != 0 else np.zeros(len(assets))
        turnover = float(np.abs(weights - drifted).sum())
        cost = cost_rate * turnover * start_value
        after_cost = start_value - cost
        holdings = weights * after_cost
        cash = after_cost - holdings.sum()

        holdings = holdings * (1 + returns.iloc[t].to_numpy())
        cash = cash * (1 + rf.iloc[t])
        end_value = holdings.sum() + cash
        row = {"date": date, "value": end_value, "return": end_value / start_value - 1,
               "turnover": turnover, "cost": cost, "contribution": added}
        row.update({f"w_{a}": w for a, w in zip(assets, weights)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# 4.2 The client over time
# =============================================================================

def client_path(params, excess_returns, start=None):
    """The client month by month during the backtest (proposal, section 4.1).

    excess_returns: monthly excess returns with the career's bond asset and sector asset
                    (params data: career_bond_asset, career_sector_asset)
    start:          first month of the backtest (default: the first date of excess_returns)

    For month t (k months after the start):
      age = age_0 + k/12, salary = salary_0 x (1 + salary_growth)^(k/12) (annual, CHF)
      career shock = beta_bond x bond excess + beta_sector x sector excess - expected
                     career excess return (the premia of the career model), i.e. what bonds and
                     the sector did BEYOND what was expected
      H = model value at this age and salary (step 2) x the product of (1 + past shocks):
          the model value, revalued with the realised returns, but anchored to the model.
      H_start (start of month t) only contains the shocks up to t-1: it is what a decision
      for month t can use. H_end contains the shock of month t.
      P = legal BVG estimate at this age and salary (no market risk)
      contribution = savings_rate x salary / 12, until retirement
    Returns a DataFrame by month."""
    from src.optimizer import pension_savings
    client, career = params["client"], params["career"]
    bond = params["data"]["career_bond_asset"]
    sector = params["data"]["career_sector_asset"]
    data = excess_returns if start is None else excess_returns.loc[start:]
    retire = client["retirement_age"]
    has_pension = client.get("pillar2", True)
    savings_rate = params["backtest"].get("savings_rate", 0.0)

    def state(k):
        age = client["age"] + k / 12
        salary = client["salary_chf"] * (1 + career["salary_growth"]) ** (k / 12)
        return age, salary

    rows, level = [], 1.0
    for k, date in enumerate(data.index):
        age, salary = state(k)
        working = age < retire
        beta_bond, beta_sector, _ = career_betas(age, params)
        sector_part = beta_sector * data[sector].iloc[k] if sector is not None else 0.0
        realised = beta_bond * data[bond].iloc[k] + sector_part
        expected = (beta_bond * career["bond_premium"] + beta_sector * career["sector_premium"]) / 12
        shock = realised - expected if working else 0.0
        H_start = human_capital(age, salary, params) * level
        level *= 1 + shock
        next_age, next_salary = state(k + 1)
        H_end = human_capital(next_age, next_salary, params) * level
        P = pension_savings(age, salary, params) if has_pension else 0.0
        rows.append({"date": date, "age": age, "salary": salary, "beta_bond": beta_bond,
                     "beta_sector": beta_sector, "career_excess": realised, "career_sector_part": sector_part,
                     "career_shock": shock,
                     "H_start": H_start, "H_end": H_end, "P": P,
                     "contribution": savings_rate * salary / 12 if working else 0.0})
    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# 4.3 Strategies (A), (B), (C), (D), run month by month
# =============================================================================

STRATEGIES = {"A": "(A) standard: career ignored", "B": "(B) 100 - age rule",
              "D": "(D) total wealth, no hedge", "C": "(C) total wealth"}


def make_strategies(params, excess_returns, path):
    """The four strategy functions for run_backtest (proposal, section 4.1).

    excess_returns: monthly excess returns of the client's assets (src.optimizer.client_assets)
    path:           client_path for the same months as the backtest (H_start, P, age, betas)
    gamma is calibrated ONCE on the first window (the months before the backtest starts),
    so no future data is used; every strategy shares it. Each month, (A), (C), (D) use the same
    Ledoit-Wolf covariance and Black-Litterman returns of the past window and the same
    constraints, lock-up included. Returns ({"A": f, "B": f, "C": f, "D": f}, gamma)."""
    from src.career import career_covariance
    from src.optimizer import (EMPLOYER, build_constraints, client_assets, estimate_covariance,
                               expected_returns, risk_tolerance, solve_portfolio)
    assets = client_assets(params)
    window = int(params["backtest"]["window_months"])
    gamma, _ = risk_tolerance(params, excess_returns.iloc[:window])
    deferred = float(params["client"].get("deferred_shares_chf") or 0.0)
    cache = {}

    def market(window_excess, date):
        if date not in cache:                                 # same inputs for (A), (C), (D)
            cov, _ = estimate_covariance(window_excess[assets], params)
            mu, _ = expected_returns(cov, params)
            cache[date] = (cov, mu.to_numpy())
        return cache[date]

    def lockup_share(value):
        return min(deferred / value, 1.0) if EMPLOYER in assets and value > 0 else 0.0

    def total_wealth(kind):
        def strategy(window_excess, date, value):
            cov, mu = market(window_excess, date)
            state = path.loc[date]
            H, P = state["H_start"], state["P"]
            W = H + P + value
            omega, omega_H = (H + P) / W, H / W
            sigma_H = np.zeros(len(assets))
            if kind == "C" and omega > 0:
                career = career_covariance(cov, state["beta_bond"], state["beta_sector"], params)
                sigma_H = omega_H / omega * career
            om = 0.0 if kind == "A" else omega
            constraints = build_constraints(assets, params, lockup_share(value))
            return solve_portfolio(mu, cov.to_numpy(), sigma_H, om, gamma, constraints)["x"]
        return strategy

    def hundred_minus_age(window_excess, date, value):
        x = np.zeros(len(assets))
        lock = lockup_share(value)
        equity = min(max((100 - path.loc[date, "age"]) / 100, 0.0), 1.0)
        x[assets.index("EQ_WORLD")] = equity * (1 - lock)
        x[assets.index(params["data"]["career_bond_asset"])] = (1 - equity) * (1 - lock)
        if lock > 0:
            x[assets.index(EMPLOYER)] = lock
        return x

    return {"A": total_wealth("A"), "B": hundred_minus_age,
            "C": total_wealth("C"), "D": total_wealth("D")}, gamma


def run_strategies(params=None, returns=None, rf=None, strategies=("A", "B", "C", "D")):
    """Backtest of the strategies for the client in params, on the saved data by default.

    returns: monthly CHF returns with the client's assets (core + sector if needed); rf: CHF rate.
    Returns ({name: DataFrame by month}, client path, gamma). Each DataFrame is the output of
    run_backtest plus: age, H (career value, end of month), P (pension), W = portfolio + H + P."""
    from src.config import load_params
    from src.optimizer import client_assets
    if params is None:
        params = load_params()
    if returns is None or rf is None:
        from src.data import load_returns, load_rf, load_sector_returns
        returns, rf = load_returns().join(load_sector_returns()), load_rf()
    assets = client_assets(params)
    returns = returns[assets]
    excess = returns.sub(rf, axis=0)
    window = int(params["backtest"]["window_months"])
    path = client_path(params, excess, returns.index[window])
    functions, gamma = make_strategies(params, excess, path)
    results = {}
    for name in strategies:
        r = run_backtest(returns, rf, functions[name], params,
                         initial_value=params["client"]["portfolio_chf"], contributions=path["contribution"])
        r["age"], r["H"], r["P"] = path["age"], path["H_end"], path["P"]
        r["W"] = r["value"] + r["H"] + r["P"]
        results[name] = r
    return results, path, gamma


# =============================================================================
# 4.4 Metrics (proposal, section 4.1)
# =============================================================================

CRISES = {"2008 financial crisis": ("2008-08", "2009-02"), "2011 euro crisis": ("2011-06", "2011-09"),
          "Jan 2015 CHF shock": ("2014-12", "2015-01"), "COVID 2020": ("2020-01", "2020-03"),
          "2022 rates and inflation": ("2021-12", "2022-09"), "Credit Suisse 2023": ("2023-02", "2023-03")}


def certainty_equivalent_return(monthly_excess, gamma):
    """Annual certainty equivalent of a series of monthly excess returns (lab convention,
    gamma = risk tolerance): CE = mean x 12 - variance x 12 / (2 gamma)."""
    r = pd.Series(monthly_excess).dropna()
    return 12 * r.mean() - 12 * r.var() / (2 * gamma)


def max_drawdown(monthly_returns):
    """Largest fall from a peak of the cumulated returns (a negative number, 0 if none)."""
    index = (1 + pd.Series(monthly_returns).fillna(0)).cumprod()
    return float((index / index.cummax() - 1).min())


def total_wealth_excess(result, path, rf):
    """Monthly excess return of total wealth, each part weighted by its share at the start of the
    month: (F/W) x portfolio excess + (H/W) x career excess (its market part, as in the optimizer)
    + (P/W) x 0 (the pension has no market risk)."""
    F = result["value"] / (1 + result["return"])               # value after the contribution, before the month
    H, P = path.loc[result.index, "H_start"], path.loc[result.index, "P"]
    W = F + H + P
    rf = rf.loc[result.index]
    return F / W * (result["return"] - rf) + H / W * path.loc[result.index, "career_excess"]


def compute_metrics(results, path, rf, gamma, bad_share=0.10):
    """One row per strategy (proposal, section 4.1).
    Headline: certainty equivalent of total wealth (annual, and CHF per year on the average W),
    and its gain over (A). Readable: correlation of the portfolio with the SECTOR part of the
    career, average CHF result of the portfolio in the worst `bad_share` months of that sector
    part (the same months for every strategy). Secondary: volatility and maximum drawdown of the
    portfolio and of total wealth, turnover, costs, tracking error against (A)."""
    sector = path["career_sector_part"]
    has_sector = sector.abs().sum() > 0
    bad = sector.nsmallest(max(1, int(round(bad_share * len(sector))))).index if has_sector else []
    rows = {}
    for name, r in results.items():
        F_start = r["value"] / (1 + r["return"])
        excess_F = r["return"] - rf.loc[r.index]
        excess_W = total_wealth_excess(r, path, rf)
        ce = certainty_equivalent_return(excess_W, gamma)
        W_avg = r["W"].mean()
        rows[name] = {
            "CE total wealth (annual)": ce, "CE (CHF per year)": ce * W_avg,
            "correlation portfolio / sector part": excess_F.corr(sector.loc[r.index]) if has_sector else np.nan,
            "avg CHF in the sector's worst months": float((F_start * r["return"]).loc[bad].mean()) if has_sector else np.nan,
            "portfolio volatility": excess_F.std() * np.sqrt(12), "portfolio max drawdown": max_drawdown(r["return"]),
            "total wealth volatility": excess_W.std() * np.sqrt(12),
            "turnover per year": r["turnover"].mean() * 12, "costs (CHF)": r["cost"].sum(),
            "final portfolio (CHF)": r["value"].iloc[-1], "average W (CHF)": W_avg}
    table = pd.DataFrame(rows).T
    if "A" in results:
        table["CE gain vs (A) (CHF per year)"] = table["CE (CHF per year)"] - table.loc["A", "CE (CHF per year)"]
        a = results["A"]["return"]
        table["tracking error vs (A)"] = [(results[n]["return"] - a).std() * np.sqrt(12) for n in table.index]
    return table


def crisis_table(results, crises=CRISES):
    """CHF change of the portfolio F and of total wealth W in each crisis, per strategy: from the
    end of the month before the first crisis month to the end of the last one. Computed from the
    backtest, never assumed (proposal, section 4.1)."""
    rows = []
    for crisis, (first, last) in crises.items():
        for name, r in results.items():
            before = r.index[r.index < pd.Timestamp(first)]
            if len(before) == 0 or pd.Timestamp(last) > r.index[-1]:
                continue
            start, end = before[-1], r.loc[:last].index[-1]
            rows.append({"crisis": crisis, "strategy": name,
                         "portfolio change (CHF)": r.loc[end, "value"] - r.loc[start, "value"],
                         "total wealth change (CHF)": r.loc[end, "W"] - r.loc[start, "W"]})
    return pd.DataFrame(rows).set_index(["crisis", "strategy"])
