"""Backtest engine (proposal, section 4; contract, section 3).

Built step by step:
    4.1 Walk-forward engine (any strategy, no look-ahead, costs, savings)
    4.2 The client over time (age, salary, career value H, pension, savings)
    4.3 Strategies (A) standard, (B) 100 - age, (C) total wealth, (D) no hedge
    4.4 Metrics (certainty equivalent in CHF, correlation with the career, bad months)
    4.5 Robustness (wrong beta, job-loss threshold, sector-hedge fragility)

Timing (no look-ahead): the weights for month t are decided with the data of the
`window_months` months BEFORE t only, then held during month t. Cash earns the CHF
risk-free rate of the month (it can be negative). Everything is monthly.
"""
import numpy as np
import pandas as pd

from src.career import career_betas, human_capital


def run_backtest(returns, rf, strategy, params, initial_value=None, contributions=None):
    """Walk-forward backtest of one strategy (contract, section 3).

    returns:   DataFrame of monthly asset returns in CHF (total, not excess), month-end dates
    rf:        Series of the monthly CHF risk-free rate, same dates
    strategy:  function(window_excess, date) -> weights of the risky assets (same columns as
               `returns`); the rest is cash. window_excess holds ONLY the months before `date`.
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

        weights = np.asarray(strategy(excess.iloc[t - window:t], date), dtype=float)
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
        realised = beta_bond * data[bond].iloc[k]
        if sector is not None:
            realised += beta_sector * data[sector].iloc[k]
        expected = (beta_bond * career["bond_premium"] + beta_sector * career["sector_premium"]) / 12
        shock = realised - expected if working else 0.0
        H_start = human_capital(age, salary, params) * level
        level *= 1 + shock
        next_age, next_salary = state(k + 1)
        H_end = human_capital(next_age, next_salary, params) * level
        P = pension_savings(age, salary, params) if has_pension else 0.0
        rows.append({"date": date, "age": age, "salary": salary, "beta_bond": beta_bond,
                     "beta_sector": beta_sector, "career_excess": realised, "career_shock": shock,
                     "H_start": H_start, "H_end": H_end, "P": P,
                     "contribution": savings_rate * salary / 12 if working else 0.0})
    return pd.DataFrame(rows).set_index("date")
