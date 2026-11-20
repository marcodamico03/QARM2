"""Backtest engine (proposal, section 4; contract, section 3).

Built step by step:
    4.1 Walk-forward engine (any strategy, no look-ahead, costs, savings)   <- this step
    4.2 The client over time (age, career value H, pension)
    4.3 Strategies (A) standard, (B) 100 - age, (C) total wealth, (D) no hedge
    4.4 Metrics (certainty equivalent in CHF, correlation with the career, bad months)
    4.5 Robustness (wrong beta, job-loss threshold, sector-hedge fragility)

Timing (no look-ahead): the weights for month t are decided with the data of the
`window_months` months BEFORE t only, then held during month t. Cash earns the CHF
risk-free rate of the month (it can be negative). Everything is monthly.
"""
import numpy as np
import pandas as pd


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
