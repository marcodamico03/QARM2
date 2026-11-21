"""Version 2 of the advice: risk budgeting on the savings (LN5-LN6), built around the client's job.

Why a version 2 (docs, pivot plan): with plain mean-variance on total wealth the savings are a few
percent of wealth, so the optimizer always ends at a corner (all in the asset with the highest
return). Here the savings are spread by risk budgets, as in lab LN5, and the job enters in four
places:

  1. The client's industry is left out of the purchases when the pay depends on it (variable pay)
     or when the client already holds shares of it (the stake).
  2. The stake is a fixed holding until it can be sold: the rest is built around it.
  3. Bonds: the risk profile's bond share, reduced in proportion to the part of the client's wealth
     that already behaves like a bond (the career's bond part and the pension): "your job is your
     bond" (total-wealth view of steps 2-3). Filling the whole total-wealth gap with the savings would
     again give corners (0% or 100% bonds), because the savings are small next to the career.
  4. A loss limit before retirement: in a bad year (1 in 20) the savings may not lose more than the
     profile accepts plus what the client can still save before retiring. If they would, cash.

Risk budgeting (lab LN5, with budgets and a fixed holding):

    min_y  1/2 e' Sigma e - lambda sum_i b_i ln y_i      e = y + fixed (CHF exposures), y > 0

The first-order condition is y_i (Sigma e)_i = lambda b_i: each asset bought carries a share of
the risk proportional to its budget. It is solved by cyclical coordinate descent (Griveau-Billon,
Richard and Roncalli, 2013; in the course reading on optimization algorithms), and lambda is set
so that the CHF amounts add up to the money to invest.

Budgets: the growth part (equities, real assets) is split by risk profile; inside a group budgets are
in proportion to the Black-Litterman Sharpe ratios (LN4), with a floor so no asset is dropped. The
bond part is risk budgeting among the bonds.

Strategies compared in the app and the backtest:
  (A) typical advice: the profile's safe share applied to the savings; it knows neither the job nor
      the stake
  (B) 100 minus age: age% in bonds, the rest in the growth assets
  (C) our plan: the four points above
No Streamlit here.
"""
import copy

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import optimizer as O
from src.career import career_betas, human_capital, params_for_persona
from src.config import load_params

MONTHS = 12
CASH = "CASH"
STRATEGIES = {"A": "Typical advice", "B": "100 minus age", "C": "Our plan"}


# =============================================================================
# 1. The universe and the client
# =============================================================================

def assets(params):
    """The assets of version 2, in the order of the groups (equities, real, bonds)."""
    return [a for members in params["allocation"]["groups"].values() for a in members]


def group_of(asset, params):
    for group, members in params["allocation"]["groups"].items():
        if asset in members:
            return group
    return "cash"


def load_data(params=None, start=None, end=None):
    """Monthly CHF total returns of the version 2 assets and the CHF risk-free rate, on the months
    where every asset has data. Returns (returns, rf)."""
    from src.data import load_returns, load_rf, load_universe2_returns
    params = params or load_params()
    names = assets(params)
    table = load_returns(start, end, params).join(load_universe2_returns(start, end, params), how="inner")
    missing = [a for a in names if a not in table.columns]
    if missing:
        raise ValueError(f"missing assets in the saved data: {missing}. Run: python -m src.data")
    table = table[names].dropna()
    rf = load_rf(start, end, params).reindex(table.index)
    keep = rf.notna()
    return table[keep], rf[keep]


def client_params(persona, changes=None, base=None):
    """params for a persona of version 2, with the changes made in the app.

    changes (all optional): age, salary_chf, portfolio_chf (all savings, stake included), stake_chf,
    unlock_years, industry (a Fama-French industry or None), variable_pay (share of pay that moves
    with the industry), profile, pillar2 (False = self-employed, no pension fund), pension_chf
    (None = legal estimate), view {asset, q, sd}.
    The career's sector beta is the variable pay: a bonus moves one for one with the industry."""
    base = base or load_params()
    p = params_for_persona(copy.deepcopy(base), persona)
    person = base["personas"][persona]
    client = p["client"]
    client.update({"industry": person.get("industry"), "variable_pay": float(person.get("variable_pay", 0.0)),
                   "stake_chf": float(person.get("stake_chf", 0.0)),
                   "unlock_years": float(person.get("unlock_years", 0.0)),
                   "profile": base["allocation"]["default_profile"]})
    changes = changes or {}
    for key in ("age", "salary_chf", "portfolio_chf", "stake_chf", "unlock_years", "variable_pay", "profile",
                "pillar2"):
        if changes.get(key) is not None:
            client[key] = changes[key]
    if "industry" in changes:
        client["industry"] = changes["industry"]
    if "pension_chf" in changes:
        client["pension_chf"] = "auto" if changes["pension_chf"] in (None, "auto") else float(changes["pension_chf"])
    view = changes.get("view")
    if view:
        p["black_litterman"]["views"] = [{"assets": {view["asset"]: 1.0}, "q": float(view["q"]),
                                          "sd": float(view["sd"])}]
    industry = client["industry"]
    if industry is None:
        client["variable_pay"], client["stake_chf"] = 0.0, 0.0
    client["stake_chf"] = min(float(client["stake_chf"]), float(client["portfolio_chf"]))
    client["deferred_shares_chf"] = 0.0                        # version 1 lock-up not used here
    p["data"]["career_sector_asset"] = industry
    p["career"]["beta_sector"] = float(client["variable_pay"]) if industry else 0.0
    return p


def balance_sheet(params):
    """H, P, F, W today and the bond-like share of wealth: (beta_bond x H + P) / W."""
    client = params["client"]
    age, salary, F = client["age"], client["salary_chf"], float(client["portfolio_chf"])
    H = human_capital(age, salary, params)
    if not client.get("pillar2", True):
        P = 0.0
    elif client.get("pension_chf") in (None, "auto"):
        P = O.pension_savings(age, salary, params)
    else:
        P = float(client["pension_chf"])
    W = H + P + F
    beta_bond, beta_sector, rate = career_betas(age, params)
    return {"H": H, "P": P, "F": F, "W": W, "beta_bond": beta_bond, "beta_sector": beta_sector,
            "discount_rate": rate, "bond_like": (beta_bond * H + P) / W,
            "career_bonds": beta_bond * H, "career_sector": beta_sector * H}


def future_savings(params, age=None, salary=None):
    """What the client can still save before retirement (no growth on it): the savings rate of
    params times the salary of each remaining year (salary grows with salary_growth)."""
    client, career = params["client"], params["career"]
    age = client["age"] if age is None else age
    salary = client["salary_chf"] if salary is None else salary
    years = max(int(np.ceil(client["retirement_age"] - age)), 0)
    growth = (1 + career["salary_growth"]) ** np.arange(years)
    return float(params["backtest"]["savings_rate"] * salary * growth.sum())


# =============================================================================
# 2. Expected returns and risk (LN3-LN4)
# =============================================================================

def market_weights(params, names):
    """Black-Litterman prior on the version 2 universe (params allocation market_weights)."""
    mw = params["allocation"]["market_weights"]
    sectors = pd.Series(mw["sector_weights"], dtype=float)
    w = pd.Series(0.0, index=names)
    w[sectors.index] = mw["equities_share"] * sectors / sectors.sum()
    for name in ("BOND_GLB_H", "BOND_CORP", "REIT_CH", "GOLD", "COMMOD"):
        w[name] = float(mw.get(name, 0.0))
    return w / w.sum()


def estimates(window_excess, params):
    """Annual covariance (Ledoit-Wolf, LN3) and expected excess returns (Black-Litterman, LN4)
    from a window of monthly excess returns. Returns (mu, cov, info)."""
    names = list(window_excess.columns)
    cov_m, shrinkage = O.ledoit_wolf(window_excess)
    cov = cov_m * MONTHS
    bl = params["black_litterman"]
    w = market_weights(params, names).to_numpy()
    phi = O.implied_risk_aversion(cov.to_numpy(), w, bl["sharpe_ratio"])
    P, Q, omega, used = O.views_matrices(names, params)
    mu = O.bl_expected_returns(cov.to_numpy(), w, phi, P, Q, omega, bl["tau"])
    return pd.Series(mu, index=names), cov, {"phi": float(phi), "shrinkage": shrinkage, "views": used,
                                             "market_weights": pd.Series(w, index=names)}


# =============================================================================
# 3. Risk budgeting with a fixed holding (LN5, CCD algorithm)
# =============================================================================

def risk_budget_amounts(cov, budgets, fixed, total, sweeps=200, tol=1e-10):
    """CHF amounts y >= 0 to buy, with sum(y) = total, such that y_i (Sigma e)_i is proportional to
    the budget b_i, where e = y + fixed. Assets with a zero budget are not bought.

    cov: annual covariance (DataFrame); budgets, fixed: Series on the same assets (CHF for fixed).
    Cyclical coordinate descent on 1/2 e'Sigma e - lambda sum b ln y: each coordinate solves
    Sigma_ii y^2 + c_i y - lambda b_i = 0, c_i = sum_{j != i} Sigma_ij e_j. lambda by bisection."""
    names = list(cov.columns)
    S = cov.to_numpy()
    b = budgets.reindex(names).fillna(0.0).to_numpy()
    f = fixed.reindex(names).fillna(0.0).to_numpy()
    buy = b > 0
    if total <= 0 or not buy.any():
        return pd.Series(0.0, index=names)
    diag = np.diag(S)

    def solve(lam, y):
        for _ in range(sweeps):
            previous = y.copy()
            for i in np.flatnonzero(buy):
                e = y + f
                c = S[i] @ e - S[i, i] * y[i]
                y[i] = (-c + np.sqrt(c * c + 4 * diag[i] * lam * b[i])) / (2 * diag[i])
            if np.abs(y - previous).sum() <= tol * max(total, 1.0):
                break
        return y

    y = np.where(buy, total / buy.sum(), 0.0)
    scale = float(total) ** 2 * np.mean(diag[buy])           # lambda has units of CHF^2 x variance
    low, high = 1e-12 * scale, 1e3 * scale
    for _ in range(100):
        lam = np.sqrt(low * high)
        y = solve(lam, y)
        if y.sum() > total:
            high = lam
        else:
            low = lam
        if abs(y.sum() - total) <= 1e-7 * total:
            break
    y = y * total / y.sum()                                    # exact total (the error is tiny)
    return pd.Series(y, index=names)


def risk_contributions(weights, cov, idio_var=0.0):
    """Relative risk contributions of each holding to the volatility of the savings (LN5), with the
    stake's company-specific variance (idio_var, in weight units) added to its industry."""
    names = list(cov.columns)
    w = weights.reindex(names).fillna(0.0).to_numpy()
    total_var = w @ cov.to_numpy() @ w + idio_var
    if total_var <= 0:
        return pd.Series(0.0, index=names)
    rc = w * (cov.to_numpy() @ w) / total_var
    return pd.Series(rc, index=names)


def diversification(weights, cov, idio_var=0.0):
    """Diversification measures of LN6: effective number of bets (inverse Herfindahl of the risk
    contributions) and diversification ratio (weighted volatilities / portfolio volatility)."""
    names = list(cov.columns)
    w = weights.reindex(names).fillna(0.0).to_numpy()
    vol = np.sqrt(w @ cov.to_numpy() @ w + idio_var)
    rc = risk_contributions(weights, cov, idio_var).to_numpy()
    rc = rc[rc > 0]
    enb = 1.0 / np.sum(rc ** 2) if len(rc) else 0.0
    ratio = (np.abs(w) @ np.sqrt(np.diag(cov.to_numpy()))) / vol if vol > 0 else 0.0
    return {"effective bets": float(enb), "diversification ratio": float(ratio), "volatility": float(vol)}


# =============================================================================
# 4. The plans (A), (B), (C)
# =============================================================================

def budgets(params, mu, cov, groups_used, exclude=()):
    """Risk budgets of the assets to buy in the groups used: the profile's split between the groups
    (bonds alone: 1), inside a group in proportion to the Black-Litterman Sharpe ratios with a floor."""
    alloc = params["allocation"]
    profile = alloc["profiles"][params["client"]["profile"]]
    sharpe = mu / np.sqrt(np.diag(cov))
    b = pd.Series(0.0, index=list(cov.columns))
    for group in groups_used:
        members = [m for m in alloc["groups"][group] if m not in exclude and m in b.index]
        share = profile.get(group, 1.0) if len(groups_used) > 1 else 1.0
        if share <= 0 or not members:
            continue
        s = sharpe[members].clip(lower=0)
        floor = alloc["sharpe_floor"] * max(s.mean(), 1e-12)
        s = s.clip(lower=floor)
        b[members] = share * s / s.sum()
    return b / b.sum() if b.sum() > 0 else b


def safe_share(params, kind, sheet):
    """Share of the savings in bonds: (A) the profile's safe share; (B) age%; (C) the profile's safe
    share x (1 - bond-like share of wealth)."""
    client = params["client"]
    profile = params["allocation"]["profiles"][client["profile"]]
    if kind == "A":
        return profile["safe_share"]
    if kind == "B":
        return min(max(client["age"] / 100, 0.0), 1.0)
    return float(profile["safe_share"] * min(max(1 - sheet["bond_like"], 0.0), 1.0))


def _locked(params, months_from_now=0.0):
    client = params["client"]
    return client["industry"] is not None and client["stake_chf"] > 0 and \
        months_from_now < MONTHS * float(client["unlock_years"])


def plan(params, mu, cov, kind="C", stake_value=None, months_from_now=0.0, sheet=None, savings_left=None):
    """Weights of the savings (Series on the assets plus CASH, sum 1) for strategy kind A, B or C,
    and a dict with the inputs that explain them.

    stake_value: CHF value of the stake now (default: params); it is held while locked.
    months_from_now: time since today (the backtest uses it for the unlock date).
    sheet, savings_left: balance sheet and future savings (default: computed from params)."""
    alloc, client = params["allocation"], params["client"]
    names = list(cov.columns)
    F = float(client["portfolio_chf"])
    industry = client["industry"]
    stake = float(client["stake_chf"] if stake_value is None else stake_value)
    locked = _locked(params, months_from_now) and stake > 0
    stake = min(stake, F) if locked else 0.0
    sheet = sheet or balance_sheet(params)
    reserve = min(alloc["reserve_months"] / MONTHS * client["salary_chf"], F - stake)
    invest = F - stake - reserve
    fixed = pd.Series(0.0, index=names)
    if locked:
        fixed[industry] = stake
    idio = (stake / F * alloc["stake_idio_vol"]) ** 2 if locked else 0.0

    if kind not in ("A", "B", "C"):
        raise ValueError(f"unknown strategy {kind}")
    exclude = [industry] if kind == "C" and industry and (client["variable_pay"] > 0 or client["stake_chf"] > 0) else []
    d = safe_share(params, kind, sheet)
    to_bonds = min(d * F, invest)
    to_growth = invest - to_bonds
    b_growth = budgets(params, mu, cov, ["equities", "real"], exclude)
    b_bonds = budgets(params, mu, cov, ["bonds"])
    y = risk_budget_amounts(cov, b_growth, fixed, to_growth) + \
        risk_budget_amounts(cov, b_bonds, pd.Series(0.0, index=names), to_bonds)
    b = b_growth * (1 - d) + b_bonds * d
    w = (y + fixed) / F
    cash = reserve / F
    info = {"budgets": b, "reserve": reserve, "stake": stake, "locked": locked, "idio_var": idio,
            "bond_like": sheet["bond_like"], "safe_share": d, "excluded": exclude, "cash_from_loss_limit": 0.0}

    if kind == "C":                                            # loss limit before retirement
        z = norm.ppf(alloc["loss_confidence"])
        tolerance = alloc["profiles"][client["profile"]]["loss_tolerance"]
        left = future_savings(params) if savings_left is None else savings_left
        allowed = tolerance * F + left
        free = w.copy()
        if locked:
            free[industry] -= stake / F

        def loss(scale):
            x = free * scale + (w - free)
            return F * (z * np.sqrt(x @ cov.to_numpy() @ x + idio) - x @ mu)

        if loss(1.0) > allowed:
            low, high = 0.0, 1.0
            for _ in range(60):
                mid = (low + high) / 2
                if loss(mid) <= allowed:
                    low = mid
                else:
                    high = mid
            moved = float(free.sum() * (1 - low))            # share of the savings moved to cash
            w = free * low + (w - free)
            cash += moved
            info["cash_from_loss_limit"] = moved
        info.update(loss=float(loss(1.0) if info["cash_from_loss_limit"] == 0 else allowed), allowed=allowed,
                    savings_left=left, z=float(z))

    weights = pd.concat([w, pd.Series({CASH: cash})])
    weights = weights.clip(lower=0)
    return weights / weights.sum(), info


def risk_by_part(weights, cov, detail, params):
    """Shares of the variance of the savings from: the employer shares (their industry part and their
    company-specific part), the other equities, the real assets and the bonds."""
    names = list(cov.columns)
    w = weights.reindex(names).fillna(0.0)
    S = cov.to_numpy()
    marginal = pd.Series(S @ w.to_numpy(), index=names)
    var = float(w @ marginal) + detail["idio_var"]
    out = {}
    stake_w = detail["stake"] / float(params["client"]["portfolio_chf"]) if detail["locked"] else 0.0
    industry = params["client"]["industry"]
    out["stake"] = (stake_w * marginal[industry] + detail["idio_var"]) / var if stake_w > 0 else 0.0
    for group, members in params["allocation"]["groups"].items():
        part = w[members] * marginal[members]
        if stake_w > 0 and industry in members:
            part[industry] -= stake_w * marginal[industry]
        out[group] = float(part.sum()) / var
    return pd.Series(out)


def groups(weights, params):
    """Weights summed by group: equities, real, bonds, cash."""
    out = {g: float(weights.reindex(m).fillna(0).sum()) for g, m in params["allocation"]["groups"].items()}
    out["cash"] = float(weights.get(CASH, 0.0))
    return pd.Series(out)


def analyze(params, data=None):
    """Everything the app shows about today: the three plans, their risk, the budgets, the balance
    sheet, the expected numbers. data: (returns, rf) of load_data (default: saved data)."""
    returns, rf = load_data(params) if data is None else data
    excess = returns.sub(rf, axis=0)
    mu, cov, info = estimates(excess, params)
    sheet = balance_sheet(params)
    client = params["client"]
    F = float(client["portfolio_chf"])
    out = {"params": params, "mu": mu, "cov": cov, "estimates": info, "sheet": sheet, "plans": {}, "months": len(excess)}
    for kind in ("A", "B", "C"):
        w, detail = plan(params, mu, cov, kind, sheet=sheet)
        x = w.drop(CASH)
        idio = detail["idio_var"]
        out["plans"][kind] = {
            "weights": w, "detail": detail, "groups": groups(w, params),
            "risk_parts": risk_by_part(x, cov, detail, params),
            "risk": risk_contributions(x, cov, idio), "diversification": diversification(x, cov, idio),
            "expected excess return": float(x @ mu), "chf": w * F}
    return out


# =============================================================================
# 5. The plan by age (expected path, no market surprise)
# =============================================================================

def glide_path(params, mu, cov, step_years=None):
    """Plan (C) and (A) solved again at each age until retirement, with the balance sheet expected
    then: salary grown, pension estimated with the legal rules (or scaled from the certificate),
    savings grown at the plan's expected return plus the savings; the stake grows with its industry
    and joins the free savings when it unlocks. Returns a DataFrame by age."""
    client, career = params["client"], params["career"]
    step = step_years or params["app"]["glide_step_years"]
    age0, salary0 = float(client["age"]), float(client["salary_chf"])
    retire = client["retirement_age"]
    sheet0 = balance_sheet(params)
    P_scale = 1.0
    if client.get("pillar2", True) and client.get("pension_chf") not in (None, "auto"):
        legal = O.pension_savings(age0, salary0, params)
        P_scale = sheet0["P"] / legal if legal > 0 else 1.0
    rf = career["rf_long_run"]
    F, stake = float(client["portfolio_chf"]), float(client["stake_chf"])
    industry = client["industry"]
    rows, age = {}, age0
    while age < retire:
        years = age - age0
        salary = salary0 * (1 + career["salary_growth"]) ** years
        p = copy.deepcopy(params)
        p["client"].update(age=age, salary_chf=salary, portfolio_chf=F,
                           stake_chf=stake if years < client["unlock_years"] else 0.0,
                           unlock_years=max(client["unlock_years"] - years, 0.0))
        if client.get("pillar2", True):
            p["client"]["pension_chf"] = O.pension_savings(age, salary, p) * P_scale
        sheet = balance_sheet(p)
        row = {}
        for kind in ("C", "A"):
            w, _ = plan(p, mu, cov, kind, sheet=sheet)
            for g, v in groups(w, p).items():
                row[(kind, g)] = v
            if kind == "C":
                growth = rf + float(w.drop(CASH) @ mu)
        row[("", "savings")] = F
        row[("", "bond-like share")] = sheet["bond_like"]
        rows[round(age, 2)] = row
        # move on: savings grow with the plan's expected return and receive the savings of the period
        saved = params["backtest"]["savings_rate"] * salary * step
        F = F * (1 + growth) ** step + saved
        if industry and stake > 0 and years < client["unlock_years"]:
            stake = stake * (1 + rf + float(mu[industry])) ** step
        age += step
    table = pd.DataFrame(rows).T
    table.index.name = "age"
    return table


# =============================================================================
# 6. Backtest (step 4 engine, version 2 strategies)
# =============================================================================

def backtest(params, data=None, strategies=("A", "B", "C")):
    """Walk-forward backtest of (A), (B), (C) with the engine of step 4 (src/backtest.py): each
    decision uses only the past window; the plans are solved again every rebalance_months and kept
    in between; the stake is held while locked and grows with its industry. Returns the same dict as
    backtest.run_strategies, so its metric tables work unchanged."""
    from src import backtest as B
    returns, rf = load_data(params) if data is None else data
    names = list(returns.columns)
    excess = returns.sub(rf, axis=0)
    window = int(params["backtest"]["window_months"])
    every = int(params["allocation"]["rebalance_months"])
    start = returns.index[window]
    v1_excess = O.load_excess_returns().reindex(returns.index)
    core = [a for a in params["data"]["assets"] if a != O.EMPLOYER]
    gamma, _ = O.risk_tolerance(params, v1_excess[core].iloc[:window].dropna())
    path = B.client_path(params, excess, rf, start)
    industry = params["client"]["industry"]
    stake0 = float(params["client"]["stake_chf"])
    cache = {}

    def make(kind):
        state = {"stake": stake0, "last": None, "weights": None, "count": 0}

        def strategy(window_excess, date, info):
            client, value = info["client"], info["value"]
            t = returns.index.get_loc(date)
            if state["last"] is not None and industry:                 # stake grows with its industry
                state["stake"] *= 1 + returns[industry].iloc[t - 1]
            state["last"] = date
            months = state["count"]
            state["count"] += 1
            if months % every and state["weights"] is not None:
                return state["weights"]
            if date not in cache:
                cache[date] = estimates(window_excess[names], params)
            mu, cov, _ = cache[date]
            p = copy.deepcopy(params)
            p["client"].update(age=float(client["age"]), salary_chf=float(client["salary"]), portfolio_chf=value,
                               pension_chf=float(client["P_start"]))
            H, P = float(client["H_start"]), float(client["P_start"])
            W = H + P + value
            sheet = {"H": H, "P": P, "F": value, "W": W, "bond_like": (client["beta_bond"] * H + P) / W}
            left = future_savings(p)
            w, _ = plan(p, mu, cov, kind, stake_value=min(state["stake"], value), months_from_now=months,
                        sheet=sheet, savings_left=left)
            if not _locked(params, months):
                state["stake"] = 0.0
            state["weights"] = w.drop(CASH).reindex(names).to_numpy()
            return state["weights"]
        return strategy

    results = {k: B.run_backtest(returns[names], rf, make(k), params, path) for k in strategies}
    return {"results": results, "path": path, "gamma": gamma, "excess": excess, "rf": rf, "params": params}


def backtest_summary(params, data=None):
    """The backtest reduced to what the app shows (small enough to keep on disk)."""
    from src import backtest as B
    run = backtest(params, data)
    pairs = (("C", "A"), ("C", "B"), ("B", "A"))
    return {"values": pd.DataFrame({k: r["value"] for k, r in run["results"].items()}),
            "weights": {k: r[[c for c in r.columns if c.startswith("w_")]].rename(columns=lambda c: c[2:])
                        for k, r in run["results"].items()},
            "gains": B.gain_table(run, pairs), "metrics": B.metrics_table(run).rename(
                columns={B.STRATEGIES[k]: STRATEGIES[k] for k in run["results"]}),
            "crises": B.crisis_table(run), "gamma": run["gamma"],
            "savings": float(run["path"]["savings"].sum()), "start_age": float(run["path"]["age"].iloc[0])}
