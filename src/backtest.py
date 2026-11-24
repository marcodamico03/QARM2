"""
Walk-forward backtest (proposal section 4).

At every month-end only past data is used: rolling 60-month window -> Ledoit-Wolf + Black-Litterman,
gamma re-calibrated to 60/40, re-optimisation, weights held for one month.
The client's balance sheet moves every month:

    career   H_t = PV(remaining salary at age t) * S_t
             S_t = S_{t-1} (1 + surprise_t),  surprise = beta_B (R_B - bond premium) + beta_S_true (R_S - sector premium)
             (when bank stocks fall more than expected, the market value of a banking career falls)
    pension  P_t = P_{t-1} (1 + BVG rate / 12) + savings credit
    portfolio F_t = F_{t-1} (1 + rf + x'R) - costs + savings

Return on total wealth:  r_W = (H r_H + P r_P + F r_F - lost salary) / W
'beta_S_true' can differ from the beta_S used to build the hedge: that is the wrong-beta test (4.2).
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

import career
from config import (BACKTEST_WINDOW, BOND_PREMIUM, BVG_COORDINATION_DEDUCTION, BVG_MIN_INTEREST,
                    CAREER_BOND_ASSET, EQUITY_ASSETS, JOB_LOSS_MONTHS, POSITION_CAPS,
                    SECTOR_PREMIUM, TARGET_EQUITY_SHARE, TRANSACTION_COST_BP)
from estimation import MarketInputs, estimate_inputs
from optimizer import Constraints, rule_100_minus_age
from personas import Persona

STRATEGIES = ["A", "B", "C", "D"]
LABELS = {"A": "A standard", "B": "B 100-age", "C": "C total wealth", "D": "D no hedge"}


# ---------------------------------------------------------------- fast re-usable solver
class FastSolver:
    """Same problem as optimizer.solve, written with cvxpy Parameters so that it is compiled once
    and re-solved thousands of times quickly.   max c1'x - ||c2' x||^2  s.t. constraints."""

    def __init__(self, assets: list, cons: Constraints):
        n = len(assets)
        self.assets = assets
        self.x = cp.Variable(n)
        self.c1 = cp.Parameter(n)
        self.c2 = cp.Parameter((n, n))
        c = []
        if cons.long_only:
            c.append(self.x >= 0)
        if cons.no_leverage:
            c.append(cp.sum(self.x) <= 1)
        for a, cap in cons.caps.items():
            if a in assets:
                c.append(self.x[assets.index(a)] <= cap)
        for a, low in cons.lockup.items():
            if a in assets:
                c.append(self.x[assets.index(a)] >= low)
        self.prob = cp.Problem(cp.Maximize(self.c1 @ self.x - cp.sum_squares(self.c2.T @ self.x)), c)

    def solve(self, m: MarketInputs, gamma, omega_H, omega, sigma_H) -> np.ndarray:
        a = 1 - omega
        L = np.linalg.cholesky(m.Sigma.values)
        self.c1.value = a * m.mu.values - gamma * a * omega_H * np.asarray(sigma_H)
        self.c2.value = np.sqrt(gamma / 2) * a * L
        self.prob.solve(solver=cp.CLARABEL)
        w = np.asarray(self.x.value).copy()
        w[np.abs(w) < 1e-7] = 0.0
        return np.clip(w, 0, None)


def _calibrate_gamma(solver: FastSolver, m: MarketInputs, target=TARGET_EQUITY_SHARE,
                     lo=0.1, hi=100.0) -> float:
    eq = [m.assets.index(a) for a in EQUITY_ASSETS]
    zero = np.zeros(len(m.assets))
    while hi / lo > 1.002:
        mid = np.sqrt(lo * hi)
        if solver.solve(m, mid, 0.0, 0.0, zero)[eq].sum() > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ---------------------------------------------------------------- inputs at each date
@dataclass
class Precomputed:
    dates: list                # decision months t (weights used DURING month t)
    inputs: dict               # t -> MarketInputs estimated on [t-60, t-1]
    gamma: dict                # t -> calibrated gamma


def precompute(excess: pd.DataFrame, assets: list, window: int = BACKTEST_WINDOW,
               start=None, end=None) -> Precomputed:
    """Estimate inputs for every month using ONLY the previous `window` months (no look-ahead)."""
    X = excess[assets].dropna(how="all")
    first_full = X.dropna().index[0]
    dates = [t for t in X.index if t - window >= first_full]
    if start:
        dates = [t for t in dates if t >= pd.Period(start, "M")]
    if end:
        dates = [t for t in dates if t <= pd.Period(end, "M")]
    cal = FastSolver(assets, Constraints())
    inputs, gamma = {}, {}
    for t in dates:
        m = estimate_inputs(excess, assets, start=t - window, end=t - 1)
        inputs[t] = m
        gamma[t] = _calibrate_gamma(cal, m)
    return Precomputed(dates, inputs, gamma)


# ---------------------------------------------------------------- one run
def run(strategy: str, p: Persona, pre: Precomputed, excess: pd.DataFrame, rates: pd.DataFrame,
        beta_S_true: float | None = None, job_loss_threshold: float | None = None,
        cost_bp: float = TRANSACTION_COST_BP) -> pd.DataFrame:
    """Simulate the client month by month under one strategy. Returns one row per month."""
    assets = pre.inputs[pre.dates[0]].assets
    bS_true = p.beta_S if beta_S_true is None else beta_S_true
    solver = FastSolver(assets, Constraints.for_persona(p)) if strategy in "ACD" else None

    ages = np.arange(p.age, p.retirement_age + 1)
    pv = np.array([career.human_capital(p, a) for a in ages])
    pv_at = lambda age: float(np.interp(age, ages, pv))     # PV of remaining salary at any age
    shock = 1.0                                               # career shock index S_t
    H = pv_at(p.age)
    P, F = p.pension, p.portfolio
    x_prev = np.zeros(len(assets))
    sector_index = 1.0
    sector_hist = []      # drawdown measured from the highest level of the last 24 months
    job_loss_left = 0
    armed = True          # one job loss per crisis: re-armed once the sector recovers half-way
    rows = []

    for k, t in enumerate(pre.dates):
        age = p.age + k / 12
        m, gamma = pre.inputs[t], pre.gamma[t]
        W = H + P + F
        omega_H, omega_P = H / W, P / W

        # ---- decision at the end of month t-1
        if strategy == "B":
            x = rule_100_minus_age(int(age), m).weights.values
        else:
            sigma_H = career.career_covariance(m.Sigma, p).values if strategy == "C" else np.zeros(len(assets))
            om_H, om = (0.0, 0.0) if strategy == "A" else (omega_H, omega_H + omega_P)
            x = solver.solve(m, gamma, om_H, om, sigma_H)

        # ---- what happens during month t
        R = excess.loc[t, assets].fillna(0.0).values
        rf = float(rates.loc[t, "rf_chf"])
        turnover = np.abs(x - x_prev).sum()
        cost = cost_bp / 1e4 * turnover
        r_F = rf + x @ R - cost

        R_B, R_S = float(excess.loc[t, CAREER_BOND_ASSET]), float(excess.loc[t, p.sector_asset])
        sector_part = bS_true * R_S
        surprise = p.beta_B * (R_B - BOND_PREMIUM / 12) + bS_true * (R_S - SECTOR_PREMIUM / 12)
        r_P = BVG_MIN_INTEREST / 12

        salary = p.salary * (1 + p.salary_growth) ** (age - p.age) / 12
        sector_index *= 1 + R_S + rf
        sector_hist = (sector_hist + [sector_index])[-24:]
        sector_peak = max(sector_hist)
        drawdown = sector_index / sector_peak - 1
        if job_loss_threshold is not None:
            if armed and job_loss_left == 0 and drawdown < -job_loss_threshold:
                job_loss_left, armed = JOB_LOSS_MONTHS, False
            elif not armed and drawdown > -job_loss_threshold / 2:
                armed = True
        employed = job_loss_left == 0
        if job_loss_left:
            job_loss_left -= 1
        paid = salary if employed else 0.0
        lost = salary - paid

        shock *= 1 + surprise
        H_new = pv_at(age + 1 / 12) * shock
        r_H = (H_new + salary) / H - 1 if H > 0 else 0.0
        r_W = (H * r_H + P * r_P + F * r_F - lost) / W

        credit = career.bvg_credit_rate(int(age)) * max(paid * 12 - BVG_COORDINATION_DEDUCTION, 0) / 12
        H = H_new
        P = P * (1 + r_P) + credit
        F_before = F
        F = F * (1 + r_F) + p.savings_rate * paid

        gross = 1 + rf + x @ R
        x_prev = x * (1 + rf + R) / gross if gross > 0 else x     # weights drift during the month

        rows.append({"date": t, "age": age, "sector_drawdown": drawdown, "H": H, "P": P, "F": F, "W": H + P + F,
                     "omega": omega_H + omega_P, "r_F": r_F, "r_W": r_W, "rf": rf,
                     "pnl_F": F_before * (r_F - rf), "sector_part": sector_part,
                     "turnover": turnover, "employed": employed, "gamma": gamma,
                     **{f"w_{a}": w for a, w in zip(assets, x)}})
    return pd.DataFrame(rows).set_index("date")


def run_all(p: Persona, pre: Precomputed, excess, rates, **kw) -> dict:
    return {s: run(s, p, pre, excess, rates, **kw) for s in STRATEGIES}


# ---------------------------------------------------------------- metrics
CRISES = {
    "2008 GFC": ("2007-11", "2009-02"),
    "2011 Euro crisis": ("2011-05", "2011-09"),
    "Jan 2015 CHF shock": ("2015-01", "2015-01"),
    "2020 COVID": ("2020-02", "2020-03"),
    "2022 rates shock": ("2022-01", "2022-09"),
}


def certainty_equivalent(r: pd.Series, gamma: float) -> float:
    """Annual mean - gamma/2 * annual variance (what the strategy maximises)."""
    return float(r.mean() * 12 - gamma / 2 * r.var() * 12)


def max_drawdown(r: pd.Series) -> float:
    idx = (1 + r).cumprod()
    return float((idx / idx.cummax() - 1).min())


def metrics(res: dict) -> pd.DataFrame:
    gamma = float(np.mean(res["A"]["gamma"]))
    base = res["A"]
    out = {}
    for s, df in res.items():
        worst = df["sector_part"] <= df["sector_part"].quantile(0.10)
        ce = certainty_equivalent(df["r_W"], gamma)
        out[LABELS[s]] = {
            "CE return on total wealth": ce,
            "CE gain vs A (CHF/yr)": (ce - certainty_equivalent(base["r_W"], gamma)) * df["W"].mean(),
            "corr(portfolio, sector part of career)": (df["r_F"].corr(df["sector_part"])
                                                       if df["sector_part"].std() > 0 else np.nan),
            "avg portfolio P&L in worst 10% sector months (CHF)": df.loc[worst, "pnl_F"].mean(),
            "vol of total wealth": df["r_W"].std() * np.sqrt(12),
            "max drawdown of total wealth": max_drawdown(df["r_W"]),
            "vol of portfolio": df["r_F"].std() * np.sqrt(12),
            "max drawdown of portfolio": max_drawdown(df["r_F"]),
            "turnover per year": df["turnover"].mean() * 12,
            "tracking error vs A": (df["r_F"] - base["r_F"]).std() * np.sqrt(12),
            "final portfolio F (CHF)": df["F"].iloc[-1],
        }
    return pd.DataFrame(out)


def crisis_table(res: dict, what: str = "pnl_F") -> pd.DataFrame:
    """CHF profit/loss in each crisis window (portfolio by default), computed in the backtest."""
    out = {}
    for s, df in res.items():
        out[LABELS[s]] = {c: df.loc[a:b, what].sum() for c, (a, b) in CRISES.items()
                          if pd.Period(a, "M") >= df.index[0]}
    return pd.DataFrame(out)


# ---------------------------------------------------------------- wrong-beta test (4.2)
def wrong_beta_curve(p: Persona, pre: Precomputed, excess, rates,
                     multipliers=(0.0, 0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Hedge built with the assumed beta_S; career simulated with beta_S_true = m * beta_S.
    The CE gain of C over D is split into
      * risk benefit   = -gamma/2 * (Var_C - Var_D)   (what the hedge is designed to deliver)
      * return effect  = mean_C - mean_D              (depends on the one historical path)"""
    gamma = float(np.mean(list(pre.gamma.values())))
    rows = []
    for mult in multipliers:
        bt = mult * p.beta_S
        c = run("C", p, pre, excess, rates, beta_S_true=bt)
        d = run("D", p, pre, excess, rates, beta_S_true=bt)
        risk = -gamma / 2 * (c["r_W"].var() - d["r_W"].var()) * 12 * 1e4
        ret = (c["r_W"].mean() - d["r_W"].mean()) * 12 * 1e4
        rows.append({"true beta / assumed beta": mult,
                     "risk benefit C vs D (bp/yr)": risk,
                     "return effect C vs D (bp/yr)": ret,
                     "CE gain C vs D (bp/yr)": risk + ret})
    return pd.DataFrame(rows).set_index("true beta / assumed beta")


def theoretical_hedge_gain(h: float, beta_true: np.ndarray) -> np.ndarray:
    """Variance reduction of a hedge of size h against a true exposure beta: h (2 beta - h)
    (in units of the sector variance). Positive only when 0 < h < 2 beta."""
    return h * (2 * np.asarray(beta_true) - h)


def sample_hedge_gain(sector: pd.Series, h: float, beta_true: float) -> float:
    """The same quantity measured on data: Var(beta s) - Var((beta - h) s), divided by Var(s).
    Used to check the code against the formula."""
    s = sector.dropna()
    return float((np.var(beta_true * s) - np.var((beta_true - h) * s)) / np.var(s))
