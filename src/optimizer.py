"""
Phase 1: the total-wealth portfolio (proposal 3.2).

Total wealth W = H + P + F. The optimizer chooses the risky weights x of the portfolio F
(the rest, 1 - sum(x), earns the CHF risk-free rate). With omega = (H + P) / W:

    max_x  (1-omega) x'mu - gamma/2 * [ (1-omega)^2 x'Sigma x + 2 (1-omega) omega_H x'sigma_H ]

    x* = 1/(gamma (1-omega)) Sigma^-1 mu  -  omega_H/(1-omega) Sigma^-1 sigma_H
         (speculative demand)              (hedging demand)

Strategies (proposal 4.1):
    A  standard MVO on the portfolio only (career and pension ignored)
    B  "100 - age" rule
    C  total-wealth portfolio (our strategy)
    D  C without the hedging term (only the "your job is bond-like" effect)
"""
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

import career
from config import BOND_ASSETS, EQUITY_ASSETS, POSITION_CAPS, TARGET_EQUITY_SHARE
from estimation import MarketInputs
from personas import Persona


# ---------------------------------------------------------------- client state
@dataclass(frozen=True)
class ClientState:
    age: int
    H: float
    P: float
    F: float
    W: float
    omega_H: float
    omega_P: float

    @property
    def omega(self) -> float:
        return self.omega_H + self.omega_P


def client_state(p: Persona, age: int | None = None) -> ClientState:
    age = p.age if age is None else age
    row = career.balance_sheet(p).loc[age]
    return ClientState(age, row.H, row.P, row.F, row.W, row.omega_H, row.omega_P)


# ---------------------------------------------------------------- constraints
@dataclass
class Constraints:
    long_only: bool = True
    no_leverage: bool = True
    caps: dict = field(default_factory=lambda: dict(POSITION_CAPS))
    lockup: dict = field(default_factory=dict)    # {"ubs": 0.10} minimum weights

    @classmethod
    def for_persona(cls, p: Persona) -> "Constraints":
        lock = {p.employer_asset: p.employer_lockup} if p.employer_asset and p.employer_lockup else {}
        return cls(lockup=lock)

    def without(self, name: str) -> "Constraints":
        """Copy with one constraint removed (for its shadow price)."""
        c = Constraints(self.long_only, self.no_leverage, dict(self.caps), dict(self.lockup))
        if name == "long_only":
            c.long_only = False
        elif name == "no_leverage":
            c.no_leverage = False
        elif name == "caps":
            c.caps = {}
        elif name == "lockup":
            c.lockup = {}
        return c


# ---------------------------------------------------------------- the problem
@dataclass
class Solution:
    weights: pd.Series      # risky weights x (share of the portfolio F)
    cash: float             # 1 - sum(x), earns the risk-free rate
    utility: float          # certainty-equivalent excess return of TOTAL wealth (annual)
    duals: dict             # constraint name -> dual value(s), utility units
    status: str


def utility(x, inputs: MarketInputs, gamma, omega_H, omega, sigma_H) -> float:
    x = np.asarray(x)
    mu, S, sH = inputs.mu.values, inputs.Sigma.values, np.asarray(sigma_H)
    a = 1 - omega
    return float(a * x @ mu - gamma / 2 * (a ** 2 * x @ S @ x + 2 * a * omega_H * x @ sH))


def closed_form(inputs: MarketInputs, gamma, omega_H, omega, sigma_H) -> tuple[pd.Series, pd.Series]:
    """Unconstrained x* split into (speculative, hedging) demand."""
    S = inputs.Sigma.values
    spec = np.linalg.solve(S, inputs.mu.values) / (gamma * (1 - omega))
    hedge = -omega_H / (1 - omega) * np.linalg.solve(S, np.asarray(sigma_H))
    return (pd.Series(spec, index=inputs.assets, name="speculative"),
            pd.Series(hedge, index=inputs.assets, name="hedging"))


def solve(inputs: MarketInputs, gamma, omega_H, omega, sigma_H, cons: Constraints) -> Solution:
    n, assets = len(inputs.assets), inputs.assets
    x = cp.Variable(n)
    a = 1 - omega
    S = cp.psd_wrap(inputs.Sigma.values)
    obj = a * inputs.mu.values @ x - gamma / 2 * (
        a ** 2 * cp.quad_form(x, S) + 2 * a * omega_H * np.asarray(sigma_H) @ x)
    named = {}
    if cons.long_only:
        named["long_only"] = x >= 0
    if cons.no_leverage:
        named["no_leverage"] = cp.sum(x) <= 1
    for asset, cap in cons.caps.items():
        if asset in assets:
            named[f"cap:{asset}"] = x[assets.index(asset)] <= cap
    for asset, low in cons.lockup.items():
        if asset in assets:
            named[f"lockup:{asset}"] = x[assets.index(asset)] >= low
    prob = cp.Problem(cp.Maximize(obj), list(named.values()))
    prob.solve(solver=cp.CLARABEL)
    w = pd.Series(x.value, index=assets)
    w[w.abs() < 1e-7] = 0.0                  # solver noise like -1e-9 -> exactly 0
    if cons.long_only:
        w = w.clip(lower=0.0)
    duals = {k: (pd.Series(c.dual_value, index=assets) if k == "long_only" else float(c.dual_value))
             for k, c in named.items()}
    return Solution(w, 1 - w.sum(), utility(w.values, inputs, gamma, omega_H, omega, sigma_H),
                    duals, prob.status)


# ---------------------------------------------------------------- strategies
def strategy(name: str, p: Persona, inputs: MarketInputs, gamma: float,
             age: int | None = None, cons: Constraints | None = None) -> Solution:
    """Weights of strategy A, B, C or D for persona p at `age`."""
    st = client_state(p, age)
    cons = cons or Constraints.for_persona(p)
    zero = np.zeros(len(inputs.assets))
    if name == "A":
        return solve(inputs, gamma, 0.0, 0.0, zero, cons)
    if name == "B":
        return rule_100_minus_age(st.age, inputs)
    sigma_H = career.career_covariance(inputs.Sigma, p).values
    if name == "C":
        return solve(inputs, gamma, st.omega_H, st.omega, sigma_H, cons)
    if name == "D":
        return solve(inputs, gamma, st.omega_H, st.omega, zero, cons)
    raise ValueError(name)


def rule_100_minus_age(age: int, inputs: MarketInputs) -> Solution:
    """Equity share = (100 - age)%, split by market weights; the rest in bonds."""
    w = pd.Series(0.0, index=inputs.assets)
    eq = inputs.w_mkt[EQUITY_ASSETS] / inputs.w_mkt[EQUITY_ASSETS].sum()
    bd = inputs.w_mkt[BOND_ASSETS] / inputs.w_mkt[BOND_ASSETS].sum()
    share = np.clip((100 - age) / 100, 0, 1)
    w[EQUITY_ASSETS] = share * eq
    w[BOND_ASSETS] = (1 - share) * bd
    return Solution(w, 0.0, np.nan, {}, "rule")


# ---------------------------------------------------------------- calibration of gamma
def equity_share(w: pd.Series) -> float:
    return float(w[EQUITY_ASSETS].sum())


def calibrate_gamma(inputs: MarketInputs, target: float = TARGET_EQUITY_SHARE,
                    cons: Constraints | None = None, lo=0.1, hi=100.0) -> float:
    """gamma such that a client with NO career and NO pension holds `target` in equities
    (proposal 3.2: one gamma for all strategies)."""
    cons = cons or Constraints()
    zero = np.zeros(len(inputs.assets))
    f = lambda g: equity_share(solve(inputs, g, 0.0, 0.0, zero, cons).weights) - target
    for _ in range(60):                       # bisection on log(gamma)
        mid = np.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0005:
            break
    return float(np.sqrt(lo * hi))


# ---------------------------------------------------------------- shadow prices
def constraint_costs(p: Persona, inputs: MarketInputs, gamma: float,
                     age: int | None = None) -> pd.DataFrame:
    """What each constraint costs the client (strategy C):
    re-solve without it and compare certainty-equivalent returns of total wealth."""
    st = client_state(p, age)
    cons = Constraints.for_persona(p)
    sigma_H = career.career_covariance(inputs.Sigma, p).values
    base = solve(inputs, gamma, st.omega_H, st.omega, sigma_H, cons)
    labels = {"long_only": "No short selling", "no_leverage": "No borrowing",
              "caps": "Position caps", "lockup": "Deferred-share lock-up"}
    rows = []
    for key, label in labels.items():
        if key == "lockup" and not cons.lockup:
            continue
        relaxed = solve(inputs, gamma, st.omega_H, st.omega, sigma_H, cons.without(key))
        gain = max(relaxed.utility - base.utility, 0.0)
        rows.append({"constraint": label, "cost (bp/yr of total wealth)": gain * 1e4,
                     "cost (CHF/yr)": gain * st.W})
    return pd.DataFrame(rows).set_index("constraint")
