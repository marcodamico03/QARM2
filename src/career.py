"""
Career model (proposal section 3.1).

    R_H - rf = beta_B * (R_B - rf) + beta_S * (R_S - rf) + eps

* human_capital()      value H of the remaining salary (CHF)
* pension_projection() 2nd-pillar account P by age (CHF)
* balance_sheet()      H, P, F and total wealth W by age (Figure 1 of the proposal)
* career_excess_returns() monthly R_H - rf from the asset data
* career_covariance()  sigma_H = Cov(R, R_H), the input of the hedging demand
"""
import numpy as np
import pandas as pd

from config import (BOND_PREMIUM, BVG_COORDINATION_DEDUCTION, BVG_CREDIT_RATES,
                    BVG_MIN_INTEREST, CAREER_BOND_ASSET, PORTFOLIO_GROWTH,
                    RF_LONG_RUN, SECTOR_PREMIUM)
from personas import Persona


# ---------------------------------------------------------------- value of the career
def discount_rate(p: Persona) -> float:
    """rf + beta_B * bond premium + beta_S * sector premium (same betas as the risk)."""
    return RF_LONG_RUN + p.beta_B * BOND_PREMIUM + p.beta_S * SECTOR_PREMIUM


def salary_path(p: Persona, age: int | None = None) -> pd.Series:
    """Expected salary for each remaining working year, starting at `age`."""
    age = p.age if age is None else age
    ages = np.arange(age, p.retirement_age)
    salaries = p.salary * (1 + p.salary_growth) ** (ages - p.age)
    return pd.Series(salaries, index=pd.Index(ages, name="age"), name="salary")


def human_capital(p: Persona, age: int | None = None) -> float:
    """H at `age`: present value of the remaining salaries (paid at the end of each year)."""
    s = salary_path(p, age)
    if s.empty:
        return 0.0
    t = np.arange(1, len(s) + 1)
    return float((s.values / (1 + discount_rate(p)) ** t).sum())


# ---------------------------------------------------------------- pension (2nd pillar)
def bvg_credit_rate(age: int) -> float:
    rate = 0.0
    for start, r in sorted(BVG_CREDIT_RATES.items()):
        if age >= start:
            rate = r
    return rate


def pension_projection(p: Persona) -> pd.Series:
    """2nd-pillar account from today to retirement: interest + age-dependent savings credits.
    Assumption: the whole salary above the coordination deduction is insured
    (typical for banks, which offer more than the legal minimum)."""
    ages = np.arange(p.age, p.retirement_age + 1)
    P = np.empty(len(ages))
    P[0] = p.pension
    for i in range(1, len(ages)):
        prev_age = ages[i - 1]
        salary = p.salary * (1 + p.salary_growth) ** (prev_age - p.age)
        insured = max(salary - BVG_COORDINATION_DEDUCTION, 0.0)
        P[i] = P[i - 1] * (1 + BVG_MIN_INTEREST) + bvg_credit_rate(prev_age) * insured
    return pd.Series(P, index=pd.Index(ages, name="age"), name="P")


# ---------------------------------------------------------------- balance sheet
def balance_sheet(p: Persona) -> pd.DataFrame:
    """H, P, F, W = H + P + F and omega = (H + P) / W for every age until retirement.
    F grows at PORTFOLIO_GROWTH plus yearly savings (a projection, not an optimisation)."""
    ages = np.arange(p.age, p.retirement_age + 1)
    P = pension_projection(p)
    F = np.empty(len(ages))
    F[0] = p.portfolio
    for i in range(1, len(ages)):
        salary = p.salary * (1 + p.salary_growth) ** (ages[i - 1] - p.age)
        F[i] = F[i - 1] * (1 + PORTFOLIO_GROWTH) + p.savings_rate * salary
    df = pd.DataFrame({
        "H": [human_capital(p, a) for a in ages],
        "P": P.values,
        "F": F,
    }, index=pd.Index(ages, name="age"))
    df["W"] = df[["H", "P", "F"]].sum(axis=1)
    df["omega_H"] = df["H"] / df["W"]
    df["omega_P"] = df["P"] / df["W"]
    df["omega"] = df["omega_H"] + df["omega_P"]
    return df


# ---------------------------------------------------------------- risk of the career
def career_betas(p: Persona, assets) -> pd.Series:
    """Vector b with beta_B on the bond asset and beta_S on the sector asset, 0 elsewhere."""
    b = pd.Series(0.0, index=list(assets))
    b[CAREER_BOND_ASSET] += p.beta_B
    b[p.sector_asset] += p.beta_S
    return b


def career_excess_returns(excess: pd.DataFrame, p: Persona) -> pd.Series:
    """Monthly R_H - rf implied by the model (eps left out: it is not hedgeable)."""
    return (p.beta_B * excess[CAREER_BOND_ASSET] + p.beta_S * excess[p.sector_asset]).rename("career")


def career_covariance(Sigma: pd.DataFrame, p: Persona) -> pd.Series:
    """sigma_H = Cov(R, R_H) = Sigma @ b. Then Sigma^-1 sigma_H = b exactly (proposal 3.2)."""
    return (Sigma @ career_betas(p, Sigma.columns)).rename("sigma_H")


def hedging_demand_unconstrained(p: Persona, omega_H: float, omega: float, assets) -> pd.Series:
    """Career part of x* without constraints (proposal 3.2, with pension):
    -omega_H / (1 - omega) * (beta_B e_B + beta_S e_S), where omega = (H + P) / W.
    The pension is bond-like with sigma_P ~ 0, so it adds no hedging term."""
    return -omega_H / (1 - omega) * career_betas(p, assets)
