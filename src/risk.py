"""
Euler risk contributions on TOTAL wealth (proposal: "how much of your risk is your job?").
Positions: (1-omega) x in each asset, omega_H in the career, omega_P in the pension (riskless).
"""
import numpy as np
import pandas as pd

import career
from estimation import MarketInputs
from optimizer import ClientState
from personas import Persona


def total_wealth_risk_contributions(x: pd.Series, p: Persona, st: ClientState,
                                    inputs: MarketInputs) -> pd.Series:
    """Share of the variance of total wealth coming from each asset, the career and the pension.
    Sums to 1."""
    S = inputs.Sigma
    b = career.career_betas(p, S.columns)
    sH = (S @ b).values
    var_H = float(b @ S @ b)
    n = len(S)
    Om = np.zeros((n + 2, n + 2))            # assets..., career, pension
    Om[:n, :n] = S.values
    Om[:n, n] = Om[n, :n] = sH
    Om[n, n] = var_H
    v = np.concatenate([(1 - st.omega) * x.values, [st.omega_H, st.omega_P]])
    marg = Om @ v
    total = v @ marg
    rc = v * marg / total
    return pd.Series(rc, index=list(S.columns) + ["career", "pension"], name="risk share")


def portfolio_risk_contributions(x: pd.Series, inputs: MarketInputs) -> pd.Series:
    """Classic Euler contributions inside the portfolio only (what a standard tool shows)."""
    S = inputs.Sigma.values
    marg = S @ x.values
    return pd.Series(x.values * marg / (x.values @ marg), index=x.index, name="risk share")
