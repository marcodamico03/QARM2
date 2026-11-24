"""
Inputs of the optimizer (proposal 3.2): covariance, expected returns, risk aversion.

* ledoit_wolf_cov()   shrunk covariance matrix (LN3)
* equilibrium returns pi = delta * Sigma * w_mkt with the MARKET's delta (not the client's gamma)
* black_litterman()   prior + optional client views (LN4)
* estimate_inputs()   everything above in one call, annualised
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import INVESTABLE, MARKET_EXCESS_RETURN, MARKET_WEIGHTS, TAU


@dataclass(frozen=True)
class MarketInputs:
    assets: list
    Sigma: pd.DataFrame    # annual covariance of excess returns
    pi: pd.Series          # annual equilibrium excess returns (prior)
    mu: pd.Series          # annual expected excess returns used by the optimizer (BL posterior)
    delta: float           # market risk aversion
    w_mkt: pd.Series       # market weights of the prior
    sample: str            # estimation window, for display


def ledoit_wolf_cov(X: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage of the sample covariance (same units as X)."""
    lw = LedoitWolf().fit(X.values)
    return pd.DataFrame(lw.covariance_, index=X.columns, columns=X.columns)


def market_weights(assets) -> pd.Series:
    w = pd.Series({a: MARKET_WEIGHTS[a] for a in assets})
    return w / w.sum()


def implied_delta(Sigma: pd.DataFrame, w: pd.Series, premium: float = MARKET_EXCESS_RETURN) -> float:
    """delta = market excess return / market variance (annual units)."""
    return premium / float(w @ Sigma @ w)


def black_litterman(Sigma: pd.DataFrame, pi: pd.Series, P=None, Q=None, Omega=None,
                    tau: float = TAU) -> pd.Series:
    """Posterior expected returns. P: k x n view matrix, Q: k views (annual).
    No views -> returns pi. Omega defaults to diag(tau * P Sigma P') (He-Litterman)."""
    if P is None or len(Q) == 0:
        return pi.copy()
    P, Q = np.atleast_2d(np.asarray(P, float)), np.asarray(Q, float)
    S, p = Sigma.values, pi.values
    if Omega is None:
        Omega = np.diag(np.diag(tau * P @ S @ P.T))
    tS = tau * S
    mid = np.linalg.inv(P @ tS @ P.T + Omega)
    post = p + tS @ P.T @ mid @ (Q - P @ p)
    return pd.Series(post, index=pi.index)


def view_matrix(views: list[dict], assets) -> tuple[np.ndarray, np.ndarray]:
    """views = [{"long": "smi", "short": "world_eq", "q": 0.01}, {"long": "gold", "q": 0.03}]
    'q' is an annual excess return (absolute view) or outperformance (relative view)."""
    P = np.zeros((len(views), len(assets)))
    Q = np.zeros(len(views))
    idx = {a: i for i, a in enumerate(assets)}
    for k, v in enumerate(views):
        P[k, idx[v["long"]]] = 1.0
        if v.get("short"):
            P[k, idx[v["short"]]] = -1.0
        Q[k] = v["q"]
    return P, Q


def estimate_inputs(excess: pd.DataFrame, assets=INVESTABLE, start=None, end=None,
                    views: list[dict] | None = None) -> MarketInputs:
    """Annual Sigma (Ledoit-Wolf), prior pi and BL posterior mu on the window [start, end]."""
    X = excess.loc[start:end, list(assets)].dropna()
    Sigma = ledoit_wolf_cov(X) * 12
    w = market_weights(assets)
    delta = implied_delta(Sigma, w)
    pi = (delta * Sigma @ w).rename("pi")
    if views:
        P, Q = view_matrix(views, assets)
        mu = black_litterman(Sigma, pi, P, Q)
    else:
        mu = pi.copy()
    sample = f"{X.index[0]} to {X.index[-1]} ({len(X)} months)"
    return MarketInputs(list(assets), Sigma, pi, mu.rename("mu"), delta, w, sample)
