"""Optimization engine (contract, section 3).

Built step by step, reusing the professor's lab code and notation (contract, section 1):
    1. Black-Litterman expected returns (lab LN4)
    2. Covariance on our data: sample and Ledoit-Wolf (labs LN1, LN3)
    3. Calibration of gamma (a client with no career holds 60/40)
    4. Total-wealth optimization with the career (proposal, section 3.2)
    5. Shadow prices and risk contributions (lab LN5)

Notation (contract, section 1.2 and 1.3):
    gamma   risk TOLERANCE, as in the labs: QP(x) = 0.5 x' Sigma x - gamma x' mu
    phi     implied risk aversion of the market, phi = 1 / gamma
    tau, P, Q, omega_views   Black-Litterman inputs, as in lab LN4
Everything is in excess returns over the risk-free rate, unless stated otherwise.
"""
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, fsolve, minimize


def _array(a):
    """Accept numpy arrays or pandas objects."""
    return np.asarray(a, dtype=float)


# =============================================================================
# 1. Black-Litterman (lab LN4, lecture notes 4 slides 6 to 18)
# =============================================================================

def portfolio_volatility(x, cov):
    """sigma(x) = sqrt(x' Sigma x)."""
    x, cov = _array(x), _array(cov)
    return float(np.sqrt(x @ cov @ x))


def implied_risk_aversion(cov, x0, sharpe):
    """phi = SR(x0) / sigma(x0): the risk aversion that makes x0 optimal (LN4, slide 6)."""
    return sharpe / portfolio_volatility(x0, cov)


def implied_excess_returns(cov, x0, phi):
    """Implied risk premia pi = phi x Sigma x0 (LN4, slide 7). Add rf for total returns."""
    return phi * _array(cov) @ _array(x0)


def bl_expected_returns(cov, w_mkt, phi, P, Q, omega_views, tau):
    """Black-Litterman expected EXCESS returns (contract, section 3):

        mu_bar = pi + Gamma P' (P Gamma P' + Omega)^-1 (Q - P pi),  with Gamma = tau x Sigma

    pi = phi x Sigma x w_mkt are the implied excess returns of the prior portfolio.
    Q must be in excess returns too: an absolute view "asset i returns q" becomes q - rf;
    a relative view (P row sums to 0) does not change."""
    cov, P, Q = _array(cov), np.atleast_2d(_array(P)), _array(Q)
    pi = implied_excess_returns(cov, w_mkt, phi)
    gamma_matrix = tau * cov                               # lab LN4: Gamma = tau x Sigma (not the client's gamma)
    correction = gamma_matrix @ P.T @ np.linalg.solve(P @ gamma_matrix @ P.T + _array(omega_views), Q - P @ pi)
    return pi + correction


def tracking_error(x, x0, cov):
    """Tracking error volatility of x against x0: sqrt((x - x0)' Sigma (x - x0))."""
    return portfolio_volatility(_array(x) - _array(x0), cov)


# =============================================================================
# 2. Covariance of monthly excess returns (annualised: x 12, contract section 1.4)
# =============================================================================

def estimate_covariance(excess_returns, method="ledoit_wolf"):
    """Annualised covariance matrix of monthly excess returns, as a DataFrame with the
    asset names (contract section 1.5).

    method="sample":       the plain estimate of the labs, excess_returns.cov() x 12.
    method="ledoit_wolf":  Ledoit-Wolf shrinkage (Ledoit and Wolf, 2004) applied to the
                           CORRELATIONS only: it is fitted on standardised returns, then scaled
                           back with each asset's own volatility. Steadier weights (lecture notes
                           3, stability) without changing any volatility. (Applied to raw returns,
                           scikit-learn pulls every variance towards the average: CHF bond
                           volatility would rise from 5.4% to 6.1%, which would distort the
                           career hedge.)
    Rows with a missing value are dropped first."""
    data = excess_returns.dropna()
    if method == "sample":
        cov = data.cov().to_numpy() * 12
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        vol = data.std().to_numpy()
        correl = LedoitWolf().fit(((data - data.mean()) / vol).to_numpy()).covariance_
        correl = correl / np.sqrt(np.outer(np.diag(correl), np.diag(correl)))   # exact 1 on the diagonal
        cov = correl * np.outer(vol, vol) * 12
    else:
        raise ValueError(f"Unknown method {method}: use 'sample' or 'ledoit_wolf'")
    return pd.DataFrame(cov, index=data.columns, columns=data.columns)


def shrinkage_intensity(excess_returns):
    """How much Ledoit-Wolf pulls the sample correlations towards the target (0 = none, 1 = all)."""
    from sklearn.covariance import LedoitWolf
    data = excess_returns.dropna()
    return float(LedoitWolf().fit(((data - data.mean()) / data.std()).to_numpy()).shrinkage_)


def correlation_from_covariance(cov):
    """Correlation matrix from a covariance matrix (same labels)."""
    vol = np.sqrt(np.diag(cov))
    return pd.DataFrame(_array(cov) / np.outer(vol, vol), index=cov.index, columns=cov.columns)


# =============================================================================
# 3. Mean-variance solver (lab LN1 code)
# =============================================================================

def QP(x, sigma, mu, gamma):
    """Objective of every lab: 0.5 x' Sigma x - gamma x' mu (minimised)."""
    return 0.5 * x.T @ sigma @ x - gamma * x.T @ mu


# scipy's default precision (ftol = 1e-6) stops too early on these flat objectives: in lab LN4
# it gives a tracking error of 0.01321 where the exact optimum is 0.01532. We use a tight
# precision by default; precise=False reproduces the lab's expected outputs.
PRECISE_OPTIONS = {"disp": False, "ftol": 1e-12, "maxiter": 1000}
LAB_OPTIONS = {"disp": False}


def mean_variance_weights(mu, cov, gamma, x0=None, fully_invested=True, long_only=True, precise=True):
    """Weights minimising QP, as in the labs (scipy minimize with LinearConstraint).
    fully_invested: sum of weights = 1.  long_only: no short selling.
    precise=False uses scipy's default precision, as in the labs."""
    mu, cov = _array(mu), _array(cov)
    n = len(mu)
    x0 = np.ones(n) / n if x0 is None else _array(x0)
    constraints = []
    if fully_invested:
        constraints += [LinearConstraint(np.ones(n), ub=1), LinearConstraint(-np.ones(n), ub=-1)]
    if long_only:
        constraints += [LinearConstraint(np.eye(n), 0)]
    options = PRECISE_OPTIONS if precise else LAB_OPTIONS
    res = minimize(QP, x0, args=(cov, mu, gamma), options=options, constraints=constraints)
    return res.x


def tau_for_tracking_error(target, cov, x0, phi, P, Q, omega_views, tau_start=0.05, precise=True):
    """tau such that the Black-Litterman portfolio has the target tracking error
    against x0 (LN4, slide 18 step 6; lab LN4 last cell)."""
    def gap(tau):
        mu_bar = bl_expected_returns(cov, x0, phi, P, Q, omega_views, float(np.atleast_1d(tau)[0]))
        x = mean_variance_weights(mu_bar, cov, 1 / phi, x0=x0, precise=precise)
        return tracking_error(x, x0, cov) - target
    return float(fsolve(gap, tau_start)[0])
