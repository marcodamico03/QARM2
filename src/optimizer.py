"""Optimization engine (contract, section 3).

Built step by step, reusing the professor's lab code and notation (contract, section 1):
    1. Black-Litterman expected returns (lab LN4)
    2. Covariance on our data: sample and Ledoit-Wolf (labs LN1, LN3)
    3. Neutral portfolio, Black-Litterman prior and gamma (a client with no career holds it)
    4. Total-wealth optimization with the career (proposal, section 3.2; contract 1.2)
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
# 3. Neutral portfolio, Black-Litterman prior and gamma
# =============================================================================

def neutral_weights(params, assets):
    """Neutral portfolio (params: optimization -> neutral_portfolio) as weights summing to 1,
    aligned with `assets`. Assets not listed (for example EMPLOYER or a sector) get 0."""
    raw = pd.Series(params["optimization"]["neutral_portfolio"], dtype=float)
    unknown = set(raw.index) - set(assets)
    if unknown:
        raise ValueError(f"Neutral portfolio assets not in the data: {sorted(unknown)}")
    w = raw.reindex(assets).fillna(0.0)
    return w / w.sum()


def market_prior(excess_returns, params):
    """Everything the optimizer starts from, computed on the data:
        cov     annualised covariance (method from params)
        w0      neutral portfolio (weights sum to 1)
        phi     implied risk aversion = prior_sharpe / volatility of w0         (LN4, slide 6)
        pi      implied excess returns = phi x Sigma x w0                       (LN4, slide 7)
        gamma   risk tolerance: 1 / phi if params say "auto", so that a client with no career
                and no pension holds exactly w0; otherwise the number in params.
    Returned as a dictionary."""
    opt = params["optimization"]
    cov = estimate_covariance(excess_returns, opt.get("covariance_method", "ledoit_wolf"))
    w0 = neutral_weights(params, list(cov.columns))
    phi = implied_risk_aversion(cov, w0, opt["prior_sharpe"])
    pi = pd.Series(implied_excess_returns(cov, w0, phi), index=cov.columns)
    gamma = 1.0 / phi if opt.get("gamma", "auto") == "auto" else float(opt["gamma"])
    return {"cov": cov, "w0": w0, "phi": phi, "pi": pi, "gamma": gamma}


# =============================================================================
# 4. Total-wealth optimization with the career (contract, section 1.2)
# =============================================================================
#
#   Total wealth W = H + F (career + portfolio), omega = H / W (non-tradable share).
#   R_W = (1 - omega) x'R + omega R_H, all in excess returns. Lab convention (gamma = tolerance):
#       maximise gamma E[R_W] - 0.5 Var(R_W)
#   Dividing by (1 - omega)^2 and dropping constants, the client minimises
#       0.5 x' Sigma x + omega/(1-omega) x' sigma_H - gamma/(1-omega) x' mu
#   whose unconstrained solution is the formula of the contract:
#       x* = gamma/(1-omega) inv(Sigma) mu  -  omega/(1-omega) inv(Sigma) sigma_H
#   The weights x are shares of the PORTFOLIO F; the rest (1 - sum x) is cash.

def omega_share(human_capital, portfolio, other_non_tradable=0.0):
    """omega = non-tradable wealth / total wealth. other_non_tradable: e.g. pension wealth P."""
    non_tradable = human_capital + other_non_tradable
    return non_tradable / (non_tradable + portfolio)


def formula_portfolio(mu, cov, sigma_H, omega, gamma):
    """Unconstrained optimum x* (contract 1.2), split into its two parts.
    Returns (x*, speculative demand, hedging demand) as arrays."""
    cov, mu, sigma_H = _array(cov), _array(mu), _array(sigma_H)
    speculative = gamma / (1 - omega) * np.linalg.solve(cov, mu)
    hedging = -omega / (1 - omega) * np.linalg.solve(cov, sigma_H)
    return speculative + hedging, speculative, hedging


def constraints_from_params(params, assets):
    """The client's constraints from params (section optimization), for the given assets:
    {"long_only": bool, "max_total": float or None, "upper": {asset: cap}, "lower": {asset: floor}}."""
    opt = params["optimization"]
    caps = dict(opt.get("position_caps", {}))
    for a in assets:
        if a.startswith("SEC_") and a not in caps:
            caps[a] = opt.get("sector_cap", 1.0)
    return {"assets": list(assets), "long_only": opt.get("long_only", True),
            "max_total": opt.get("max_leverage", 1.0),
            "upper": {a: c for a, c in caps.items() if a in assets}, "lower": {}}


def optimal_portfolio(mu, cov, sigma_H, omega, gamma, constraints=None, x0=None, precise=True):
    """Total-wealth optimal weights of the risky assets (contract, section 3); the rest is cash.

    mu:       expected EXCESS returns (e.g. Black-Litterman), same order as cov
    sigma_H:  Cov(R, R_H), from src.career.career_covariance (zeros = career ignored)
    omega:    non-tradable share of total wealth (0 = a client with no career)
    gamma:    risk tolerance (lab convention), the same in every strategy
    constraints: dict from constraints_from_params; None = no constraint (gives the formula)
    Returns an ndarray of weights."""
    mu, cov, sigma_H = _array(mu), _array(cov), _array(sigma_H)
    n = len(mu)
    if not 0 <= omega < 1:
        raise ValueError(f"omega must be in [0, 1), got {omega}")
    k_hedge, k_spec = omega / (1 - omega), gamma / (1 - omega)

    def objective(x):
        return 0.5 * x @ cov @ x + k_hedge * x @ sigma_H - k_spec * x @ mu

    def gradient(x):
        return cov @ x + k_hedge * sigma_H - k_spec * mu

    c = constraints or {}
    names = c.get("assets")
    lower = np.full(n, 0.0 if c.get("long_only", False) else -np.inf)
    upper = np.full(n, np.inf)
    if names is not None:
        for a, v in c.get("upper", {}).items():
            upper[names.index(a)] = v
        for a, v in c.get("lower", {}).items():
            lower[names.index(a)] = v
    linear = []
    if c.get("max_total") is not None:
        linear.append(LinearConstraint(np.ones(n), ub=c["max_total"]))
    # Start from equal weights (never from the formula, so the test "solver = formula" is a real test)
    start = np.clip(np.ones(n) / (2 * n) if x0 is None else _array(x0), np.maximum(lower, -1e6), np.minimum(upper, 1e6))
    bounds = None if constraints is None else list(zip(np.where(np.isinf(lower), None, lower), np.where(np.isinf(upper), None, upper)))
    if bounds is None and not linear:
        # No constraint at all: scipy would silently switch to BFGS, which ignores "ftol" and stops
        # too early (error ~6e-4). We call BFGS explicitly with a tight gradient tolerance instead.
        res = minimize(objective, start, jac=gradient, method="BFGS", options={"gtol": 1e-12, "maxiter": 10_000})
    else:
        res = minimize(objective, start, jac=gradient, method="SLSQP", bounds=bounds, constraints=linear,
                       options=PRECISE_OPTIONS if precise else LAB_OPTIONS)
    return res.x


# =============================================================================
# 5. Mean-variance solver (lab LN1 code)
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
