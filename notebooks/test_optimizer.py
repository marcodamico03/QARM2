"""Tests of src/optimizer.py with known answers (contract, section 6).

Reference values:
- the "Expected output" cells of the professor's lab LN4 (Black-Litterman, Example 4 of
  lecture notes 4), reproduced with scipy's default precision (precise=False);
- the EXACT solution from a closed formula. The lab's tracking error (0.01321) and tau (0.0961)
  come from scipy stopping too early; the exact values are 0.01532 and 0.0823.

Run from the project folder:
    python notebooks/test_optimizer.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.optimizer as O

# Example 4 of LN4 (assets of Example 1 / lab LN1)
VOL = np.array([0.15, 0.20, 0.25, 0.30])
CORREL = np.array([[1, 0.1, 0.4, 0.5], [0.1, 1, 0.7, 0.4], [0.4, 0.7, 1, 0.8], [0.5, 0.4, 0.8, 1]])
COV = CORREL * np.outer(VOL, VOL)
X0 = np.array([0.40, 0.30, 0.20, 0.10])
SHARPE, RF = 0.25, 0.03
# Views (LN4 slide 19): absolute view on asset 1 (4%), relative view asset 2 - asset 3 = -1%
P = np.array([[1, 0, 0, 0], [0, 1, -1, 0]])
Q_TOTAL = np.array([0.04, -0.01])
Q_EXCESS = Q_TOTAL - P @ np.ones(4) * RF        # absolute view minus rf; relative view unchanged
OMEGA = np.diag([0.10 ** 2, 0.05 ** 2])


def test_implied_returns():
    phi = O.implied_risk_aversion(COV, X0, SHARPE)
    assert abs(O.portfolio_volatility(X0, COV) - 0.15349267083479914) < 1e-12      # lab: vol_x0
    assert abs(phi - 1.6287422626782593) < 1e-12                                 # lab: implied_phi
    mu_total = RF + O.implied_excess_returns(COV, X0, phi)
    assert np.allclose(mu_total, [0.0547, 0.0668, 0.0870, 0.0906], atol=5e-5)    # LN4 slide 8
    print("implied returns OK")


def exact_weights(mu, cov, gamma):
    """Closed-form solution of min 0.5 x'Sx - gamma x'mu subject to sum(x) = 1 (Lagrange).
    Valid when no weight is negative, so long-only does not bind."""
    inv, one = np.linalg.inv(cov), np.ones(len(mu))
    lam = (1 - gamma * one @ inv @ mu) / (one @ inv @ one)
    return gamma * inv @ mu + lam * inv @ one


def test_black_litterman():
    phi = O.implied_risk_aversion(COV, X0, SHARPE)
    # With no confidence in the views (tau = 0), Black-Litterman returns the implied premia
    assert np.allclose(O.bl_expected_returns(COV, X0, phi, P, Q_EXCESS, OMEGA, 0.0),
                       O.implied_excess_returns(COV, X0, phi))
    mu_bar = O.bl_expected_returns(COV, X0, phi, P, Q_EXCESS, OMEGA, 0.05)

    # 1. Lab LN4 reproduced with scipy's default precision
    x_lab = O.mean_variance_weights(mu_bar, COV, 1 / phi, x0=X0, precise=False)
    assert abs(O.tracking_error(x_lab, X0, COV) - 0.013212510637878089) < 1e-6   # lab: tracking_error_optimized
    tau_lab = O.tau_for_tracking_error(0.02, COV, X0, phi, P, Q_EXCESS, OMEGA, precise=False)
    assert abs(tau_lab - 0.09612848941361753) < 1e-4                             # lab: opti_tau

    # 2. Our default (precise) matches the exact closed-form optimum
    x = O.mean_variance_weights(mu_bar, COV, 1 / phi, x0=X0)
    exact = exact_weights(mu_bar, COV, 1 / phi)
    assert (exact > 0).all() and np.allclose(x, exact, atol=1e-5), (x, exact)
    assert abs(x.sum() - 1) < 1e-8 and (x > -1e-8).all()
    assert abs(O.tracking_error(x, X0, COV) - 0.015322621582734643) < 1e-5       # exact, not 0.01321
    tau = O.tau_for_tracking_error(0.02, COV, X0, phi, P, Q_EXCESS, OMEGA)
    assert abs(tau - 0.08226052909915386) < 1e-4                                 # exact, not 0.0961
    print("Black-Litterman OK (lab LN4 reproduced, and exact optimum with the precise solver)")


def test_mean_variance():
    # With the implied returns and gamma = 1 / phi, the optimal portfolio is x0 itself
    phi = O.implied_risk_aversion(COV, X0, SHARPE)
    x = O.mean_variance_weights(O.implied_excess_returns(COV, X0, phi), COV, 1 / phi, x0=np.ones(4) / 4)
    assert np.allclose(x, X0, atol=1e-4), x
    print("mean-variance OK (implied returns give back x0)")


def test_covariance():
    import pandas as pd
    # Known answer on simulated data: "sample" equals pandas .cov() x 12
    rng = np.random.default_rng(0)
    names = ["A", "B", "C"]
    data = pd.DataFrame(rng.normal(0, [0.04, 0.02, 0.06], size=(240, 3)), columns=names)
    sample = O.estimate_covariance(data, "sample")
    assert np.allclose(sample, data.cov() * 12) and list(sample.columns) == names
    lw = O.estimate_covariance(data, "ledoit_wolf")
    assert 0 <= O.shrinkage_intensity(data) <= 1
    assert np.allclose(lw, lw.T) and (np.linalg.eigvalsh(lw) > 0).all()        # symmetric, positive definite
    assert np.allclose(np.diag(O.correlation_from_covariance(lw)), 1)
    assert np.allclose(np.diag(lw), np.diag(sample))                              # volatilities unchanged

    # On our saved data: same assets and dates as the data layer, and shrinkage makes the
    # matrix better conditioned (more stable to invert, which the optimizer does)
    from src.data import load_returns, load_rf
    excess = load_returns().sub(load_rf(), axis=0)
    s, l = O.estimate_covariance(excess, "sample"), O.estimate_covariance(excess, "ledoit_wolf")
    assert list(s.columns) == list(excess.columns)
    assert np.linalg.cond(l) <= np.linalg.cond(s)
    assert np.allclose(np.sqrt(np.diag(s)), excess.std() * np.sqrt(12))          # annualised volatilities
    assert np.allclose(np.diag(l), np.diag(s))                                    # shrinkage keeps every volatility
    off = ~np.eye(len(s), dtype=bool)
    assert (np.abs(O.correlation_from_covariance(l).to_numpy()[off])
            <= np.abs(O.correlation_from_covariance(s).to_numpy()[off]) + 1e-12).all()   # correlations pulled to 0
    print(f"covariance OK (Ledoit-Wolf shrinkage on our data: {O.shrinkage_intensity(excess):.3f})")


def test_market_prior():
    import copy
    from src.config import load_params
    from src.data import load_returns, load_rf
    params = load_params()
    excess = load_returns().sub(load_rf(), axis=0)
    prior = O.market_prior(excess, params)
    w0, cov, pi, gamma, phi = prior["w0"], prior["cov"], prior["pi"], prior["gamma"], prior["phi"]

    # Neutral weights: sum to 1, in the data's asset order, 0 for assets not listed (EMPLOYER)
    assert abs(w0.sum() - 1) < 1e-12 and list(w0.index) == list(excess.columns) and w0["EMPLOYER"] == 0
    assert abs(w0["BOND_CHF"] - 30 / 90) < 1e-12 and w0["BANKS_EU"] > 0     # the hedge needs CHF bonds and banks
    # phi = Sharpe / volatility, gamma = 1 / phi, and the neutral portfolio earns Sharpe x volatility
    vol = O.portfolio_volatility(w0, cov)
    assert abs(phi - params["optimization"]["prior_sharpe"] / vol) < 1e-12 and abs(gamma - 1 / phi) < 1e-12
    assert abs(w0 @ pi - params["optimization"]["prior_sharpe"] * vol) < 1e-12

    # Key property: a client with no career and no pension holds exactly the neutral portfolio.
    # (a) closed form, no constraint: gamma x inv(Sigma) x pi = w0
    assert np.allclose(gamma * np.linalg.solve(cov.to_numpy(), pi.to_numpy()), w0.to_numpy())
    # (b) the lab solver, fully invested and long-only
    x = O.mean_variance_weights(pi, cov, gamma)
    assert np.allclose(x, w0.to_numpy(), atol=1e-5), x

    # A number in params overrides gamma; an asset that is not in the data is refused
    p = copy.deepcopy(params); p["optimization"]["gamma"] = 0.5
    assert O.market_prior(excess, p)["gamma"] == 0.5
    p = copy.deepcopy(params); p["optimization"]["neutral_portfolio"]["XYZ"] = 1.0
    try:
        O.market_prior(excess, p)
        raise AssertionError("an unknown asset should be refused")
    except ValueError:
        pass
    print(f"market prior OK (phi {phi:.2f}, gamma {gamma:.3f}, neutral client holds the neutral portfolio)")


def test_total_wealth():
    import src.career as C
    from src.config import load_params
    from src.data import load_returns, load_rf
    params = load_params()
    excess = load_returns().sub(load_rf(), axis=0)
    prior = O.market_prior(excess, params)
    cov, pi, gamma, w0 = prior["cov"], prior["pi"], prior["gamma"], prior["w0"]
    names = list(cov.columns)
    zero = np.zeros(len(names))

    # 1. No career (omega = 0): the client holds the neutral portfolio, with or without constraints
    assert np.allclose(O.optimal_portfolio(pi, cov, zero, 0.0, gamma), w0, atol=1e-5)
    cons = O.constraints_from_params(params, names)
    assert np.allclose(O.optimal_portfolio(pi, cov, zero, 0.0, gamma, cons), w0, atol=1e-5)

    # 2. The banker (age 35, salary 130k, portfolio 100k): without constraints the solver
    #    finds the formula of the contract, starting from equal weights
    p = C.params_for_persona(params, "banker")
    beta_bond, beta_sector, _ = C.career_betas(35, p)
    sigma_h = C.career_covariance(cov, beta_bond, beta_sector, p)
    omega = O.omega_share(C.human_capital(35, 130_000, p), 100_000)
    x_star, spec, hedge = O.formula_portfolio(pi, cov, sigma_h, omega, gamma)
    x = O.optimal_portfolio(pi, cov, sigma_h, omega, gamma)
    assert np.allclose(x, x_star, rtol=1e-6, atol=1e-6), np.abs(x - x_star).max()
    # the hedge is "hold beta_bond less CHF bonds and beta_sector less banks", scaled by omega/(1-omega)
    k = omega / (1 - omega)
    expected_hedge = np.zeros(len(names))
    expected_hedge[names.index("BOND_CHF")] = -k * beta_bond
    expected_hedge[names.index("BANKS_EU")] = -k * beta_sector
    assert np.allclose(hedge, expected_hedge, atol=1e-8)

    # 3. With the client's constraints: valid weights, and the hedge holds no MORE banks or
    #    CHF bonds than the same client without the hedge (strategy D = C without hedging term)
    c = O.optimal_portfolio(pi, cov, sigma_h, omega, gamma, cons)
    d = O.optimal_portfolio(pi, cov, zero, omega, gamma, cons)
    for w in (c, d):
        assert (w > -1e-8).all() and w.sum() < 1 + 1e-8
        for a, cap in cons["upper"].items():
            assert w[names.index(a)] <= cap + 1e-8
    for a in ("BANKS_EU", "BOND_CHF"):
        assert c[names.index(a)] <= d[names.index(a)] + 1e-6
    assert d[names.index("BANKS_EU")] > c[names.index("BANKS_EU")] + 0.05      # visible: 10% -> 0%

    # 4. A client with no sector (civil servant) is not pushed out of banks by the hedge
    q = C.params_for_persona(params, "civil_servant")
    bb, bs, _ = C.career_betas(35, q)
    sh = C.career_covariance(cov, bb, bs, q)
    om = O.omega_share(C.human_capital(35, 100_000, q), 100_000)
    c2 = O.optimal_portfolio(pi, cov, sh, om, gamma, cons)
    assert c2[names.index("BANKS_EU")] >= d[names.index("BANKS_EU")] - 1e-6
    print(f"total-wealth optimization OK (banker: banks {d[names.index('BANKS_EU')]:.0%} without hedge, "
          f"{c[names.index('BANKS_EU')]:.0%} with hedge)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # fsolve progress messages
    test_implied_returns()
    test_black_litterman()
    test_mean_variance()
    test_covariance()
    test_market_prior()
    test_total_wealth()
    print("ALL TESTS PASSED")
