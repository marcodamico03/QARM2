"""Run from the project folder:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import career                                       # noqa: E402
import optimizer as o                               # noqa: E402
import risk                                         # noqa: E402
from data_loader import load_excess_returns         # noqa: E402
from estimation import estimate_inputs              # noqa: E402
from personas import BANKER, CIVIL_SERVANT          # noqa: E402


@pytest.fixture(scope="module")
def inputs():
    return estimate_inputs(load_excess_returns())


@pytest.fixture(scope="module")
def gamma(inputs):
    return o.calibrate_gamma(inputs)


NO_CONSTRAINTS = o.Constraints(long_only=False, no_leverage=False, caps={}, lockup={})


def test_cvxpy_matches_closed_form_without_constraints(inputs, gamma):
    st = o.client_state(BANKER)
    sH = career.career_covariance(inputs.Sigma, BANKER).values
    spec, hedge = o.closed_form(inputs, gamma, st.omega_H, st.omega, sH)
    sol = o.solve(inputs, gamma, st.omega_H, st.omega, sH, NO_CONSTRAINTS)
    assert sol.weights.values == pytest.approx((spec + hedge).values, rel=1e-4, abs=1e-4)


def test_no_views_no_career_holds_scaled_market_portfolio(inputs, gamma):
    # with mu = pi = delta Sigma w_mkt, unconstrained x = (delta / gamma) w_mkt
    sol = o.solve(inputs, gamma, 0.0, 0.0, np.zeros(len(inputs.assets)), NO_CONSTRAINTS)
    assert sol.weights.values == pytest.approx((inputs.delta / gamma * inputs.w_mkt).values, abs=1e-5)


def test_gamma_gives_60_40_for_client_without_career(inputs, gamma):
    w = o.solve(inputs, gamma, 0.0, 0.0, np.zeros(len(inputs.assets)), o.Constraints()).weights
    assert o.equity_share(w) == pytest.approx(0.60, abs=0.005)


def test_hedge_has_the_sign_theory_predicts(inputs, gamma):
    # C holds no more of the client's sector than D (proposal 'done when')
    c = o.strategy("C", BANKER, inputs, gamma).weights
    d = o.strategy("D", BANKER, inputs, gamma).weights
    assert c["eu_banks"] <= d["eu_banks"] + 1e-6
    # the unconstrained hedging demand is negative in bonds and in the sector
    st = o.client_state(BANKER)
    _, hedge = o.closed_form(inputs, gamma, st.omega_H, st.omega,
                             career.career_covariance(inputs.Sigma, BANKER).values)
    assert hedge["eu_banks"] < 0 and hedge["chf_bonds"] < 0


def test_banker_holds_less_of_his_sector_than_civil_servant(inputs, gamma):
    b = o.strategy("C", BANKER, inputs, gamma, cons=o.Constraints()).weights
    c = o.strategy("C", CIVIL_SERVANT, inputs, gamma, cons=o.Constraints()).weights
    assert b["eu_banks"] <= c["eu_banks"] + 1e-6


def test_constraints_respected_and_costs_nonnegative(inputs, gamma):
    w = o.strategy("C", BANKER, inputs, gamma).weights
    assert (w >= -1e-8).all() and w.sum() <= 1 + 1e-8
    assert w["ubs"] >= BANKER.employer_lockup - 1e-8
    assert (o.constraint_costs(BANKER, inputs, gamma).iloc[:, 0] >= 0).all()


def test_risk_contributions_sum_to_one(inputs, gamma):
    st = o.client_state(BANKER)
    w = o.strategy("C", BANKER, inputs, gamma).weights
    rc = risk.total_wealth_risk_contributions(w, BANKER, st, inputs)
    assert rc.sum() == pytest.approx(1.0)
    assert rc["career"] > 0.5            # the job dominates a young banker's risk
