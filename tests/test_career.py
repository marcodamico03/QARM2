"""Run from the project folder:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import career as c                      # noqa: E402
from data_loader import load_excess_returns  # noqa: E402
from personas import BANKER, CIVIL_SERVANT   # noqa: E402

CORE = ["world_eq", "smi", "eu_banks", "ubs", "chf_bonds",
        "global_bonds_hedged", "swiss_re", "gold"]


def test_human_capital_is_large_and_falls_to_zero():
    bs = c.balance_sheet(BANKER)
    assert bs["H"].iloc[0] > 2_000_000                  # proposal: "more than CHF 2 million"
    assert bs["H"].iloc[-1] == 0                         # nothing left at retirement
    assert bs["omega"].iloc[0] > 0.9                     # proposal: omega above 0.9 when young


def test_pension_grows_until_retirement():
    P = c.pension_projection(BANKER)
    assert (P.diff().dropna() > 0).all()


def test_riskier_career_is_worth_less():
    # same salary, higher beta_S -> higher discount rate -> lower H
    assert c.human_capital(BANKER.with_(beta_S=1.0)) < c.human_capital(BANKER)


def test_sigma_inverse_sigma_H_recovers_the_betas():
    # proposal 3.2: Sigma^-1 sigma_H = beta_B e_B + beta_S e_S
    X = load_excess_returns()[CORE].dropna()
    Sigma = X.cov()
    sigma_H = c.career_covariance(Sigma, BANKER)
    recovered = np.linalg.solve(Sigma.values, sigma_H.values)
    expected = c.career_betas(BANKER, CORE).values
    assert recovered == pytest.approx(expected, abs=1e-8)


def test_hedge_sells_bonds_and_sector_only():
    h = c.hedging_demand_unconstrained(BANKER, omega_H=0.85, omega=0.9, assets=CORE)
    assert h["chf_bonds"] < 0 and h["eu_banks"] < 0
    assert (h.drop(["chf_bonds", "eu_banks"]) == 0).all()
    assert (c.hedging_demand_unconstrained(CIVIL_SERVANT, 0.85, 0.9, CORE)["eu_banks"]) == 0
