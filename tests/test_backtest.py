"""Run from the project folder:  python -m pytest -q   (this file takes ~15 seconds)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backtest as bt                                    # noqa: E402
from config import INVESTABLE                            # noqa: E402
from data_loader import load_excess_returns, load_rates  # noqa: E402
from personas import BANKER                              # noqa: E402

X, R = load_excess_returns(), load_rates()


@pytest.fixture(scope="module")
def pre():
    return bt.precompute(X, INVESTABLE)


@pytest.fixture(scope="module")
def res(pre):
    return bt.run_all(BANKER, pre, X, R)


def test_inputs_use_only_past_data(pre):
    for t in pre.dates[:24]:
        assert pre.inputs[t].sample.split(" to ")[1].startswith(str(t - 1))


def test_no_look_ahead_future_data_does_not_change_past_weights(pre):
    # cut the data 2 years after the start: the first 24 decisions must be identical
    cut = pre.dates[23]
    pre_cut = bt.precompute(X.loc[:cut], INVESTABLE)
    a = bt.run("C", BANKER, pre, X, R).loc[:cut].filter(like="w_")
    b = bt.run("C", BANKER, pre_cut, X, R).loc[:cut].filter(like="w_")
    assert np.allclose(a.values, b.values, atol=1e-7)


def test_same_numbers_on_every_run(pre, res):
    again = bt.run("C", BANKER, pre, X, R)
    assert np.allclose(again["W"].values, res["C"]["W"].values)


def test_balance_sheet_adds_up(res):
    df = res["C"]
    assert np.allclose(df["W"], df["H"] + df["P"] + df["F"])


def test_weights_feasible(res):
    for s in "ACD":
        w = res[s].filter(like="w_")
        assert (w.values >= -1e-8).all() and (w.sum(axis=1) <= 1 + 1e-6).all()
        assert (w["w_ubs"] >= BANKER.employer_lockup - 1e-6).all()


def test_C_holds_less_of_the_sector_than_D(res):
    assert res["C"]["w_eu_banks"].mean() <= res["D"]["w_eu_banks"].mean()


def test_hedge_gain_matches_h_2beta_minus_h():
    # proposal 4.2: a hedge h against a true exposure beta reduces variance by h(2 beta - h)
    h = 0.3
    for mult in [0, 0.5, 1, 1.5, 2]:
        sample = bt.sample_hedge_gain(X["eu_banks"], h, mult * h)
        theory = bt.theoretical_hedge_gain(h, mult * h)
        assert sample == pytest.approx(theory, abs=1e-12)


def test_risk_benefit_of_hedge_grows_with_true_beta(pre):
    # proposal 4.2: the more the career really moves with the sector, the more the hedge reduces risk
    wb = bt.wrong_beta_curve(BANKER, pre, X, R, multipliers=(0.0, 1.0, 2.0))
    risk = wb["risk benefit C vs D (bp/yr)"].values
    assert risk[0] < risk[1] < risk[2]
