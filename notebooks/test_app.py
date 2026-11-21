"""Tests of the app (step 5): the numbers in src/dashboard.py, the results kept on disk, the
background jobs, then the app itself, run headless with Streamlit's own test tool (both pages,
every persona).

Run from the project folder (about two minutes the first time, then the backtests are saved):
    python notebooks/test_app.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.backtest as B
import src.dashboard as D
import src.optimizer as O
import src.theme as T
from src.jobs import Jobs
from src.career import params_for_persona
from src.config import load_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_client_params():
    base = load_params()
    # No change: exactly the persona of step 2
    assert D.client_params("dentist") == params_for_persona(base, "dentist")
    p = D.client_params("banker", {"age": 50, "salary_chf": 200_000, "portfolio_chf": 500_000,
                                   "deferred_shares_chf": 20_000, "pension_chf": 300_000, "beta_sector": 0.4,
                                   "wage_rate_link": 0.2, "long_only": False, "no_leverage": False,
                                   "sector_cap": 0.25, "view": {"asset": "EQ_CH", "q": 0.06, "sd": 0.05}})
    c = p["client"]
    assert (c["age"], c["salary_chf"], c["portfolio_chf"], c["deferred_shares_chf"], c["pension_chf"]) == \
        (50, 200_000, 500_000, 20_000, 300_000)
    assert p["career"]["beta_sector"] == 0.4 and p["career"]["wage_rate_link"] == 0.2
    assert p["optimization"]["long_only"] is False and p["optimization"]["max_leverage"] is None
    assert set(p["optimization"]["position_caps"].values()) == {0.25}
    assert p["black_litterman"]["views"] == [{"assets": {"EQ_CH": 1.0}, "q": 0.06, "sd": 0.05}]
    assert base["client"]["age"] == 35                       # the shared params are not changed
    # A civil servant has no sector: the sector beta stays 0; locked shares never exceed the portfolio
    assert D.client_params("civil_servant", {"beta_sector": 0.5})["career"]["beta_sector"] == 0.0
    assert D.client_params("banker", {"portfolio_chf": 10_000})["client"]["deferred_shares_chf"] == 10_000
    print("client settings OK")


def test_analysis_and_lifecycle():
    excess = O.load_excess_returns()
    for name in load_params()["personas"]:
        p = D.client_params(name)
        r = D.analysis(p, excess)
        sheet, story = r["balance_sheet"], r["story"]
        assert abs(story["portfolio_share"] - sheet["F"] / sheet["W"]) < 1e-15
        assert 0 <= story["career_risk_share"] <= 1 and story["bonds_A"] >= story["bonds_D"] - 1e-12
        assert story["sector_C"] <= story["sector_D"] + 1e-12
        # unconstrained demands in CHF: the hedging part is minus the career's hidden holdings
        dem = D.demand_chf(r)
        for asset, value in r["career_holdings"].items():
            assert abs(dem.loc[asset, "hedging"] + value) < 1e-6 * max(1.0, value)
        # life cycle: starts at today's balance sheet, the career is worth 0 at retirement
        lc = D.lifecycle(p, r)
        assert abs(lc["H"].iloc[0] - sheet["H"]) < 1e-6 and abs(lc["P"].iloc[0] - sheet["P"]) < 1e-6
        assert lc["F"].iloc[0] == sheet["F"] and lc["H"].iloc[-1] == 0.0
        assert lc.index[-1] == p["client"]["retirement_age"] and (np.diff(lc["F"]) > 0).all()
    print("analysis and life cycle OK")


def test_plan_numbers():
    excess = O.load_excess_returns()
    p = D.client_params("banker")
    r = D.analysis(p, excess)
    # the bars of "How we value your career" add up to H
    years, rate = D.career_years(p)
    assert abs(years["value today"].sum() - r["balance_sheet"]["H"]) < 1e-6
    assert years.index[0] == p["client"]["age"] and years.index[-1] == p["client"]["retirement_age"] - 1
    # families: each portfolio sums to 1; the plan by age starts with today's advice
    for s in ("A", "C"):
        assert abs(D.groups(r["weights"][s]).sum() - 1) < 1e-9
    glide = D.glide_path(p, r, excess)
    assert glide.index[0] == p["client"]["age"] and glide.index[-1] < p["client"]["retirement_age"]
    assert abs(glide.loc[glide.index[0], ("C", "Bonds")] - D.groups(r["weights"]["C"])["Bonds"]) < 1e-12
    shares = [c for c in glide.columns if c[0] == "C"]
    assert np.allclose(glide[shares].sum(axis=1), 1)
    # the banker holds bonds again near retirement, never more than standard advice
    assert glide[("C", "Bonds")].iloc[-1] > glide[("C", "Bonds")].iloc[0]
    assert (glide[("C", "Bonds")] <= glide[("A", "Bonds")] + 1e-9).all()
    print("plan numbers OK")


def test_saved_results():
    # same client written as 130000 or 130000.0: the same saved file; another client: another file
    a = D.client_params("banker")
    b = D.client_params("banker", {"salary_chf": 130_000.0, "age": 35.0})
    assert D.cache_key("backtest", a) == D.cache_key("backtest", b)
    assert D.cache_key("backtest", a) != D.cache_key("backtest", D.client_params("banker", {"age": 36}))
    first = D.remember("backtest", a)                          # computed (or read) once ...
    again = D.remember("backtest", b)                          # ... then read from disk
    assert first["gains"].equals(again["gains"])
    fresh = D.backtest(a, B.load_backtest_data())
    assert np.allclose(fresh["values"].to_numpy(), first["values"].to_numpy())
    # the background queue: a job returns its result; an older job that has not started is dropped
    jobs = Jobs()
    slow = jobs.get("slow", lambda: __import__("time").sleep(0.5) or 1, group="g")
    old = jobs.get("old", lambda: 2, group="g")
    new = jobs.get("new", lambda: 3, group="g")
    assert old.cancelled() and new.result() == 3 and slow.result() == 1 and not jobs.pending()
    print("saved results and background jobs OK")


def test_theme():
    assert T.chf(3_289_732) == "CHF 3.29 m" and T.chf(493_460) == "CHF 493k" and T.chf(1250) == "CHF 1'250"
    assert T.chf(-2500, short=False) == "−CHF 2'500" and T.pct(0.0291, 1) == "2.9%"
    assert "\n\n" not in T.css()                       # a blank line would end the HTML block in Markdown
    sheet = {"H": 3e6, "P": 5e4, "F": 1e5, "W": 3.15e6}
    assert T.ribbon(sheet).count('class="career"') == 1 and "CHF 3.00 m" in T.ribbon(sheet)
    print("theme OK")


def test_app_headless():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("streamlit not installed, app test skipped")
        return
    base = load_params()
    for name in base["personas"]:                             # saved first: the pages then never wait
        for kind in ("backtest", "prior"):
            D.remember(kind, D.client_params(name, None, base))
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    for name in base["personas"]:                             # page "Your plan", every job
        at.selectbox(key="plan_job").set_value(name).run()
        assert not at.exception, (name, at.exception)
    at.selectbox(key="plan_job").set_value("banker").run()
    at.number_input(key="plan_age_banker").set_value(55).run()
    assert not at.exception, at.exception
    at.switch_page("views/analyst.py").run()                 # page "Analyst view"
    assert not at.exception, at.exception
    at.slider(key="age_banker").set_value(45).run()
    assert not at.exception, at.exception
    print("both pages run for every persona, OK")


if __name__ == "__main__":
    test_client_params()
    test_analysis_and_lifecycle()
    test_plan_numbers()
    test_saved_results()
    test_theme()
    test_app_headless()
    print("ALL TESTS PASSED")
