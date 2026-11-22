"""Tests of app_v3 (the story page) and of what it adds to the model.

Run from the project folder:  python notebooks/test_app_v3.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.allocation as AL
from src import look as L
from src.config import load_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_typical_and_fewer_bonds():
    """Typical advice is the profile's model portfolio (balanced 60/40); our plan starts from the same
    bonds; (M) has the bonds of our plan but keeps the industry."""
    data = AL.load_data()
    for persona in ("banker", "civil_servant"):
        for profile in ("prudent", "balanced", "dynamic"):
            p = AL.client_params(persona, {"profile": profile})
            r = AL.analyze(p, data)
            gA, gM, gC = (r["plans"][k]["groups"] for k in "AMC")
            safe = p["allocation"]["profiles"][profile]["safe_share"]
            assert abs(gA["bonds"] - safe) < 1e-9, (persona, profile, gA["bonds"])
            assert abs(r["plans"]["C"]["detail"]["safe_share"] - safe * (1 - r["sheet"]["bond_like"])) < 1e-12
            assert abs(gM["bonds"] - gC["bonds"]) < 1e-9
            assert gC["bonds"] < gA["bonds"]
            for k in "AMC":
                w = r["plans"][k]["weights"]
                assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all(), k
    r = AL.analyze(AL.client_params("banker"), data)
    wM, wC = r["plans"]["M"]["weights"], r["plans"]["C"]["weights"]
    assert r["plans"]["C"]["detail"]["excluded"] == ["IND_MONEY"] and r["plans"]["M"]["detail"]["excluded"] == []
    assert abs(wC["IND_MONEY"] - 0.15) < 1e-9 and wM["IND_MONEY"] > 0.15     # (M) buys finance on top of the stake
    sheet = r["sheet"]
    assert abs(r["plans"]["C"]["detail"]["safe_share"] - 0.40 * (1 - sheet["bond_like"])) < 1e-12
    assert load_params()["allocation"]["profiles"]["balanced"]["safe_share"] == 0.40     # balanced = 60/40
    print("typical advice = model portfolio of the profile, fewer bonds only OK")


def test_story_backtest():
    """The two steps add up to the total gain (all measured with the wealth of typical advice)."""
    s = AL.backtest_story(AL.client_params("banker"))
    g = s["gains"]["CE gain (CHF/yr)"]
    assert abs(g["(M) vs (A)"] + g["(C) vs (M)"] - g["(C) vs (A)"]) < 1e-6 * abs(g["(C) vs (A)"])
    assert set(s["metrics"].columns) == {"A", "B", "M", "C"} and set(s["values"].columns) == {"A", "B", "M", "C"}
    assert g["(M) vs (A)"] > 0
    print(f"story backtest OK: fewer bonds {g['(M) vs (A)']:+,.0f}, no own industry {g['(C) vs (M)']:+,.0f}, "
          f"total {g['(C) vs (A)']:+,.0f} CHF a year")


def test_look():
    for dark in (False, True):
        c = L.palette(dark)
        assert "\n\n" not in L.css(c)                     # a blank line would end the HTML block in Markdown
        assert L.streamlit_theme(c)["base"] == ("dark" if dark else "light")
    assert L.signed_chf(3704) == "+CHF 3'704" and L.signed_chf(-85).startswith("−")
    assert L.chance(0.001) == "more than 99% of replayed histories" and L.chance(0.4) == "60% of replayed histories"
    assert L.bar(L.LIGHT, "", "", [(0.5, "#000", "a"), (0.5, "#fff", "b")]).startswith('<div class="v3-bar">')
    print("look OK")


def test_app_headless():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app_v3.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    for name in load_params()["personas"]:
        at.selectbox(key="job").set_value(name).run()
        assert not at.exception, (name, at.exception)
    at.selectbox(key="job").set_value("banker").run()
    at.number_input(key="age_banker").set_value(60).run()
    assert not at.exception, at.exception
    at.toggle(key="dark").set_value(True).run()
    assert not at.exception, at.exception
    at.radio(key="profile_banker").set_value("prudent").run()
    assert not at.exception, at.exception
    print("app_v3 runs for every example, at 60, prudent, in dark mode OK")


if __name__ == "__main__":
    os.chdir(ROOT)
    test_typical_and_fewer_bonds()
    test_story_backtest()
    test_look()
    test_app_headless()
    print("all tests passed")
