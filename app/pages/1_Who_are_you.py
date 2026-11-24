import common  # noqa: F401  -- must stay first: tells Python where src/ is
import streamlit as st

import career
from common import PRESETS, chf, get_persona, setup

setup("Who are you?", "👤")
st.title("1 · Who are you?")
st.markdown("Pick a client, then adjust any assumption. Every other page updates automatically.")

FIELDS = ["age", "salary", "salary_growth", "savings_rate", "portfolio", "pension",
          "beta_B", "beta_S", "employer_lockup"]


def load_preset():
    p = PRESETS[st.session_state["preset"]]
    for f in FIELDS:
        st.session_state[f"in_{f}"] = getattr(p, f)


if "in_age" not in st.session_state:
    st.session_state["preset"] = get_persona().name if get_persona().name in PRESETS else list(PRESETS)[0]
    load_preset()

st.selectbox("Client preset", list(PRESETS), key="preset", on_change=load_preset)
base = PRESETS[st.session_state["preset"]]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### Life and income")
    st.slider("Age", 25, 60, key="in_age")
    st.number_input("Gross salary (CHF / year)", 30_000, 1_000_000, step=5_000, key="in_salary")
    st.slider("Salary growth per year", 0.0, 0.05, step=0.005, format="%.3f", key="in_salary_growth")
    st.slider("Savings rate (share of salary invested)", 0.0, 0.40, step=0.01, key="in_savings_rate")
with c2:
    st.markdown("#### Wealth today")
    st.number_input("Free portfolio (CHF)", 0, 5_000_000, step=10_000, key="in_portfolio")
    st.number_input("Pension account, 2nd pillar (CHF)", 0, 5_000_000, step=10_000, key="in_pension")
    st.slider("Deferred employer shares (minimum weight, locked)", 0.0, 0.30, step=0.01,
              key="in_employer_lockup", disabled=base.employer_asset is None,
              help="Shares paid as compensation that cannot be sold for several years.")
with c3:
    st.markdown("#### What your career behaves like")
    st.slider("β_B: how bond-like is your job?", 0.0, 5.0, step=0.1, key="in_beta_B",
              help="In units of a CHF bond index. Long careers behave like long bonds, so this can exceed 1.")
    st.slider("β_S: how much does your job move with your sector?", 0.0, 1.5, step=0.05, key="in_beta_S",
              help="0 = civil servant, 0.3 = banker (conservative), higher = bonus-heavy or equity-paid jobs.")
    st.caption("These betas are **assumptions**, not estimates. Page 6 shows what happens if they are wrong.")

p = base.with_(**{f: st.session_state[f"in_{f}"] for f in FIELDS})
if base.employer_asset is None:
    p = p.with_(employer_lockup=0.0)
st.session_state["persona"] = p

st.divider()
bs = career.balance_sheet(p)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Value of your career (H)", chf(bs["H"].iloc[0]))
m2.metric("Discount rate for the career", f"{career.discount_rate(p):.2%}")
m3.metric("Total wealth (H + P + F)", chf(bs["W"].iloc[0]))
m4.metric("Non-tradable share ω", f"{bs['omega'].iloc[0]:.0%}")
st.success(f"Your portfolio is only **{p.portfolio / bs['W'].iloc[0]:.0%}** of what you really own. "
           "The rest is your career and your pension.")
