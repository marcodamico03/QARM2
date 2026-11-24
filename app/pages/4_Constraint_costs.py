import common  # noqa: F401  -- must stay first: tells Python where src/ is
import plotly.express as px
import streamlit as st

import optimizer as o
from common import chf, get_persona, market, setup

setup("What your constraints cost", "🔐")
p = get_persona()
X, m, gamma = market()
st.title("4 · What your constraints cost")
st.markdown("For each rule we re-solve the problem **without** it and measure how much better off you would be, "
            "as a certainty-equivalent return on your **total wealth**.")

costs = o.constraint_costs(p, m, gamma)
costs = costs.rename(columns={"cost (bp/yr of total wealth)": "bp per year", "cost (CHF/yr)": "CHF per year"})
fig = px.bar(costs.reset_index(), x="bp per year", y="constraint", orientation="h", text_auto=".0f",
             color_discrete_sequence=["#1f4e79"])
fig.update_layout(height=320, margin=dict(t=10), yaxis_title="")
st.plotly_chart(fig, width="stretch")
st.dataframe(costs.style.format({"bp per year": "{:,.0f}", "CHF per year": lambda v: chf(v)}),
             width="stretch")

top = costs["bp per year"].idxmax()
st.markdown(f"""
**In plain words.** The rule that matters most for you is **{top.lower()}**.
Your job already makes you long bonds and long your sector; the ideal portfolio would *offset* that by selling them short,
which a private client cannot do. The cost is large because the ideal is extreme, so read it as
**"this is the rule that binds"**, not as money you are losing today.
""")
if p.employer_lockup > 0:
    lock = costs.loc["Deferred-share lock-up", "CHF per year"] if "Deferred-share lock-up" in costs.index else 0
    st.info(f"The lock-up on your employer shares ({p.employer_lockup:.0%} of the portfolio) costs you about "
            f"**{chf(lock)} per year** at your current age.")
