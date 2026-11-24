import common  # noqa: F401  -- must stay first: tells Python where src/ is
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import optimizer as o
import risk
from common import ASSET_COLORS, STRAT_LABELS, get_persona, label, market, setup

setup("Where your risk comes from", "🧭")
p = get_persona()
X, m, gamma = market()
st_ = o.client_state(p)
st.title("5 · Where your risk comes from")
st.markdown("Euler risk contributions on **total wealth**: how much of the ups and downs of everything you own "
            "comes from your career, your pension and each asset.")

rows = {}
for k in ["A", "C"]:
    rows[STRAT_LABELS[k]] = risk.total_wealth_risk_contributions(o.strategy(k, p, m, gamma).weights, p, st_, m)
rc = label(pd.DataFrame(rows))
rc = rc[(rc.abs() > 1e-4).any(axis=1)]

fig = go.Figure()
for name in rc.index:
    fig.add_trace(go.Bar(y=rc.columns, x=rc.loc[name], name=name, orientation="h",
                         marker_color=ASSET_COLORS.get(name)))
fig.update_layout(barmode="stack", height=300, xaxis=dict(tickformat=".0%", title="share of total-wealth risk"),
                  legend=dict(orientation="h", y=-0.35), margin=dict(t=10))
st.plotly_chart(fig, width="stretch")

career_share = rc.loc["Career", STRAT_LABELS["C"]]
st.metric("Share of your total risk coming from your job", f"{career_share:.0%}")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**What a standard tool shows** (portfolio only, strategy C)")
    pr = label(risk.portfolio_risk_contributions(o.strategy("C", p, m, gamma).weights, m))
    st.dataframe(pr[pr.abs() > 1e-3].to_frame("share of portfolio risk").style.format("{:.0%}"),
                 width="stretch")
with c2:
    st.markdown("**What really matters** (total wealth)")
    st.dataframe(rc.style.format("{:.1%}"), width="stretch")
st.caption("A standard tool would tell you that equities are most of your risk. On total wealth, your career dominates: "
           "the portfolio can only change a small part of your risk, and the goal is to change it in the right direction.")
