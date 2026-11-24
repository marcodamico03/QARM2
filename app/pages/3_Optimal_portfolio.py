import common  # noqa: F401  -- must stay first: tells Python where src/ is
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import career
import optimizer as o
from common import ASSET_COLORS, STRAT_LABELS, chf, get_persona, label, market, setup

setup("Your optimal portfolio", "🎯")
p = get_persona()
X, m, gamma = market()
st.title("3 · Your optimal portfolio")

st_ = o.client_state(p)
cols = st.columns(4)
cols[0].metric("Risk aversion γ (same for all)", f"{gamma:.2f}", help="Set so that a client with no career holds 60/40.")
cols[1].metric("Career share ω_H", f"{st_.omega_H:.0%}")
cols[2].metric("Pension share ω_P", f"{st_.omega_P:.0%}")
cols[3].metric("Total wealth", chf(st_.W))

sols = {k: o.strategy(k, p, m, gamma) for k in ["A", "B", "D", "C"]}
W = pd.DataFrame({STRAT_LABELS[k]: s.weights for k, s in sols.items()})
W.loc["cash"] = (1 - W.sum()).clip(lower=0)
W = label(W)

fig = go.Figure()
for asset in W.index:
    if W.loc[asset].abs().max() > 1e-4:
        fig.add_trace(go.Bar(x=W.columns, y=W.loc[asset], name=asset, marker_color=ASSET_COLORS.get(asset)))
fig.update_layout(barmode="stack", height=460, yaxis=dict(tickformat=".0%", title="share of the portfolio"),
                  legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

st.dataframe(W[(W.abs() > 1e-4).any(axis=1)].style.format("{:.0%}"), width="stretch")
if True:
    st.markdown("""
**How to read it**
* **A · Standard** ignores your job and pension.
* **D · No hedge** knows your job is bond-like, so it takes **more equity risk**…
* …but it may buy your **own sector**, doubling a bet you already hold through your job.
* **C · Total wealth** does both: more risk *and* none of your own sector.
* **B · 100 − age** is the rule of thumb, for comparison.
""")

st.divider()
st.subheader("Why: speculative demand vs hedging demand")
st.latex(r"x^* = \underbrace{\frac{1}{\gamma(1-\omega)}\Sigma^{-1}\mu}_{\text{speculative}}"
         r"\;-\;\underbrace{\frac{\omega_H}{1-\omega}\Sigma^{-1}\sigma_H}_{\text{hedging}}")
sH = career.career_covariance(m.Sigma, p).values
spec, hedge = o.closed_form(m, gamma, st_.omega_H, st_.omega, sH)
U = label(pd.DataFrame({"speculative": spec, "hedging": hedge, "total (no constraints)": spec + hedge}))
st.dataframe(U.style.format("{:+,.0%}"), width="stretch")
st.caption("Without limits, a young client would short huge amounts of bonds and of his own sector. "
           "Long-only and no-borrowing rules stop this (a corner solution); page 4 shows what those rules cost.")
