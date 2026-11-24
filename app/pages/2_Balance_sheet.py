import common  # noqa: F401  -- must stay first: tells Python where src/ is
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import career
from common import chf, get_persona, setup

setup("Your real balance sheet", "📊")
p = get_persona()
st.title("2 · Your real balance sheet")
st.markdown("Only the portfolio is chosen by the optimizer, but it is judged on **total wealth** W = H + P + F.")

bs = career.balance_sheet(p)
fig = go.Figure()
for col, name, color in [("H", "Career H", "#1f4e79"), ("P", "Pension P", "#e377c2"), ("F", "Portfolio F", "#2ca02c")]:
    fig.add_trace(go.Scatter(x=bs.index, y=bs[col], name=name, stackgroup="w", line=dict(width=0.5, color=color)))
fig.add_trace(go.Scatter(x=bs.index, y=bs["omega"], name="ω (non-tradable share)", yaxis="y2",
                         line=dict(color="black", width=3)))
fig.update_layout(height=480, xaxis_title="age", yaxis=dict(title="CHF", tickformat=",.0f"),
                  yaxis2=dict(title="ω", overlaying="y", side="right", range=[0, 1], tickformat=".0%"),
                  legend=dict(orientation="h", y=1.08), margin=dict(t=40))
st.plotly_chart(fig, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric(f"Career today (age {p.age})", chf(bs["H"].iloc[0]))
c2.metric(f"Pension at {p.retirement_age}", chf(bs["P"].iloc[-1]))
c3.metric(f"Portfolio at {p.retirement_age} (projection)", chf(bs["F"].iloc[-1]))

st.markdown("""
**Reading.** While you work, your career is by far your largest asset. As you age it turns into pension and savings,
so the share you cannot trade (ω, black line) falls. That is why the right portfolio changes with age.
The portfolio line is a simple projection (fixed return plus savings), not the optimizer.
""")

with st.expander("How sensitive is the value of the career to the assumptions?"):
    betas = [0.0, 0.15, 0.3, 0.6, 1.0]
    growths = [0.0, 0.01, 0.02, 0.03]
    tab = pd.DataFrame({f"salary growth {g:.0%}": [career.human_capital(p.with_(beta_S=b, salary_growth=g))
                                                     for b in betas] for g in growths},
                       index=[f"β_S = {b}" for b in betas])
    st.dataframe(tab.style.format(lambda v: chf(v)), width="stretch")
    st.caption("A riskier career is discounted at a higher rate, so it is worth less today.")
