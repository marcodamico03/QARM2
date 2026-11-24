import common  # noqa: F401  -- must stay first: tells Python where src/ is
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtest as bt
from common import STRAT_COLORS, STRAT_LABELS, backtest_results, chf, get_persona, setup, wrong_beta

setup("Did it work?", "📈")
p = get_persona()
st.title("6 · Did it work? Walk-forward backtest")
st.markdown(f"""
The client starts at **age {p.age} in October 2006** and follows each strategy month by month until September 2026.
At every month-end only **past** data is used (rolling 60 months), the portfolio is re-optimised and held for one month.
Costs of 10 bp per unit of turnover; negative CHF rates included.
""")

jl = st.radio("Job loss in banking crises", ["none", "20%", "30%", "40%"], horizontal=True,
              help="12 months without salary when the client's sector falls this much below its 2-year high.")
th = None if jl == "none" else int(jl[:-1]) / 100
res = backtest_results(p, th)


def shade(fig):
    for a, b in bt.CRISES.values():
        fig.add_vrect(x0=pd.Period(a, "M").start_time, x1=pd.Period(b, "M").end_time,
                      fillcolor="grey", opacity=0.12, line_width=0)


fig = go.Figure()
for k in ["A", "B", "D", "C"]:
    d = res[k]
    fig.add_trace(go.Scatter(x=d.index.to_timestamp(), y=d["F"], name=STRAT_LABELS[k],
                             line=dict(color=STRAT_COLORS[k], width=3 if k == "C" else 1.8)))
shade(fig)
fig.update_layout(height=420, yaxis=dict(title="portfolio value (CHF)", tickformat=",.0f"),
                  legend=dict(orientation="h", y=1.1), margin=dict(t=30))
st.plotly_chart(fig, width="stretch")
if th:
    u = res["C"]["employed"]
    starts = [str(d) for d, a, b in zip(u.index, u.shift(fill_value=True), u) if a and not b]
    st.caption("Job losses start in: " + (", ".join(starts) if starts else "none"))

st.subheader("Scorecard")
M = bt.metrics(res).rename(columns={v: STRAT_LABELS[k] for k, v in bt.LABELS.items()})
M = M[[STRAT_LABELS[k] for k in ["A", "B", "D", "C"]]]
fmt = {"CE return on total wealth": "{:.2%}", "CE gain vs A (CHF/yr)": "{:+,.0f}",
       "corr(portfolio, sector part of career)": "{:.2f}",
       "avg portfolio P&L in worst 10% sector months (CHF)": "{:+,.0f}",
       "vol of total wealth": "{:.1%}", "max drawdown of total wealth": "{:.1%}", "vol of portfolio": "{:.1%}",
       "max drawdown of portfolio": "{:.1%}", "turnover per year": "{:.0%}", "tracking error vs A": "{:.1%}",
       "final portfolio F (CHF)": "{:,.0f}"}
st.dataframe(M.apply(lambda r: r.map(lambda v: "–" if pd.isna(v) else fmt[r.name].format(v)), axis=1),
             width="stretch")
st.caption("Headline: certainty-equivalent (CE) return on total wealth, the quantity every strategy maximises.")

if True:
    st.subheader("Crises, in CHF")
    ct = bt.crisis_table(res).rename(columns={v: STRAT_LABELS[k] for k, v in bt.LABELS.items()})
    st.dataframe(ct[[STRAT_LABELS[k] for k in ["A", "B", "D", "C"]]].style.format("{:+,.0f}"),
                 width="stretch")
    st.caption("Portfolio profit or loss during each crisis, computed in the backtest.")
if True:
    st.subheader("What if our β_S is wrong?")
    wb = wrong_beta(p)
    f2 = go.Figure()
    for col, dash in [("risk benefit C vs D (bp/yr)", "solid"), ("return effect C vs D (bp/yr)", "dash"),
                      ("CE gain C vs D (bp/yr)", "dot")]:
        f2.add_trace(go.Scatter(x=wb.index, y=wb[col], name=col.replace(" (bp/yr)", ""), line=dict(dash=dash)))
    f2.add_hline(y=0, line_color="grey")
    f2.update_layout(height=330, xaxis_title="true β_S / assumed β_S", yaxis_title="bp per year",
                     legend=dict(orientation="h", y=-0.3), margin=dict(t=10))
    st.plotly_chart(f2, width="stretch")
    st.caption("The risk benefit of avoiding your sector grows with your true exposure, as theory predicts. "
               "The return effect depends on this one historical path.")

st.warning("**Honesty note.** A backtest is one historical path, and part of the career return is modelled, "
           "not observed. It tests the logic and the robustness of the strategy, not a guaranteed extra return.")
