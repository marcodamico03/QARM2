"""Page "Analyst view": every number behind the plan, for the adviser and the class (step 5).

The side panel changes the client (it starts from the client of "Your plan"); every tab updates
at once. The backtest and the robustness tests run by themselves in the background and are kept
on disk (python -m src.precompute computes them in advance).
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import backtest as B
from src import dashboard as D
from src import optimizer as O
from src import theme as T
from views import common as C

html, show = C.html, C.show
STRATEGY_NAMES = {"A": "Standard advice", "B": "100 minus age", "D": "Total wealth, no hedge",
                  "C": "Total wealth (our advice)"}
STRATEGY_COLORS = {"A": "#9C98B3", "B": "#86AEDD", "D": T.SAFFRON, "C": T.PLUM}


def crisis_bands(fig, crises):
    """Shaded crises with their names on two rows, so that close crises do not overlap."""
    for i, (crisis, (first, last)) in enumerate(crises.items()):
        x0 = pd.Timestamp(first) - pd.offsets.MonthEnd(1)
        fig.add_vrect(x0=x0, x1=pd.Timestamp(last) + pd.offsets.MonthEnd(0), fillcolor=T.RED, opacity=0.08,
                      line_width=0)
        fig.add_annotation(x=x0, y=1.0 if i % 2 == 0 else 0.92, yref="paper", text=crisis, showarrow=False,
                           xanchor="left", font=dict(size=10, color=T.MUTED))


# =============================================================================
# Side panel: who is the client?
# =============================================================================

params0 = C.base_params()
APP = params0["app"]
personas = params0["personas"]
plan_persona, plan_changes = st.session_state.get("plan_client", [params0["client"]["persona"], {}])
with st.sidebar:
    html('<div style="font-family:' + T.DISPLAY + ';font-weight:800;font-size:1.6rem;letter-spacing:-.02em;'
         'margin:0 0 .2rem">Beyond the Portfolio</div><div class="btp-quiet" style="margin-bottom:1.1rem">'
         'Pick a client, then change anything.</div>')
    persona = st.selectbox("Client", list(personas), index=list(personas).index(plan_persona),
                           format_func=lambda k: personas[k]["label"], label_visibility="collapsed")
    start = plan_changes if persona == plan_persona else {}          # the client of "Your plan"
    p = personas[persona]
    has_sector = p["sector"] is not None
    has_shares = p.get("deferred_shares_chf", 0) > 0
    k = f"_{persona}"                                   # widgets restart from the persona's values

    if st.button("Back to this client's defaults"):
        for name in list(st.session_state):
            if name.endswith(k):
                del st.session_state[name]
        st.rerun()

    with st.expander("Life and salary", expanded=True):
        age = st.slider("Age", *APP["age_range"], int(start.get("age") or p["age"]), key="age" + k)
        salary = st.slider("Gross salary (CHF a year)", *APP["salary_range"],
                           int(start.get("salary_chf") or p["salary_chf"]), APP["salary_step"], key="salary" + k,
                           format="%d")
    with st.expander("Savings", expanded=True):
        choices = sorted(set(APP["portfolio_choices"]) | {int(start.get("portfolio_chf")
                                                               or params0["client"]["portfolio_chf"])})
        portfolio = st.select_slider("Portfolio (CHF)", choices, value=int(start.get("portfolio_chf")
                                                                           or params0["client"]["portfolio_chf"]),
                                     key="portfolio" + k, format_func=lambda v: f"{v:,.0f}".replace(",", "'"))
        shares = None
        if has_shares:
            shares = st.slider("Locked employer shares (CHF)", 0, int(portfolio),
                               min(int(start.get("deferred_shares_chf") or p["deferred_shares_chf"]), int(portfolio)),
                               APP["shares_step"], key="shares" + k, format="%d")
        own_pension = None
        if p.get("pillar2", True) and st.toggle("Enter the pension from the certificate",
                                                value=start.get("pension_chf") is not None, key="pension_on" + k):
            estimate = O.pension_savings(age, salary, D.client_params(persona, None, params0))
            own_pension = st.number_input("Pension savings (CHF)", 0, APP["pension_max"],
                                          int(start.get("pension_chf") or round(estimate, -3)), APP["pension_step"],
                                          key="pension" + k)
    with st.expander("How the job moves with markets"):
        beta_sector = st.slider("Link with the sector (beta)", 0.0, 1.0,
                                float(start["beta_sector"] if start.get("beta_sector") is not None
                                      else p["beta_sector"]), 0.05,
                                key="beta" + k, disabled=not has_sector,
                                help="0 = the salary ignores the sector; 1 = it moves one for one")
        link = st.slider("Wages follow interest rates", 0.0, 1.0, float(params0["career"]["wage_rate_link"]),
                         0.05, key="link" + k,
                         help="0 = a fixed salary, like a bond; 1 = wages rise fully with rates")
    with st.expander("Rules"):
        long_only = st.toggle("No short selling", True, key="long" + k)
        no_leverage = st.toggle("No borrowing", True, key="lev" + k)
        caps = list(params0["optimization"]["position_caps"].values())
        cap = st.slider("Most in one sector or stock", 0.05, 1.0, float(caps[0]) if caps else 1.0, 0.05,
                        key="cap" + k, format="%.2f")
    with st.expander("A market view"):
        view = None
        if st.toggle("Add a view", key="view_on" + k):
            choices = [a for a in O.client_assets(D.client_params(persona, None, params0))]
            asset = st.selectbox("Asset", choices, format_func=lambda a: T.asset_name(a, params0), key="vasset" + k)
            low, high, step = APP["view_return_range"]
            q = st.slider("Expected return above cash (a year)", low, high, APP["view_default"], step, key="vq" + k,
                          format="%.3f")
            levels = list(APP["view_confidence"])
            confidence = st.radio("Confidence", levels, index=len(levels) // 2, horizontal=True, key="vc" + k)
            view = {"asset": asset, "q": q, "sd": APP["view_confidence"][confidence]}

changes = {"age": age, "salary_chf": salary, "portfolio_chf": portfolio, "deferred_shares_chf": shares,
           "pension_chf": own_pension, "beta_sector": beta_sector if has_sector else None, "wage_rate_link": link,
           "long_only": long_only, "no_leverage": no_leverage, "sector_cap": cap, "view": view}
key = C.key_of(persona, changes)
try:
    with st.spinner("Optimising..."):
        res = C.analysis(key)
except Exception as error:                                   # an impossible combination of settings
    st.error(f"These settings have no solution: {error}. Change a rule or the locked shares.")
    st.stop()

params, sheet, story = res["params"], res["balance_sheet"], res["story"]
sector = params["data"]["career_sector_asset"]
sector_name = T.asset_name(sector, params) if sector else None

# =============================================================================
# Hero
# =============================================================================

html(f'<p class="btp-who">{T.esc(p["label"])}, {age} years old, earns {T.chf(salary, short=False)} a year</p>')
share = story["portfolio_share"]
html(f'<div class="btp-hero">Your portfolio is {T.pct(share, 1)} of what you own.'
     f'<span class="soft">The rest is your job and your pension.</span></div>')
html(T.ribbon(sheet))

tabs = st.tabs(["Your wealth", "Your portfolio", "What the rules cost", "Where risk comes from",
                "Did it work?", "What we assume"])

# =============================================================================
# 1. Your wealth
# =============================================================================
with tabs[0]:
    holdings = res["career_holdings"]
    bond = params["data"]["career_bond_asset"]
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        html(T.plate(T.chf(holdings.get(bond, 0)), "of Swiss government bonds are already inside your job: "
                     "when interest rates fall, your future salary is worth more, like a bond.",
                     f"Bond beta {sheet['beta_bond']:.2f}", tint="blue", color=T.BLUE))
    with c2:
        if sector and sheet["beta_sector"] > 0:
            html(T.plate(T.chf(holdings.get(sector, 0)), f"of exposure to {sector_name} is already inside your "
                         "job: a bad year for your sector is a bad year for your career.",
                         f"Sector beta {sheet['beta_sector']:.2f}", tint="red", color=T.RED))
        else:
            html(T.plate("No sector", "Your salary does not move with any industry, so there is nothing "
                         "to avoid in the portfolio.", None, tint="red", color=T.RED))
    with c3:
        html(T.plate(T.pct(story["career_risk_share"]), "of the ups and downs of your total wealth come from "
                     "your career, even with the portfolio invested in equities.", None, tint="plum", color=T.PLUM))

    html(T.section("Your balance sheet until retirement",
                   f"Expected values, with no market surprise. Today the career is worth {T.chf(sheet['H'])}: the "
                   "remaining salaries, discounted. Every year one salary is paid out, so the career shrinks to zero "
                   "at retirement, while part of the salary is saved into the portfolio and the pension receives "
                   "its legal credits. The total falls because most of each salary is spent."))
    lc = res["lifecycle"]
    fig = go.Figure()
    for col, name, color in (("H", "Career", T.PLUM), ("P", "Pension", T.GREEN), ("F", "Portfolio", T.SAFFRON)):
        fig.add_trace(go.Scatter(x=lc.index, y=lc[col], name=name, stackgroup="w", mode="lines",
                                 line=dict(width=0, color=color, shape="spline"), fillcolor=color,
                                 hovertemplate=name + ": CHF %{y:,.0f}<extra>age %{x:.0f}</extra>"))
    fig.update_layout(**T.plot_layout(400, hovermode="x unified"))
    fig.update_xaxes(title="Age", dtick=5)
    fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
    show(fig)
    html(T.facts([(f"{sheet['omega']:.1%}", "of wealth cannot be traded (omega)"),
                  (f"{sheet['discount_rate']:.2%}", "discount rate of the career"),
                  (f"{res['gamma']:.3f}", "risk tolerance, the same for everybody"),
                  (T.chf(sheet["P"]), "pension today" + (" (legal estimate)" if own_pension is None else ""))]))

# =============================================================================
# 2. Your portfolio
# =============================================================================
with tabs[1]:
    html(T.section(f"Three ways to invest {T.chf(sheet['F'])}",
                   "Same expected returns (Black-Litterman), same risk tolerance, same rules. Only what the "
                   "adviser knows about your job changes."))
    weights = res["weights"]
    order = [a for a in weights.index if weights.loc[a].abs().max() > 0.0005]
    perf = res["performance"]
    notes = {"A": "career ignored", "D": f"{T.chf(story['gain_D_A'])} a year better than standard",
             "C": f"{T.chf(perf.loc['gain over (A), CHF per year', 'C'])} a year better than standard"}
    for s in ("A", "D", "C"):
        html(T.allocation(STRATEGY_NAMES[s], weights.loc[order, s], params, notes[s]))
    html(T.chips(order, params))

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(T.plate(f"{T.pct(story['bonds_A'])} → {T.pct(story['bonds_D'])}",
                     "in bonds. Your job already behaves like a large bond, so the portfolio can carry more "
                     f"equity risk. Worth {T.chf(story['gain_D_A'])} a year in certainty equivalent.",
                     "Standard advice compared with total wealth, no hedge", tint="blue", color=T.BLUE))
    with c2:
        if sector and story["sector_D"] > 0.0005:
            html(T.plate(f"{T.pct(story['sector_D'])} → {T.pct(story['sector_C'])}",
                         f"in {sector_name}. Your job already holds this exposure, so the portfolio holds less. "
                         f"Worth {T.chf(story['gain_C_D'])} a year.",
                         "Total wealth, no hedge compared with our advice", tint="red", color=T.RED))
        else:
            html(T.plate("Nothing to hedge", "Even without the hedge, the portfolio would hold none of your "
                         "sector, so avoiding it changes nothing here.", None, tint="red", color=T.RED))

    with st.expander("What the model would buy with no rule at all"):
        dem = D.demand_chf(res)
        dem = dem[(dem.abs() > 1).any(axis=1)]
        names = [T.asset_name(a, params) for a in dem.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=names, x=dem["speculative"], name="To earn the risk premium", orientation="h",
                             marker_color=T.SAFFRON, hovertemplate="CHF %{x:,.0f}<extra></extra>"))
        fig.add_trace(go.Bar(y=names, x=dem["hedging"], name="To offset your job", orientation="h",
                             marker_color=T.PLUM, hovertemplate="CHF %{x:,.0f}<extra></extra>"))
        fig.update_layout(**T.plot_layout(60 + 42 * len(names), barmode="relative"))
        fig.update_xaxes(tickprefix="CHF ", tickformat="~s", showgrid=True, gridcolor="#ECEAF3",
                         zeroline=True, zerolinecolor=T.INK)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig)
        st.caption("Positions in CHF. The second part is exactly minus your job's hidden holdings. "
                   "Without rules the client would borrow and sell short large amounts: the rules make the answer "
                   "realistic, and the next page shows what they cost.")
    with st.expander("Expected returns and risk"):
        er = res["expected_returns"].copy()
        er.index = [T.asset_name(a, params) for a in er.index]
        f = lambda v: T.pct(v, 1)
        html(T.table(er, {c: f for c in er.columns}))
        st.caption(f"Annual, above the CHF risk-free rate, {res['months']} months of data. Prior: global market "
                   f"portfolio (Doeswijk and Swinkels, 2024), market risk aversion {res['phi']:.2f}; "
                   f"covariance with Ledoit-Wolf shrinkage ({res['shrinkage']:.2f}).")

# =============================================================================
# 3. What the rules cost
# =============================================================================
with tabs[2]:
    html(T.section("What each rule costs you",
                   "The value of one percentage point more room, per year, from the solver's Lagrange "
                   "multipliers. Rules that do not bind cost nothing and are not shown."))
    marginal = res["shadow_marginal"].sort_values("CHF per year", ascending=False)
    words = {"No leverage": ("Borrowing", "Being allowed to borrow 1% of the portfolio and invest it"),
             "Lock-up": ("Locked shares", "Being allowed to sell 1% of the locked employer shares"),
             "Long-only": ("Short selling", "Being allowed to sell short 1% in {}"),
             "Position caps": ("Cap", "Holding 1% more than the cap in {}")}
    tags = []
    for _, row in marginal.iterrows():
        asset = row["constraint"].split(" at ")[0]
        what = words[row["group"]][1].format(T.esc(T.asset_name(asset, params)))
        color = {"No leverage": T.PLUM, "Lock-up": "#8A5A44", "Long-only": T.BLUE}.get(row["group"], T.RED)
        tags.append((T.chf(row["CHF per year"], short=False), "a year", what, color))
    html(T.price_tags(tags) if tags else T.plate("Free", "No rule binds for this client.", tint="green"))

    html(T.section("If a rule disappeared completely",
                   "The problem solved again without the whole rule. These are upper bounds: without the borrowing "
                   "rule a young client would borrow many times the portfolio."))
    total = res["shadow_total"].set_index("group")
    total = total[["binding", "CHF per year", "invested without it"]]
    total["binding"] = total["binding"].map({True: "yes", False: "no"})
    html(T.table(total.rename(columns={"binding": "Binds today", "CHF per year": "Gain, CHF a year",
                                       "invested without it": "Invested without it"}),
                 {"Gain, CHF a year": lambda v: T.chf(v, short=False),
                  "Invested without it": lambda v: T.pct(v)}))

# =============================================================================
# 4. Where risk comes from
# =============================================================================
with tabs[3]:
    html(T.section("Where the risk of your total wealth comes from",
                   "Euler risk contributions on the whole balance sheet: each piece's share of the volatility "
                   "of total wealth."))
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        pick = st.radio("Portfolio", ["C", "A"], horizontal=True, format_func=lambda s: STRATEGY_NAMES[s],
                        label_visibility="collapsed")
        risk = res["risk"][pick]["share of risk"]
        main = risk[risk.abs() >= 0.005]
        rest = risk[risk.abs() < 0.005].sum()
        labels = [T.asset_name(a, params) for a in main.index] + (["Other"] if abs(rest) > 1e-9 else [])
        values = list(main.to_numpy()) + ([rest] if abs(rest) > 1e-9 else [])
        colors = [T.PLUM if a == O.CAREER else T.GREEN if a == O.PENSION else T.asset_color(a)
                  for a in main.index] + ["#C9C7D6"]
        fig = go.Figure(go.Pie(labels=labels, values=np.clip(values, 0, None), hole=0.74, sort=False,
                               marker=dict(colors=colors, line=dict(color="#fff", width=3)), textinfo="none",
                               hovertemplate="%{label}: %{percent}<extra></extra>"))
        fig.update_layout(**T.plot_layout(380, showlegend=True, legend=dict(orientation="v", x=1, y=0.5),
                                          annotations=[dict(text=f"<b>{T.pct(risk.get(O.CAREER, 0))}</b><br>"
                                                            "your career", showarrow=False,
                                                            font=dict(family=T.DISPLAY, size=26, color=T.PLUM))]))
        show(fig)
    with c2:
        cov = res["covariance"].to_numpy()
        rows = {}
        for s in ("A", "C"):
            x = res["solutions"][s]["x"]
            rc = O.risk_contributions(x, cov)
            rows[s] = rc / rc.sum() if rc.sum() > 0 else rc
        rc = pd.DataFrame(rows, index=res["assets"])
        rc = rc[(rc.abs() > 0.001).any(axis=1)]
        fig = go.Figure()
        for s in ("A", "C"):
            fig.add_trace(go.Bar(y=[T.asset_name(a, params) for a in rc.index], x=rc[s], orientation="h",
                                 name=STRATEGY_NAMES[s], marker_color=STRATEGY_COLORS[s],
                                 hovertemplate="%{x:.1%}<extra></extra>"))
        html(T.chart_title("Inside the portfolio only"))
        fig.update_layout(**T.plot_layout(340, barmode="group"))
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor="#ECEAF3")
        fig.update_yaxes(autorange="reversed")
        show(fig)
    st.caption("Left: total wealth, career and pension included. Right: the classic view of lab LN5, the "
               "portfolio alone.")

# =============================================================================
# 5. Did it work?
# =============================================================================
with tabs[4]:
    html(T.section("Would it have worked from 2005 to 2026?",
                   f"The same client, {age} years old in September 2005, followed month by month: each decision "
                   "uses only the five years before it, with costs, savings and the real CHF interest rate."))
    bt = C.background("backtest", params)
    if bt is None:
        C.wait_for("Running the backtest for this client (a few seconds the first time).")
    else:
        gains = bt["gains"]
        c1, c2 = st.columns([1.15, 1], gap="medium")
        with c1:
            html(T.chart_title("Gain over standard advice, CHF a year", "With the 95% interval (block bootstrap)."))
            labels = {"(D) vs (A)": "Take more risk", "(C) vs (A)": "Total wealth", "(C) vs (D)": "The hedge"}
            g = gains.loc[list(labels)]
            fig = go.Figure(go.Bar(
                x=[f"{labels[i]}<br><span style='color:{T.MUTED}'>p = {g.loc[i, 'p-value']:.2f}</span>"
                   for i in g.index], y=g["CE gain (CHF/yr)"],
                marker_color=[T.SAFFRON, T.PLUM, T.RED],
                error_y=dict(type="data", symmetric=False, array=g["95% high (CHF/yr)"] - g["CE gain (CHF/yr)"],
                             arrayminus=g["CE gain (CHF/yr)"] - g["95% low (CHF/yr)"], color=T.INK, thickness=1.2,
                             width=8),
                hovertemplate="CHF %{y:,.0f} a year<extra></extra>"))
            fig.update_layout(**T.plot_layout(340))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s", zeroline=True, zerolinecolor=T.INK)
            show(fig)
        with c2:
            m = bt["metrics"]
            mm = {s: m[B.STRATEGIES[s]] for s in "ABDC"}
            worst = "CHF in the sector's worst 10% months"
            html(T.plate(T.chf(gains.loc["(C) vs (A)", "CE gain (CHF/yr)"]),
                         "a year more certainty equivalent than standard advice, "
                         f"p = {gains.loc['(C) vs (A)', 'p-value']:.3f} (block bootstrap).",
                         "Same wealth shares for every strategy, so the comparison is fair.", tint="plum",
                         color=T.PLUM))
            html("<div style='height:12px'></div>")
            if sector:
                html(T.plate(f"{mm['D']['correlation with the sector part']:.2f} → "
                             f"{mm['C']['correlation with the sector part']:.2f}",
                             "correlation of the portfolio with the sector part of the career, without and with "
                             "the hedge.", f"In the sector's worst months: {T.chf(mm['D'][worst], short=False)} "
                             f"against {T.chf(mm['C'][worst], short=False)} a month.", tint="red", color=T.RED))

        vals = bt["values"]
        html(T.chart_title(f"Portfolio value, with {T.chf(bt['savings'])} of savings added over the years",
                           "Shaded: the crises of the period."))
        fig = go.Figure()
        crisis_bands(fig, params["backtest"]["crises"])
        for s in ("A", "B", "D", "C"):
            fig.add_trace(go.Scatter(x=vals.index, y=vals[s], name=STRATEGY_NAMES[s], mode="lines",
                                     line=dict(color=STRATEGY_COLORS[s], width=3.2 if s == "C" else 2),
                                     hovertemplate=STRATEGY_NAMES[s] + ": CHF %{y:,.0f}<extra></extra>"))
        fig.update_layout(**T.plot_layout(440, hovermode="x unified",
                                          legend=dict(orientation="h", yanchor="top", y=-0.08, x=0)))
        fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
        show(fig)

        crises = bt["crises"]["portfolio return"].unstack("strategy")
        order = [c for c in params["backtest"]["crises"] if c in crises.index]
        crises = crises.loc[order, ["A", "B", "D", "C"]].rename(columns=STRATEGY_NAMES)
        html(T.section("In the crises", "Return of each portfolio over the crisis."))
        html(T.table(crises, {c: (lambda v: T.pct(v, 1)) for c in crises.columns}, signed=list(crises.columns)))

        html(T.section("Robustness", "Wrong beta, job loss, another prior, other clients. About half a minute "
                       "the first time for a new client, then instant."))
        rob = C.background("robustness", params)
        if rob is None:
            C.wait_for("Running the robustness tests in the background (about half a minute).")
        else:
            if rob["wrong_beta"] is None:
                html(T.plate("No sector", "The salary follows no sector, so there is no hedge to test: wrong beta, "
                             "job loss and hedge map do not apply. The prior test is below.", tint="red",
                             color=T.RED))
            else:
                wb = rob["wrong_beta"]
                html(T.chart_title("Wrong beta", "The decisions use the assumed beta; the career is replayed with the "
                                   "true one."))
                fig = go.Figure()
                for col, name, color, dash in (("CE gain (C) vs (D) (CHF/yr)", "Gain of the hedge", T.PLUM, "solid"),
                                               ("risk part (CHF/yr)", "Risk part", T.GREEN, "dash"),
                                               ("return part (CHF/yr)", "Return part", T.SAFFRON, "dot")):
                    fig.add_trace(go.Scatter(x=wb["true beta_sector"], y=wb[col], name=name, mode="lines+markers",
                                             line=dict(color=color, width=3, dash=dash), marker=dict(size=8),
                                             hovertemplate="CHF %{y:,.0f}<extra></extra>"))
                fig.add_vline(x=params["career"]["beta_sector"], line_color=T.MUTED, line_dash="dot",
                              annotation_text="assumed", annotation_font=dict(color=T.MUTED))
                fig.update_layout(**T.plot_layout(340))
                fig.update_xaxes(title="True sector beta")
                fig.update_yaxes(tickprefix="CHF ", zeroline=True, zerolinecolor=T.INK)
                show(fig)
                st.caption(f"The risk part is a straight line in the true beta (R² "
                           f"{wb.attrs['risk part: R2 of the straight line']:.4f}), as the theory h(2β − h) says; "
                           f"it turns positive at a true beta of {wb.attrs['risk part: break-even true beta']:.3f}.")

                hm = rob["hedge_map"]["risk part (CHF/yr)"].unstack("beta_sector")
                html(T.chart_title("Hedge map", "Risk part of the hedge, CHF a year, by starting portfolio and "
                                   "sector beta."))
                fig = go.Figure(go.Heatmap(z=hm.to_numpy(), x=[f"beta {b:.2f}" for b in hm.columns],
                                           y=[T.chf(v) for v in hm.index], colorscale=[[0, "#F3EEF8"], [1, T.PLUM]],
                                           text=[[T.chf(v) for v in row] for row in hm.to_numpy()],
                                           texttemplate="%{text}", showscale=False, xgap=4, ygap=4,
                                           hovertemplate="%{y}, %{x}: risk part %{text} a year<extra></extra>"))
                fig.update_layout(**T.plot_layout(300))
                show(fig)

                jl = rob["job_loss"]
                cols = ["jobs lost", "(C) vs (A) CHF/yr", "(C) vs (D) CHF/yr", "hedge: risk part"]
                html(T.section("Losing the job", "Twelve months without salary when the sector falls this much "
                               "below its peak."))
                html(T.table(jl[cols].rename(columns={"jobs lost": "Jobs lost", "(C) vs (A) CHF/yr": "Total wealth, CHF/yr",
                                                      "(C) vs (D) CHF/yr": "Hedge, CHF/yr",
                                                      "hedge: risk part": "Hedge risk part"}),
                             {c: (lambda v: T.chf(v, short=False)) for c in ("Total wealth, CHF/yr", "Hedge, CHF/yr",
                                                                             "Hedge risk part")},
                             signed=["Total wealth, CHF/yr", "Hedge, CHF/yr", "Hedge risk part"]))
            pr = rob["prior"]
            cols = ["(C) vs (A) CHF/yr", "(C) vs (A) p-value", "(C) vs (D) CHF/yr", "hedge: risk part"]
            html(T.section("Another prior", "The global market portfolio against a Swiss pension fund. With the "
                           "Swiss prior the gain of total wealth is smaller and only borderline significant."))
            html(T.table(pr[cols].rename(columns={"(C) vs (A) CHF/yr": "Total wealth, CHF/yr",
                                                  "(C) vs (A) p-value": "p-value",
                                                  "(C) vs (D) CHF/yr": "Hedge, CHF/yr",
                                                  "hedge: risk part": "Hedge risk part"}),
                         {"Total wealth, CHF/yr": lambda v: T.chf(v, short=False), "p-value": lambda v: f"{v:.3f}",
                          "Hedge, CHF/yr": lambda v: T.chf(v, short=False),
                          "Hedge risk part": lambda v: T.chf(v, short=False)},
                         signed=["Total wealth, CHF/yr", "Hedge, CHF/yr", "Hedge risk part"]))

# =============================================================================
# 6. What we assume
# =============================================================================
with tabs[5]:
    html(T.section("Every assumption on one page", "Values used for this client, and where they come from. "
                   "All of them live in config/params.yaml."))
    c, pen, bl, bt_p, opt = (params["career"], params["pension"], params["black_litterman"], params["backtest"],
                             params["optimization"])
    rows = [
        ("Retirement age", f"{params['client']['retirement_age']}", "Swiss reference age"),
        ("Salary growth", T.pct(c["salary_growth"], 1) + " a year", "Assumption"),
        ("Long-run CHF rate", T.pct(c["rf_long_run"], 1), "Assumption, used to value the career"),
        ("Bond and sector premia", f"{T.pct(c['bond_premium'], 1)} and {T.pct(c['sector_premium'], 1)}",
         "Assumption, used to value the career"),
        ("Bond beta of the career", "duration of the salary / duration of CHF bonds (9)", "Step 2, career model"),
        ("Wages follow rates", f"{c['wage_rate_link']:.2f}", "Roberto's fix: a civil servant is not riskier than a banker"),
        ("Sector beta", f"{c['beta_sector']:.2f}", "Persona preset, cautious (robustness test)"),
        ("Pension credits", "7, 10, 15, 18% of the coordinated salary", "BVG art. 16"),
        ("Coordination deduction, upper limit", f"CHF {pen['coordination_deduction']:,} and CHF "
         f"{pen['max_insured_salary']:,}", "BSV key figures 2025-2026"),
        ("Pension interest", T.pct(pen["interest_rate"], 2), "BVG minimum 2024-2026 (2.0% from 2027)"),
        ("Market portfolio", "equities 49.7%, bonds 43.5%, real estate 4.4%, commodities 2.4%",
         "Doeswijk and Swinkels (2024), Table 1"),
        ("Market Sharpe ratio", f"{bl['sharpe_ratio']:.3f}", "Doeswijk and Swinkels (2024), Table 2"),
        ("Black-Litterman tau", f"{bl['tau']:.2f}", "Lab LN4"),
        ("Risk tolerance", f"{res['gamma']:.3f}", "A client with no career is as risky as 60/40 (fsolve, LN1)"),
        ("Covariance", "Ledoit-Wolf, constant correlation", "LN3"),
        ("Rules", f"no short selling, no borrowing, at most {T.pct(cap)} in one sector or stock", "Assumption"),
        ("Locked employer shares", T.chf(params["client"]["deferred_shares_chf"], short=False), "Assumption"),
        ("Savings", f"{T.pct(bt_p['savings_rate'])} of the gross salary", "Assumption (backtest)"),
        ("Trading costs", f"{bt_p['transaction_cost_bp']} bp of turnover", "Proposal"),
        ("Estimation window", f"{bt_p['window_months']} months", "Proposal"),
    ]
    html(T.table(pd.DataFrame(rows, columns=["Assumption", "Value", "Source"]).set_index("Assumption")))
    st.caption("Data: Yahoo Finance, Kenneth French, FRED and the SNB data portal, monthly in CHF, saved in data/.")
