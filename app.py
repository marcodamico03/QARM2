"""Beyond the Portfolio: the Streamlit app (step 5, version 2 of the advice).

Run from the project folder:
    streamlit run app.py

One page. The client is in the side panel: industry, age, pay, savings, employer shares and when
they can be sold, risk profile. The answer fits on one screen: the plan in CHF and why, the
backtest since 2005, where the risk of the savings comes from, and the plan by age. The tabs below
hold every detail. Everything updates by itself; the backtest runs in the background.
The numbers come from src/allocation.py (risk budgeting, LN5-LN6) and steps 1 to 4; the look is in
src/theme.py. The version of the app before the pivot is the git tag app-v1.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit import config

from src import allocation as A
from src import dashboard as D
from src import optimizer as O
from src import theme as T
from src.config import load_params
from src.jobs import Jobs

T.use("light" if st.session_state.get("light", False) else "dark")   # dark unless the switch top right is on
st.set_page_config(page_title="Beyond the Portfolio", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")
THEME = T.streamlit_theme()
if any(config.get_option(f"theme.{name}") != value for name, value in THEME.items()):
    for name, value in THEME.items():                       # Streamlit's own widgets follow the same palette
        config.set_option(f"theme.{name}", value)
    st.rerun()
st.markdown(T.css(), unsafe_allow_html=True)
CHART_CONFIG = {"displayModeBar": False, "responsive": True}
NAMES = A.STRATEGIES
COLORS = {"A": T.TYPICAL, "B": T.STRATEGY_B, "C": T.SAVINGS}


# =============================================================================
# Data and results (kept in memory; the backtests also on disk, data/cache/)
# =============================================================================

@st.cache_data(show_spinner=False)
def base_params():
    return load_params()


@st.cache_data(show_spinner=False)
def market_data():
    return A.load_data(base_params())


@st.cache_resource(show_spinner=False)
def jobs():
    """One queue of background jobs for the whole app."""
    return Jobs()


@st.cache_data(show_spinner=False, max_entries=64)
def analysis(key):
    persona, changes = json.loads(key)
    return A.analyze(A.client_params(persona, changes, base_params()), market_data())


@st.cache_data(show_spinner=False, max_entries=64)
def glide(key):
    persona, changes = json.loads(key)
    params = A.client_params(persona, changes, base_params())
    res = analysis(key)
    return A.glide_path(params, res["mu"], res["cov"], mu_prior=res["mu_prior"])


def background(name, params):
    """Result of D.remember(name, params): at once if it is saved, otherwise started in the background and
    None for now. The page then checks every second and reruns when it is ready (poll at the end)."""
    result = D.saved(name, params)
    if result is not None:
        return result
    future = jobs().get(D.cache_key(name, params), lambda: D.remember(name, params, market_data()), group=name)
    return future.result() if future.done() else None       # an error in the computation is raised here


def html(text):
    st.markdown(text, unsafe_allow_html=True)


def show(fig):
    """Plotly chart at full width, with the new and the old Streamlit API."""
    try:
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def choice(label, options, default, key, format_func=str):
    """One choice among a few, as a segmented switch (a radio on an old Streamlit)."""
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(label, options, default=default, key=key, format_func=format_func)
        return value if value is not None else default        # a click on the chosen one unselects it
    return st.radio(label, options, index=list(options).index(default), horizontal=True, key=key,
                    format_func=format_func)


def crisis_bands(fig, crises):
    """Shaded crises, with their names on two rows so that close crises do not overlap."""
    for i, (crisis, (first, last)) in enumerate(crises.items()):
        x0 = pd.Timestamp(first) - pd.offsets.MonthEnd(1)
        fig.add_vrect(x0=x0, x1=pd.Timestamp(last) + pd.offsets.MonthEnd(0), fillcolor=T.SECTOR, opacity=0.10,
                      line_width=0)
        fig.add_annotation(x=x0, y=1.0 if i % 2 == 0 else 0.9, yref="paper", text=crisis, showarrow=False,
                           xanchor="left", font=dict(size=9, color=T.FAINT))


# =============================================================================
# Side panel: the client
# =============================================================================

P0 = base_params()
APP, PERSONAS, ALLOC = P0["app"], P0["personas"], P0["allocation"]
INDUSTRIES = APP["industry_names"]
examples = list(PERSONAS)
with st.sidebar:
    html(T.brand())
    persona = st.selectbox("Start from an example", examples, format_func=lambda k: PERSONAS[k]["label"], key="job")
    p = PERSONAS[persona]
    k = f"_{persona}"                                       # each example starts from its own values
    options = list(INDUSTRIES) + [None]
    industry = st.selectbox("I work in", options, index=options.index(p.get("industry")),
                            format_func=lambda i: INDUSTRIES.get(i, APP["no_industry"]), key="industry" + k)
    c1, c2 = st.columns([0.8, 1.2])
    age = c1.number_input("Age", *APP["age_range"], int(p["age"]), 1, key="age" + k)
    salary = c2.number_input("Salary, CHF a year", *APP["salary_range"], int(p["salary_chf"]), APP["salary_step"],
                             key="salary" + k)
    savings = st.number_input("Savings to invest, CHF (employer shares included)", *APP["savings_range"],
                              int(P0["client"]["portfolio_chf"]), APP["savings_step"], key="savings" + k)
    stake, unlock, variable = 0.0, 0.0, 0.0
    if industry:
        stake = st.number_input("Of which shares of my employer, CHF", 0, int(savings),
                                min(int(p.get("stake_chf", 0)), int(savings)), APP["shares_step"], key="stake" + k)
        if stake > 0:
            unlock = st.slider("They can be sold in (years)", 0, APP["unlock_max_years"], int(p.get("unlock_years", 0)),
                               key="unlock" + k)
        variable = st.slider("Part of my pay that depends on my industry (bonus)", 0.0, 1.0,
                             float(p.get("variable_pay", 0.0)), APP["variable_pay_step"], key="variable" + k,
                             format="%.2f")
    self_employed = st.toggle("I am self-employed (no pension fund)", value=not p.get("pillar2", True),
                              key="self" + k)
    pension = None
    if not self_employed:
        how = choice("Pension fund (2nd pillar)", ["Estimate", "My certificate"], "Estimate", "pension_how" + k)
        if how == "My certificate":
            estimate = O.pension_savings(age, salary, A.client_params(persona, None, P0))
            pension = st.number_input("Savings on the certificate, CHF", 0, APP["pension_max"],
                                      int(round(estimate, -3)), APP["pension_step"], key="pension" + k)
    profiles = list(ALLOC["profiles"])
    profile = choice("Risk profile", profiles, ALLOC["default_profile"], "profile" + k, format_func=str.capitalize)
    with st.expander("Advanced"):
        view = None
        if st.toggle("Add my own view on one market", key="view_on" + k):
            asset = st.selectbox("Market", A.assets(P0), format_func=T.asset_name, key="vasset" + k)
            low, high, step = APP["view_return_range"]
            q = st.slider("Expected return above cash, a year", low, high, APP["view_default"], step,
                          key="vq" + k, format="%.3f")
            levels = list(APP["view_confidence"])
            confidence = choice("Confidence", levels, levels[len(levels) // 2], "vc" + k)
            view = {"asset": asset, "q": q, "sd": APP["view_confidence"][confidence]}
            st.caption(f"A view is about the next years: it fades out over {ALLOC['view_years']} years in the "
                       "plan by age, and the backtest does not use it.")
    html('<p class="btp-side-note">Everything updates as you type. The backtest takes a few seconds the first '
         "time for a new client.</p>")

changes = {"industry": industry, "age": age, "salary_chf": salary, "portfolio_chf": savings,
           "stake_chf": float(stake), "unlock_years": float(unlock), "variable_pay": float(variable),
           "pillar2": not self_employed, "pension_chf": pension, "profile": profile, "view": view}
key = json.dumps([persona, changes], sort_keys=True)
try:
    res = analysis(key)
except Exception as error:
    st.error(f"No plan for these settings ({error}). Change the savings or the employer shares.")
    st.stop()

params, sheet = res["params"], res["sheet"]
client = params["client"]
F, W = sheet["F"], sheet["W"]
plans = res["plans"]
C_, A_ = plans["C"], plans["A"]
wC, wA = C_["weights"], A_["weights"]
gC, gA = C_["groups"], A_["groups"]
retire = client["retirement_age"]
ind_name = INDUSTRIES.get(industry, "") if industry else None
bt = background("backtest2", A.without_client_view(params))  # the backtest never uses the client's view
first_month = market_data()[0].index[int(params["backtest"]["window_months"])]

# =============================================================================
# Header
# =============================================================================
who, switch = st.columns([6, 1], vertical_alignment="center")
with who:
    html(f'<div class="btp-who">{T.esc(ind_name or APP["no_industry"])}, {age} years old, '
         f'{T.chf(salary, short=False)} a year {T.status(bt is None, "Backtest ready" if bt is not None else "Updating the backtest")}</div>')
with switch.container(key="theme_switch"):
    st.toggle("Light", key="light", help="Light or dark page")
parts = [f"{T.pct(gC['equities'])} in shares", f"{T.pct(gC['real'])} in real assets",
         f"{T.pct(gC['bonds'])} in bonds"]
if gC["cash"] >= APP["min_weight"] * 10:
    parts.append(f"{T.pct(gC['cash'])} in cash")
mix = ", ".join(parts[:-1]) + " and " + parts[-1]
html(f'<div class="btp-headline">Invest your {T.chf(F, short=False)}: <em>{mix}</em>.</div>')
dC, dA = C_["detail"]["safe_share"], A_["detail"]["safe_share"]
lede = (f"{T.pct(sheet['bond_like'])} of what you own already behaves like a bond (your future salary and "
        f"pension), so your savings need {T.pct(dC)} in bonds instead of the usual {T.pct(dA)}")
lede += f", and we leave out {ind_name.lower()}." if C_["detail"]["excluded"] else "."
html(f'<p class="btp-lede">{lede}</p>')
html(T.wealth({"H": sheet["H"], "P": sheet["P"], "F": F, "W": W}))

# =============================================================================
# The four panels
# =============================================================================
names = A.assets(params)
order = ["STAKE"] + [a for a in names if max(wC.get(a, 0), wA.get(a, 0)) >= APP["min_weight"]] + ["CASH"]


def with_stake(weights, detail):
    """The weights with the locked employer shares shown apart (in rose), not inside their industry."""
    w = weights.copy()
    w["STAKE"] = 0.0
    if detail["locked"] and industry:
        share = detail["stake"] / F
        w[industry] -= share
        w["STAKE"] = share
    return w.reindex(order).fillna(0).clip(lower=0)



left, right = st.columns([1, 1.25], gap="small")
with left:
    with st.container(key="panel_plan"):
        html(T.panel_title("Your portfolio", T.chf(F, short=False)))
        sC, sA = with_stake(wC, C_["detail"]), with_stake(wA, A_["detail"])
        html('<div class="btp-bars">' + T.bar_row("Our plan", sC, params, "ours", APP["min_weight"], labels=False)
             + T.bar_row("Typical", sA, params, "typical", APP["min_weight"], labels=False) + "</div>")
        rows = []
        if sC["STAKE"] > 0:
            rows.append((T.SECTOR, "Your employer shares", T.chf(sC["STAKE"] * F, short=False), T.pct(sC["STAKE"])))
        industries = [a for a in ALLOC["groups"]["equities"] if a.startswith("IND_") and sC.get(a, 0) >= APP["min_weight"]]
        shares = float(sC.reindex(ALLOC["groups"]["equities"]).fillna(0).sum())
        rows.append((T.SAVINGS, f"Shares in {len(industries)} industries and Switzerland", T.chf(shares * F, short=False),
                     T.pct(shares)))
        for g, label in (("real", "Real estate, gold, commodities"), ("bonds", "Bonds: Swiss, global, corporate")):
            if gC[g] >= APP["min_weight"]:
                rows.append((T.GROUP_COLORS[g], label, T.chf(gC[g] * F, short=False), T.pct(gC[g])))
        if gC["cash"] >= APP["min_weight"]:
            rows.append((T.FAINT, "Cash", T.chf(gC["cash"] * F, short=False), T.pct(gC["cash"])))
        html(T.holdings(rows))
        why = []
        if C_["detail"]["excluded"]:
            reason = []
            if client["variable_pay"] > 0:
                reason.append(f"{T.pct(client['variable_pay'])} of your pay")
            if client["stake_chf"] > 0:
                reason.append("your employer shares")
            why.append((T.SECTOR, f"<b>No {ind_name.lower()} in the rest.</b> {' and '.join(reason).capitalize()} "
                                  "already depend on it: a bad year there would hit you twice."))
        if C_["detail"]["locked"]:
            why.append((T.SECTOR, f"<b>Your employer shares stay {unlock:.0f} more year{'s' if unlock != 1 else ''}.</b> "
                                  "The rest is built around them; then they join the plan."))
        why.append((T.BONDS, f"<b>Your job works like a bond.</b> So {T.pct(dC)} in bonds, not {T.pct(dA)}."))
        moved = C_["detail"]["cash_from_loss_limit"]
        if moved > 0:
            why.append((T.SAVINGS, f"<b>Close to retirement.</b> A bad year could cost more than you can still save, "
                                   f"so {T.pct(moved)} goes to cash."))
        else:
            why.append((T.SAVINGS, f"<b>A bad year (1 in 20) costs about {T.chf(C_['detail']['loss'])}.</b> "
                                   f"Within your {profile} profile, with {T.chf(C_['detail']['savings_left'])} still "
                                   "to save before retiring."))
        html(T.why(why))

with right:
    with st.container(key="panel_backtest"):
        html(T.panel_title(f"Since {first_month:%B %Y}", "you then, our plan against typical advice"))
        if bt is None:
            html(T.waiting("Replaying the past for this client..."))
        else:
            values = bt["values"]
            fig = go.Figure()
            crisis_bands(fig, params["backtest"]["crises"])
            for s, width in (("A", 1.8), ("C", 2.6)):
                fig.add_trace(go.Scatter(x=values.index, y=values[s], name=NAMES[s], mode="lines",
                                         line=dict(color=COLORS[s], width=width),
                                         hovertemplate=NAMES[s] + ": CHF %{y:,.0f}<extra></extra>"))
            fig.update_layout(**T.plot_layout(214, hovermode="x unified", showlegend=False))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
            show(fig)
            m = bt["metrics"]
            g = bt["gains"].loc["(C) vs (A)"]
            html(T.stats([(T.chf(values["C"].iloc[-1]), f"today with our plan, against {T.chf(values['A'].iloc[-1])} "
                           "with typical advice", T.SAVINGS),
                          (f"{T.chf(g['CE gain (CHF/yr)'])} a year", f"better once risk is counted, p = {g['p-value']:.2f}",
                           None),
                          (T.pct(m.loc["portfolio max drawdown", NAMES["C"]]), "worst fall, against "
                           f"{T.pct(m.loc['portfolio max drawdown', NAMES['A']])} with typical advice", None)]))
            html(f'<p class="btp-panel-note">Someone like you, {bt["start_age"]:.0f} in {first_month:%B %Y}, saving '
                 f"{T.pct(params['backtest']['savings_rate'])} of the salary; each decision uses only the past "
                 f"{params['backtest']['window_months']} months, with trading costs.</p>")

left, right = st.columns([1, 1.25], gap="small")
with left:
    with st.container(key="panel_risk"):
        html(T.panel_title("Where the risk of your savings comes from",
                           f"{C_['diversification']['effective bets']:.0f} independent bets against "
                           f"{A_['diversification']['effective bets']:.0f}"))
        fig = go.Figure()
        for g, colour in (("stake", T.SECTOR), ("equities", T.SAVINGS), ("real", T.PENSION), ("bonds", T.BONDS)):
            values = [float(plans[s]["risk_parts"][g]) for s in ("C", "A")]
            if max(values) < APP["min_weight"]:
                continue
            label = "Employer shares" if g == "stake" else T.GROUP_NAMES[g]
            fig.add_trace(go.Bar(y=[NAMES["C"], NAMES["A"]], x=values, name=label, orientation="h", marker_color=colour,
                                 hovertemplate=label + ": %{x:.0%}<extra></extra>"))
        fig.update_layout(**T.plot_layout(170, barmode="stack", legend_traceorder="normal",
                                          legend=dict(orientation="h", yanchor="top", y=-0.3, x=0,
                                                      font=dict(color=T.MUTED, size=11)),
                                          margin=dict(l=4, r=8, t=4, b=40)))
        fig.update_xaxes(tickformat=".0%", range=[0, 1], showgrid=True, gridcolor=T.LINE)
        fig.update_yaxes(autorange="reversed")
        show(fig)
        dv = C_["diversification"]
        html(f'<p class="btp-panel-note">Risk contributions: the share of the ups and downs of your savings '
             f"that comes from each part. Volatility {T.pct(dv['volatility'], 1)} a year; diversification ratio "
             f"{dv['diversification ratio']:.2f} against {A_['diversification']['diversification ratio']:.2f} "
             "for typical advice.</p>")

with right:
    with st.container(key="panel_glide"):
        gl = glide(key).copy()
        html(T.panel_title("As you get older", "bonds grow as your career shrinks"))
        if len(gl) > 1:
            gl.loc[retire] = gl.iloc[-1]
            fig = go.Figure()
            for grp in ("equities", "real", "bonds", "cash"):
                if gl[("C", grp)].max() > APP["min_weight"]:
                    fig.add_trace(go.Scatter(x=gl.index, y=gl[("C", grp)], name=T.GROUP_NAMES[grp], stackgroup="mix",
                                             mode="lines", line=dict(width=0, color=T.GROUP_COLORS[grp]),
                                             fillcolor=T.GROUP_COLORS[grp],
                                             hovertemplate=T.GROUP_NAMES[grp] + ": %{y:.0%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=gl.index, y=1 - gl[("A", "bonds")] - gl[("A", "cash")],
                                     name="Typical advice, not bonds", mode="lines",
                                     line=dict(color=T.TEXT, width=1.5, dash="dot"),
                                     hovertemplate="Typical advice: %{y:.0%} outside bonds<extra></extra>"))
            fig.update_layout(**T.plot_layout(190, hovermode="x unified", legend_traceorder="normal"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
            show(fig)
        html(f'<p class="btp-panel-note">Our plan solved again every {APP["glide_step_years"]} years with the wealth '
             f"you are expected to have then. Every year a salary is paid out, so less of your wealth behaves like "
             f"a bond and the savings hold more bonds; near retirement the loss limit adds cash.</p>")

# =============================================================================
# Details
# =============================================================================
tabs = st.tabs(["Holdings", "Budgets and returns", "Backtest", "Assumptions"])

with tabs[0]:
    html('<div class="btp-h3">Every holding, in CHF</div><p class="btp-p">Our plan, typical advice (it knows '
         "neither your job nor your shares) and 100 minus age.</p>")
    table = pd.DataFrame({NAMES[s]: plans[s]["weights"] * F for s in ("C", "A", "B")})
    table = table[(table.abs() > 1).any(axis=1)]
    table.index = [T.asset_name(a) for a in table.index]
    html(T.table(table, {c: (lambda v: T.chf(v, short=False)) for c in table.columns}))

with tabs[1]:
    html('<div class="btp-h3">How the plan is built</div><p class="btp-p">Risk budgeting: every asset we buy '
         "carries a share of the risk equal to its budget. Between groups the budgets come from your risk "
         "profile; inside a group they follow the Sharpe ratios of the Black-Litterman expected returns, with a "
         "floor so no asset is dropped. Covariance with Ledoit-Wolf shrinkage.</p>")
    vol = np.sqrt(np.diag(res["cov"]))
    info = pd.DataFrame({"Market weight (prior)": res["estimates"]["market_weights"],
                         "Expected return above cash": res["mu"], "Volatility": vol,
                         "Sharpe ratio": res["mu"] / vol, "Risk budget, our plan": C_["detail"]["budgets"],
                         "Risk contribution, our plan": C_["risk"]})
    info.index = [T.asset_name(a) for a in info.index]
    fmt = {c: (lambda v: T.pct(v, 1)) for c in info.columns}
    fmt["Sharpe ratio"] = lambda v: f"{v:.2f}"
    html(T.table(info, fmt))
    st.caption(f"{res['months']} months of data. Market risk aversion {res['estimates']['phi']:.2f}, "
               f"shrinkage {res['estimates']['shrinkage']:.2f}.")

with tabs[2]:
    if bt is None:
        html(T.waiting("Running the backtest for this client..."))
    else:
        values = bt["values"]
        html(f'<div class="btp-h3">Three strategies since {first_month:%B %Y}</div><p class="btp-p">The same client '
             f"followed month by month with real CHF data; savings added: {T.chf(bt['savings'])}. Gains compare "
             "certainty equivalents on total wealth with the same wealth shares for every strategy.</p>")
        c1, c2 = st.columns([1.1, 1.5], gap="large")
        with c1:
            labels = {"(C) vs (A)": "Ours vs<br>typical", "(C) vs (B)": "Ours vs<br>100 - age",
                      "(B) vs (A)": "100 - age<br>vs typical"}
            g = bt["gains"].loc[list(labels)]
            fig = go.Figure(go.Bar(
                x=[f"{labels[i]}<br>p = {g.loc[i, 'p-value']:.2f}" for i in g.index], y=g["CE gain (CHF/yr)"],
                marker_color=[T.SAVINGS, T.SAVINGS, T.TYPICAL],
                error_y=dict(type="data", symmetric=False, array=g["95% high (CHF/yr)"] - g["CE gain (CHF/yr)"],
                             arrayminus=g["CE gain (CHF/yr)"] - g["95% low (CHF/yr)"], color=T.MUTED, thickness=1.2,
                             width=6),
                hovertemplate="CHF %{y:,.0f} a year<extra></extra>"))
            fig.update_layout(**T.plot_layout(300))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s", zeroline=True, zerolinecolor=T.MUTED)
            show(fig)
            st.caption("CHF a year, with the 95% interval (block bootstrap).")
        with c2:
            fig = go.Figure()
            crisis_bands(fig, params["backtest"]["crises"])
            for s in ("A", "B", "C"):
                fig.add_trace(go.Scatter(x=values.index, y=values[s], name=NAMES[s], mode="lines",
                                         line=dict(color=COLORS[s], width=2.6 if s == "C" else 1.6),
                                         hovertemplate=NAMES[s] + ": CHF %{y:,.0f}<extra></extra>"))
            fig.update_layout(**T.plot_layout(300, hovermode="x unified",
                                              legend=dict(orientation="h", yanchor="top", y=-0.08, x=0)))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
            show(fig)
        crises = bt["crises"]["portfolio return"].unstack("strategy")
        order_c = [c for c in params["backtest"]["crises"] if c in crises.index]
        crises = crises.loc[order_c, ["A", "B", "C"]].rename(columns=NAMES)
        html('<div class="btp-h3">In the crises</div>')
        html(T.table(crises, {c: (lambda v: T.pct(v, 1)) for c in crises.columns}, signed=list(crises.columns)))
        rows = ["portfolio excess return (annual)", "portfolio volatility", "portfolio max drawdown",
                "average share outside bonds and cash", "turnover per year"]
        metrics = bt["metrics"][[NAMES[s] for s in ("A", "B", "C")]]
        html('<div class="btp-h3">Numbers</div>')
        html(T.table(metrics.loc[[r for r in rows if r in metrics.index]].rename(index=str.capitalize),
                     {c: (lambda v: f"{v:.2f}") for c in metrics.columns}))

with tabs[3]:
    prof = ALLOC["profiles"][profile]
    c = params["career"]
    rows = [
        ("Risk profile", f"{profile}: bonds {T.pct(prof['safe_share'])} of the savings for someone with no career; "
                         f"growth risk {T.pct(prof['equities'])} shares, {T.pct(prof['real'])} real assets", "Group choice"),
        ("Bonds in our plan", "profile's bond share x (1 - bond-like share of wealth)", "Total wealth (steps 2-3)"),
        ("Bond-like share of wealth", "(bond beta of the career x career value + pension) / total wealth",
         "Career model (step 2)"),
        ("Own industry", "left out when part of the pay depends on it or employer shares are held", "Pivot plan"),
        ("Loss limit", f"bad year = 1 in {1 / (1 - ALLOC['loss_confidence']):.0f}; accepted loss "
                       f"{T.pct(prof['loss_tolerance'])} of the savings plus what can still be saved", "Gaussian value at risk"),
        ("Employer shares", f"industry return plus {T.pct(ALLOC['stake_idio_vol'])} company-specific volatility",
         "Assumption"),
        ("Industries", "12 Fama-French industry portfolios, in CHF", "Kenneth French data library"),
        ("Prior of Black-Litterman", "Doeswijk and Swinkels (2024) shares; equities by MSCI World sectors, Sep 2026",
         "Doeswijk and Swinkels (2024); MSCI"),
        ("Your market view", f"counts fully today, fades out over {ALLOC['view_years']} years in the plan by age, "
                             "not used in the backtest", "Group choice"),
        ("Salary growth", T.pct(c["salary_growth"], 1) + " a year", "Assumption"),
        ("Savings", f"{T.pct(params['backtest']['savings_rate'])} of the gross salary", "Assumption"),
        ("Trading costs", f"{params['backtest']['transaction_cost_bp']} bp of turnover", "Proposal"),
        ("Rebalancing (backtest)", f"plan solved every {ALLOC['rebalance_months']} months, "
                                   f"window {params['backtest']['window_months']} months", "Proposal"),
    ]
    html(T.table(pd.DataFrame(rows, columns=["Assumption", "Value", "Source"]).set_index("Assumption")))
    html('<p class="btp-foot">Every value lives in config/params.yaml. Data: Yahoo Finance, Kenneth French, FRED and '
         "the Swiss National Bank, monthly in CHF. Beyond the Portfolio is a student project (HEC Lausanne, "
         "Quantitative Asset and Risk Management II), not financial advice.</p>")


# =============================================================================
# While a backtest runs in the background: check every second, then show it
# =============================================================================
@st.fragment(run_every=APP["poll_seconds"])
def poll():
    if not jobs().pending():
        st.rerun()


if jobs().pending():
    poll()
