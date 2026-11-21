"""Beyond the Portfolio: the Streamlit app (step 5).

Run from the project folder:
    streamlit run app.py

One page. The client is in the side panel; the answer fits on one screen: the advice, the
backtest since 2005, the value of the career and the plan by age. The tabs below hold every
detail (why, risk, rules, backtest, robustness, assumptions). Everything updates by itself;
the backtest runs in the background and appears after a few seconds.
The numbers come from src/ (steps 1 to 4) through src/dashboard.py; the look is in src/theme.py.
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
from src.config import load_params
from src.jobs import Jobs

st.set_page_config(page_title="Beyond the Portfolio", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(T.css(), unsafe_allow_html=True)
CHART_CONFIG = {"displayModeBar": False, "responsive": True}
NAMES = {"A": "Typical advice", "B": "100 minus age", "D": "Total wealth, no hedge", "C": "Our plan"}


# =============================================================================
# Data and results (kept in memory; the backtests also on disk, data/cache/)
# =============================================================================

@st.cache_data(show_spinner=False)
def base_params():
    return load_params()


@st.cache_data(show_spinner=False)
def excess_returns():
    return O.load_excess_returns()


@st.cache_data(show_spinner=False)
def backtest_period():
    """First month decided, last month and number of months of the backtest (step 4)."""
    returns, _ = B.load_backtest_data()
    months = returns.index[int(base_params()["backtest"]["window_months"]):]
    return months[0], months[-1], len(months)


@st.cache_resource(show_spinner=False)
def jobs():
    """One queue of background jobs for the whole app."""
    return Jobs()


@st.cache_data(show_spinner=False, max_entries=64)
def analysis(key):
    """Optimizer of step 3 for the client, with the expected balance sheet until retirement."""
    persona, changes = json.loads(key)
    params = D.client_params(persona, changes, base_params())
    result = D.analysis(params, excess_returns())
    result["lifecycle"] = D.lifecycle(params, result)
    return result


@st.cache_data(show_spinner=False, max_entries=64)
def glide_path(key):
    persona, changes = json.loads(key)
    return D.glide_path(D.client_params(persona, changes, base_params()), analysis(key), excess_returns())


def background(name, params):
    """Result of D.remember(name, params): at once if it is saved, otherwise started in the background and
    None for now. The page then checks every second and reruns when it is ready (poll at the end)."""
    result = D.saved(name, params)
    if result is not None:
        return result
    future = jobs().get(D.cache_key(name, params), lambda: D.remember(name, params, B.load_backtest_data()),
                        group=name)
    return future.result() if future.done() else None       # an error in the computation is raised here


def html(text):
    st.markdown(text, unsafe_allow_html=True)


def show(fig):
    """Plotly chart at full width, with the new and the old Streamlit API."""
    try:
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def choice(label, options, default, key):
    """One choice among a few, as a segmented switch (a radio on an old Streamlit)."""
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(label, options, default=default, key=key)
        return value if value is not None else default        # a click on the chosen one unselects it
    return st.radio(label, options, index=list(options).index(default), horizontal=True, key=key)


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
APP, PERSONAS = P0["app"], P0["personas"]
job_names = list(APP["jobs"])
with st.sidebar:
    html(T.brand())
    persona = st.selectbox("I work in", job_names, index=job_names.index(P0["client"]["persona"]),
                           format_func=APP["jobs"].get, key="job")
    p = PERSONAS[persona]
    k = f"_{persona}"                                       # each job starts from its own typical client
    c1, c2 = st.columns([0.8, 1.2])
    age = c1.number_input("Age", *APP["age_range"], int(p["age"]), 1, key="age" + k)
    salary = c2.number_input("Salary, CHF a year", *APP["salary_range"], int(p["salary_chf"]), APP["salary_step"],
                             key="salary" + k)
    savings = st.number_input("Savings to invest, CHF", *APP["savings_range"], int(P0["client"]["portfolio_chf"]),
                              APP["savings_step"], key="savings" + k)
    pension = None
    if p.get("pillar2", True):
        how = choice("Pension fund (2nd pillar)", ["Estimate it", "From my certificate"], "Estimate it",
                     "pension_how" + k)
        if how == "From my certificate":
            estimate = O.pension_savings(age, salary, D.client_params(persona, None, P0))
            pension = st.number_input("Savings on the certificate, CHF", 0, APP["pension_max"],
                                      int(round(estimate, -3)), APP["pension_step"], key="pension" + k)
    shares = None
    if p.get("deferred_shares_chf", 0) > 0:
        shares = st.number_input("Employer shares you cannot sell yet, CHF", 0, int(savings),
                                 min(int(p["deferred_shares_chf"]), int(savings)), APP["shares_step"],
                                 key="shares" + k)
    sector = p["sector"]
    sector_words = APP["sector_words"].get(sector, T.asset_name(sector, P0)) if sector else None
    beta = None
    if sector:
        link = choice(f"How much does your pay follow {sector_words}?", list(APP["sector_link"]),
                      APP["sector_link_default"], "link" + k)
        beta = p["beta_sector"] * APP["sector_link"][link]

    with st.expander("Advanced"):
        wage_link = st.slider("Wages follow interest rates", 0.0, 1.0, float(P0["career"]["wage_rate_link"]), 0.05,
                              key="wage" + k, help="0 = a fixed salary, like a bond; 1 = wages rise fully with rates")
        long_only = st.toggle("No short selling", True, key="long" + k)
        no_leverage = st.toggle("No borrowing", True, key="lev" + k)
        caps = list(P0["optimization"]["position_caps"].values())
        cap = st.slider("Most in one sector or stock", APP["cap_range"][0], 1.0, float(caps[0]) if caps else 1.0,
                        APP["cap_range"][1], key="cap" + k, format="%.2f")
        view = None
        if st.toggle("Add my own view on one market", key="view_on" + k):
            assets = O.client_assets(D.client_params(persona, None, P0))
            asset = st.selectbox("Market", assets, format_func=lambda a: T.asset_name(a, P0), key="vasset" + k)
            low, high, step = APP["view_return_range"]
            q = st.slider("Expected return above cash, a year", low, high, APP["view_default"], step,
                          key="vq" + k, format="%.3f")
            levels = list(APP["view_confidence"])
            confidence = choice("Confidence", levels, levels[len(levels) // 2], "vc" + k)
            view = {"asset": asset, "q": q, "sd": APP["view_confidence"][confidence]}
    html('<p class="btp-side-note">Everything updates as you type. The backtest takes a few seconds the first '
         "time for a new client.</p>")

changes = {"age": age, "salary_chf": salary, "portfolio_chf": savings, "deferred_shares_chf": shares,
           "pension_chf": pension, "beta_sector": beta, "wage_rate_link": wage_link, "long_only": long_only,
           "no_leverage": no_leverage, "sector_cap": cap, "view": view}
key = json.dumps([persona, changes], sort_keys=True)
try:
    res = analysis(key)
except Exception as error:                                  # an impossible combination of settings
    st.error(f"No portfolio satisfies these settings ({error}). Change a rule or the locked shares.")
    st.stop()

params, sheet, story = res["params"], res["balance_sheet"], res["story"]
weights, F, W = res["weights"], sheet["F"], sheet["W"]
retire = params["client"]["retirement_age"]
holds = res["career_holdings"]
bond = params["data"]["career_bond_asset"]
mix_c, mix_a = D.groups(weights["C"]), D.groups(weights["A"])
gain = res["performance"].loc["gain over (A), CHF per year", "C"]
bt = background("backtest", params)
prior = background("prior", params) if bt is not None else None
first_month, last_month, n_months = backtest_period()

# =============================================================================
# Header: the answer in one sentence, and everything the client owns
# =============================================================================
ready = bt is not None and prior is not None
html(f'<div class="btp-who">{T.esc(APP["jobs"][persona])}, {age} years old, {T.chf(salary, short=False)} a year '
     f'{T.status(not ready, "Backtest ready" if ready else "Updating the backtest")}</div>')
levered = (weights["C"] < -APP["min_weight"]).any()            # borrowing or short selling allowed and used
if levered:
    advice = (f"{T.pct(mix_c['Shares'])} in shares and {T.pct(mix_c['Bonds'])} in bonds, "
              f"borrowing {T.chf(max(-mix_c['Cash'], 0) * F)}")
elif mix_c["Bonds"] < APP["bonds_visible"]:
    advice = "all in shares"
else:
    advice = f"{T.pct(mix_c['Shares'])} in shares and {T.pct(mix_c['Bonds'])} in bonds"
html(f'<div class="btp-headline">Invest your {T.chf(F, short=False)} <em>{advice}</em>.</div>')

reasons = []                                                # (colour, one line), shown in the panel
if mix_a["Bonds"] - mix_c["Bonds"] > APP["bond_gap"] and sheet["P"] > sheet["H"]:
    lede = "Your pension fund is already a large safe asset, so your savings can take the risk that pays."
    reasons.append((T.PENSION, f"<b>Your pension works like a bond.</b> Its {T.chf(sheet['P'])} earn a guaranteed "
                    f"interest, so you do not need the {T.pct(mix_a['Bonds'])} in bonds of typical advice."))
elif mix_a["Bonds"] - mix_c["Bonds"] > APP["bond_gap"]:
    lede = "Your job already works like a large Swiss bond, so your savings can take the risk that pays."
    reasons.append((T.BONDS, f"<b>Your job works like a bond.</b> Your salary moves like {T.chf(holds.get(bond, 0))} "
                    f"of Swiss government bonds, so you do not need the {T.pct(mix_a['Bonds'])} in bonds of "
                    "typical advice."))
else:
    lede = "With few working years left, your savings now carry most of what you own, so they hold bonds again."
    reasons.append((T.BONDS, "<b>Your savings take over from your job.</b> Your salary holds few bonds for you "
                    "now, so your savings hold them instead."))
if sector and story["sector_D"] - story["sector_C"] > APP["min_weight"]:
    reasons.append((T.SECTOR, f"<b>Your pay already depends on {sector_words}.</b> We leave them out, so a bad year "
                    "does not hit you twice."))
elif sector:
    reasons.append((T.SECTOR, f"<b>Nothing to offset.</b> We would not buy {sector_words} anyway, so a bad year "
                    "for them touches only your income."))
else:
    reasons.append((T.SECTOR, "<b>Nothing to offset.</b> Your pay does not follow any industry."))
reasons.append((T.SAVINGS, f"<b>Worth {T.chf(gain, short=False)} a year to you</b> compared with typical advice, once "
                "the extra risk is counted: like receiving it every year, with no risk."))
if levered:
    lede = ("Without the rules on borrowing and short selling the plan takes positions many times your savings: "
            "the theory is right, but no bank would allow it. Switch the rules back on in Advanced.")
html(f'<p class="btp-lede">{lede}</p>')
html(T.wealth(sheet))

# =============================================================================
# The four panels
# =============================================================================
shown = [a for a in weights.index if weights.loc[a].max() >= APP["min_weight"]]
left, right = st.columns([1, 1.25], gap="small")
with left:
    with st.container(key="panel_plan"):
        html(T.panel_title("Your portfolio", T.chf(F, short=False)))
        html('<div class="btp-bars">' + T.bar_row("Our plan", weights.loc[shown, "C"], params, "ours",
                                                    APP["min_weight"])
             + T.bar_row("Typical", weights.loc[shown, "A"], params, "typical", APP["min_weight"], labels=False)
             + "</div>")
        rows = [(T.asset_color(a), T.asset_name(a, params) + (" (locked)" if a == O.EMPLOYER else ""),
                 T.chf(weights.loc[a, "C"] * F, short=False), T.pct(weights.loc[a, "C"]))
                for a in shown if weights.loc[a, "C"] >= APP["min_weight"]]
        html(T.holdings(rows))
        if levered:
            html('<p class="btp-panel-note">The bars show only what is bought; the borrowing and the short '
                 "positions are in the tab Why this plan.</p>")
        html(T.why(reasons))

with right:
    with st.container(key="panel_backtest"):
        html(T.panel_title(f"Since {first_month:%B %Y}", f"you at {age}, our plan against typical advice"))
        if bt is None:
            html(T.waiting(f"Replaying {n_months} months for this client..."))
        else:
            values = bt["values"]
            fig = go.Figure()
            crisis_bands(fig, params["backtest"]["crises"])
            for s, width in (("A", 1.8), ("C", 2.6)):
                fig.add_trace(go.Scatter(x=values.index, y=values[s], name=NAMES[s], mode="lines",
                                         line=dict(color=T.STRATEGY_COLORS[s], width=width),
                                         hovertemplate=NAMES[s] + ": CHF %{y:,.0f}<extra></extra>"))
            fig.update_layout(**T.plot_layout(214, hovermode="x unified", showlegend=False))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
            show(fig)
            m = {s: bt["metrics"][B.STRATEGIES[s]] for s in ("A", "C")}
            g = bt["gains"].loc["(C) vs (A)"]
            html(T.stats([(T.chf(values["C"].iloc[-1]), f"today with our plan, against {T.chf(values['A'].iloc[-1])} "
                           "with typical advice", T.SAVINGS),
                          (f"{T.chf(g['CE gain (CHF/yr)'])} a year", f"better once risk is counted, p = {g['p-value']:.2f}",
                           None),
                          (T.pct(m["C"]["portfolio max drawdown"]), f"worst fall, against "
                           f"{T.pct(m['A']['portfolio max drawdown'])}: more shares, a bumpier ride", None)]))
            if prior is not None:
                swiss = prior.iloc[-1]
                html(f'<p class="btp-panel-note">With a more cautious Swiss pension-fund view of returns the gain is '
                     f"{T.chf(swiss['(C) vs (A) CHF/yr'], short=False)} a year, p = "
                     f"{swiss['(C) vs (A) p-value']:.2f}"
                     + (", so less certain." if swiss["(C) vs (A) p-value"] > APP["significance"] else ".") + "</p>")

left, right = st.columns([1, 1.25], gap="small")
with left:
    with st.container(key="panel_career"):
        years, rate = D.career_years(params)
        html(T.panel_title("What your career is worth", f"{T.chf(sheet['H'])} today"))
        if len(years):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=years.index, y=years["salary"], name="Salary that year",
                                 marker_color="rgba(142,155,255,0.22)",
                                 hovertemplate="Salary: CHF %{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Bar(x=years.index, y=years["value today"], name="Worth today", marker_color=T.CAREER,
                                 hovertemplate="Worth today: CHF %{y:,.0f}<extra></extra>"))
            fig.update_layout(**T.plot_layout(190, barmode="overlay", bargap=0.2, hovermode="x unified"))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
            show(fig)
            html(f'<p class="btp-panel-note">{len(years)} more years of salary, {T.chf(years["salary"].sum())} in all. '
                 f"A franc earned later is worth less today: discounted at {T.pct(rate, 2)} a year, the bars add "
                 f"up to <b>{T.chf(sheet['H'])}</b>.</p>")
        else:
            html('<p class="btp-panel-note">You are at retirement age: no salary left to value.</p>')

with right:
    with st.container(key="panel_glide"):
        glide = glide_path(key).copy()
        bonds_c = glide[("C", "Bonds")] if len(glide) else pd.Series(dtype=float)
        first = next((a for a, v in bonds_c.items() if v > APP["bonds_visible"]), None)
        if first is None:
            note = "shares until you retire"
        elif first > age:
            note = f"shares until about {first:.0f}, then bonds come in"
        else:
            note = "more bonds as retirement gets closer"
        html(T.panel_title("As you get older", note))
        if len(glide) > 1:
            glide.loc[retire] = glide.iloc[-1]               # the last plan is kept until retirement
            fig = go.Figure()
            for grp in D.GROUPS:
                if glide[("C", grp)].max() > APP["min_weight"]:
                    fig.add_trace(go.Scatter(x=glide.index, y=glide[("C", grp)], name=grp, stackgroup="mix",
                                             mode="lines", line=dict(width=0, color=T.GROUP_COLORS[grp]),
                                             fillcolor=T.GROUP_COLORS[grp],
                                             hovertemplate=grp + ": %{y:.0%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=glide.index, y=glide[("A", "Shares")], name="Typical advice, shares",
                                     mode="lines", line=dict(color=T.TEXT, width=1.5, dash="dot"),
                                     hovertemplate="Typical advice: %{y:.0%} shares<extra></extra>"))
            fig.update_layout(**T.plot_layout(190, hovermode="x unified", legend_traceorder="normal"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
            show(fig)
        html(f'<p class="btp-panel-note">Your salary is paid out year by year, so your career shrinks while your '
             f"savings grow; near retirement they must hold the bonds your job held. Saving "
             f"{T.pct(params['backtest']['savings_rate'])} of your salary, you reach about "
             f"<b>{T.chf(res['lifecycle']['F'].iloc[-1])}</b> at {retire} if markets give their average return.</p>")

# =============================================================================
# Details
# =============================================================================
tabs = st.tabs(["Why this plan", "Risk", "Rules", "Backtest", "Robustness", "Assumptions"])

with tabs[0]:
    perf = res["performance"]
    html('<div class="btp-h3">Three ways to invest the same savings</div><p class="btp-p">Same expected returns, '
         "same appetite for risk, same rules. Only what the adviser knows about your job changes: typical advice "
         "ignores it, the second plan counts your career but does not avoid your sector, our plan does both.</p>")
    html('<div class="btp-bars">' + "".join(
        T.bar_row(NAMES[s], weights.loc[shown, s], params, "ours" if s == "C" else "", APP["min_weight"])
        for s in ("A", "D", "C")) + "</div>")
    items = [(f"{T.pct(story['bonds_A'])} → {T.pct(story['bonds_D'])}",
              f"in bonds once the career is counted: it already works like a bond. Worth "
              f"{T.chf(story['gain_D_A'], short=False)} a year.", T.BONDS)]
    if sector and story["sector_D"] > APP["min_weight"]:
        items.append((f"{T.pct(story['sector_D'])} → {T.pct(story['sector_C'])}",
                      f"in {sector_words} once we avoid what your pay already depends on. Worth "
                      f"{T.chf(story['gain_C_D'], short=False)} a year.", T.SECTOR))
    items.append((T.pct(story["career_risk_share"]), "of the ups and downs of everything you own come from your "
                  "career, with our plan.", T.CAREER))
    html(T.cards(items))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html('<div class="btp-h3">What the model would buy with no rule</div><p class="btp-p">In CHF. The second '
             "part is exactly minus what your job already holds. Without rules it would borrow and sell short large "
             "amounts: the rules keep the plan realistic (tab Rules).</p>")
        dem = D.demand_chf(res)
        dem = dem[(dem.abs() > 1).any(axis=1)]
        names = [T.asset_name(a, params) for a in dem.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=names, x=dem["speculative"], name="To earn the risk premium", orientation="h",
                             marker_color=T.SAVINGS, hovertemplate="CHF %{x:,.0f}<extra></extra>"))
        fig.add_trace(go.Bar(y=names, x=dem["hedging"], name="To offset your job", orientation="h",
                             marker_color=T.CAREER, hovertemplate="CHF %{x:,.0f}<extra></extra>"))
        fig.update_layout(**T.plot_layout(60 + 36 * len(names), barmode="relative"))
        fig.update_xaxes(tickprefix="CHF ", tickformat="~s", showgrid=True, gridcolor=T.LINE, zeroline=True,
                         zerolinecolor=T.MUTED)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig)
    with c2:
        html(f'<div class="btp-h3">Expected returns</div><p class="btp-p">A year, above the CHF risk-free rate, '
             f"{res['months']} months of data. Black-Litterman with the world market portfolio as the prior "
             f"(Doeswijk and Swinkels, 2024), market risk aversion {res['phi']:.2f}; covariance with Ledoit-Wolf "
             f"shrinkage ({res['shrinkage']:.2f}).</p>")
        er = res["expected_returns"].copy()
        er.index = [T.asset_name(a, params) for a in er.index]
        html(T.table(er, {c: (lambda v: T.pct(v, 1)) for c in er.columns}))

with tabs[1]:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        risk = res["risk"]["C"]["share of risk"]
        main = risk[risk.abs() >= APP["risk_other"]]
        rest = risk[risk.abs() < APP["risk_other"]].sum()
        labels = [T.asset_name(a, params) for a in main.index] + (["Other"] if rest > 0 else [])
        values = list(main.to_numpy()) + ([rest] if rest > 0 else [])
        colors = [T.CAREER if a == O.CAREER else T.PENSION if a == O.PENSION else T.asset_color(a)
                  for a in main.index] + [T.FAINT]
        html('<div class="btp-h3">Where the risk of everything you own comes from</div><p class="btp-p">Each '
             "piece's share of the ups and downs of total wealth, with our plan (Euler risk contributions).</p>")
        fig = go.Figure(go.Pie(labels=labels, values=np.clip(values, 0, None), hole=0.72, sort=False,
                               marker=dict(colors=colors, line=dict(color=T.BG, width=2)), textinfo="none",
                               hovertemplate="%{label}: %{percent}<extra></extra>"))
        fig.update_layout(**T.plot_layout(300, showlegend=True, legend=dict(orientation="v", x=1, y=0.5),
                                          annotations=[dict(text=f"<b>{T.pct(risk.get(O.CAREER, 0))}</b><br>career",
                                                            showarrow=False, font=dict(size=20, color=T.CAREER))]))
        show(fig)
    with c2:
        cov = res["covariance"].to_numpy()
        rows = {}
        for s in ("A", "C"):
            rc = O.risk_contributions(res["solutions"][s]["x"], cov)
            rows[s] = rc / rc.sum() if rc.sum() > 0 else rc
        rc = pd.DataFrame(rows, index=res["assets"])
        rc = rc[(rc.abs() >= APP["min_weight"]).any(axis=1)]
        html('<div class="btp-h3">Inside the portfolio only</div><p class="btp-p">The classic view (lab LN5): '
             "each holding's share of the risk of the savings alone.</p>")
        fig = go.Figure()
        for s in ("A", "C"):
            fig.add_trace(go.Bar(y=[T.asset_name(a, params) for a in rc.index], x=rc[s], orientation="h",
                                 name=NAMES[s], marker_color=T.STRATEGY_COLORS[s],
                                 hovertemplate="%{x:.1%}<extra></extra>"))
        fig.update_layout(**T.plot_layout(300, barmode="group"))
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=T.LINE)
        fig.update_yaxes(autorange="reversed")
        show(fig)

with tabs[2]:
    html('<div class="btp-h3">What each rule costs you</div><p class="btp-p">The value of one percentage point more '
         "room, CHF a year, from the solver's Lagrange multipliers. Rules that do not bind cost nothing.</p>")
    words = {"No leverage": "borrowing 1% of the savings and investing it",
             "Lock-up": "selling 1% of the locked employer shares",
             "Long-only": "selling short 1% of {}", "Position caps": "holding 1% more than the cap in {}"}
    colors = {"No leverage": T.CAREER, "Lock-up": T.ASSET_COLORS["EMPLOYER"], "Long-only": T.BONDS}
    marginal = res["shadow_marginal"].sort_values("CHF per year", ascending=False)
    items = [(f"{T.chf(row['CHF per year'], short=False)} a year",
              "Value of " + words[row["group"]].format(T.esc(T.asset_name(row["constraint"].split(" at ")[0], params))),
              colors.get(row["group"], T.SECTOR)) for _, row in marginal.iterrows()]
    html(T.cards(items) if items else '<p class="btp-p">No rule binds for this client.</p>')
    html('<div class="btp-h3">If a rule disappeared completely</div><p class="btp-p">The problem solved again '
         "without the whole rule. Upper bounds: without the borrowing rule a young client would borrow many times "
         "the savings.</p>")
    total = res["shadow_total"].set_index("group")[["binding", "CHF per year", "invested without it"]]
    total["binding"] = total["binding"].map({True: "yes", False: "no"})
    html(T.table(total.rename(columns={"binding": "Binds today", "CHF per year": "Gain, CHF a year",
                                       "invested without it": "Invested without it"}),
                 {"Gain, CHF a year": lambda v: T.chf(v, short=False), "Invested without it": lambda v: T.pct(v)}))

with tabs[3]:
    if bt is None:
        html(T.waiting("Running the backtest for this client..."))
    else:
        html(f'<div class="btp-h3">Four strategies, {first_month:%B %Y} to {last_month:%B %Y}</div><p class="btp-p">'
             f"The same client followed month by month: each decision uses only the {params['backtest']['window_months']}"
             f" months before it, with trading costs, savings ({T.chf(bt['savings'])} in all) and the real CHF "
             "interest rate. The gains compare certainty equivalents with the same wealth shares for every "
             "strategy, so the comparison is fair.</p>")
        c1, c2 = st.columns([1, 1.6], gap="large")
        with c1:
            labels = {"(D) vs (A)": "Count the career", "(C) vs (A)": "Our plan", "(C) vs (D)": "Avoid the sector"}
            g = bt["gains"].loc[list(labels)]
            fig = go.Figure(go.Bar(
                x=[f"{labels[i]}<br>p = {g.loc[i, 'p-value']:.2f}" for i in g.index], y=g["CE gain (CHF/yr)"],
                marker_color=[T.STRATEGY_COLORS["D"], T.SAVINGS, T.SECTOR],
                error_y=dict(type="data", symmetric=False, array=g["95% high (CHF/yr)"] - g["CE gain (CHF/yr)"],
                             arrayminus=g["CE gain (CHF/yr)"] - g["95% low (CHF/yr)"], color=T.MUTED, thickness=1.2,
                             width=6),
                hovertemplate="CHF %{y:,.0f} a year<extra></extra>"))
            fig.update_layout(**T.plot_layout(300))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s", zeroline=True, zerolinecolor=T.MUTED)
            show(fig)
            st.caption("Gain over typical advice, CHF a year, with the 95% interval (block bootstrap).")
        with c2:
            values = bt["values"]
            fig = go.Figure()
            crisis_bands(fig, params["backtest"]["crises"])
            for s in ("A", "B", "D", "C"):
                fig.add_trace(go.Scatter(x=values.index, y=values[s], name=NAMES[s], mode="lines",
                                         line=dict(color=T.STRATEGY_COLORS[s], width=2.6 if s == "C" else 1.6),
                                         hovertemplate=NAMES[s] + ": CHF %{y:,.0f}<extra></extra>"))
            fig.update_layout(**T.plot_layout(300, hovermode="x unified",
                                              legend=dict(orientation="h", yanchor="top", y=-0.08, x=0)))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
            show(fig)
        crises = bt["crises"]["portfolio return"].unstack("strategy")
        order = [c for c in params["backtest"]["crises"] if c in crises.index]
        crises = crises.loc[order, ["A", "B", "D", "C"]].rename(columns=NAMES)
        html('<div class="btp-h3">In the crises</div>')
        html(T.table(crises, {c: (lambda v: T.pct(v, 1)) for c in crises.columns}, signed=list(crises.columns)))
        metrics = bt["metrics"][[B.STRATEGIES[s] for s in ("A", "B", "D", "C")]]
        metrics.columns = [NAMES[s] for s in ("A", "B", "D", "C")]
        rows = ["portfolio excess return (annual)", "portfolio volatility", "portfolio max drawdown",
                "average share outside bonds and cash", "turnover per year", "correlation with the sector part"]
        html('<div class="btp-h3">Numbers</div>')
        html(T.table(metrics.loc[[r for r in rows if r in metrics.index]].rename(index=str.capitalize),
                     {c: (lambda v: f"{v:.2f}") for c in metrics.columns}))

with tabs[4]:
    rob = background("robustness", params) if bt is not None else None
    if rob is None:
        html(T.waiting("Running the robustness tests in the background (about half a minute)..."))
    else:
        if rob["wrong_beta"] is None:
            html('<p class="btp-p">Your pay follows no sector, so there is no hedge to test: wrong beta, job loss and '
                 "the hedge map do not apply. The prior test is below.</p>")
        else:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                wb = rob["wrong_beta"]
                html('<div class="btp-h3">Wrong beta</div><p class="btp-p">The decisions use the assumed link with the '
                     "sector; the career is replayed with the true one. The risk part is a straight line, as the "
                     f"theory h(2β − h) says (R² {wb.attrs['risk part: R2 of the straight line']:.4f}); it turns "
                     f"positive at a true beta of {wb.attrs['risk part: break-even true beta']:.3f}.</p>")
                fig = go.Figure()
                for col, name, color, dash in (("CE gain (C) vs (D) (CHF/yr)", "Gain of avoiding the sector",
                                                T.SAVINGS, "solid"),
                                               ("risk part (CHF/yr)", "Risk part", T.PENSION, "dash"),
                                               ("return part (CHF/yr)", "Return part", T.CAREER, "dot")):
                    fig.add_trace(go.Scatter(x=wb["true beta_sector"], y=wb[col], name=name, mode="lines+markers",
                                             line=dict(color=color, width=2.4, dash=dash), marker=dict(size=7),
                                             hovertemplate="CHF %{y:,.0f}<extra></extra>"))
                fig.add_vline(x=params["career"]["beta_sector"], line_color=T.MUTED, line_dash="dot",
                              annotation_text="assumed", annotation_font=dict(color=T.MUTED))
                fig.update_layout(**T.plot_layout(280))
                fig.update_xaxes(title="True sector beta")
                fig.update_yaxes(tickprefix="CHF ", zeroline=True, zerolinecolor=T.MUTED)
                show(fig)
            with c2:
                hm = rob["hedge_map"]["risk part (CHF/yr)"].unstack("beta_sector")
                html('<div class="btp-h3">Hedge map</div><p class="btp-p">Risk part of avoiding the sector, CHF a '
                     "year, by starting savings and link with the sector.</p>")
                fig = go.Figure(go.Heatmap(z=hm.to_numpy(), x=[f"beta {b:.2f}" for b in hm.columns],
                                           y=[T.chf(v) for v in hm.index], colorscale=[[0, T.RAISED], [1, T.SAVINGS]],
                                           text=[[T.chf(v) for v in row] for row in hm.to_numpy()],
                                           texttemplate="%{text}", showscale=False, xgap=3, ygap=3,
                                           hovertemplate="%{y}, %{x}: %{text} a year<extra></extra>"))
                fig.update_layout(**T.plot_layout(280))
                show(fig)
            jl = rob["job_loss"]
            cols = {"jobs lost": "Jobs lost", "(C) vs (A) CHF/yr": "Our plan, CHF a year",
                    "(C) vs (D) CHF/yr": "Avoiding the sector, CHF a year", "hedge: risk part": "Risk part"}
            html('<div class="btp-h3">Losing the job</div><p class="btp-p">Twelve months without salary when the '
                 "sector falls this much below its peak.</p>")
            html(T.table(jl[list(cols)].rename(columns=cols), {c: (lambda v: T.chf(v, short=False))
                                                              for c in list(cols.values())[1:]},
                         signed=list(cols.values())[1:]))
        pr = rob["prior"]
        cols = {"(C) vs (A) CHF/yr": "Our plan, CHF a year", "(C) vs (A) p-value": "p-value",
                "(C) vs (D) CHF/yr": "Avoiding the sector, CHF a year", "hedge: risk part": "Risk part"}
        html('<div class="btp-h3">Another prior</div><p class="btp-p">The world market portfolio against a Swiss '
             "pension fund (LPP-40) as the starting view of returns. With the Swiss prior the gain is smaller and "
             "only borderline significant.</p>")
        html(T.table(pr[[c for c in cols if c in pr.columns]].rename(columns=cols),
                     {"Our plan, CHF a year": lambda v: T.chf(v, short=False), "p-value": lambda v: f"{v:.3f}",
                      "Avoiding the sector, CHF a year": lambda v: T.chf(v, short=False),
                      "Risk part": lambda v: T.chf(v, short=False)},
                     signed=["Our plan, CHF a year", "Avoiding the sector, CHF a year", "Risk part"]))

with tabs[5]:
    c, pen, bl, btp, opt = (params["career"], params["pension"], params["black_litterman"], params["backtest"],
                            params["optimization"])
    rows = [
        ("Retirement age", f"{retire}", "Swiss reference age"),
        ("Salary growth", T.pct(c["salary_growth"], 1) + " a year", "Assumption"),
        ("Long-run CHF rate", T.pct(c["rf_long_run"], 1), "Assumption, used to value the career"),
        ("Bond and sector premia", f"{T.pct(c['bond_premium'], 1)} and {T.pct(c['sector_premium'], 1)}",
         "Assumption, used to value the career"),
        ("Bond beta of the career", f"{sheet['beta_bond']:.2f}: duration of the salary / duration of CHF bonds",
         "Career model (step 2)"),
        ("Wages follow rates", f"{c['wage_rate_link']:.2f}", "Group choice: a civil servant is not riskier than a banker"),
        ("Sector beta", f"{c['beta_sector']:.2f}", "Persona preset, cautious on purpose"),
        ("Pension credits", "7, 10, 15, 18% of the coordinated salary", "BVG art. 16"),
        ("Coordination deduction, upper limit", f"CHF {pen['coordination_deduction']:,} and CHF "
         f"{pen['max_insured_salary']:,}".replace(",", "'"), "BSV key figures 2025-2026"),
        ("Pension interest", T.pct(pen["interest_rate"], 2), "BVG minimum 2024-2026"),
        ("Market portfolio", "equities 49.7%, bonds 43.5%, real estate 4.4%, commodities 2.4%",
         "Doeswijk and Swinkels (2024), Table 1"),
        ("Market Sharpe ratio", f"{bl['sharpe_ratio']:.3f}", "Doeswijk and Swinkels (2024), Table 2"),
        ("Black-Litterman tau", f"{bl['tau']:.2f}", "Lab LN4"),
        ("Appetite for risk", f"{res['gamma']:.3f}", "Same risk as 60/40 for a client with no career (fsolve, LN1)"),
        ("Covariance", "Ledoit-Wolf, constant correlation", "Lab LN3"),
        ("Rules", f"no short selling {'on' if opt['long_only'] else 'off'}, no borrowing "
         f"{'on' if opt['max_leverage'] is not None else 'off'}, at most {T.pct(cap)} in one sector or stock",
         "Assumption"),
        ("Savings", f"{T.pct(btp['savings_rate'])} of the gross salary", "Assumption"),
        ("Trading costs", f"{btp['transaction_cost_bp']} bp of turnover", "Proposal"),
        ("Estimation window", f"{btp['window_months']} months", "Proposal"),
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
