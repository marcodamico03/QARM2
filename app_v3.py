"""Beyond the Portfolio, version 3 of the app: one page that tells the result as a story.

Run from the project folder:
    streamlit run app_v3.py

The client is in the side panel. The page answers, top to bottom:
    1. What to do with the savings (our plan against the typical 60/40)
    2. Why: the client is more than the savings (career, pension, savings)
    3. The two changes from typical advice and what each was worth in the backtest
    4. What the client gives up (more swings in the savings)
    5. Since 2005, someone like the client
    6. As the client gets older
    Details (closed): every holding, the full backtest numbers, the assumptions.
Light or dark with the switch at the top of the side panel. Same model as app.py (src/allocation.py);
the look is in src/look.py. app.py (version 2) is not changed.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit import config

from src import allocation as A
from src import dashboard as D
from src import look as L
from src import optimizer as O
from src.config import load_params
from src.jobs import Jobs

# =============================================================================
# Light or dark: Streamlit's own theme follows the switch (it applies on the next run, hence the rerun)
# =============================================================================
DARK = bool(st.session_state.get("dark", False))
C = L.palette(DARK)
_theme = L.streamlit_theme(C)
if any(config.get_option(f"theme.{k}") != v for k, v in _theme.items()):
    for k, v in _theme.items():
        config.set_option(f"theme.{k}", v)
    st.rerun()

st.set_page_config(page_title="Beyond the Portfolio", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(L.css(C), unsafe_allow_html=True)
CHART = {"displayModeBar": False, "responsive": True}
SHORT_CRISES = {"2008 financial crisis": "2008", "2011 euro crisis": "Euro 2011", "Jan 2015 CHF shock": "CHF 2015",
                "COVID 2020": "COVID", "2022 rates and inflation": "2022", "Credit Suisse 2023": "CS 2023"}


# =============================================================================
# Data and results (in memory; the backtests also on disk, data/cache/)
# =============================================================================

@st.cache_data(show_spinner=False)
def base_params():
    return load_params()


@st.cache_data(show_spinner=False)
def market_data():
    return A.load_data(base_params())


@st.cache_resource(show_spinner=False)
def jobs():
    return Jobs()


@st.cache_data(show_spinner=False, max_entries=64)
def analysis(key):
    persona, changes = json.loads(key)
    return A.analyze(A.client_params(persona, changes, base_params()), market_data())


@st.cache_data(show_spinner=False, max_entries=64)
def glide(key):
    persona, changes = json.loads(key)
    res = analysis(key)
    return A.glide_path(A.client_params(persona, changes, base_params()), res["mu"], res["cov"])


def background(name, params):
    """The saved result at once, or None while it is computed in the background (the page then
    checks every second and reruns when it is ready)."""
    result = D.saved(name, params)
    if result is not None:
        return result
    future = jobs().get(D.cache_key(name, params), lambda: D.remember(name, params, market_data()), group=name)
    return future.result() if future.done() else None


def html(text):
    st.markdown(text, unsafe_allow_html=True)


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch", config=CHART)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=CHART)


def p_text(p):
    return "p < 0.01" if p < 0.01 else f"p = {p:.2f}"


# =============================================================================
# Side panel: the client
# =============================================================================
P0 = base_params()
APP, PERSONAS, ALLOC = P0["app"], P0["personas"], P0["allocation"]
INDUSTRIES = APP["industry_names"]
with st.sidebar:
    html(L.brand(C))
    st.toggle("Dark mode", key="dark")
    persona = st.selectbox("Start from an example", list(PERSONAS), format_func=lambda k: PERSONAS[k]["label"], key="job")
    p = PERSONAS[persona]
    k = f"_{persona}"                                         # each example starts from its own values
    options = list(INDUSTRIES) + [None]
    industry = st.selectbox("I work in", options, index=options.index(p.get("industry")),
                            format_func=lambda i: INDUSTRIES.get(i, APP["no_industry"]), key="industry" + k)
    c1, c2 = st.columns([0.8, 1.2])
    age = c1.number_input("Age", *APP["age_range"], int(p["age"]), 1, key="age" + k)
    salary = c2.number_input("Salary, CHF a year", *APP["salary_range"], int(p["salary_chf"]), APP["salary_step"],
                             key="salary" + k)
    savings = st.number_input("Savings to invest, CHF", *APP["savings_range"], int(P0["client"]["portfolio_chf"]),
                              APP["savings_step"], key="savings" + k)
    stake, unlock, variable = 0.0, 0.0, 0.0
    if industry:
        stake = st.number_input("Of which shares of my employer, CHF", 0, int(savings),
                                min(int(p.get("stake_chf", 0)), int(savings)), APP["shares_step"], key="stake" + k)
        if stake > 0:
            unlock = st.slider("They can be sold in (years)", 0, APP["unlock_max_years"], int(p.get("unlock_years", 0)),
                               key="unlock" + k)
        variable = st.slider("Share of my pay that is a bonus linked to my industry", 0.0, 1.0,
                             float(p.get("variable_pay", 0.0)), APP["variable_pay_step"], key="variable" + k,
                             format="%.2f")
    self_employed = st.toggle("Self-employed (no pension fund)", value=not p.get("pillar2", True), key="self" + k)
    pension = None
    if not self_employed:
        how = st.radio("Pension fund (2nd pillar)", ["Estimate it", "From my certificate"], horizontal=True,
                       key="pension_how" + k)
        if how == "From my certificate":
            estimate = O.pension_savings(age, salary, A.client_params(persona, None, P0))
            pension = st.number_input("Savings on the certificate, CHF", 0, APP["pension_max"],
                                      int(round(estimate, -3)), APP["pension_step"], key="pension" + k)
    profiles = list(ALLOC["profiles"])
    profile = st.radio("Risk profile", profiles, index=profiles.index(ALLOC["default_profile"]), horizontal=True,
                       format_func=str.capitalize, key="profile" + k)
    html('<p class="v3-side-note">The page updates as you type. The backtest of a new client takes a few seconds.</p>')

changes = {"industry": industry, "age": age, "salary_chf": salary, "portfolio_chf": savings,
           "stake_chf": float(stake), "unlock_years": float(unlock), "variable_pay": float(variable),
           "pillar2": not self_employed, "pension_chf": pension, "profile": profile, "view": None}
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
pC, pA = plans["C"], plans["A"]
gC, gA = pC["groups"], pA["groups"]
dC = pC["detail"]
excluded = bool(dC["excluded"])
ind_name = INDUSTRIES.get(industry) if industry else None
ind_short = ind_name.split(" (")[0].lower() if ind_name else ""
story = background("story", params)
first_month = market_data()[0].index[int(params["backtest"]["window_months"])]
typical_bonds = float(ALLOC.get("typical_safe_share", 0.4))

# =============================================================================
# The answer
# =============================================================================
who = f"{ind_name or 'Pay not linked to an industry'} · {age} years old · {L.chf(salary, short=False)} a year"
html(f'<div class="v3-kicker">{L.esc(who)}</div>')
parts = [f"{L.pct(gC['equities'])} in shares", f"{L.pct(gC['real'])} in real assets", f"{L.pct(gC['bonds'])} in bonds"]
if gC["cash"] >= 0.005:
    parts.append(f"{L.pct(gC['cash'])} in cash")
html(f'<div class="v3-title">Invest your {L.chf(F, short=False)}: <em>{", ".join(parts[:-1])} and {parts[-1]}</em>.</div>')
lede = (f"Typical advice gives everyone 60% growth and 40% bonds. We start from the same mix and look at "
        f"your job: <b>your future salary already works like a bond</b>, so your savings need only "
        f"{L.pct(gC['bonds'])} in bonds")
lede += f", and <b>no extra {ind_short}</b>, because your pay already depends on it." if excluded else "."
html(f'<p class="v3-lede">{lede}</p>')


def mix_parts(g, values=True):
    return [(g[k], L.group_color(C, k), L.pct(g[k]) if values else "") for k in ("equities", "real", "bonds", "cash")]


row1 = st.columns([1.15, 1], gap="medium")
with row1[0]:
    with st.container(key="card_plan"):
        html(L.heading("1 · Your savings", "Our plan against typical advice"))
        html(L.bar(C, "Typical", "60/40 for everyone", mix_parts(gA))
             + L.bar(C, "Our plan", "built around your job", mix_parts(gC)))
        html(L.legend([(L.group_color(C, k), L.GROUPS[k]) for k in ("equities", "real", "bonds")
                       + (("cash",) if gC["cash"] >= 0.005 else ())]))
        items = []
        stake_now = dC["stake"] if dC["locked"] else 0.0
        shares_txt = "in 11 industries and Switzerland" if excluded else "in 12 industries and Switzerland"
        items.append(("Shares", shares_txt, L.chf(gC["equities"] * F - stake_now, short=False)))
        if stake_now > 0:
            items.append(("Your employer shares", f"locked {unlock:.0f} more year{'s' if unlock != 1 else ''}",
                          L.chf(stake_now, short=False)))
        items.append(("Real assets", "Swiss real estate, gold, commodities", L.chf(gC["real"] * F, short=False)))
        items.append(("Bonds", "Swiss, global hedged, corporate", L.chf(gC["bonds"] * F, short=False)))
        if gC["cash"] >= 0.005:
            items.append(("Cash", "close to retirement: the loss limit", L.chf(gC["cash"] * F, short=False)))
        html(L.rows(items))

with row1[1]:
    with st.container(key="card_wealth"):
        html(L.heading("2 · Why", "You are more than your savings"))
        html(f'<div class="v3-big">{L.chf(W)}</div><p class="v3-note" style="margin:0 0 0.7rem">everything you own, '
             "today's value</p>")
        html(L.bar(C, "", "", [(sheet["H"] / W, C["career"], ""), (sheet["P"] / W, C["pension"], ""),
                               (F / W, C["ours"], "")], show_values=False))
        html(L.legend([(C["career"], f"Career {L.chf(sheet['H'])}"), (C["pension"], f"Pension {L.chf(sheet['P'])}"),
                       (C["ours"], f"Savings {L.chf(F)}")]))
        safe = sheet["career_bonds"] + sheet["P"]
        html(f'<p class="v3-p" style="margin-top:0.8rem">Your savings are only {L.pct(F / W)} of your wealth. '
             f"<b>{L.chf(safe)} ({L.pct(sheet['bond_like'])}) works like a bond</b>: the steady part of your future "
             f"salary ({L.chf(sheet['career_bonds'])}) plus your pension ({L.chf(sheet['P'])}). "
             f"That is why your savings can hold fewer bonds: {L.pct(typical_bonds)} "
             f"× (1 − {L.pct(sheet['bond_like'])}) = {L.pct(dC['safe_share'])}.</p>")

# =============================================================================
# What it was worth, and what it costs
# =============================================================================
row2 = st.columns([1.15, 1], gap="medium")
with row2[0]:
    with st.container(key="card_changes"):
        html(L.heading("3 · The result", "The two changes, and what each was worth",
                       f"Replayed month by month since {first_month:%B %Y} for someone like you, in CHF a year "
                       "once risk is counted."))
        if story is None:
            html(L.waiting("Replaying the past for this client..."))
        else:
            g = story["gains"]
            g1, g2, gt = (g.loc[i] for i in ("(M) vs (A)", "(C) vs (M)", "(C) vs (A)"))
            v1, v2, vt = (float(x["CE gain (CHF/yr)"]) for x in (g1, g2, gt))
            labels = ["Typical 60/40", "Fewer bonds"] + ([f"No extra {ind_short}"] if excluded else []) + ["Our plan"]
            ys = [0, v1] + ([v2] if excluded else []) + [0]
            measures = ["absolute", "relative"] + (["relative"] if excluded else []) + ["total"]
            texts = ["0", L.signed_chf(v1)] + ([L.signed_chf(v2)] if excluded else []) + [L.signed_chf(vt)]
            fig = go.Figure(go.Waterfall(
                x=labels, y=ys, measure=measures, text=texts, textposition="outside",
                textfont=dict(color=C["text"], size=13), cliponaxis=False,
                connector=dict(line=dict(color=C["line"], width=1)),
                increasing=dict(marker=dict(color=C["good"])), decreasing=dict(marker=dict(color=C["bad"])),
                totals=dict(marker=dict(color=C["ours"])),
                hovertemplate="%{x}: %{text} a year<extra></extra>"))
            top_v = max(v1, vt, v1 + (v2 if excluded else 0))
            fig.update_layout(**L.plot_layout(C, 250, margin=dict(l=4, r=8, t=28, b=8)))
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s", range=[min(0, v2 if excluded else 0, v1 + v2) * 1.3,
                                                                       top_v * 1.25])
            show(fig)
            html(L.change(1, "Fewer bonds: your job is a bond",
                          f"Bonds go from {L.pct(typical_bonds)} to {L.pct(dC['safe_share'])}; the money goes to shares "
                          "and real assets, which earn more over time.", L.signed_chf(v1),
                          C["good"] if v1 >= 0 else C["bad"], g1["p-value"] <= 0.05, L.chance(g1["p-value"])))
            if excluded:
                html(L.change(2, f"No extra {ind_short}: no second bet on your industry",
                              f"A bad year for {ind_short} already hits your bonus"
                              f"{' and your employer shares' if stake_now > 0 else ''}. Leaving it out lowers the "
                              "risk but gives up some return: it is insurance, not a source of profit.",
                              L.signed_chf(v2), C["good"] if v2 >= 0 else C["bad"], g2["p-value"] <= 0.05,
                              L.chance(g2["p-value"])))
            html(f'<p class="v3-note">Certainty equivalent on total wealth (gain in return minus the cost of the extra '
                 f"risk), same wealth for every strategy, 10 bp trading costs, block bootstrap. "
                 f"Total {L.signed_chf(vt)} a year, {p_text(gt['p-value'])}.</p>")

with row2[1]:
    with st.container(key="card_cost"):
        html(L.heading("4 · The price", "What you give up"))
        if story is None:
            html(L.waiting("Replaying the past for this client..."))
        else:
            m = story["metrics"]
            crises = story["crises"]["portfolio return"].unstack("strategy")
            gfc = "2008 financial crisis"
            tl = [("Your savings swing, a year", L.pct(m.loc["portfolio volatility", "C"], 1),
                   f"typical {L.pct(m.loc['portfolio volatility', 'A'], 1)}"),
                  ("Worst fall of the savings", L.pct(m.loc["portfolio max drawdown", "C"]),
                   f"typical {L.pct(m.loc['portfolio max drawdown', 'A'])}")]
            if gfc in crises.index:
                tl.append(("In the 2008 crisis", L.pct(crises.loc[gfc, "C"]), f"typical {L.pct(crises.loc[gfc, 'A'])}"))
            tl.append(("Everything you own swings, a year", L.pct(m.loc["total wealth volatility", "C"], 1),
                       f"typical {L.pct(m.loc['total wealth volatility', 'A'], 1)}"))
            html(L.tiles(tl))
            html(f'<p class="v3-p" style="margin-top:0.6rem"><b>Your savings will move more.</b> You are paid for it '
                 f"(step 3), and your salary, which keeps coming in a crash, absorbs it: everything you own "
                 f"swings almost as little as before.</p>")
            if dC.get("loss") is not None and dC["cash_from_loss_limit"] == 0:
                html(f'<p class="v3-note">A bad year (1 in 20) would cost your savings about {L.chf(dC["loss"])} today, '
                     f"within your {profile} profile.</p>")

# =============================================================================
# The past and the future
# =============================================================================
row3 = st.columns([1.15, 1], gap="medium")
with row3[0]:
    with st.container(key="card_history"):
        html(L.heading("5 · The past", f"Since {first_month:%B %Y}, someone like you",
                       f"Aged {story['start_age']:.0f} in {first_month:%Y}, saving "
                       f"{L.pct(params['backtest']['savings_rate'])} of the salary; each decision uses only the "
                       f"past {params['backtest']['window_months']} months." if story else None))
        if story is None:
            html(L.waiting("Replaying the past for this client..."))
        else:
            values = story["values"]
            fig = go.Figure()
            for i, (crisis, (first, last)) in enumerate(params["backtest"]["crises"].items()):
                x0 = pd.Timestamp(first) - pd.offsets.MonthEnd(1)
                fig.add_vrect(x0=x0, x1=pd.Timestamp(last) + pd.offsets.MonthEnd(0), fillcolor=C["band"], line_width=0,
                              layer="below")
                fig.add_annotation(x=x0, y=1.0 - 0.07 * (i % 2), yref="paper", text=SHORT_CRISES.get(crisis, crisis),
                                   showarrow=False, xanchor="left", font=dict(size=9, color=C["faint"]))
            for s, name, col, width in (("A", "Typical 60/40", C["typical"], 1.8), ("C", "Our plan", C["ours"], 2.6)):
                fig.add_trace(go.Scatter(x=values.index, y=values[s], name=name, mode="lines",
                                         line=dict(color=col, width=width),
                                         hovertemplate=name + ": CHF %{y:,.0f}<extra></extra>"))
            fig.update_layout(**L.plot_layout(C, 260, hovermode="x unified", showlegend=True, legend_traceorder="reversed",
                                              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=C["muted"])),
                                              margin=dict(l=4, r=8, t=30, b=8)))
            fig.update_xaxes(range=[values.index[0], values.index[-1]])
            fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
            show(fig)
            html(L.tiles([("Today with our plan", L.chf(values["C"].iloc[-1]), f"typical {L.chf(values['A'].iloc[-1])}"),
                          ("Saved along the way", L.chf(story["savings"]), "the same for both")]))

with row3[1]:
    with st.container(key="card_glide"):
        html(L.heading("6 · The future", "As you get older",
                       "Each year one salary is paid out, so less of your wealth works like a bond and your "
                       "savings hold more bonds."))
        gl = glide(key).copy()
        if len(gl) > 1:
            gl.loc[client["retirement_age"]] = gl.iloc[-1]
            fig = go.Figure()
            for grp in ("bonds", "cash", "real", "equities"):        # bonds at the bottom, like the 40% line
                if gl[("C", grp)].max() > 0.005:
                    col = L.group_color(C, grp)
                    fig.add_trace(go.Scatter(x=gl.index, y=gl[("C", grp)], name=L.GROUPS[grp], stackgroup="mix",
                                             mode="lines", line=dict(width=0, color=col), fillcolor=col,
                                             hovertemplate=L.GROUPS[grp] + ": %{y:.0%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=gl.index, y=gl[("A", "bonds")] + gl[("A", "cash")], name="Typical: bonds",
                                     mode="lines", line=dict(color=C["text"], width=1.4, dash="dot"),
                                     hovertemplate="Typical: %{y:.0%} in bonds<extra></extra>"))
            fig.update_layout(**L.plot_layout(C, 250, hovermode="x unified", showlegend=True,
                                              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=C["muted"])),
                                              margin=dict(l=4, r=8, t=30, b=8)))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.0])
            fig.update_xaxes(title=None)
            show(fig)
        html(f'<p class="v3-note">The bonds band (teal, at the bottom) grows with age. The dotted line is the '
             f"{L.pct(typical_bonds)} in bonds of typical advice, the same at every age.</p>")

# =============================================================================
# Details
# =============================================================================
with st.expander("Details: every holding, the full backtest, the assumptions"):
    names = {"A": "Typical 60/40", "M": "Fewer bonds only", "C": "Our plan", "B": "100 minus age"}
    html('<div class="v3-sub">Every holding, in CHF</div>')
    hold = pd.DataFrame({names[s]: plans[s]["weights"] * F for s in ("A", "M", "C", "B")})
    hold = hold[(hold.abs() > 1).any(axis=1)]
    from src.theme import asset_name
    hold.index = [asset_name(a) for a in hold.index]
    html(L.table(hold, {c: (lambda v: L.chf(v, short=False)) for c in hold.columns}))
    if story is not None:
        html('<div class="v3-sub">Backtest, every comparison</div>')
        gl_ = story["gains"].rename(index={"(M) vs (A)": "Fewer bonds vs typical", "(C) vs (M)": "No own industry vs fewer bonds",
                                           "(C) vs (A)": "Our plan vs typical", "(B) vs (A)": "100 minus age vs typical"})
        cols = {"CE gain (CHF/yr)": "Gain, CHF a year", "return part (CHF/yr)": "from return",
                "risk part (CHF/yr)": "from risk", "95% low (CHF/yr)": "95% low", "95% high (CHF/yr)": "95% high",
                "p-value": "p-value"}
        gl_ = gl_[list(cols)].rename(columns=cols)
        fm = {c: (lambda v: L.signed_chf(v)) for c in gl_.columns}
        fm["p-value"] = p_text
        html(L.table(gl_, fm))
        rows_m = {"portfolio excess return (annual)": "Return above cash, a year", "portfolio volatility": "Volatility",
                  "portfolio max drawdown": "Worst fall", "total wealth volatility": "Total wealth volatility",
                  "average share outside bonds and cash": "Average share outside bonds", "turnover per year": "Turnover, a year"}
        mt = story["metrics"].loc[list(rows_m), ["A", "M", "C", "B"]].rename(index=rows_m, columns=names)
        html('<div class="v3-sub">The savings in the backtest</div>')
        html(L.table(mt, {c: (lambda v: L.pct(v, 1)) for c in mt.columns}))
        cr = story["crises"]["portfolio return"].unstack("strategy")
        cr = cr.loc[[c for c in params["backtest"]["crises"] if c in cr.index], ["A", "M", "C", "B"]].rename(columns=names)
        html('<div class="v3-sub">In the crises</div>')
        html(L.table(cr, {c: (lambda v: L.pct(v, 1)) for c in cr.columns}))
    prof = ALLOC["profiles"][profile]
    assumptions = [
        ("Typical advice", f"{L.pct(1 - typical_bonds)} growth (shares and real assets), {L.pct(typical_bonds)} bonds, "
                           "for everybody", "Classic 60/40"),
        ("Bonds in our plan", f"{L.pct(prof['safe_share'])} ({profile}) × (1 − share of wealth that works like a bond)",
         "Total wealth, steps 2 and 3"),
        ("Works like a bond", "bond beta of the career × career value + pension", "Career model, step 2"),
        ("Own industry", "left out when part of the pay depends on it or employer shares are held", "Pivot plan"),
        ("Inside each group", "risk budgeting (lab LN5), budgets from Black-Litterman Sharpe ratios (LN4)", "LN4, LN5"),
        ("Risk", "Ledoit-Wolf shrinkage of the covariance", "LN3"),
        ("Loss limit", f"bad year = 1 in 20; accepted loss {L.pct(prof['loss_tolerance'])} of the savings plus what "
                       "can still be saved", "LN6"),
        ("Backtest", f"{params['backtest']['window_months']}-month windows, rebalanced every "
                     f"{ALLOC['rebalance_months']} months, {params['backtest']['transaction_cost_bp']} bp costs, "
                     f"saving {L.pct(params['backtest']['savings_rate'])} of the salary", "Proposal, step 4"),
    ]
    html('<div class="v3-sub">Assumptions</div>')
    html(L.table(pd.DataFrame(assumptions, columns=["Assumption", "Value", "Source"]).set_index("Assumption")))

html('<p class="v3-foot">Every value lives in config/params.yaml. Data: Yahoo Finance, Kenneth French, FRED and the Swiss '
     "National Bank, monthly in CHF. Beyond the Portfolio is a student project (HEC Lausanne, Quantitative Asset and "
     "Risk Management II), not financial advice.</p>")


# =============================================================================
# While a backtest runs in the background: check every second, then show it
# =============================================================================
@st.fragment(run_every=APP["poll_seconds"])
def poll():
    if not jobs().pending():
        st.rerun()


if jobs().pending():
    poll()
