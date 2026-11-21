"""Page "Your plan": the app as the client sees it (step 5).

One page, read from top to bottom: the client says who they are, gets the advice and the
reasons in plain words, sees where their wealth really is, how the plan changes with age, and
how it would have done since 2005. The backtest runs by itself in the background.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import dashboard as D
from src import optimizer as O
from src import theme as T
from views import common as C

html = C.html
P0 = C.base_params()
APP, PERSONAS = P0["app"], P0["personas"]
GREY = "#9C98B3"


def choice(label, options, default, key):
    """One choice among a few, as a segmented switch (a radio on an old Streamlit)."""
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(label, options, default=default, key=key)
        return value if value is not None else default        # a click on the chosen one unselects it
    return st.radio(label, options, index=list(options).index(default), horizontal=True, key=key)


# =============================================================================
# About you
# =============================================================================
html(T.brand())
jobs = APP["jobs"]
names = list(jobs)
start = st.session_state.get("plan_job", P0["client"]["persona"])
with st.container(key="about"):
    c1, c2, c3, c4 = st.columns([1.35, 0.7, 1, 1], gap="medium")
    persona = c1.selectbox("I work in", names, index=names.index(start), format_func=jobs.get, key="plan_job")
    p = PERSONAS[persona]
    k = f"_{persona}"                                      # each job starts from its own typical client
    age = c2.number_input("Age", *APP["age_range"], int(p["age"]), 1, key="plan_age" + k)
    salary = c3.number_input("Gross salary, CHF a year", *APP["salary_range"], int(p["salary_chf"]),
                             APP["salary_step"], key="plan_salary" + k)
    savings = c4.number_input("Savings to invest, CHF", *APP["savings_range"], int(P0["client"]["portfolio_chf"]),
                              APP["savings_step"], key="plan_savings" + k)

    sector = p["sector"]
    sector_name = APP["sector_words"].get(sector, T.asset_name(sector, P0)) if sector else None
    c1, c2, c3 = st.columns([1.35, 1, 1.7], gap="medium")
    pension = None
    with c1:
        if p.get("pillar2", True):
            how = choice("Pension fund (2nd pillar)", ["Estimate it for me", "From my certificate"],
                         "Estimate it for me", "plan_pension_how" + k)
            if how == "From my certificate":
                estimate = O.pension_savings(age, salary, D.client_params(persona, None, P0))
                pension = st.number_input("Savings on the certificate, CHF", 0, APP["pension_max"],
                                          int(round(estimate, -3)), APP["pension_step"], key="plan_pension" + k)
        else:
            st.caption("Self-employed: no pension fund, so your savings and your business are all you own "
                       "besides your future income.")
    shares = None
    with c2:
        if p.get("deferred_shares_chf", 0) > 0:
            shares = st.number_input("Employer shares you cannot sell yet, CHF", 0, int(savings),
                                     min(int(p["deferred_shares_chf"]), int(savings)), APP["shares_step"],
                                     key="plan_shares" + k)
    with c3:
        beta = None
        if sector:
            link = choice(f"How much does your pay follow {sector_name}?", list(APP["sector_link"]),
                          APP["sector_link_default"], "plan_link" + k)
            beta = p["beta_sector"] * APP["sector_link"][link]

changes = {"age": age, "salary_chf": salary, "portfolio_chf": savings, "deferred_shares_chf": shares,
           "pension_chf": pension, "beta_sector": beta}
st.session_state["plan_client"] = [persona, changes]        # the analyst view starts from this client
key = C.key_of(persona, changes)
try:
    res = C.analysis(key)
except Exception as error:
    st.error(f"We could not build a plan with these numbers ({error}). Try other values.")
    st.stop()

params, sheet, story = res["params"], res["balance_sheet"], res["story"]
weights, F, W = res["weights"], sheet["F"], sheet["W"]
retire = params["client"]["retirement_age"]
shown = [a for a in weights.index if weights.loc[a, ["A", "C"]].max() >= APP["min_weight"]]
mix_c, mix_a = D.groups(weights["C"]), D.groups(weights["A"])

# =============================================================================
# The advice
# =============================================================================
html(f'<p class="btp-kicker">Our advice for your {T.chf(F, short=False)}</p>')
bonds_line = "No bonds." if mix_c["Bonds"] < APP["bonds_visible"] else f"{T.pct(mix_c['Bonds'])} bonds."
html(f'<div class="btp-verdict">{T.pct(mix_c["Shares"])} shares.<br>{bonds_line}</div>')
if mix_a["Bonds"] - mix_c["Bonds"] > APP["bond_gap"]:
    sub = (f"A typical adviser would put {T.pct(mix_a['Bonds'])} of your savings in bonds. "
           "Your job already does that for you.")
else:
    sub = (f"Typical advice: {T.pct(mix_a['Shares'])} shares and {T.pct(mix_a['Bonds'])} bonds. Close to ours, "
           "because your savings now carry most of your wealth.")
html(f'<p class="btp-sub">{sub}</p>')
html(T.band(weights.loc[shown, "C"], params, min_weight=APP["min_weight"]))
html(T.band_row("Typical advice", weights.loc[shown, "A"], params, APP["min_weight"]))

rows = []
for asset in shown:
    w = weights.loc[asset, "C"]
    if w < APP["min_weight"]:
        continue
    note = "locked: you cannot sell them yet" if asset == O.EMPLOYER else None
    rows.append((T.asset_color(asset), T.asset_name(asset, params), note, T.chf(w * F, short=False), T.pct(w)))
html(T.holdings(rows))

holdings = res["career_holdings"]
bond = params["data"]["career_bond_asset"]
cards = []
if mix_a["Bonds"] - mix_c["Bonds"] > APP["bond_gap"] and sheet["P"] > sheet["H"]:
    cards.append(("Your pension works like a bond",
                  f"Your pension fund pays a guaranteed interest: its <b>{T.chf(sheet['P'])}</b> are as safe as "
                  "government bonds. With that much already safe, your savings can earn more in shares.", "blue"))
elif mix_a["Bonds"] - mix_c["Bonds"] > APP["bond_gap"]:
    cards.append(("Your job works like a bond",
                  f"Your salary is a steady income in francs, a bit like owning "
                  f"<b>{T.chf(holdings.get(bond, 0))}</b> of Swiss government bonds. You already have more than "
                  "enough of them, so your savings can earn more in shares.", "blue"))
else:
    cards.append(("Your savings take over from your job",
                  "With few working years left, your salary no longer holds many bonds for you. "
                  "Your savings hold them instead.", "blue"))
if sector and story["sector_D"] - story["sector_C"] > APP["min_weight"]:
    cards.append((f"Your pay already depends on {sector_name}",
                  f"When {sector_name} do badly, bonuses and jobs go first. We leave them out of your "
                  "savings, so a bad year does not hit you twice.", "red"))
elif sector:
    cards.append(("Nothing to offset",
                  f"Even without your job in the picture we would not buy {sector_name}, so a bad year for "
                  "them touches only your income.", "red"))
else:
    cards.append(("Nothing to offset",
                  "Your pay does not follow any industry, so your savings do not need to avoid one.", "red"))
gain = res["performance"].loc["gain over (A), CHF per year", "C"]
cards.append((f"Worth {T.chf(gain, short=False)} a year to you",
              "That is how much better this plan is than typical advice, once the extra risk is counted: "
              "like receiving this amount every year, with no risk.", "plum"))
html(T.reasons(cards))

# =============================================================================
# What you really own
# =============================================================================
career_share = sheet["H"] / W
share_text = "Almost all" if T.pct(career_share) == T.pct(1) else T.pct(career_share)
html(f'<div class="btp-h2">You own {T.chf(W)}. {share_text} of it is your future salary.</div>')
pension_text = f", and your pension fund holds {T.chf(sheet['P'])}" if sheet["P"] > 0 else ""
html(f'<p class="btp-p">Your career will pay you a salary for {retire - age} more years{pension_text}. '
     f"An adviser who only looks at your savings sees {T.pct(F / W, 1)} of what you own.</p>")
html(T.ribbon(sheet))

years, rate = D.career_years(params)
if len(years):
    html(T.chart_title("How we value your career", "Each bar is one year of salary until retirement."))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years.index, y=years["salary"], name="Salary that year", marker_color="#E2D8EE",
                         hovertemplate="Salary: CHF %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=years.index, y=years["value today"], name="What it is worth today",
                         marker_color=T.PLUM, hovertemplate="Worth today: CHF %{y:,.0f}<extra></extra>"))
    fig.update_layout(**T.plot_layout(330, barmode="overlay", bargap=0.18, hovermode="x unified"))
    fig.update_xaxes(title="Age", dtick=5)
    fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
    C.show(fig)
    sector_text = (f" and {T.chf(holdings.get(sector, 0))} of {sector_name}"
                   if sector and holdings.get(sector, 0) > 0 else "")
    html(f'<p class="btp-p">You will earn about {T.chf(years["salary"].sum())} until {retire}. A franc in twenty '
         f"years is worth less than a franc today, so each year is discounted at {T.pct(rate, 2)} a year, "
         f"the rate that matches how risky your salary is. The dark bars add up to <b>{T.chf(sheet['H'])}</b>: "
         f"your career today. It moves with markets like {T.chf(holdings.get(bond, 0))} of Swiss government "
         f"bonds{sector_text}, and that is what your savings have to balance.</p>")

# =============================================================================
# The plan by age
# =============================================================================
glide = C.glide_path(key).copy()
if len(glide) > 1:
    bonds_c = glide[("C", "Bonds")]
    first = next((a for a, v in bonds_c.items() if v > APP["bonds_visible"]), None)
    if first is None:
        title = "Shares until you retire."
    elif first > age:
        title = f"Shares until about {first:.0f}, then bonds come in."
    else:
        title = "More bonds as retirement gets closer."
    html(f'<div class="btp-h2">{title}</div>')
    html('<p class="btp-p">Every year one year of salary is paid out: your career is worth less and holds fewer '
         "bonds for you, while your savings grow. Near retirement your savings have to do the job your salary "
         "did, and bonds come back.</p>")
    html(T.chart_title("Your savings by age", f"Our plan, solved again every {APP['glide_step_years']} years with "
                       "the wealth you are expected to have then. The dashed line is typical advice."))
    glide.loc[retire] = glide.iloc[-1]                     # the last plan is kept until retirement
    fig = go.Figure()
    for g in D.GROUPS:
        if glide[("C", g)].max() > APP["min_weight"]:
            fig.add_trace(go.Scatter(x=glide.index, y=glide[("C", g)], name=g, stackgroup="mix", mode="lines",
                                     line=dict(width=0, color=T.GROUP_COLORS[g]),
                                     fillcolor=T.GROUP_COLORS[g], hovertemplate=g + ": %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=glide.index, y=glide[("A", "Shares")], name="Typical advice, shares", mode="lines",
                             line=dict(color=T.INK, width=2, dash="dash"),
                             hovertemplate="Typical advice: %{y:.0%} shares<extra></extra>"))
    fig.update_layout(**T.plot_layout(340, hovermode="x unified"))
    fig.update_layout(legend_traceorder="normal")
    fig.update_xaxes(title="Age", dtick=5)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    C.show(fig)
    end = res["lifecycle"]["F"].iloc[-1]
    html(f'<p class="btp-p">If markets give their average return and you save '
         f"{T.pct(params['backtest']['savings_rate'])} of your salary, your savings reach about "
         f"<b>{T.chf(end)}</b> at {retire}.</p>")

# =============================================================================
# Since 2005
# =============================================================================
first_month, last_month, n_months = C.backtest_period()
html(f'<div class="btp-h2">Would it have worked? We replayed {first_month.year} to {last_month.year}.</div>')
html(f'<p class="btp-p">Someone like you: {age} in {first_month:%B %Y}, earning {T.chf(salary, short=False)} a year and '
     f"saving {T.pct(params['backtest']['savings_rate'])} of it every month. Each month the plan used only what "
     "was known at the time, and paid trading costs.</p>")
bt = C.background("backtest", params)
prior = C.background("prior", params) if bt is not None else None
if bt is None:
    C.wait_for(f"Replaying {n_months} months for you. The rest of the page is ready.")
else:
    values = bt["values"]
    html(T.chart_title("Your savings, month by month", "Shaded: the crises of the period."))
    fig = go.Figure()
    for i, (crisis, (first, last)) in enumerate(params["backtest"]["crises"].items()):
        x0 = pd.Timestamp(first) - pd.offsets.MonthEnd(1)
        x1 = pd.Timestamp(last) + pd.offsets.MonthEnd(0)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=T.RED, opacity=0.08, line_width=0)
        fig.add_annotation(x=x0, y=1.0 if i % 2 == 0 else 0.92, yref="paper", text=crisis, showarrow=False,
                           xanchor="left", font=dict(size=10, color=T.MUTED))
    end_a, end_c = values["A"].iloc[-1], values["C"].iloc[-1]
    apart = 0 if abs(end_c - end_a) > APP["labels_apart"] * max(end_a, end_c) else 10
    for s, name, color, width, shift in (("A", "Typical advice", GREY, 2, -apart if end_a < end_c else apart),
                                         ("C", "Our plan", T.PLUM, 3.2, apart if end_c >= end_a else -apart)):
        fig.add_trace(go.Scatter(x=values.index, y=values[s], name=name, mode="lines",
                                 line=dict(color=color, width=width),
                                 hovertemplate=name + ": CHF %{y:,.0f}<extra></extra>"))
        fig.add_annotation(x=values.index[-1], y=values[s].iloc[-1], text=f"<b>{name}</b><br>{T.chf(values[s].iloc[-1])}",
                           showarrow=False, xanchor="left", align="left", xshift=8, yshift=shift,
                           font=dict(color=color, size=13))
    fig.update_layout(**T.plot_layout(420, hovermode="x unified", showlegend=False,
                                      margin=dict(l=8, r=150, t=36, b=8)))
    fig.update_yaxes(tickprefix="CHF ", tickformat="~s")
    C.show(fig)

    m = {s: bt["metrics"][name] for s, name in (("A", "(A) standard: career ignored"),
                                                 ("D", "(D) total wealth, no hedge"), ("C", "(C) total wealth"))}
    g = bt["gains"].loc["(C) vs (A)"]
    caveat = None
    if prior is not None:
        swiss = prior.iloc[-1]
        caveat = (f"With a more cautious Swiss pension-fund view of future returns, the gain is "
                  f"{T.chf(swiss['(C) vs (A) CHF/yr'], short=False)} a year (p = {swiss['(C) vs (A) p-value']:.2f})"
                  + (", less certain." if swiss["(C) vs (A) p-value"] > APP["significance"] else "."))
    items = [(T.chf(values["C"].iloc[-1]), f"your savings today with our plan, against {T.chf(values['A'].iloc[-1])} "
              "with typical advice.", f"More shares means a bumpier ride: the worst fall was "
              f"{T.pct(m['C']['portfolio max drawdown'])} against {T.pct(m['A']['portfolio max drawdown'])}.", T.PLUM),
             (f"{T.chf(g['CE gain (CHF/yr)'], short=False)} a year",
              "better than typical advice, once the extra risk is counted on everything you own.",
              f"p = {g['p-value']:.2f}: a gain this large would rarely come from luck. " + (caveat or ""), T.PLUM)]
    worst = "CHF in the sector's worst 10% months"
    saved_chf = m["C"][worst] - m["D"][worst]
    if sector and story["sector_D"] > APP["min_weight"] and saved_chf > 0:
        items.append((T.chf(saved_chf, short=False) + " a month", f"less lost in an average bad month for {sector_name}, "
                      "by leaving them out of your savings.", "Leaving your sector out lowers risk. It does not "
                      "always raise the return.", T.RED))
    html(T.tiles(items))
    if prior is None:
        C.wait_for("Checking the result with a second view of future returns.")

# =============================================================================
# Fine print
# =============================================================================
with st.expander("How we work this out"):
    caps = list(params["optimization"]["position_caps"].values())
    html(f"""<p class="btp-p">We look at everything you own: your savings, your pension fund and the value of your
career. Expected returns come from how the world market itself is invested (Black-Litterman), not from our
guesses. Your appetite for risk is that of a balanced investor with 60% in shares and 40% in bonds.</p>
<p class="btp-p">Our rules: no borrowing, no short selling, and at most {T.pct(caps[0]) if caps else "100%"} in one
sector or single stock. Employer shares you cannot sell yet stay in your savings until you can. The page
<b>Analyst view</b> shows every number, what each rule costs you and every test of the model.</p>""")
html('<p class="btp-foot">Beyond the Portfolio is a student project (HEC Lausanne, Quantitative Asset and Risk '
     "Management II). It is not financial advice. Data: Yahoo Finance, Kenneth French, FRED and the Swiss "
     "National Bank, monthly in CHF.</p>")
