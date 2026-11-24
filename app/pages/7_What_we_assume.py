import common  # noqa: F401  -- must stay first: tells Python where src/ is
import pandas as pd
import streamlit as st

import career
import config as c
from common import ASSET_LABELS, chf, get_persona, market, setup

setup("What we assume", "📋")
p = get_persona()
X, m, gamma = market()
st.title("7 · What we assume")
st.markdown("Every assumption behind the numbers, with its value and where it comes from.")

rows = [
    # client
    ("Client", "Age / retirement age", f"{p.age} / {p.retirement_age}", "Persona preset (page 1)"),
    ("Client", "Salary, growth", f"{chf(p.salary)}, {p.salary_growth:.1%} per year", "Our assumption"),
    ("Client", "Savings rate", f"{p.savings_rate:.0%} of salary", "Our assumption"),
    ("Client", "Portfolio / pension today", f"{chf(p.portfolio)} / {chf(p.pension)}", "Our assumption"),
    ("Client", "Deferred employer shares", f"{p.employer_lockup:.0%} minimum weight", "Our assumption"),
    # career
    ("Career", "Model", "R_H − rf = β_B (R_B − rf) + β_S (R_S − rf) + ε",
     "Viceira (2001); Cocco, Gomes & Maenhout (2005); Campbell & Viceira (2002)"),
    ("Career", "β_B (bond-like part)", f"{p.beta_B}", "Assumption: long career duration, around 3 (proposal 3.1)"),
    ("Career", "β_S (sector part)", f"{p.beta_S}", "Assumption, conservative on purpose (page 6, wrong-beta test)"),
    ("Career", "Bond index R_B / sector R_S", f"{ASSET_LABELS[c.CAREER_BOND_ASSET]} / {ASSET_LABELS[p.sector_asset]}", "Data"),
    ("Career", "Discount rate for the career", f"{career.discount_rate(p):.2%}",
     f"rf {c.RF_LONG_RUN:.1%} + β_B × {c.BOND_PREMIUM:.1%} + β_S × {c.SECTOR_PREMIUM:.1%}"),
    # pension
    ("Pension (BVG 2026)", "Coordination deduction", chf(c.BVG_COORDINATION_DEDUCTION), "BSV / federal law"),
    ("Pension (BVG 2026)", "Minimum interest rate", f"{c.BVG_MIN_INTEREST:.2%}", "BSV / federal law"),
    ("Pension (BVG 2026)", "Savings credits by age", "7% / 10% / 15% / 18%", "BVG art. 16"),
    ("Pension (BVG 2026)", "Risk of the pension", "σ_P ≈ 0 (credited at a guaranteed rate)", "Proposal 3.2"),
    ("Pension (BVG 2026)", "Insured salary", "whole salary above the coordination deduction",
     "Assumption: banks insure more than the legal minimum"),
    # markets
    ("Markets", "Data", "monthly, CHF, Jan 1999 to Sep 2026", "yfinance, FRED, Ken French data library"),
    ("Markets", "Expected returns", "Black-Litterman, market prior δ Σ w_mkt", "He & Litterman; Doeswijk, Lam & Swinkels"),
    ("Markets", "Market risk aversion δ", f"{m.delta:.2f}", f"market excess return {c.MARKET_EXCESS_RETURN:.1%} / market variance"),
    ("Markets", "Market weights", "≈ 40% equities, 54% bonds, 4% real estate, 1% gold",
     "Approximate, from Doeswijk, Lam & Swinkels"),
    ("Markets", "Covariance", "Ledoit-Wolf shrinkage", "Ledoit & Wolf (2004)"),
    ("Markets", "CHF bonds", f"built from SNB/OECD 10y yields, duration {c.DURATION_CHF_BONDS}", "FRED IRLTLT01CHM156N"),
    ("Markets", "Global bonds", "US 10y yields, hedged to CHF with the short-rate difference", "FRED GS10"),
    ("Markets", "European banks before 2008", "US banks (Ken French), converted to CHF", "Proxy: EXV1.DE starts in 2008"),
    ("Markets", "Swiss equities", f"SMI price index + {c.SMI_DIVIDEND_YIELD:.0%} dividend yield", "Assumption"),
    # optimisation
    ("Optimisation", "Risk aversion γ", f"{gamma:.2f}", "Set so a client with no career holds 60/40; same for all strategies"),
    ("Optimisation", "Constraints", "long-only, no borrowing, caps "
     + ", ".join(f"{ASSET_LABELS[k]} ≤ {v:.0%}" for k, v in c.POSITION_CAPS.items()), "Client constraints"),
    # backtest
    ("Backtest", "Window / rebalancing", f"{c.BACKTEST_WINDOW} months rolling, monthly", "Walk-forward, no look-ahead"),
    ("Backtest", "Transaction costs", f"{c.TRANSACTION_COST_BP} bp × turnover", "Assumption"),
    ("Backtest", "Job loss", f"{c.JOB_LOSS_MONTHS} months without salary if the sector falls "
     f"{c.JOB_LOSS_THRESHOLD:.0%} below its 2-year high (varied 20–40%)", "Assumption"),
]
df = pd.DataFrame(rows, columns=["Area", "Assumption", "Value", "Source"])
for area, g in df.groupby("Area", sort=False):
    st.markdown(f"#### {area}")
    st.dataframe(g.drop(columns="Area"), hide_index=True, width="stretch")
