"""Shared helpers for all pages: loading data once, the current client, formatting, colours."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import streamlit as st                  # noqa: E402

import backtest as bt                   # noqa: E402
import optimizer as o                   # noqa: E402
from config import INVESTABLE           # noqa: E402
from data_loader import load_excess_returns, load_rates  # noqa: E402
from estimation import estimate_inputs  # noqa: E402
from personas import BANKER, CIVIL_SERVANT, Persona  # noqa: E402

PRESETS = {BANKER.name: BANKER, CIVIL_SERVANT.name: CIVIL_SERVANT}

ASSET_LABELS = {
    "world_eq": "World equities", "smi": "Swiss equities (SMI)", "eu_banks": "European banks",
    "ubs": "UBS (employer)", "chf_bonds": "CHF bonds", "global_bonds_hedged": "Global bonds (CHF-hedged)",
    "swiss_re": "Swiss real estate", "gold": "Gold", "cash": "Cash (CHF)",
    "career": "Career", "pension": "Pension (2nd pillar)",
}
ASSET_COLORS = {
    "World equities": "#1f77b4", "Swiss equities (SMI)": "#d62728", "European banks": "#ff7f0e",
    "UBS (employer)": "#8c564b", "CHF bonds": "#2ca02c", "Global bonds (CHF-hedged)": "#98df8a",
    "Swiss real estate": "#9467bd", "Gold": "#bcbd22", "Cash (CHF)": "#c7c7c7",
    "Career": "#1f4e79", "Pension (2nd pillar)": "#e377c2",
}
STRAT_LABELS = {"A": "A · Standard", "B": "B · 100 − age", "D": "D · No hedge", "C": "C · Total wealth (ours)"}
STRAT_COLORS = {"A": "#7f7f7f", "B": "#bcbd22", "D": "#ff7f0e", "C": "#1f4e79"}


def setup(title: str, icon: str = "💼"):
    st.set_page_config(page_title=f"{title} · Beyond the Portfolio", page_icon=icon, layout="wide")
    p = get_persona()
    with st.sidebar:
        st.markdown("### Current client")
        st.markdown(f"**{p.name}**, age {p.age}  \nSalary CHF {p.salary:,.0f}  \n"
                    f"Career betas: β_B = {p.beta_B:.1f}, β_S = {p.beta_S:.2f}")
        st.caption("Change the client on page 1 · Who are you?")
        st.caption("🔒 Running locally on this computer only.")


def get_persona() -> Persona:
    return st.session_state.get("persona", BANKER)


@st.cache_resource(show_spinner="Estimating expected returns and risk…")
def market():
    """Full-sample inputs (Black-Litterman + Ledoit-Wolf) and the calibrated gamma."""
    X = load_excess_returns()
    m = estimate_inputs(X)
    gamma = o.calibrate_gamma(m)
    return X, m, gamma


@st.cache_resource(show_spinner="Preparing the walk-forward backtest (first time only, ~10 s)…")
def backtest_inputs():
    X, R = load_excess_returns(), load_rates()
    return X, R, bt.precompute(X, INVESTABLE)


@st.cache_data(show_spinner="Running the backtest for this client…")
def backtest_results(p: Persona, job_loss_threshold=None):
    X, R, pre = backtest_inputs()
    return bt.run_all(p, pre, X, R, job_loss_threshold=job_loss_threshold)


@st.cache_data(show_spinner="Running the wrong-beta robustness test…")
def wrong_beta(p: Persona):
    X, R, pre = backtest_inputs()
    return bt.wrong_beta_curve(p, pre, X, R)


def label(s: pd.Series | pd.DataFrame):
    return s.rename(index=ASSET_LABELS)


def chf(x: float) -> str:
    return f"CHF {x:,.0f}".replace(",", "'")


def pct(x: float, d: int = 0) -> str:
    return f"{x:.{d}%}"
