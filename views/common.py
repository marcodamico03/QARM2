"""What both pages share: data and results kept in memory, background jobs, small helpers."""
import json

import streamlit as st

from src import backtest as B
from src import dashboard as D
from src import optimizer as O
from src.config import load_params
from src.jobs import Jobs

CHART_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner=False)
def base_params():
    return load_params()


@st.cache_data(show_spinner=False)
def excess_returns():
    return O.load_excess_returns()


@st.cache_resource(show_spinner=False)
def jobs():
    """One queue of background jobs for the whole app."""
    return Jobs()


@st.cache_data(show_spinner=False)
def backtest_period():
    """First month decided, last month and number of months of the backtest (step 4)."""
    returns, _ = B.load_backtest_data()
    months = returns.index[int(base_params()["backtest"]["window_months"]):]
    return months[0], months[-1], len(months)


def key_of(persona, changes):
    return json.dumps([persona, changes], sort_keys=True)


@st.cache_data(show_spinner=False, max_entries=64)
def analysis(key):
    """Optimizer of step 3 for the client, with the yearly plan until retirement."""
    persona, changes = json.loads(key)
    params = D.client_params(persona, changes, base_params())
    result = D.analysis(params, excess_returns())
    result["lifecycle"] = D.lifecycle(params, result)
    return result


@st.cache_data(show_spinner=False, max_entries=64)
def glide_path(key):
    persona, changes = json.loads(key)
    params = D.client_params(persona, changes, base_params())
    return D.glide_path(params, analysis(key), excess_returns())


def background(name, params):
    """Result of D.remember(name, params): at once if it is saved on disk, otherwise started in the
    background and None for now (then call wait_for)."""
    result = D.saved(name, params)
    if result is not None:
        return result
    future = jobs().get(D.cache_key(name, params), lambda: D.remember(name, params, B.load_backtest_data()),
                        group=name)
    if future.done():
        return future.result()             # an error in the computation is raised here
    return None


def wait_for(text):
    """A waiting bar that checks every second and reruns the page when the background jobs are done."""
    from src import theme as T

    @st.fragment(run_every=base_params()["app"]["poll_seconds"])
    def check():
        html(T.waiting(text))
        if not jobs().pending():
            st.rerun()
    check()


def html(text):
    st.markdown(text, unsafe_allow_html=True)


def show(fig):
    """Plotly chart at full width, with the new and the old Streamlit API."""
    try:
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
