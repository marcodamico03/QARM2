"""Beyond the Portfolio: the Streamlit app (step 5).

Run from the project folder:
    streamlit run app.py

Two pages: "Your plan" (views/plan.py), the app as a client uses it, and "Analyst view"
(views/analyst.py), every number behind it. The numbers come from src/ (steps 1 to 4) through
src/dashboard.py; the look is in src/theme.py and .streamlit/config.toml.
Before a demo: python -m src.precompute (the backtests are then instant).
"""
import streamlit as st

from src import theme as T

st.set_page_config(page_title="Beyond the Portfolio", page_icon="◐", layout="wide",
                   initial_sidebar_state="auto")
st.markdown(T.css(), unsafe_allow_html=True)

pages = [st.Page("views/plan.py", title="Your plan", default=True, url_path="plan"),
         st.Page("views/analyst.py", title="Analyst view", url_path="analyst")]
try:
    page = st.navigation(pages, position="top")
except TypeError:                                     # Streamlit before 1.46: the menu goes in the side panel
    page = st.navigation(pages)
page.run()
