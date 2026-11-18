"""Streamlit home page of the app.

Run with:  streamlit run app.py
The other pages go in the pages/ folder; Streamlit adds them to the sidebar automatically.
"""
import streamlit as st

st.set_page_config(page_title="Beyond the Portfolio", layout="wide")

st.title("Beyond the Portfolio")
st.write("Life-cycle, total-wealth portfolio optimizer. Work in progress.")
