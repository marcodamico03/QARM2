import streamlit as st

from common import setup

setup("Home", "🏔️")

st.title("Beyond the Portfolio")
st.subheader("A life-cycle, total-wealth optimizer: from hedging your career to choosing your pension")

st.markdown("""
Standard portfolio tools optimize only what sits in your brokerage account.
But for most people the biggest assets are things they **cannot trade**: their **career** while they work,
and their **pension** when they retire.

This app builds the financial portfolio that **complements** those assets, and tests on 20 years of
Swiss-franc data whether it would have protected the client better than a standard portfolio.
""")

c1, c2, c3 = st.columns(3)
c1.metric("Our core client", "Zurich banker, 35")
c2.metric("Portfolio", "CHF 100'000")
c3.metric("Value of his career", "> CHF 2 million")

st.markdown("""
### How to use the app
1. **Who are you?** pick a client and adjust the assumptions
2. **Your real balance sheet**: career, pension and portfolio over life
3. **Your optimal portfolio**: standard vs total-wealth portfolio
4. **What your constraints cost**: shadow prices in plain language
5. **Where your risk comes from**: how much of your risk is your job?
6. **Did it work?** walk-forward backtest 2006–2026
7. **What we assume**: every assumption, its value and its source
""")
st.info("All returns are in CHF. Expected returns come from Black-Litterman, risk from Ledoit-Wolf shrinkage. "
        "Results are a model, not a promise: see page 7 for every assumption.")
