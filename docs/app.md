# The app and version 2 of the advice (step 5)

- Code: `app.py` (the page), `src/allocation.py` (version 2 of the advice), `src/theme.py` (the look),
  `src/jobs.py` and `src/dashboard.py` (the backtest in the background, kept on disk in `data/cache/`).
- Settings: sections `allocation` and `app` of `config/params.yaml`; data: `universe2` (12 industries,
  corporate bonds, commodities) in the same file, saved by `python -m src.data` in `data/universe2_chf.csv`.
- Tests: `notebooks/test_app.py` (also runs the app headless for every example client).
- The app before the pivot (mean-variance on total wealth): git tag `app-v1`.

```bash
python -m src.data
streamlit run app.py
python notebooks/test_app.py
```

## Why version 2

With plain mean-variance on total wealth the savings are a few percent of wealth, so the client is almost
risk-neutral on them and the optimizer ends at a corner: about 100% world equities for every client
(the pivot plan has the numbers). Version 2 keeps the total-wealth idea but spreads the savings with risk
budgeting (LN5-LN6), and the job decides four things.

## How the plan is built (`src/allocation.py`)

| Step | What | Course |
|---|---|---|
| Universe | 12 Fama-French industries, Swiss equities, Swiss real estate, gold, commodities, Swiss government, global government (hedged) and corporate bonds (hedged), all in CHF | LN1 |
| Risk and returns | Ledoit-Wolf covariance; Black-Litterman with Doeswijk-Swinkels shares, equities split by MSCI World sector weights | LN3, LN4 |
| 1. Own industry | Left out of the purchases when part of the pay depends on it or employer shares are held | |
| 2. Employer shares | Held while locked (fixed holding); the rest is built around them; they join the plan when they can be sold | |
| 3. Bonds | Profile's bond share x (1 - bond-like share of wealth); bond-like share = (bond beta of the career x career value + pension) / total wealth | Steps 2-3 |
| Growth part | Risk budgeting with the stake fixed: y_i (Sigma e)_i proportional to the budget; budgets by profile between shares and real assets, by Black-Litterman Sharpe ratio inside each group | LN5, LN4 |
| 4. Loss limit | Bad year (1 in 20, Gaussian) loss <= profile tolerance x savings + savings still to come before retirement; else cash | LN6 |
| Measures | Risk contributions by part; effective number of bets; diversification ratio | LN5, LN6 |

Solver: cyclical coordinate descent for risk budgeting (Griveau-Billon, Richard and Roncalli, 2013, in the
course reading), with the multiplier set so the amounts add up to the money to invest.

Strategies: (A) typical advice = the profile's model portfolio, without the job or the shares; (B) 100 minus
age; (C) our plan. Backtest: the step 4 engine, plans solved every 3 months on the past 60 months.

## The page

- **Side panel.** Example client, industry (12, or none), age, salary, savings to invest, employer shares and
  when they can be sold, part of pay that depends on the industry, self-employed, pension, risk profile, a market view.
- **Answer.** One sentence with the mix, why in one line, everything the client owns as one bar.
- **Four panels.** Portfolio (our plan and typical advice as bars, holdings by group, reasons); the backtest;
  where the risk of the savings comes from (employer shares, shares, real assets, bonds) with the effective number
  of bets; the plan by age.
- **Tabs.** Every holding in CHF; budgets, expected returns and risk contributions; the full backtest; assumptions.

Everything updates by itself; the backtest runs in the background (a few seconds) and is kept on disk.
`python -m src.precompute` computes the example clients in advance (optional).

## Demo (3 minutes)

1. **Banker, 35.** 68% shares, 23% real assets, 9% bonds (typical advice: 35% bonds). No finance in the rest,
   the UBS shares stay three years. Risk: the shares are a third of the risk of the savings.
2. **Profile and age.** Dynamic against prudent; then age 62 and CHF 2 m: bonds and the loss limit come in.
3. **Another job.** Technology or civil servant: the left-out industry changes, or nothing is left out.
4. **Backtest and tabs.** Our plan against typical advice, with the p-value; the budgets tab for the method.
