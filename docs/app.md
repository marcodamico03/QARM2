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
| Growth part | Risk budgeting with the stake fixed: y_i (Sigma e)_i proportional to the budget; budgets by profile between shares and real assets, by Black-Litterman expected return per unit of risk inside each group | LN5, LN4 |
| 4. Loss limit | Bad year (1 in 20, Gaussian) loss <= profile tolerance x savings + savings still to come before retirement; else cash | LN6 |
| Measures | Risk contributions by part; effective number of bets | LN5, LN6 |

Solver: cyclical coordinate descent for risk budgeting (Griveau-Billon, Richard and Roncalli, 2013, in the
course reading), with the multiplier set so the amounts add up to the money to invest.

Strategies on the page: (A) typical advice = the profile's model portfolio, without the job or the shares;
(M) fewer bonds only (A plus the career as a bond, own industry kept); (C) our plan. The code also computes
(B) 100 minus age, not shown. Backtest: the step 4 engine, plans solved every 3 months on the past 60 months.

## The page

- **Side panel.** Example client, industry (12, or none), age, salary, savings to invest, employer shares and
  when they can be sold, part of pay that depends on the industry, self-employed (pension fund estimated from age
  and salary otherwise), risk profile, a market view under Advanced.
- **Answer.** One sentence with the mix, why in one line, everything the client owns as one bar.
- **Panels.** Portfolio (our plan and typical advice as bars of shares, real assets and bonds, holdings by group,
  reasons); the backtest since 2005; the two changes (fewer bonds, no own industry) and what each was worth, with
  how often it helped in the bootstrap; the plan by age.
- **Tabs.** Every holding in CHF; risk and budgets (where the risk comes from, effective number of bets, market
  weights, expected returns, budgets, risk contributions); what you give up and the crises; assumptions.

Everything updates by itself; the backtest runs in the background (a few seconds) and is kept on disk.
`python -m src.precompute` computes the example clients in advance (optional).

## Demo (3 minutes)

1. **Banker, 35.** 66% shares, 24% real assets, 10% bonds (typical advice: 40% bonds). No finance in the rest,
   the employer shares stay three years.
2. **Profile and age.** Dynamic against prudent; then age 62 and CHF 2 m: bonds and the loss limit come in.
3. **Another job.** Technology or civil servant: the left-out industry changes, or nothing is left out.
4. **The two changes.** Fewer bonds earns most of the gain and is reliable; leaving out the own industry is
   insurance. Then the tabs: risk and budgets for the method, what you give up.
