# Data layer (step 1): choices to discuss

Code: `src/data.py`. Settings: `config/params.yaml` (section `data`).

## How to use it

```bash
pip install -r requirements.txt
python -m src.data          # downloads everything, prints a coverage report, saves data/returns_chf.csv
```

```python
from src.data import load_returns, load_rf, excess_returns
returns = load_returns()        # monthly CHF returns, one column per asset + "rf" (reads the saved file)
returns_2008 = load_returns("2008-01-01", "2008-12-31")   # optional period
rf = load_rf()                  # CHF risk-free rate only
excess = excess_returns(returns)
```

## Output

`data/returns_chf.csv`: month-end to month-end total returns **in CHF**, in decimals, one column per asset, plus `rf` (CHF risk-free rate for the month, can be negative). Only months where every column has data are kept, so there are no holes.
Raw downloads are saved in `data/raw/`. If a website is down, the saved copy is used automatically.
`data/coverage.csv` shows which source was used for each asset and period.

## Universe and sources

| Asset (column) | Investable series | Long-history proxy (only before the investable series starts) |
|---|---|---|
| `world_equity` | URTH, iShares MSCI World (USD, from 2012) | Ken French developed-markets factor file: Mkt-RF + RF (USD, from 1990) |
| `euro_banks` (the banker's sector, R_S) | EXV1.DE, iShares STOXX Europe 600 Banks (EUR) | Equal-weight basket of 8 euro-area banks (BNP.PA, SAN.MC, DBK.DE, INGA.AS, ISP.MI, BBVA.MC, GLE.PA, UCG.MI; EUR, Dec 1999 to Jan 2008), then Ken French "Banks" (US banks, USD) before that |
| `swiss_equity` | EWL, iShares MSCI Switzerland (USD, from 1996) | not needed |
| `chf_bonds` (R_B) | synthetic index from the SNB 10y Confederation yield, duration 8: new cube `rendeiduebd` (2025 methodology), joined with the old cube `rendoblim` (stops on 2025-07-31) | not needed |
| `global_bonds_hedged` | synthetic US 10y Treasury index (FRED DGS10), duration 8, hedged to CHF | not needed |
| `swiss_real_estate` | equal-weight basket PSPN.SW, SPSN.SW, ALLN.SW (CHF) | not needed |
| `gold` | GLD (USD, from 2004) | GC=F gold futures (from 2000) |
| `employer_stock` | UBSG.SW (CHF) | UBS on NYSE (USD) |

Rates and FX: CHF risk-free = SARON from the SNB data portal (cube `zimoma`). USDCHF and EURUSD from FRED daily series (DEXSZUS, DEXUSEU), month-end value.

## Formulas

- Unhedged CHF return: (1 + r_CHF) = (1 + r_local)(1 + change in CHF per foreign unit).
- Synthetic bond return: r_t = y_{t-1}/12 − D × (y_t − y_{t-1}).
- Hedged return: r_local + (CHF rate − foreign rate)/12, both rates at the start of the month. From 2015 to 2022 this carry is negative.
- Risk-free: rf_t = SARON at the end of month t−1 / 12.

## Data cleaning

- Any monthly return above `max_abs_monthly_return` (60%) is flagged.
- In a basket (several tickers), each member is checked separately and a bad month is dropped **for that member only**, because one bad price is diluted in the average and can pass the final check. Example: PSPN.SW in May 2004 (+133%, an unadjusted share operation on Yahoo) had pushed the real estate basket up 45% that month.

## Open points for the group

1. **World equities include financials.** An ex-financials world index with history from 2000 is not freely available. Banks are about 10% of MSCI World, so the hedge appears as "zero in euro_banks", not as "zero financials".
2. **Global bonds = US Treasuries hedged.** It is a proxy for global bonds, and the history is consistent from 2000. Alternative: a global aggregate ETF, but those start around 2008.
3. **Bond duration = 8** for both bond indices (`duration` in params.yaml).
4. **Swiss real estate = three listed property companies** instead of the SXI real estate funds index, which has no free long history.
5. **Proxy for euro_banks before 2008 was US banks.** EXV1.DE on Yahoo starts in February 2008, so in the first build US banks covered September 2000 to January 2008 (89 months), including the whole 60-month window before the 2008 crisis. This is now replaced by an equal-weight basket of 8 large euro-area banks, so from the start of the sample (September 2000) no US data is used. (EXX1.DE and the ^SX7E index were tried first: Yahoo has no data for them before 2008.) The basket has some survivorship bias (only banks that still exist today) and is equal-weight rather than cap-weight; acceptable for a proxy used only before 2008.
6. **The SNB changed its bond-yield methodology in July 2025.** The old cube `rendoblim` stops on 2025-07-31, which cut the whole table there (only complete months are kept). The new cube `rendeiduebd` is now tried first and needs a dimension selection (`dim_sel` in params.yaml). If its codes are wrong, the error message lists the available ones.
7. **SNB codes `10J` and `SARON` must be checked on the first real run.** If a code is wrong, the error message lists the available codes. Also check whether the SNB 10y yield is a month-end value or a monthly average; an average would make bond volatility look too low.
