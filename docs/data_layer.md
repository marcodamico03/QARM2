# Data layer (step 1): choices to discuss

Code: `src/data.py`. Settings: `config/params.yaml` (section `data`).

## How to use it

```bash
pip install -r requirements.txt
python -m src.data          # downloads everything, prints a coverage report, saves data/returns_chf.csv
```

```python
from src.data import load_returns, excess_returns
returns = load_returns()        # monthly CHF returns, one column per asset + "rf" (reads the saved file)
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
| `euro_banks` (the banker's sector, R_S) | EXV1.DE, iShares STOXX Europe 600 Banks (EUR) | Ken French 49 industries, "Banks" (US banks, USD) |
| `swiss_equity` | EWL, iShares MSCI Switzerland (USD, from 1996) | not needed |
| `chf_bonds` (R_B) | synthetic index from the SNB 10y Confederation yield, duration 8 | not needed |
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

## Open points for the group

1. **World equities include financials.** An ex-financials world index with history from 2000 is not freely available. Banks are about 10% of MSCI World, so the hedge appears as "zero in euro_banks", not as "zero financials".
2. **Global bonds = US Treasuries hedged.** It is a proxy for global bonds, and the history is consistent from 2000. Alternative: a global aggregate ETF, but those start around 2008.
3. **Bond duration = 8** for both bond indices (`duration` in params.yaml).
4. **Swiss real estate = three listed property companies** instead of the SXI real estate funds index, which has no free long history.
5. **Proxy for euro_banks before 2001 is US banks.** Only a few months are affected.
6. **SNB codes `10J` and `SARON` must be checked on the first real run.** If a code is wrong, the error message lists the available codes. Also check whether the SNB 10y yield is a month-end value or a monthly average; an average would make bond volatility look too low.
