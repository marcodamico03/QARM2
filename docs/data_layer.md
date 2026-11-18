# Data layer (step 1): choices to discuss

Code: `src/data.py` (follows `docs/CONTRACT.md`). Settings: `config/params.yaml`, section `data`. Tests: `notebooks/test_data.py`.

## How to use it

```bash
python -m src.data              # downloads everything, prints a coverage report, saves the files in data/
python notebooks/test_data.py   # offline tests with known answers
```

```python
from src.data import load_returns, load_rf
returns = load_returns("2000-01-01", "2026-12-31")   # DataFrame, monthly CHF returns
rf = load_rf("2000-01-01", "2026-12-31")             # Series RF_CHF
excess = returns.sub(rf, axis=0)                     # what the optimizer uses
```

## Output (contract, sections 1.4 and 1.5)

- `data/returns_chf.csv`: monthly simple returns in CHF, in decimals. Index: month-end dates named `date`. Columns: `EQ_WORLD, BANKS_EU, EQ_CH, BOND_CHF, BOND_GLB_H, REIT_CH, GOLD, EMPLOYER`.
- `data/rf_chf.csv`: `RF_CHF`, the CHF risk-free rate for the month (it can be negative).
- Only months where every asset and the risk-free rate have data are kept, so there are no holes.
- Only finished months are kept: on 18 November the last month is October, never the current month.
- `data/coverage.csv`: which source is used for each asset and period.
- `data/raw/`: raw downloads. If a website is down, the saved copy is used automatically.

## Universe and sources

For each asset, the first source has priority. The next sources only fill the months it is missing: before it starts, after it ends, or a month removed by the cleaning.

| Column | Sources, in order of priority |
|---|---|
| `EQ_WORLD` | URTH, iShares MSCI World (USD, from 2012) → Ken French developed markets, Mkt-RF + RF (USD) |
| `BANKS_EU` (R_S) | EXV1.DE, iShares STOXX Europe 600 Banks (EUR; Yahoo history from 2008) → equal-weight basket of BNP, Santander, Deutsche Bank, ING, BBVA, Société Générale, Intesa (EUR) → Ken French US banks (last resort) |
| `EQ_CH` | EWL, iShares MSCI Switzerland (USD, from 1996) |
| `BOND_CHF` (R_B) | CSBGC0.SW, iShares Swiss Domestic Government Bond 7-15 (CHF, month-end prices) → before the ETF exists: synthetic index, duration 9, from the SNB 10y Confederation yield (monthly cube `rendoblim`) |
| `BOND_GLB_H` | synthetic US 10y Treasury index (FRED DGS10, daily, month-end value), duration 8, hedged to CHF |
| `REIT_CH` | equal-weight basket of PSPN.SW, SPSN.SW, ALLN.SW (CHF) |
| `GOLD` | GLD (USD, from 2004) → GC=F gold futures (from 2000) |
| `EMPLOYER` | UBSG.SW (CHF, from 1999) → UBS on NYSE (fallback only) |

Rates and FX:
- `RF_CHF`: SARON from the SNB data portal (cube `zimoma`).
- USDCHF and EURUSD: FRED daily series DEXSZUS and DEXUSEU, month-end value.

## Formulas

- Unhedged CHF return: (1 + r_CHF) = (1 + r_local)(1 + change in CHF per foreign unit).
- Synthetic bond return: r_t = y_{t−1}/12 − D × (y_t − y_{t−1}).
- Hedged return: r_local + (CHF rate − foreign rate)/12, both rates at the start of the month. From 2015 to 2022 this carry is negative.
- Risk-free: RF_CHF_t = SARON at the end of month t−1, divided by 12.

## Cleaning

- A single stock or ETF with |monthly return| above 60% is treated as a data error and set to NaN, never 0.
  - In a basket, the other tickers are used for that month.
  - For a single-ticker asset, the next source fills the month.
  - Example: PSPN.SW in May 2004 (+133% on Yahoo, a wrong adjustment) previously produced +45% for Swiss real estate.
- Asset months above 40% are printed for a manual check. They can be real, for example UBS and the banks in April 2009.

## Open points for the group

1. **World equities include financials.** An ex-financials world index with history from 2000 is not freely available. Banks are about 10% of MSCI World, so the hedge appears as "zero in BANKS_EU".
2. **Global bonds = US Treasuries hedged.** Global aggregate ETFs only start around 2008.
3. **Bond duration**: 9 for the CHF proxy (close to the 7-15 ETF), 8 for US Treasuries (`duration` in params.yaml).
4. **Swiss real estate = three listed property companies** instead of the SXI real estate funds index, which has no free long history.
5. **No monthly averages after the ETF starts**: BOND_CHF uses real month-end ETF prices. The SNB monthly yield, used only before the ETF exists, should still be checked (month-end value or monthly average).
