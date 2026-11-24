# Beyond the Portfolio

**A life-cycle, total-wealth optimizer: from hedging your career to choosing your pension**
QARM II group project, HEC Lausanne, 2026

Standard portfolio tools optimize only the brokerage account. For most people the biggest assets
are their **career** and their **pension**, which they cannot trade. We build the portfolio that
complements them, for a 35-year-old Zurich banker, and test it in a walk-forward backtest (2006–2026, in CHF).

## Project structure

```
data/
  raw/              downloaded data (FRED, Ken French, Yahoo Finance)
  processed/        clean monthly CHF returns used by everything else
src/
  config.py         ALL shared assumptions (tickers, dates, BVG values, constraints...)
  download_yahoo.py downloads Yahoo prices into data/raw/
  build_data.py     builds data/processed/ from data/raw/
  data_loader.py    load_returns(), load_excess_returns(), load_rates()
  personas.py       client presets (Zurich banker, civil servant, retiree)
  career.py         human capital H, pension P, balance sheet, career risk sigma_H
  estimation.py     Ledoit-Wolf covariance, Black-Litterman expected returns
  optimizer.py      total-wealth optimizer, strategies A/B/C/D, gamma calibration, shadow prices
  risk.py           Euler risk contributions on total wealth
  backtest.py       walk-forward backtest, crisis table, wrong-beta test, job-loss variant
tests/              automatic checks (run: python -m pytest -q)
notebooks/          01 data check · 02 career model · 03 optimal portfolio · 04 backtest
app/                Streamlit web app (runs locally, see below)
```

## Setup (once per computer)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Everyday commands

```bash
python -m pytest -q                # should print "20 passed"

cd app
streamlit run Home.py              # opens the app at http://localhost:8501 (this computer only)
```

Stop the app with **Control + C** in the terminal.

The processed data is already in the repository, so you do **not** need to download anything.
To refresh the data: `python src/download_yahoo.py` then `python src/build_data.py`.

## Status

| Layer | Content | Status |
|---|---|---|
| Core | data, career model, optimizer, shadow prices, risk sources, backtest, app pages 1–7 | done |
| Layer 2 | glide path page, stress lab, Michaud resampling bands | to do |
| Layer 3 | retirement planner (annuity vs lump sum, BFS mortality tables) | to do |
| Extras | PDF report, more personas, taxes | optional |

## Data sources

FRED (Swiss and US yields, short rates, exchange rates), Ken French data library (developed markets,
49 industries), Yahoo Finance (EXV1.DE, UBSG.SW, ^SSMI, PSPN.SW, GC=F), BSV (BVG 2026 parameters),
Doeswijk, Lam & Swinkels (global market portfolio). Every assumption is listed on page 7 of the app.
