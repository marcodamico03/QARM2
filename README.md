# Beyond the Portfolio

QARM II group project, HEC Lausanne (autumn 2026).

A life-cycle, total-wealth portfolio optimizer for a Swiss client. The financial
portfolio is built around the client's non-tradable assets: the career (human
capital) while working, and the 2nd-pillar pension at retirement. The strategy is
tested with a walk-forward backtest and delivered as a Streamlit app.

The full design is in `docs/` (concept note).

## How to run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

| Path | What it contains |
|---|---|
| `app.py` | The Streamlit app, one page (see `docs/app.md`) |
| `src/allocation.py` | Version 2 of the advice: risk budgeting around the client's job (see `docs/app.md`) |
| `src/dashboard.py`, `src/theme.py` | Saved results and look of the app |
| `src/jobs.py` | Backtests run in the background, so the app never freezes |
| `src/precompute.py` | Optional: the backtests of every persona computed in advance |
| `pages/` | Not used: the app is one page |
| `config/params.yaml` | Shared parameters used by every module |
| `src/data.py` | Downloads, cleaning, CHF conversion |
| `src/career.py` | Career model and human capital |
| `src/optimizer.py` | Black-Litterman, constrained solver, shadow prices |
| `src/backtest.py` | Walk-forward backtest and metrics |
| `data/` | Saved data, so the app works if a website is down |
| `notebooks/` | Tests and experiments |
| `docs/` | Concept note and documentation |

## How we work with Git

1. `main` always contains a version that works.
2. Everyone works on everything, each on their own branch. Pull the latest `main`
   before starting and tell the group which file you are changing.
3. Work is added to `main` through a small pull request, checked by someone else.
4. All modules read their settings from `config/params.yaml`, never from numbers
   typed inside the code.
