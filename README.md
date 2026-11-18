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

| Path | What it contains | Owner |
|---|---|---|
| `app.py` | Streamlit home page | Member 5 |
| `pages/` | One file per app page | Member 5 |
| `config/params.yaml` | Shared parameters used by every module | Everyone |
| `src/data.py` | Downloads, cleaning, CHF conversion | Member 1 |
| `src/career.py` | Career model and human capital | Member 2 |
| `src/optimizer.py` | Black-Litterman, constrained solver, shadow prices | Member 3 |
| `src/backtest.py` | Walk-forward backtest and metrics | Member 4 |
| `data/` | Saved data, so the app works if a website is down | Member 1 |
| `notebooks/` | Tests and experiments | Everyone |
| `docs/` | Concept note and documentation | Everyone |

## How we work with Git

1. `main` always contains a version that works.
2. Each member works on their own branch.
3. Work is added to `main` through a pull request, checked by another member.
4. All modules read their settings from `config/params.yaml`, never from numbers
   typed inside the code.
