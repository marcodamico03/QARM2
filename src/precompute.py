"""Compute the backtests of the app once, before showing it (step 5).

The app computes a backtest the first time a client is shown (a few seconds) and keeps it in
data/cache/. This script does it in advance for every persona with its default settings, so the
demo is instant from the first click.

Run from the project folder:
    python -m src.precompute                 backtest and prior check of every persona (about 1 minute)
    python -m src.precompute --robustness    also the robustness tests (about 5 minutes in all)
"""
import sys
import time

from src import backtest as B
from src import dashboard as D
from src.config import load_params


def main(robustness=False):
    base = load_params()
    data = B.load_backtest_data()
    names = ["backtest", "prior"] + (["robustness"] if robustness else [])
    for persona in base["personas"]:
        params = D.client_params(persona, None, base)
        for name in names:
            start = time.time()
            was_saved = D.saved(name, params) is not None
            D.remember(name, params, data)
            print(f"{persona:18s} {name:11s} {'already saved' if was_saved else 'computed'}"
                  f"  {time.time() - start:5.1f} s")
    print(f"Saved in {D.CACHE_DIR}")


if __name__ == "__main__":
    main(robustness="--robustness" in sys.argv)
