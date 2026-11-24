"""
One place to read the processed data. Every other module uses these functions.
Index: pandas PeriodIndex at monthly frequency ('1999-01', ...).
"""
from functools import lru_cache

import pandas as pd

from config import DATA_PROCESSED


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / name, index_col="date")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


@lru_cache(maxsize=None)
def load_returns() -> pd.DataFrame:
    """Total monthly returns in CHF (decimals)."""
    return _read("returns_chf.csv")


@lru_cache(maxsize=None)
def load_excess_returns() -> pd.DataFrame:
    """Monthly returns in CHF minus the CHF risk-free rate."""
    return _read("excess_returns_chf.csv")


@lru_cache(maxsize=None)
def load_rates() -> pd.DataFrame:
    """Monthly risk-free rates (rf_chf, rf_usd, rf_eur) in decimals per month."""
    return _read("rates.csv")
