"""
Download monthly prices from Yahoo Finance into data/raw/yahoo_<name>.csv

Run from the project folder:
    python src/download_yahoo.py
Needs:  pip install yfinance
"""
import pandas as pd
import yfinance as yf

from config import DATA_RAW, START, YAHOO_TICKERS


def download_one(ticker: str) -> pd.Series:
    # Daily data, then the last price of each month. Yahoo's own monthly bars
    # have missing months for some tickers (e.g. gold futures).
    df = yf.download(ticker, start=START, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index)
    close = close.groupby(close.index.to_period("M")).last()
    close.index = close.index.to_timestamp()     # first of month, like FRED
    return close


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"{'name':15s} {'ticker':10s} {'first':>8s} {'last':>8s} {'months':>6s}")
    for name, (ticker, ccy) in YAHOO_TICKERS.items():
        s = download_one(ticker)
        if s.empty:
            print(f"{name:15s} {ticker:10s}  NO DATA -> drop it or find another ticker")
            continue
        out = s.rename("close").to_frame()
        out.index.name = "date"
        out.to_csv(DATA_RAW / f"yahoo_{name}.csv")
        print(f"{name:15s} {ticker:10s} {s.index[0]:%Y-%m} {s.index[-1]:%Y-%m} {len(s):6d}")


if __name__ == "__main__":
    main()
