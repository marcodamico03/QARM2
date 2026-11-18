"""Member 1: data layer.

Downloads prices and rates, cleans them, converts all returns to CHF,
and saves copies in data/ so the app keeps working if a website is down.

Main entry point for the other modules:

    from src.data import load_returns
    returns = load_returns()          # monthly CHF returns, one column per asset + "rf"

Run this file directly to download everything again and print a coverage report:

    python -m src.data
"""
import io
import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import requests
import yaml

# Folder of the project (one level above src/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{file}_CSV.zip"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SNB_URL = "https://data.snb.ch/api/cube/{cube}/data/csv/en"


# =============================================================================
# 1. Parameters and small helpers
# =============================================================================

def load_params(path=None):
    """Read the shared parameter file config/params.yaml."""
    if path is None:
        path = os.path.join(ROOT, "config", "params.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _data_folder(params):
    folder = os.path.join(ROOT, params["data"].get("folder", "data"))
    os.makedirs(os.path.join(folder, "raw"), exist_ok=True)
    return folder


def _as_series(data):
    """Saved copies are read back as one-column tables: turn them into a Series."""
    return data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data


def _month_end(index):
    """Move any date to the last day of its month, so all series share one index."""
    return pd.DatetimeIndex(index).to_period("M").to_timestamp(how="end").normalize()


def _to_monthly_last(series):
    """Daily (or irregular) series -> last available value of each month."""
    series = series.dropna()
    monthly = series.groupby(series.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    return monthly


def _cached(params, name, download_function):
    """Download a raw series and save it in data/raw/.
    If the download fails (website down, no internet), use the saved copy."""
    path = os.path.join(_data_folder(params), "raw", f"{name}.csv")
    try:
        data = download_function()
        data.to_csv(path)
        return data
    except Exception as error:
        if os.path.exists(path):
            warnings.warn(f"Download of {name} failed ({error}). Using the saved copy {path}.")
            return pd.read_csv(path, index_col=0, parse_dates=True)   # always a DataFrame
        raise RuntimeError(f"Download of {name} failed and no saved copy exists: {error}")


# =============================================================================
# 2. Downloaders (one per website). Each returns a pandas Series/DataFrame
#    with a month-end DatetimeIndex.
# =============================================================================

def download_yahoo(tickers, start):
    """Month-end adjusted close prices (dividends included) from Yahoo Finance."""
    import yfinance as yf
    start = (pd.Timestamp(start) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")  # one extra month for the first return
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close[[t for t in tickers if t in close.columns]]
    if close.dropna(how="all").empty:
        raise ValueError(f"Yahoo returned no data for {tickers}")
    monthly = close.apply(_to_monthly_last)
    return monthly


def download_fred(series_id):
    """One series from FRED (St. Louis Fed) as a float Series, original frequency."""
    data = pd.read_csv(FRED_URL.format(series=series_id), na_values=".")
    # first column is the date (called "DATE" or "observation_date" depending on the FRED version)
    series = pd.Series(data[series_id].astype(float).values,
                       index=pd.DatetimeIndex(pd.to_datetime(data.iloc[:, 0])), name=series_id)
    return series.dropna()


def _parse_kf_csv(text):
    """Read the FIRST table of a Kenneth French CSV file (the monthly one).
    The files have a few lines of text, then a header line starting with ',',
    then rows starting with a date YYYYMM, then a blank line before the next table."""
    lines = text.splitlines()
    header, rows, started = None, [], False
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if not started:
            if line.startswith(",") and len(cells) > 1:
                header = cells
                started = True
            continue
        if len(cells[0]) == 6 and cells[0].isdigit():
            rows.append(cells)
        else:
            break                       # end of the monthly table
    if header is None or not rows:
        raise ValueError("Could not find the monthly table in the Kenneth French file")
    table = pd.DataFrame(rows, columns=["date"] + header[1:]).set_index("date")
    table = table.apply(pd.to_numeric, errors="coerce")
    table = table.replace([-99.99, -999], np.nan) / 100.0   # percent -> decimal; -99.99 = missing
    table.index = _month_end(pd.to_datetime(table.index, format="%Y%m"))
    return table


def download_kenfrench(file):
    """Monthly table of a Kenneth French data library file, in decimals."""
    response = requests.get(KF_URL.format(file=file), timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    name = [n for n in archive.namelist() if n.lower().endswith(".csv")][0]
    text = archive.read(name).decode("latin-1")
    return _parse_kf_csv(text)


def _parse_snb_csv(text, code):
    """SNB data portal CSV: a few metadata lines, then 'Date;D0;...;Value'.
    Keep the rows whose dimension equals `code` (e.g. 'SARON' or '10J')."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.replace('"', "").startswith("Date;"))
    table = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=";")
    dims = [c for c in table.columns if c not in ("Date", "Value")]
    mask = pd.Series(False, index=table.index)
    for d in dims:
        mask |= table[d].astype(str) == code
    if not mask.any():
        available = {d: sorted(table[d].astype(str).unique().tolist()) for d in dims}
        raise ValueError(f"Code '{code}' not found. Available codes: {available}")
    table = table[mask]
    series = pd.to_numeric(table["Value"], errors="coerce")
    series.index = _month_end(pd.to_datetime(table["Date"]))
    return series.dropna()


def download_snb(cube, code):
    """Monthly series from the Swiss National Bank data portal (annual %)."""
    response = requests.get(SNB_URL.format(cube=cube), timeout=60)
    response.raise_for_status()
    series = _parse_snb_csv(response.text, code)
    series.name = f"{cube}_{code}"
    return series


# =============================================================================
# 3. Rates, exchange rates and CHF conversion
# =============================================================================

def build_fx(params):
    """Month-end exchange rates: CHF per 1 unit of foreign currency."""
    fx_ids = params["data"]["fx"]
    usdchf = _to_monthly_last(_as_series(_cached(params, "fred_" + fx_ids["USDCHF"], lambda: download_fred(fx_ids["USDCHF"]))))
    eurusd = _to_monthly_last(_as_series(_cached(params, "fred_" + fx_ids["EURUSD"], lambda: download_fred(fx_ids["EURUSD"]))))
    fx = pd.DataFrame({"USD": usdchf, "EUR": usdchf * eurusd, "CHF": 1.0})
    return fx


def build_chf_rf(params):
    """CHF risk-free rate as a monthly return.
    The rate observed at the end of month t-1 is the return earned during month t.
    Negative rates (2015-2022) are kept as they are."""
    spec = params["data"]["rates"]["chf_rf"]
    level = _cached(params, f"snb_{spec['cube']}_{spec['code']}", lambda: download_snb(spec["cube"], spec["code"]))
    level = _to_monthly_last(_as_series(level))  # annual %, month-end
    rf = (level / 100.0 / 12.0).shift(1)
    rf.name = "rf"
    return rf, level


def to_chf(local_returns, currency, fx):
    """Unhedged conversion: (1 + r_CHF) = (1 + r_local) x (1 + change of the exchange rate)."""
    if currency == "CHF":
        return local_returns
    fx_return = fx[currency].pct_change()
    return (1 + local_returns) * (1 + fx_return) - 1


def synthetic_bond_returns(yield_level_pct, duration):
    """Monthly return of a constant-duration bond index built from a yield:
    carry (last month's yield / 12) minus duration x change in yield."""
    y = yield_level_pct / 100.0
    return y.shift(1) / 12.0 - duration * y.diff()


# =============================================================================
# 4. One source -> monthly CHF returns
# =============================================================================

def source_returns(params, source, fx, chf_rf_level):
    """Monthly returns in CHF of one source described in params.yaml."""
    kind = source["type"]
    start = params["data"]["start_date"]

    if kind == "yahoo":
        tickers = source["tickers"]
        prices = _cached(params, "yahoo_" + "_".join(tickers), lambda: download_yahoo(tickers, start))
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        local = prices.pct_change(fill_method=None).mean(axis=1, skipna=True)   # equal weight if several tickers
        local[prices.notna().sum(axis=1) == 0] = np.nan
        local = local.where(prices.shift(1).notna().any(axis=1))                # no return without a previous price
        return to_chf(local, source["currency"], fx)

    if kind == "kf_market":
        table = _cached(params, "kf_" + source["file"], lambda: download_kenfrench(source["file"]))
        local = table["Mkt-RF"] + table["RF"]      # market return = excess return + US T-bill
        return to_chf(local, source["currency"], fx)

    if kind == "kf_industry":
        table = _cached(params, "kf_" + source["file"], lambda: download_kenfrench(source["file"]))
        local = table[source["column"]]
        return to_chf(local, source["currency"], fx)

    if kind == "snb_yield":
        level = _cached(params, f"snb_{source['cube']}_{source['code']}",
                        lambda: download_snb(source["cube"], source["code"]))
        return synthetic_bond_returns(_to_monthly_last(_as_series(level)), source["duration"])

    if kind == "fred_yield":
        level = _to_monthly_last(_as_series(_cached(params, "fred_" + source["series"],
                                                    lambda: download_fred(source["series"]))))
        local = synthetic_bond_returns(level, source["duration"])
        if source.get("hedged", False):
            # Currency hedge with a 1-month forward: local return + (CHF rate - foreign rate) / 12,
            # both rates known at the start of the month. Negative when CHF rates are lower.
            foreign = _to_monthly_last(_as_series(_cached(params, "fred_" + source["foreign_rf"],
                                                          lambda: download_fred(source["foreign_rf"]))))
            carry = (chf_rf_level.reindex(local.index) - foreign.reindex(local.index)) / 100.0 / 12.0
            return local + carry.shift(1)
        return to_chf(local, source["currency"], fx)

    raise ValueError(f"Unknown source type: {kind}")


def _describe(source):
    if source["type"] == "yahoo":
        return "Yahoo " + "+".join(source["tickers"])
    if source["type"] in ("kf_market", "kf_industry"):
        return "Ken French " + source["file"] + (f" [{source['column']}]" if "column" in source else "")
    if source["type"] == "snb_yield":
        return f"SNB {source['cube']} {source['code']} (synthetic, D={source['duration']})"
    if source["type"] == "fred_yield":
        return f"FRED {source['series']} (synthetic, D={source['duration']}" + (", hedged)" if source.get("hedged") else ")")
    return source["type"]


def splice(series_list):
    """First series wherever it exists; earlier months filled from the next series, and so on.
    Returns the spliced series and, for each month, which source was used (0, 1, ...)."""
    index = series_list[0].index
    for s in series_list[1:]:
        index = index.union(s.index)
    combined = pd.Series(np.nan, index=index)
    used = pd.Series(np.nan, index=index)
    for i, s in enumerate(series_list):
        s = s.reindex(index)
        fill = s.notna()
        if combined.notna().any():
            fill &= index < combined.first_valid_index()   # proxies only BEFORE the better series starts
        combined[fill] = s[fill]
        used[fill] = i
    return combined, used


# =============================================================================
# 5. Build, clean, save, load
# =============================================================================

def build_returns(params=None, verbose=True):
    """Download everything and build the table of monthly CHF returns.
    Returns (returns, coverage): returns has one column per asset plus 'rf'."""
    if params is None:
        params = load_params()
    fx = build_fx(params)
    rf, chf_rf_level = build_chf_rf(params)

    columns, coverage = {}, []
    for asset, spec in params["data"]["assets"].items():
        pieces = []
        for source in spec["sources"]:
            try:
                pieces.append(source_returns(params, source, fx, chf_rf_level))
            except Exception as error:
                warnings.warn(f"{asset}: source {_describe(source)} failed: {error}")
                pieces.append(pd.Series(dtype=float))
        if all(p.dropna().empty for p in pieces):
            warnings.warn(f"{asset}: no data at all, asset skipped")
            continue
        pieces = [p if not p.dropna().empty else pd.Series(np.nan, index=fx.index) for p in pieces]
        series, used = splice(pieces)
        columns[asset] = series
        for i, source in enumerate(spec["sources"]):
            months = used[used == i].index
            if len(months):
                coverage.append({"asset": asset, "source": _describe(source),
                                 "from": months.min().date(), "to": months.max().date(), "months": len(months)})

    returns = pd.DataFrame(columns)
    returns["rf"] = rf

    # --- Keep only the period where every column has data -------------------
    start = pd.Timestamp(params["data"]["start_date"]) + pd.offsets.MonthEnd(0)
    first_full = returns.dropna().index.min()
    last_full = returns.dropna().index.max()
    returns = returns.loc[max(start, first_full):last_full]

    # --- Cleaning checks ------------------------------------------------------
    limit = params["data"].get("max_abs_monthly_return", 0.6)
    extreme = returns.drop(columns="rf").abs() > limit
    if extreme.any().any():
        for asset in extreme.columns[extreme.any()]:
            dates = [d.date() for d in returns.index[extreme[asset]]]
            warnings.warn(f"{asset}: monthly return above {limit:.0%} on {dates}. Check the source.")
    holes = returns.isna().sum()
    if holes.any():
        warnings.warn(f"Missing months inside the sample: {holes[holes > 0].to_dict()}")

    coverage = pd.DataFrame(coverage)
    if verbose:
        print_report(returns, coverage)
    return returns, coverage


def print_report(returns, coverage):
    print("\nSources used for each asset (proxies only before the investable series starts):")
    print(coverage.to_string(index=False))
    print(f"\nFinal sample: {returns.index.min().date()} to {returns.index.max().date()}, "
          f"{len(returns)} months, {returns.shape[1] - 1} assets + rf")
    stats = pd.DataFrame({
        "mean (ann.)": returns.mean() * 12,
        "vol (ann.)": returns.std() * np.sqrt(12),
        "min month": returns.min(),
        "max month": returns.max(),
    })
    print("\nSummary statistics (CHF, annualised):")
    print(stats.round(4).to_string())
    print("\nMonths with a negative CHF risk-free rate:", int((returns["rf"] < 0).sum()))


def save_returns(returns, coverage, params=None):
    if params is None:
        params = load_params()
    folder = _data_folder(params)
    returns.to_csv(os.path.join(folder, "returns_chf.csv"))
    coverage.to_csv(os.path.join(folder, "coverage.csv"), index=False)


def load_returns(refresh=False, params=None):
    """Monthly CHF returns used by every other module.
    By default reads the saved file data/returns_chf.csv (fast, works offline).
    refresh=True downloads everything again and overwrites the saved file."""
    if params is None:
        params = load_params()
    path = os.path.join(_data_folder(params), "returns_chf.csv")
    if refresh or not os.path.exists(path):
        returns, coverage = build_returns(params)
        save_returns(returns, coverage, params)
        return returns
    return pd.read_csv(path, index_col=0, parse_dates=True)


def excess_returns(returns):
    """Asset returns minus the CHF risk-free rate (what the optimizer uses)."""
    return returns.drop(columns="rf").sub(returns["rf"], axis=0)


if __name__ == "__main__":
    returns = load_returns(refresh=True)
    print(f"\nSaved to {os.path.join(_data_folder(load_params()), 'returns_chf.csv')}")
