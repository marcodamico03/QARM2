"""
Turn everything in data/raw/ into clean monthly CHF return files in data/processed/.

Run from the project folder (after download_yahoo.py):
    python src/build_data.py

Outputs
    data/processed/returns_chf.csv         total monthly returns in CHF, one column per asset
    data/processed/excess_returns_chf.csv  same minus the CHF risk-free rate
    data/processed/rates.csv               monthly risk-free rates (CHF, USD, EUR)
"""
import io
import warnings

import numpy as np
import pandas as pd

from config import (DATA_PROCESSED, DATA_RAW, DURATION_CHF_BONDS, DURATION_GLOBAL_BONDS,
                    FRENCH_BANKS_COLUMN, FRENCH_INDUSTRY_FILE, MAX_ABS_MONTHLY_RETURN,
                    SMI_DIVIDEND_YIELD, START, YAHOO_TICKERS)


# ---------------------------------------------------------------- readers
def read_fred(series_id: str) -> pd.Series:
    """FRED csv -> monthly Series indexed by Period('M'). Daily files are cut to month-end."""
    df = pd.read_csv(DATA_RAW / f"{series_id}.csv")
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]).dt.to_period("M"), name=series_id)
    return s.dropna().groupby(level=0).last()


def read_french(filename: str = "Developed_3_Factors.csv") -> pd.DataFrame:
    """Ken French csv -> monthly DataFrame in decimals (monthly block only, annual block dropped)."""
    lines = (DATA_RAW / filename).read_text().splitlines()
    first = next(i for i, l in enumerate(lines) if l.replace(" ", "").startswith(",Mkt-RF"))
    last = next(i for i, l in enumerate(lines) if "Annual" in l)
    df = pd.read_csv(io.StringIO("\n".join(lines[first:last])), skipinitialspace=True)
    df = df.rename(columns={df.columns[0]: "date"}).dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m").dt.to_period("M")
    df = df.set_index("date").astype(float).replace(-99.99, np.nan) / 100
    return df


def read_french_industry(filename: str, column: str) -> pd.Series | None:
    """Ken French industry file -> value-weighted monthly returns of one industry (decimals)."""
    path = DATA_RAW / filename
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines)
                 if "Average Value Weighted Returns -- Monthly" in l) + 1
    end = start + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m").dt.to_period("M")
    s = df.set_index("date")[column].astype(float).replace([-99.99, -999], np.nan) / 100
    return s


def read_yahoo(name: str) -> pd.Series | None:
    path = DATA_RAW / f"yahoo_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    s = pd.Series(df["close"].values, index=df["date"].dt.to_period("M"), name=name)
    return s.groupby(level=0).last()


# ---------------------------------------------------------------- helpers
def to_chf(r_local: pd.Series, fx_ret: pd.Series) -> pd.Series:
    """Unhedged: CHF return of a foreign asset."""
    return (1 + r_local) * (1 + fx_ret) - 1


def hedged_to_chf(r_local, rf_chf, rf_foreign):
    """Currency-hedged: local return plus the short-rate differential (hedge carry)."""
    return r_local + (rf_chf - rf_foreign)


def bond_return_from_yield(y_pct: pd.Series, duration: float) -> pd.Series:
    """Monthly bond return ~ carry - duration * change in yield."""
    y = y_pct / 100
    return y.shift(1) / 12 - duration * y.diff()


def annual_to_monthly_rf(rate_pct: pd.Series) -> pd.Series:
    """% p.a. quoted during month t-1 is the riskless return earned in month t."""
    return (rate_pct / 100 / 12).shift(1)


# ---------------------------------------------------------------- build
def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    notes = []

    # --- FX: prefer daily month-end (DEXSZUS/DEXUSEU), else FRED monthly averages
    if (DATA_RAW / "DEXSZUS.csv").exists() and (DATA_RAW / "DEXUSEU.csv").exists():
        chf_per_usd, usd_per_eur = read_fred("DEXSZUS"), read_fred("DEXUSEU")
        notes.append("FX: month-end daily rates (DEXSZUS, DEXUSEU)")
    else:
        chf_per_usd, usd_per_eur = read_fred("EXSZUS"), read_fred("EXUSEU")
        notes.append("FX: FRED monthly AVERAGES (EXSZUS, EXUSEU) - smooths FX moves, "
                     "download DEXSZUS + DEXUSEU for month-end rates")
    chf_per_eur = (chf_per_usd * usd_per_eur).dropna()
    fx_usd = chf_per_usd.pct_change()
    fx_eur = chf_per_eur.pct_change()

    # --- risk-free rates (monthly decimals)
    french = read_french()
    rf_usd = french["RF"]
    rf_chf = annual_to_monthly_rf(read_fred("IR3TIB01CHM156N"))
    eur_raw = read_fred("IR3TIB01EZM156N")
    rf_eur = annual_to_monthly_rf(eur_raw)
    # EUR series may stop before the others: carry the last value forward
    full = pd.period_range(rf_eur.index.min(), rf_chf.index.max(), freq="M")
    rf_eur = rf_eur.reindex(full).ffill()

    # --- returns, all in CHF
    ret = {}

    world_local = french["Mkt-RF"] + french["RF"]                     # USD total return
    ret["world_eq"] = to_chf(world_local, fx_usd)
    ret["world_eq_hedged"] = hedged_to_chf(world_local, rf_chf, rf_usd)

    ret["chf_bonds"] = bond_return_from_yield(read_fred("IRLTLT01CHM156N"), DURATION_CHF_BONDS)

    us_bonds_local = bond_return_from_yield(read_fred("GS10"), DURATION_GLOBAL_BONDS)
    ret["global_bonds"] = to_chf(us_bonds_local, fx_usd)
    ret["global_bonds_hedged"] = hedged_to_chf(us_bonds_local, rf_chf, rf_usd)

    fx_by_ccy = {"USD": fx_usd, "EUR": fx_eur}
    rf_by_ccy = {"USD": rf_usd, "EUR": rf_eur}
    for name, (ticker, ccy) in YAHOO_TICKERS.items():
        px = read_yahoo(name)
        if px is None:
            notes.append(f"{name}: no yahoo_{name}.csv yet (run download_yahoo.py)")
            continue
        r_local = px.pct_change()
        bad = r_local.abs() > MAX_ABS_MONTHLY_RETURN
        if bad.any():
            months = ", ".join(str(m) for m in r_local.index[bad])
            notes.append(f"{name}: removed impossible monthly return(s) in {months} (Yahoo data error)")
            r_local[bad] = np.nan
        if name == "smi":
            r_local = r_local + SMI_DIVIDEND_YIELD / 12
        if ccy == "CHF":
            ret[name] = r_local
        else:
            ret[name] = to_chf(r_local, fx_by_ccy[ccy])
            ret[f"{name}_hedged"] = hedged_to_chf(r_local, rf_chf, rf_by_ccy[ccy])

    # EU banks before EXV1.DE exists: US banks (Ken French) converted to CHF
    us_banks = read_french_industry(FRENCH_INDUSTRY_FILE, FRENCH_BANKS_COLUMN)
    if "eu_banks" in ret:
        first = ret["eu_banks"].dropna().index.min()
        if us_banks is None:
            notes.append(f"eu_banks starts {first}: download {FRENCH_INDUSTRY_FILE} "
                         "from Ken French to fill earlier years")
        else:
            early = us_banks[us_banks.index < first]
            ret["eu_banks"] = ret["eu_banks"].combine_first(to_chf(early, fx_usd))
            ret["eu_banks_hedged"] = ret["eu_banks_hedged"].combine_first(
                hedged_to_chf(early, rf_chf, rf_usd))
            notes.append(f"eu_banks before {first}: filled with Ken French US Banks "
                         "industry in CHF (state this on the 'What we assume' page)")

    returns = pd.DataFrame(ret).sort_index()
    returns = returns[returns.index >= pd.Period(START, "M")]

    # stop at the last month where the core series (world equities, CHF rate) both exist
    last = min(rf_chf.dropna().index.max(), french.index.max())
    returns = returns[returns.index <= last]

    # EUR rate series ends early: carry last value forward and say so
    last_eur = eur_raw.index.max()
    if last_eur < returns.index.max():
        notes.append(f"EUR 3m rate ends {last_eur}; carried forward after that "
                     "(only affects *_hedged EUR assets)")
    rates = pd.DataFrame({"rf_chf": rf_chf, "rf_usd": rf_usd, "rf_eur": rf_eur})
    rates = rates.reindex(returns.index)

    excess = returns.sub(rates["rf_chf"], axis=0)

    # --- save (index as YYYY-MM strings)
    for df, fname in [(returns, "returns_chf.csv"), (excess, "excess_returns_chf.csv"),
                      (rates, "rates.csv")]:
        out = df.copy()
        out.index = out.index.astype(str)
        out.index.name = "date"
        out.to_csv(DATA_PROCESSED / fname, float_format="%.6f")

    # --- report
    print("\nCoverage of data/processed/returns_chf.csv")
    print(f"{'column':22s} {'first':>8s} {'last':>8s} {'months':>6s} {'ann.mean':>9s} {'ann.vol':>8s}")
    for c in returns:
        s = returns[c].dropna()
        if s.empty:
            print(f"{c:22s}  empty"); continue
        print(f"{c:22s} {str(s.index[0]):>8s} {str(s.index[-1]):>8s} {len(s):6d} "
              f"{s.mean()*12:9.1%} {s.std()*np.sqrt(12):8.1%}")
    print("\nNotes:")
    for n in notes:
        print(" -", n)
    print(f"\nSaved to {DATA_PROCESSED}")


if __name__ == "__main__":
    warnings.simplefilter("ignore", FutureWarning)
    main()
