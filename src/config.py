"""
Shared settings for the whole project.
Every script imports from here, so all parts use the same assumptions.
"""
from pathlib import Path

# ---------- Paths (work wherever the project folder is) ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ---------- Sample period ----------
START = "1999-01-01"          # first estimate window (60m) ready ~2004, so 2008 is out-of-sample

# ---------- Yahoo Finance series: name -> (ticker, currency) ----------
YAHOO_TICKERS = {
    "eu_banks":      ("EXV1.DE", "EUR"),   # iShares STOXX Europe 600 Banks  -> banker's sector R_S
    "ubs":           ("UBSG.SW", "CHF"),   # employer stock
    "smi":           ("^SSMI",   "CHF"),   # Swiss equities (price index, dividends added below)
    "swiss_re":      ("PSPN.SW", "CHF"),   # Swiss real estate proxy (Swiss Prime Site)
    "gold":          ("GC=F",    "USD"),   # gold futures
    # "credit_suisse": ("CSGN.SW", "CHF"), # delisted 2023, not on Yahoo -> case study dropped
}

# ---------- Data cleaning ----------
MAX_ABS_MONTHLY_RETURN = 0.50  # |monthly return| above this is treated as a data error (set to NaN)
# Ken French US "Banks" industry (in CHF) fills EU banks before EXV1.DE exists on Yahoo (2008)
FRENCH_INDUSTRY_FILE = "49_Industry_Portfolios.csv"
FRENCH_BANKS_COLUMN = "Banks"

# ---------- Data construction assumptions ----------
SMI_DIVIDEND_YIELD = 0.03     # ^SSMI is price-only: add ~3% p.a. dividend yield
DURATION_CHF_BONDS = 7.0      # modified duration used to turn CHF 10y yields into returns
DURATION_GLOBAL_BONDS = 7.0   # same for US 10y yields (GS10)

# ---------- Career model ----------
CAREER_BOND_ASSET = "chf_bonds"     # R_B in R_H = beta_B * R_B + beta_S * R_S
CAREER_SECTOR_ASSET = "eu_banks"    # R_S for the banker
RF_LONG_RUN = 0.01                  # long-run CHF risk-free rate (annual) used to value H
BOND_PREMIUM = 0.015                # long-run CHF bond excess return (annual), rounded from our data
SECTOR_PREMIUM = 0.04               # long-run bank-sector excess return (annual), rounded from our data
PORTFOLIO_GROWTH = 0.04             # assumed portfolio return, only for the balance-sheet projection

# ---------- Swiss 2nd pillar (BVG), legal values 2026 (BSV) ----------
BVG_COORDINATION_DEDUCTION = 26_460
BVG_MIN_INTEREST = 0.0125
BVG_CONVERSION_RATE = 0.068
BVG_CREDIT_RATES = {25: 0.07, 35: 0.10, 45: 0.15, 55: 0.18}  # savings credits by age band

# ---------- Investable universe (proposal 3.2) ----------
INVESTABLE = ["world_eq", "smi", "eu_banks", "ubs",
              "chf_bonds", "global_bonds_hedged", "swiss_re", "gold"]
EQUITY_ASSETS = ["world_eq", "smi", "eu_banks", "ubs"]   # for "60/40" and "100 - age"
BOND_ASSETS = ["chf_bonds", "global_bonds_hedged"]

# ---------- Black-Litterman prior ----------
# Global market portfolio (Doeswijk, Lam & Swinkels) mapped onto our assets.
# APPROXIMATE shares: roughly 40% equities, 54% bonds, 4% real estate, 1% commodities,
# split by market cap inside each class. TODO: check against the table in the paper.
MARKET_WEIGHTS = {
    "world_eq": 0.375, "smi": 0.012, "eu_banks": 0.008, "ubs": 0.001,
    "chf_bonds": 0.010, "global_bonds_hedged": 0.534,
    "swiss_re": 0.045, "gold": 0.015,
}
MARKET_EXCESS_RETURN = 0.034   # DLS (2020): market portfolio excess return ~3.39% p.a., 1960-2017
TAU = 0.05                     # uncertainty of the prior (standard He-Litterman value)

# ---------- Client constraints (same for every strategy) ----------
POSITION_CAPS = {"ubs": 0.20, "eu_banks": 0.20, "swiss_re": 0.25, "gold": 0.15}
TARGET_EQUITY_SHARE = 0.60     # gamma is set so that a client with no career holds 60/40

# ---------- Backtest (proposal 4.1) ----------
BACKTEST_WINDOW = 60           # months of past data used at each re-optimisation
TRANSACTION_COST_BP = 10       # cost per unit of turnover
JOB_LOSS_THRESHOLD = 0.30      # sector drawdown from its 24-month high that triggers a job loss (varied 20-40%)
JOB_LOSS_MONTHS = 12           # months without salary after a job loss
