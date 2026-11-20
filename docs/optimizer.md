# Optimizer (step 3)

Code: `src/optimizer.py`. Settings: `config/params.yaml`, section `optimization`. Tests: `notebooks/test_optimizer.py`.

```bash
python notebooks/test_optimizer.py   # tests with known answers (labs LN4, LN5 and closed formulas)
```

Notation as in the labs (contract, section 1): `gamma` is risk **tolerance**, `QP(x) = 0.5 x'Sigma x - gamma x'mu`.
Everything is in excess returns over the CHF risk-free rate.

## The five pieces

| Piece | Functions | Checked against |
|---|---|---|
| 1. Black-Litterman (lab LN4) | `implied_risk_aversion`, `implied_excess_returns`, `bl_expected_returns`, `tracking_error`, `tau_for_tracking_error` | Lab LN4 expected outputs, and the exact closed-form optimum |
| 2. Covariance | `estimate_covariance` (sample or Ledoit-Wolf), `shrinkage_intensity`, `correlation_from_covariance` | pandas `.cov()`, volatilities unchanged by shrinkage |
| 3. Neutral portfolio and gamma | `neutral_weights`, `market_prior` | A client with no career holds the neutral portfolio |
| 4. Total-wealth optimization | `omega_share`, `formula_portfolio`, `constraints_from_params`, `optimal_portfolio` | The contract formula x* (solver finds it from equal weights) |
| 5. Risk and constraint costs | `risk_contributions`, `total_wealth_risk`, `certainty_equivalent`, `constraint_costs` | Lab LN5 (`vol_ew = 0.19867`), Euler contributions add up |

## Choices to discuss

1. **Neutral portfolio = Pictet LPP-40** (standard Swiss pension benchmark) mapped onto our assets: CHF bonds 33.3%, hedged global bonds 16.7%, Swiss equities 16.7%, world equities 25.0%, banks 2.8%, Swiss real estate 5.6%, gold 0%. Emerging-market bonds and hedge funds (10% of LPP-40) have no matching asset and are removed. It must hold CHF bonds and banks, otherwise the career hedge ("hold less of them") is invisible against a zero weight.
2. **gamma = 1 / phi**: a client with no career and no pension holds the neutral portfolio (replaces "60/40", which the optimizer cannot reach with a market prior and no borrowing). phi = Sharpe / volatility of the neutral portfolio, with **Sharpe 0.408** (Doeswijk and Swinkels, 2024, global market portfolio 1970 to 2022). On our data: phi = 6.04, gamma = 0.166.
3. **Ledoit-Wolf on correlations only**: volatilities are kept. Applied to raw returns, scikit-learn would raise CHF bond volatility from 5.4% to 6.1% and distort the career hedge. Shrinkage on the full sample is light (4.8%); it matters more in the 60-month backtest windows.
4. **Position caps**: banks 10%, Swiss equities 30%, real estate 20%, gold 10%, UBS 0% (only through deferred shares), any added sector 10%. Without caps the young banker (career = 97% of wealth) would hold 100% in the asset with the highest expected return.
5. **Precise solver**: scipy's default precision stops too early. In lab LN4 it gives a tracking error of 0.01321 where the exact optimum is 0.01532 (tau 0.0961 vs 0.0823). `precise=True` is our default; `precise=False` reproduces the lab.

## What it gives for the banker (age 35, salary 130k, portfolio 100k)

| | Result |
|---|---|
| omega (career share of wealth) | 0.97 |
| Without the hedge (strategy D) | world equities 90%, banks 10% |
| With the hedge (strategy C) | world equities 100%, banks 0% |
| Share of total risk from the career | 95% |
| Cost of "no short selling" | 0.44% of total wealth a year (about CHF 14,900): allowed to short, he would sell CHF bonds and banks to hedge his career |
| Cost of "no borrowing" | 0.27% a year (about CHF 9,100) |
| Cost of the caps | 0 (none binds) |

Constraint costs are certainty-equivalent returns of total wealth: the problem is solved again without
that one constraint and the gain is measured. They are the "shadow prices" of the proposal in a form a
client understands.
