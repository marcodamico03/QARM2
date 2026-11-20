# Optimization engine (step 3): choices to discuss

- Code: `src/optimizer.py`.
- Settings: `config/params.yaml`, sections `client`, `pension`, `optimization` and `black_litterman`.
- Tests: `notebooks/test_optimizer.py`.

```bash
python -m src.optimizer             # full report for the banker, then one line per persona
python notebooks/test_optimizer.py  # tests with known answers (labs, closed forms, exact solutions)
```

## What it does

The optimizer chooses only the financial portfolio F. It judges F on total wealth W = H + P + F, where H is the career (step 2) and P is the 2nd-pillar pension. Every block reuses a lab, with the lab's notation:

| Block | Method | Lab |
|---|---|---|
| Covariance | Ledoit-Wolf shrinkage to the constant-correlation matrix | LN3 |
| Expected returns | Black-Litterman: prior from the global market portfolio, optional client views | LN4 |
| Portfolio | `QP(x) = 0.5 x'Σx − γ x'μ` on total wealth; `scipy.optimize.minimize` with `Bounds` and `LinearConstraint` | LN1 |
| γ | `fsolve` on a target volatility, the volatility of a 60/40 portfolio | LN1 |
| Shadow prices | KKT multipliers of the binding constraints, plus a re-solve without each constraint | LN1 |
| Risk sources | Euler risk contributions, applied to the whole balance sheet | LN5 |

## The problem (contract, section 1.2)

ω = (H + P) / W is the share of wealth that cannot be traded. x are the weights of the risky assets inside F; the rest, 1 − 1'x, is cash at the CHF risk-free rate.

```
R_W = (1 − ω) x'R + ω R_NT
minimise 0.5 Var(R_W) − γ E(R_W)
       = 0.5 (1−ω)² x'Σx + (1−ω) ω x'σ_H − γ (1−ω) x'μ + constant

no constraints:  x* = γ/(1−ω) Σ⁻¹μ  −  ω/(1−ω) Σ⁻¹σ_H
                     speculative        hedging
```

- With ω = 0 this is exactly the lab QP.
- The pension is credited at a guaranteed rate, so it has no market risk. It makes ω larger but adds no hedging demand: ω σ_H = ω_H Cov(R, R_career).
- Inside the code everything is monthly; only the tables are annual. γ and φ are the same at both frequencies.

## Inputs and their sources

### Covariance

- Ledoit and Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix", with the constant-correlation target of LN3, slides 9–10.
- The shrinkage constant is computed with the formula of the paper. On the full sample (314 months, 8 assets) it is only 0.03, because T is large compared with N.
- On 60-month windows (the backtest) it is between 0.08 and 0.30.
- `covariance: sample` in params switches shrinkage off.

### Expected returns (Black-Litterman)

- **Prior.** The prior is π = φ Σ w_mkt, with φ = SR / σ(w_mkt), the market's risk aversion (LN4, slides 6–7).
- **Market weights.** They come from Doeswijk and Swinkels (2024), "The Risk and Reward of Investing", Robeco working paper, Table 1: average weights of the global market portfolio, 1970–2022.
  - Equities 49.7% go to `EQ_WORLD`.
  - Government 28.1% plus non-government bonds 15.4% (43.5% in total) go to `BOND_GLB_H`.
  - Real estate 4.4% goes to `REIT_CH`.
  - Commodities 2.4% go to `GOLD`.
- **Assets with zero weight.** Swiss equities, banks and UBS are already inside the world index, so giving them weight would count them twice. They get weight 0, and their expected return comes from their covariance with the market, as in the CAPM.
- **Sharpe ratio.** SR = 0.408, the Sharpe ratio of the same market portfolio over 1970–2022 (same paper, Table 2). With our covariance this gives φ = 5.30.
- **Units.** The model runs in annual units, as in the lab, so τ and the uncertainty of the views keep their usual meaning. τ = 0.05, the lab's starting value.
- **Views.** Optional views go in params, for example `{assets: {EQ_CH: 1.0}, q: 0.05, sd: 0.10}`. A view on an asset that is not in this client's universe is skipped.

| Annual | Market weight | Black-Litterman (no views) | Historical mean | Volatility |
|---|---|---|---|---|
| EQ_WORLD | 49.7% | 5.8% | 5.3% | 15.6% |
| BANKS_EU | 0 | 6.8% | 4.9% | 27.7% |
| EQ_CH | 0 | 4.2% | 4.8% | 13.5% |
| BOND_CHF | 0 | 0.4% | 2.2% | 5.4% |
| BOND_GLB_H | 43.5% | 0.4% | 1.7% | 7.2% |
| REIT_CH | 4.4% | 2.1% | 7.4% | 11.9% |
| GOLD | 2.4% | 1.0% | 7.7% | 15.1% |
| EMPLOYER | 0 | 6.2% | 6.0% | 30.7% |

### γ (one risk tolerance for everybody)

- γ is the value at which a client with no career and no pension (ω = 0) holds an optimal portfolio as volatile as 60% `EQ_WORLD` / 40% `BOND_CHF` (9.4% per year). It is found with `fsolve`, as `target_vol` in lab LN1.
- The calibration uses the core assets without the employer stock and the standard constraints, without the lock-up. So every persona and every strategy (A), (C), (D) gets the same γ = 0.267.
- If fsolve fails, a bisection takes over. If no allowed portfolio is as volatile as the benchmark, the code raises a clear error instead of returning a wrong number.

### Balance sheet

- **H** is the career value from step 2.
- **P**, the pension:
  - `pension_chf: auto` estimates it with the legal BVG minimum: age credits 7/10/15/18% from age 25 (BVG art. 16); coordination deduction CHF 26,460 and upper limit CHF 90,720 (2025–2026 key figures of the BSV); minimum interest 1.25% (2024–2026, raised to 2.0% from 2027). Past salaries are today's salary scaled back by `salary_growth`.
  - A number in `pension_chf` (from the pension certificate) replaces the estimate.
  - Self-employed personas (`pillar2: false`) have P = 0.
- **Lock-up.** The lock-up is deferred employer shares divided by F. The banker's default is CHF 15,000, so 15% of the portfolio.

## Constraints

| Constraint | params | Default |
|---|---|---|
| Long-only, x ≥ 0 | `long_only` | yes |
| No leverage, 1'x ≤ 1 (the rest is cash) | `max_leverage` | 1.0 (`null` = no limit) |
| At most 10% in one sector or one stock | `position_caps` | `BANKS_EU`, `EMPLOYER` and every `SEC_*` |
| Lock-up: x_EMPLOYER ≥ deferred shares / F | `client.deferred_shares_chf` | 15,000 for the banker |

**Why the caps.** A 35-year-old has ω ≈ 0.97, so the speculative term is multiplied by about 30: on a portfolio that is 3% of his wealth he is almost risk-neutral. Without caps, the long-only, no-leverage optimum is 100% in the single asset with the highest expected return, which is European banks (6.8%). LN3 (slide 16) says portfolio managers use constraints for this. The cost of the caps is reported like every other constraint. If the lock-up is above the cap, the deferred shares are held but no share is bought.

## Solver: scipy, then the KKT conditions

1. **scipy.** `minimize` (SLSQP) with `Bounds` and `LinearConstraint`, as in the labs. The objective is divided by (1 − ω)² × the average variance, so that the solver works with numbers of order 1.
2. **KKT conditions.** On the constraints that scipy finds binding, stationarity plus the binding constraints form one linear system (LN1).
   - Its solution is the exact optimum, and its multipliers are the shadow prices.
   - If a constraint is violated, it is added. If a multiplier is negative, that constraint is dropped.
   - At the end every KKT condition holds: the point is feasible, every multiplier is ≥ 0, and complementary slackness holds.

**Why step 2.** With scipy's default tolerance (`ftol = 1e-6`) and an objective of about 1e-2, SLSQP stops too early.

- The LN4 lab is an example. Its expected tracking error, 0.013212510637878089, is what SLSQP returns after 4 iterations from x0.
- The exact optimum of the same problem gives 0.015323. scipy itself finds the exact optimum with `ftol = 1e-15`.
- The test file reproduces the lab's number with the lab's own call, then shows that our weights satisfy the KKT conditions and have a lower objective.

## Shadow prices: what the constraints cost

- **Marginal.** The multiplier λ of each binding constraint, divided by γ, is the gain in certainty-equivalent return of total wealth for one more unit of room (envelope theorem; checked with finite differences in the tests). It is quoted for 1 percentage point of the portfolio, per year.
- **Total.** The problem is solved again without the whole group (no leverage, long-only, lock-up, caps). The cost is the gain in certainty equivalent per year.
- **Units.** Both are given in bp of W per year, in CHF per year and in bp of F per year.

Banker, age 35:

| Constraint | Marginal: 1 pp more room is worth | Total cost of the group |
|---|---|---|
| No leverage | CHF 42 per year | CHF 29,949 per year (without it: 16× the portfolio invested) |
| Long-only (CHF bonds) | CHF 51 per year | CHF 52,193 per year (all long-only constraints) |
| Lock-up (UBS ≥ 15%) | CHF 0.9 per year | CHF 6 per year |

**Read the marginal values first.**

- The total costs are upper bounds from a model that would borrow 15 times the portfolio, or short CHF 4.5 m of bonds. That is the Merton logic for a young client, not a realistic advice.
- The lock-up costs almost nothing for two reasons:
  - UBS has a high implied return (it carries market risk, which the market pays for).
  - The portfolio is only 3% of the banker's wealth, so its concentration hardly moves the risk of total wealth.

## Risk sources on total wealth

Total wealth is treated as a portfolio of the assets, with weights (1 − ω)x, the career (weight ω_H) and the pension (weight ω_P, no market risk). The LN5 formula is then applied to the whole balance sheet. For the banker in (C), the career carries **97.8%** of the risk of total wealth.

`career.idiosyncratic_vol` adds the personal career risk. It is 0 by default, as in the backtest, and it does not change the optimal portfolio.

## Results: the banker (age 35)

- H = CHF 2.97 m, P = CHF 48k, F = CHF 100k, so ω = 96.8%.
- The career already holds CHF 4.5 m of CHF bonds (1.53 × H) and CHF 0.45 m of European banks (0.15 × H).

| Share of F | (A) standard | (D) total wealth, no hedge | (C) total wealth |
|---|---|---|---|
| EQ_WORLD | 44.0% | 75.0% | 85.0% |
| BANKS_EU | 0 | 10.0% | 0 |
| BOND_GLB_H | 33.7% | 0 | 0 |
| REIT_CH | 2.2% | 0 | 0 |
| GOLD | 5.1% | 0 | 0 |
| EMPLOYER (lock-up) | 15.0% | 15.0% | 15.0% |
| Certainty equivalent vs (A), CHF per year | 0 | +1,882 | +1,888 |

- **(A) → (D).** "Your job is a bond, so you can take more risk": the bonds go and equities come in.
- **(D) → (C).** "Avoid your own sector": the banks go.
- **Total-wealth volatility** hardly moves (8.05% to 8.08%), because the career is most of the wealth (proposal, section 4.1).

## All personas

| Persona | ω | Bonds (A) | Bonds (D) | Own sector (D) | Own sector (C) | (D) vs (A), CHF/yr | (C) vs (D), CHF/yr |
|---|---|---|---|---|---|---|---|
| Banker | 96.8% | 33.7% | 0 | 10.0% | 0 | 1,882 | 6 |
| Software engineer | 97.1% | 34.2% | 0 | 10.0% | 0 | 2,067 | 36 |
| Dentist | 97.5% | 34.2% | 0 | 0 | 0 | 2,553 | 0 |
| Engineer in industry | 95.6% | 34.2% | 0 | 10.0% | 0.3% | 2,059 | 1 |
| Shop owner | 94.3% | 34.2% | 0 | 10.0% | 0 | 2,161 | 59 |
| Construction worker | 95.4% | 34.2% | 0 | 0 | 0 | 2,397 | 0 |
| Energy employee | 96.2% | 34.3% | 0 | 0 | 0 | 2,054 | 0 |
| Civil servant | 96.2% | 34.3% | 0 | – | – | 2,748 | 0 |

The hedge ((D) → (C)) matters only when (D) would hold the client's own sector. Its value is small with β_S = 0.15. For the banker it is even smaller because European banks are negatively correlated with CHF bonds (−0.25): holding banks also hedges the bond part of the career.

## For the app and the backtest

| Function | Gives |
|---|---|
| `analyze_client(params, excess_returns)` | everything above for one client, as tables (app pages 3, 4 and 5) |
| `strategy_portfolios(mu, cov, sigma_H, omega, gamma, constraints)` | weights of (A), (D), (C) for one estimation window (backtest) |
| `optimal_portfolio(mu, cov, sigma_H, omega, gamma, constraints)` | contract function |
| `bl_expected_returns(...)`, `risk_contributions(x, cov)` | contract functions |
| `build_constraints(assets, params, lockup)` | the constraint set |
| `risk_tolerance(params, excess_returns)` / `calibrate_gamma(...)` | γ |
| `shadow_prices(...)`, `total_wealth_risk_contributions(...)`, `certainty_equivalent(...)` | the pieces of pages 4 and 5 |
| `balance_sheet(params)`, `pension_savings(age, salary, params)` | H, P, F, W, ω, lock-up |

Speed: one `analyze_client` takes about 0.04 s.

Tests run on:

- 576 client variations: 8 personas × ages 22–70 × portfolios from CHF 1k to 2m × salaries from 30k to 400k;
- 344 rolling 60-month windows;
- every setting variant (no long-only, no leverage limit, sample covariance, views, β_S = 1, a 100% lock-up).

None failed.

## Open points for the group

1. **Market weights.** All bonds go to `BOND_GLB_H`, because CHF bonds are a tiny part of world bonds. As a result (A) never holds `BOND_CHF`. In the optimizer `BOND_CHF` works as the career's bond (the hedge), not as an investment. The alternative is a "Swiss" split of the bond weight, which would be a home-bias assumption.
2. **Caps at 10%** per sector and per stock. They are an assumption (slider in the app).
3. **Corner solutions** for young clients. They are a result, as the proposal says: the app should show the marginal shadow prices first.
4. **Deferred shares of CHF 15,000** for the banker. This is an assumption (slider).
5. **Pension = legal minimum** if no amount is given. Bankers usually have more (extra-mandatory plans).
6. **γ in the backtest.** Either keep the γ of the first window (no look-ahead) or calibrate it again each window. On rolling windows it moves between 0.19 and 0.32. To keep it fixed, write it in `optimization.gamma`.
7. **Discount rate of H.** H uses long-run premia (1% bond, 5% sector), the optimizer uses the Black-Litterman premia. This is kept on purpose so that H stays stable over time; it only changes the level of the certainty equivalent, never a choice.
8. **Michaud resampling bands** (Layer 2) are not done yet. They will use `optimal_portfolio` on resampled inputs.
