# Career model (step 2)

Code: `src/career.py`. Settings: `config/params.yaml`, sections `client` and `career`. Tests: `notebooks/test_career.py`.

```bash
python -m src.career              # summary for the default client (the Zurich banker)
python notebooks/test_career.py   # tests with known answers
```

## The model (proposal, section 3.1)

```
career excess return = beta_bond x (BOND_CHF - rf) + beta_sector x (BANKS_EU - rf) + noise
H = present value of the remaining salary until retirement
discount rate of H = rf_long_run + beta_bond x bond_premium + beta_sector x sector_premium
beta_bond = (1 - wage_rate_link) x duration of the salary / duration of BOND_CHF
```

- Salaries are paid at the end of each year until the retirement age and grow at `salary_growth`.
- The noise is the personal part of the career (promotion, job change). It cannot be hedged and is uncorrelated with the assets, so it does not enter the hedge.

## How beta_bond is chosen

A salary stream is a series of future cash flows, so its value moves with interest rates
by its duration. But a salary is **not a fixed nominal bond**: when rates rise because of
inflation, wages tend to rise too, which partly cancels the effect. `wage_rate_link` (0 to 1)
is the share of interest-rate moves that salaries follow:

```
beta_bond = (1 - wage_rate_link) x modified duration of the salary stream / duration of BOND_CHF (9)
```

The discount rate depends on beta_bond and the duration depends on the discount rate, so the
two are solved together.

**Why wage_rate_link is needed.** With `wage_rate_link = 0` (salary = fixed nominal bond),
every career looks like a long bond: the banker's hedgeable career risk is 8.3% a year, almost
all from interest rates, and the **civil servant (8.6%) looks riskier than the banker**, which
is backwards. The interest-rate part also drowns the sector hedge (bond hedge about 10 times the
sector hedge). With the default `wage_rate_link = 0.5`, the order is right again and the bond and
sector parts of the banker's career have about the same size. A test in
`notebooks/test_career.py` checks that the civil servant stays less risky than the banker.

Result with the default (0.5), banker, salary CHF 130k:

| Age | beta_bond | Discount rate | Duration | H (CHF) |
|---|---|---|---|---|
| 35 | 0.80 | 2.55% | 14.4 | 3.29 m |
| 45 | 0.56 | 2.31% | 10.0 | 2.36 m |
| 55 | 0.30 | 2.05% | 5.3 | 1.24 m |
| 65 | 0 | | | 0 |

Writing a number for `beta_bond` in params.yaml overrides the rule (it is still 0 after retirement).

## Functions

| Function | What it gives |
|---|---|
| `career_return(bond_ret, sector_ret, beta_bond, beta_sector)` | monthly career excess return (contract) |
| `human_capital(age, salary, params)` | H in CHF (contract) |
| `career_betas(age, params)` | beta_bond, beta_sector, discount rate |
| `career_covariance(cov, beta_bond, beta_sector, params)` | sigma_H = Cov(R, R_H) for the optimizer |
| `career_excess_returns(age, params, start, end)` | career excess returns on the saved data, for the backtest |

## Choices to discuss

1. **Salary 130,000 CHF** for the banker (gross, annual).
2. **beta_sector = 0.15 for the banker** (0.15 for tech, 0.10 for industry-related careers, 0.05 for the dentist, 0 for the civil servant): an assumption, not an estimate, kept small on purpose (proposal, section 4.2). A slider in the app, and the wrong-beta test shows what happens if it is wrong.
3. **Premia**: bond 1%, sector 5%; long-run CHF rate 1%; salary growth 1.5%.
4. **Gross salary, not net.** Using net salary or only the part that is saved would lower H.
5. **wage_rate_link = 0.5**: halfway between "salary = fixed nominal bond" (0) and "no interest-rate risk" (1). A slider in the app.
6. On the data (2000 to 2026), the banker's **hedgeable** career risk is about 5.2% per year: 4.3% from rates, 4.2% from banks (they partly offset, correlation about -0.25). The civil servant has 4.5%. The personal part of the career (promotion, job change) is not included because it cannot be hedged.

## Personas (any client, not only the banker)

`params.yaml`, section `personas`: banker, software engineer, dentist (self-employed), engineer in industry, shop owner (florist), construction worker, energy employee, civil servant. Each has an age, a salary, a sector (an asset of the universe or of the sector menu) and a beta_sector.

`params_for_persona(params, "dentist")` returns params set up for that client (also the pension fund, `pillar2`, and the deferred employer shares, `deferred_shares_chf`), and every function works unchanged. `python -m src.career` prints H, betas and career volatility for all of them.
