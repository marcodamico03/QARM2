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
```

- Salaries are paid at the end of each year until the retirement age and grow at `salary_growth`.
- The noise is the personal part of the career (promotion, job change). It cannot be hedged and is uncorrelated with the assets, so it does not enter the hedge.

## How beta_bond is chosen

A salary stream is a series of future cash flows, so like a bond its value moves with interest rates by its duration.

```
beta_bond = modified duration of the salary stream / duration of BOND_CHF (9)
```

The discount rate depends on beta_bond and the duration depends on the discount rate, so the two are solved together.

The result is a beta_bond above 1 for a young client, falling to 0 at retirement:

| Age | beta_bond | Discount rate | Duration | H (CHF, salary 130k) |
|---|---|---|---|---|
| 35 | 1.53 | 3.28% | 13.8 | 2.97 m |
| 45 | 1.09 | 2.84% | 9.8 | 2.24 m |
| 55 | 0.59 | 2.34% | 5.3 | 1.22 m |
| 65 | 0 | | | 0 |

Writing a number for `beta_bond` in params.yaml overrides the rule.

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
5. On the data (2000–2026), the career of a 35-year-old has an excess-return volatility of about 8.3% per year. Most of it comes from the bond part (rates), a smaller part from banks.

## Personas (any client, not only the banker)

`params.yaml`, section `personas`: banker, software engineer, dentist (self-employed), engineer in industry, shop owner (florist), construction worker, energy employee, civil servant. Each has an age, a salary, a sector (an asset of the universe or of the sector menu) and a beta_sector.

`params_for_persona(params, "dentist")` returns params set up for that client, and every function works unchanged. `python -m src.career` prints H, betas and career volatility for all of them.
