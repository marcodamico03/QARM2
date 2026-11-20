# Team contract

The shared rules everyone follows, so that code written separately on different
branches fits together when it is merged. **Status: proposal.** Approve or comment
in the pull request. Once merged, changes go through a new pull request only.

---

## 1. Conventions

### 1.1 Use the professor's lab code

Where a lab already solves a problem, we reuse its code and its notation, so the
project stays coherent with the course.

| Topic | Lab to reuse |
|---|---|
| Mean-variance, constraints, risk-free asset as an extra asset | LN1 Markowitz |
| Betas with OLS (`statsmodels`) | LN2 Beta |
| Efficient frontier on real data, composition maps, stability, Ledoit-Wolf shrinkage | LN3 Stability |
| Black-Litterman, tracking error, targeting with `fsolve` | LN4 Black-Litterman |
| Risk contributions, ERC | LN5 Risk Budgeting |

Solver: `scipy.optimize.minimize` with `LinearConstraint`, `NonlinearConstraint`
and `Bounds`, as in the labs. (The concept note mentioned cvxpy: replaced by scipy
to stay coherent with the course.)

### 1.2 Meaning of gamma (proposal, to confirm)

We use the **lab convention**: gamma is **risk tolerance** (higher gamma = more risk).

```
QP(x) = 0.5 * x' Sigma x - gamma * x' mu        (minimised, as in every lab)
```

Total-wealth version, with omega = non-tradable share of total wealth, (H + P) / W,
and sigma_H = Cov(R, R_NT), the covariance with the non-tradable wealth (career +
pension). The pension has no market risk, so omega x sigma_H = omega_H x Cov(R, R_career):

```
x* = gamma / (1 - omega) * inv(Sigma) mu  -  omega / (1 - omega) * inv(Sigma) sigma_H
      speculative demand                     hedging demand
```

This is the same economics as the concept note (where gamma meant risk aversion),
only the symbol changes. Calibration: gamma is chosen so that a client with no
career and no pension (omega = 0) holds an optimal portfolio as volatile as a 60/40
portfolio (`fsolve` on a target volatility, as in lab LN1), and the same gamma is
used for every client and in every strategy.

### 1.3 Black-Litterman names (as in lab LN4)

- `phi`: the **market's** implied risk aversion, used only to build the prior
  (`implied_mu`). The lab's QP then uses `gam = 1 / phi`.
- `tau`, `P`, `Q`, `omega_views`: as in the lab. We write `omega_views` (not
  `omega`) because `omega` already means the non-tradable wealth share.
- The lab's function `gamma_matrix(tau)` (= tau * Sigma) keeps its name, but it is
  **not** the client's gamma.

### 1.4 Returns and units

| Rule | Value |
|---|---|
| Frequency | Monthly |
| Return type | Simple returns (not log) |
| Currency | CHF for everything |
| Units | Decimals: 0.02 means 2% |
| Into the optimizer | Excess returns over the CHF risk-free rate |
| Annualising | Mean x 12, variance x 12, volatility x sqrt(12) |
| Missing data | `NaN`, never 0 |

### 1.5 Data format

Every time series is a `pandas.DataFrame` (or `Series`):

- Index: month-end dates (`DatetimeIndex`) named `date`.
- One column per asset, using the names in Section 2.
- Saved in `data/` as CSV, for example `data/returns_chf.csv`.

---

## 2. Investable universe

| Column name | Asset | Role |
|---|---|---|
| `EQ_WORLD` | World equities (ex-financials where available) | Core equity |
| `BANKS_EU` | European banks | The banker's sector |
| `EQ_CH` | Swiss equities | Home market |
| `BOND_CHF` | CHF government bonds | Bond-like part of the career |
| `BOND_GLB_H` | Global bonds, CHF-hedged | Diversifying bonds |
| `REIT_CH` | Swiss real estate | Real assets |
| `GOLD` | Gold | Crisis hedge |
| `EMPLOYER` | Employer stock (UBS), only when the client holds deferred shares | Lock-up constraint |

Risk-free rate: `RF_CHF` (monthly CHF short rate), stored separately.

**Sector menu** (for clients whose career is tied to another sector), stored separately
in `data/sectors_chf.csv`, same dates as the universe: `SEC_FIN`, `SEC_TECH`, `SEC_HEALTH`,
`SEC_INDU`, `SEC_CONS`, `SEC_STAPLES`, `SEC_ENERGY`, `SEC_MATERIALS`. For a client, the
universe is the table above plus the column of the client's sector.

**Personas** (`params.yaml`, section `personas`): banker, tech engineer, dentist, engineer
in industry, shop owner, construction worker, energy employee, civil servant. Each has an
age, a salary, a sector, a beta_sector, a pension fund or not (`pillar2`) and, for the banker,
deferred employer shares (`deferred_shares_chf`).

---

## 3. Function contract

Names, inputs and outputs everyone codes against. Inside a function, the author
is free. If you need a new input, propose it in a pull request first.

| File | Function | Input | Output |
|---|---|---|---|
| `src/config.py` | `load_params()` | none | `dict` with everything in `config/params.yaml` |
| `src/data.py` | `load_returns(start, end)` | dates as `"YYYY-MM-DD"` | `DataFrame` of monthly CHF returns, columns from Section 2 |
| `src/data.py` | `load_rf(start, end)` | dates | `Series` of the monthly CHF risk-free rate |
| `src/data.py` | `load_sector_returns(start, end)` | dates | `DataFrame` of monthly CHF returns of the sector menu |
| `src/career.py` | `career_return(bond_ret, sector_ret, beta_bond, beta_sector)` | monthly excess-return `Series` | `Series` of the career's monthly excess return |
| `src/career.py` | `human_capital(age, salary, params)` | age, annual salary in CHF | value of human capital H in CHF |
| `src/career.py` | `params_for_persona(params, name)` | params, persona name | copy of params set up for that persona |
| `src/career.py` | `career_betas(age, params)` | age, params | beta_bond (duration rule), beta_sector, discount rate of H |
| `src/career.py` | `career_covariance(cov, beta_bond, beta_sector, params)` | covariance `DataFrame`, betas | `sigma_H = Cov(R, R_H)`, `ndarray` |
| `src/optimizer.py` | `bl_expected_returns(cov, w_mkt, phi, P, Q, omega_views, tau)` | arrays | expected excess returns, `ndarray` (lab LN4 method) |
| `src/optimizer.py` | `optimal_portfolio(mu, cov, sigma_H, omega, gamma, constraints)` | arrays and floats | weights of the risky assets, `ndarray` (the rest is cash) |
| `src/optimizer.py` | `risk_contributions(x, cov)` | weights, covariance | `ndarray` of absolute risk contributions (lab LN5 formula) |
| `src/optimizer.py` | `build_constraints(assets, params, lockup)` | asset names, params, lock-up share | `dict` of constraints (format below) |
| `src/optimizer.py` | `strategy_portfolios(mu, cov, sigma_H, omega, gamma, constraints)` | as `optimal_portfolio` | `dict` with the solutions of (A), (D), (C) |
| `src/optimizer.py` | `analyze_client(params, excess_returns)` | params, monthly excess returns | `dict` of tables for the app: weights, demands, shadow prices, risk sources |
| `src/backtest.py` | `run_backtest(returns, rf, strategy, params, path)` | data, a strategy function `f(window_excess, date, state)`, the client path | `DataFrame` by month: weights, portfolio value, total wealth |
| `src/backtest.py` | `run_strategies(params, data)` | params, (returns, rf) | `dict`: results of (A), (B), (D), (C), client path, gamma |

Constraints (`build_constraints`): a `dict` with `assets`, `lower` and `upper` (one
bound per asset, `ndarray`, `-inf`/`inf` = no bound), `caps`, `budget` (1'x <= budget,
`None` = no limit), `long_only`, `lockup` and `fully_invested` (1'x = budget, used only
to reproduce the labs). `constraints=None` means no constraint at all.

Until a module is ready, the others use a simple placeholder with the same name
and output (for example `optimal_portfolio` returning equal weights).

---

## 4. Parameters

- Every number (dates, betas, gamma, costs, thresholds) lives in
  `config/params.yaml` and is read with `load_params()`. No numbers typed inside
  the code.
- Changing `params.yaml` needs a pull request, because it affects everyone.

---

## 5. Git rules

1. `main` always works. Nobody commits directly to `main`.
2. Everyone works on everything, each on their own branch. Before starting a piece of
   work, pull the latest `main` and tell the group which file you are changing, so two
   people do not edit the same function at the same time.
3. Shared files (`docs/CONTRACT.md`, `config/params.yaml`, `requirements.txt`,
   `src/config.py`) change only through a pull request everyone can see.
4. Bring the latest `main` into your branch often (`git pull origin main`).
5. Small pull requests: one piece of work each, tested, reviewed by someone else, and
   merged quickly, so branches do not drift apart.
6. Commit messages say what changed, for example "Add load_returns with CHF conversion".
7. Never commit passwords or API keys.

---

## 6. Tests

Every function gets a small test with a known answer, in `notebooks/` or as a
`test_*.py` file.

- **Lab expected outputs** are our reference values. Example from lab LN4:
  `vol_x0 = 0.15349267083479914`, `implied_phi = 1.6287422626782593`,
  tracking error `0.013212510637878089`. That tracking error is what scipy returns
  with its default tolerance (it stops after 4 iterations); the exact optimum gives
  0.015323 (see `docs/optimizer.md`).
- **Optimizer:** without constraints, `optimal_portfolio` must equal the formula
  in Section 1.2.
- **Backtest:** the wrong-beta curve must match `h * (2 * beta - h)` in the
  unconstrained case.
