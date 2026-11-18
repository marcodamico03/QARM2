# Team contract

The shared rules every member follows, so that code written separately on five
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
| Efficient frontier on real data, composition maps, stability | LN3 Stability |
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

Total-wealth version, with omega = non-tradable share of total wealth and
sigma_H = Cov(R, R_H):

```
x* = gamma / (1 - omega) * inv(Sigma) mu  -  omega / (1 - omega) * inv(Sigma) sigma_H
      speculative demand                     hedging demand
```

This is the same economics as the concept note (where gamma meant risk aversion),
only the symbol changes. Calibration: gamma is chosen so that a client with no
career and no pension (omega = 0) holds a 60/40 portfolio, and the same gamma is
used in every strategy.

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

Column names used everywhere (lowercase, as in `data/returns_chf.csv`). Sources,
proxies and formulas are documented in `docs/data_layer.md`.

| Column name | Asset | Role |
|---|---|---|
| `world_equity` | World equities (MSCI World, includes financials) | Core equity |
| `euro_banks` | European banks | The banker's sector, R_S |
| `swiss_equity` | Swiss equities | Home market |
| `chf_bonds` | CHF government bonds (synthetic, SNB 10y yield) | Bond-like part of the career, R_B |
| `global_bonds_hedged` | Global bonds (US Treasuries), hedged to CHF | Diversifying bonds |
| `swiss_real_estate` | Swiss listed real estate | Real assets |
| `gold` | Gold | Crisis hedge |
| `employer_stock` | UBS, only when the client holds deferred shares | Lock-up constraint |
| `rf` | CHF risk-free rate (SARON), monthly, can be negative | Not an asset |

---

## 3. Function contract

Names, inputs and outputs everyone codes against. Inside a function, each owner
is free. If you need a new input, propose it in a pull request first.

| File | Function | Input | Output |
|---|---|---|---|
| `src/config.py` | `load_params()` | none | `dict` with everything in `config/params.yaml` |
| `src/data.py` | `load_returns(start=None, end=None, refresh=False)` | optional dates `"YYYY-MM-DD"` | `DataFrame` of monthly CHF returns: the asset columns plus `rf` (Section 2) |
| `src/data.py` | `load_rf(start=None, end=None)` | optional dates | `Series` of the monthly CHF risk-free rate |
| `src/data.py` | `excess_returns(returns)` | output of `load_returns` | asset returns minus `rf` (what the optimizer uses) |
| `src/career.py` | `career_return(bond_ret, sector_ret, beta_bond, beta_sector)` | monthly excess-return `Series` | `Series` of the career's monthly excess return |
| `src/career.py` | `human_capital(age, salary, params)` | age, annual salary in CHF | value of human capital H in CHF |
| `src/optimizer.py` | `bl_expected_returns(cov, w_mkt, phi, P, Q, omega_views, tau)` | arrays | expected excess returns, `ndarray` (lab LN4 method) |
| `src/optimizer.py` | `optimal_portfolio(mu, cov, sigma_H, omega, gamma, constraints)` | arrays and floats | weights of the risky assets, `ndarray` (the rest is cash) |
| `src/optimizer.py` | `risk_contributions(x, cov)` | weights, covariance | `ndarray` of absolute risk contributions (lab LN5 formula) |
| `src/backtest.py` | `run_backtest(returns, rf, strategy, params)` | data and a strategy function | `DataFrame` by month: weights, portfolio value, total wealth |

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
2. Each member works on their own branch and edits **only their own file(s)**.
3. Shared files (`docs/CONTRACT.md`, `config/params.yaml`, `requirements.txt`,
   `src/config.py`) change only through a pull request everyone can see.
4. Bring the latest `main` into your branch often (`git pull origin main`).
5. Small pull requests: one piece of work each, tested, reviewed by one other member.
6. Commit messages say what changed, for example "Add load_returns with CHF conversion".
7. Never commit passwords or API keys.

---

## 6. Tests

Every function gets a small test with a known answer, in `notebooks/` or as a
`test_*.py` file.

- **Lab expected outputs** are our reference values. Example from lab LN4:
  `vol_x0 = 0.15349267083479914`, `implied_phi = 1.6287422626782593`,
  tracking error `0.013212510637878089`.
- **Optimizer:** without constraints, `optimal_portfolio` must equal the formula
  in Section 1.2.
- **Backtest:** the wrong-beta curve must match `h * (2 * beta - h)` in the
  unconstrained case.

---

## 7. Roles

| Role | Files | Member |
|---|---|---|
| 1. Data | `src/data.py`, `data/` | TODO |
| 2. Career | `src/career.py` | TODO |
| 3. Optimizer | `src/optimizer.py` | TODO |
| 4. Backtest | `src/backtest.py` | TODO |
| 5. App | `app.py`, `pages/` | TODO |
