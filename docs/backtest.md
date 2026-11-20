# Backtest (step 4)

Code: `src/backtest.py`. Settings: `config/params.yaml`, section `backtest`. Tests: `notebooks/test_backtest.py`.

```bash
python notebooks/test_backtest.py   # known answers, no look-ahead, theory checks, real data
```

## Design

| Element | Choice |
|---|---|
| Engine | `run_backtest`: each month the strategy sees ONLY the 60 months before, trades from the drifted holdings, pays 10 bp x turnover, holds one month. Cash earns the CHF rate (negative 2015 to 2022) |
| Period | Data from Sep 2000, so the first decision (Sep 2005) uses Sep 2000 to Aug 2005: 2008 is out-of-sample. 254 months to Oct 2026 |
| Client over time | `client_path`: age +1/12 a month, salary +1.5% a year, H = step 2 model value x (1 + realised career shocks), P = legal BVG estimate, savings = 10% of gross salary a month (`savings_rate`, assumption) |
| Strategies | (A) standard (career ignored), (B) 100 - age (world equities / CHF bonds), (D) total wealth without hedge, (C) total wealth. Same data, Black-Litterman returns, constraints and lock-up for all |
| gamma | Calibrated once on the first window only (no look-ahead) |

## Metrics (`compute_metrics`, `crisis_table`)

- Headline: certainty equivalent of total wealth, CE = 12 mean - 12 var / (2 gamma), in CHF per year.
- Readable: correlation of the portfolio with the sector part of the career, CHF result in the sector's worst 10% months, CHF change in six crises.
- **Fair comparison** (`same_wealth_excess`): two strategies are compared with the SAME wealth shares. With each strategy's own shares, the one that became richer has a smaller career share (a result of the past, not of the decision); for the banker this made the hedge look four times larger and wrongly significant (p = 0.002 instead of 0.14). A test checks the bias is gone.

## Results: the banker (age 35, portfolio 100k, Sep 2005 to Oct 2026)

| | CE gain vs (A), CHF/yr | 95% interval | p-value |
|---|---|---|---|
| (C) total wealth | +12,084 | [+1,742, +24,035] | 0.009 |
| (D) no hedge | +11,599 | [+676, +23,890] | 0.018 |
| (C) vs (D): the hedge | +455 | [-314, +1,337] | 0.14 |

Newey-West t-stats on total wealth: 2.60, 2.46, 0.59. Robust to 6, 12 or 24-month bootstrap blocks.

- The total-wealth view ("your job is a bond, so take more equity risk") adds about CHF 12k a year: significant.
- The hedge points the right way (less lost in 2008 and 2011) but is small for this client.

## Robustness (proposal, section 4.2)

1. **Wrong beta** (`wrong_beta_curve`): decisions with beta_sector 0.15, career replayed with 0 to 4 times it. The gain rises almost linearly (R2 0.9994), as the theory h(2 beta - h) says (exact in a test). At true beta 0 the hedge still gains CHF 306: European banks underperformed, so about two thirds of the CHF 455 is return luck, about CHF 150 is hedging. Significant only from about 3x our beta.
2. **Job loss** (`job_loss_table`): 12 months without salary when banks fall 20%, 30% or 40% below their peak; H drops by one year of salary, and the drop counts in the risk measures. Job lost in Jan 2008 (20%, 30%) or Jun 2008 (40%). Results barely change: the portfolio is about 3% of the banker's wealth, so even a perfect hedge moves total risk little. Banks never regained their 2007 peak, so only one episode.
3. **When does the hedge matter** (`hedge_map`): starting portfolio 100k to 3m x beta_sector 0.15 to 0.60. The **risk part is positive in every case** and grows with both (up to CHF 52k a year; total-wealth volatility 12.5% to 10.6%). The return part is often negative: to hedge a strong bank link, (C) also cuts world equities, which boomed in 2005 to 2026. No total is significant.

## Message for the presentation

The total-wealth view adds significant value. The hedge does what it is designed for, it lowers risk, more so for larger portfolios and stronger sector links, but at our cautious beta and for a young client it is small, and in a long equity bull market its return cost can outweigh the risk saved.

## Open points

1. `savings_rate` = 10% of gross salary (assumption, slider).
2. One historical path: the CE uses realised returns, so it reflects the 2005 to 2026 equity boom.
3. The job-loss rule triggers when the drawdown crosses the threshold; banks stayed far below their 2007 peak, so 2011 and 2020 could not trigger again.
4. Pension credits continue during unemployment (simplification).
