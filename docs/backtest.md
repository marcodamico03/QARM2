# Backtest (step 4): choices and results

- Code: `src/backtest.py`.
- Settings: `config/params.yaml`, section `backtest`.
- Tests: `notebooks/test_backtest.py`.

```bash
python -m src.backtest             # main backtest of the banker, then every robustness test (about 35 s)
python notebooks/test_backtest.py  # known answers, no look-ahead, theory, real data (about 20 s)
```

## The question (proposal, section 4)

Would a banker who followed the total-wealth portfolio have been better off than one who followed a standard portfolio, and for which reason?

## Design

| Element | Choice |
|---|---|
| Client | The banker (from params), 35 years old in Sep 2005, salary CHF 130k, portfolio CHF 100k, CHF 15k of UBS deferred shares |
| Period | Sep 2005 to Oct 2026, 254 months. The first decision uses Sep 2000 to Aug 2005, so 2008 is out of sample |
| Walk-forward | Each month t uses the 60 months before t only. Weights are held during month t. Trading from the drifted holdings costs 10 bp × turnover. Cash earns the CHF rate of the month (negative 2015 to 2022) |
| Strategies | (A) standard, career ignored (ω = 0); (B) 100 − age (world equities / CHF bonds); (D) total wealth without the hedge (σ_H = 0); (C) total wealth |
| Same for all | γ, the Ledoit-Wolf covariance, Black-Litterman returns and constraints (lock-up included). γ is calibrated **once, on the first window** (no look-ahead): 0.277 |
| Savings | 10% of the gross salary every month, invested at the new weights (`savings_rate`, an assumption) |
| H over time | Step 2 model value at the current age and salary, revalued every month with the realised career shocks (career excess return − expected career excess return). Between birthdays the model is interpolated, so H has no yearly jumps |
| P over time | Starts at the balance-sheet value (legal BVG estimate). Each month it earns 1.25%/12 and receives the age credit/12 while employed |
| Deferred shares | A constant CHF 15k: new grants replace the shares that vest (assumption) |

**Fair comparison.**

- With its own wealth path, a strategy that became richer has a smaller career share later. That is a result of past luck, not of the decision.
- So every strategy is measured with the **same wealth shares**, those of (A). This idea comes from Roberto's version.
- Total-wealth excess return = (F/W)(r − rf) + (H/W) × career return + (P/W) × pension excess return.
- With common shares the gains add up: (C) − (A) = ((C) − (D)) + ((D) − (A)).
- Every gain is split into a return part and a risk part.

## Results: the banker

### Headline: certainty-equivalent gain of total wealth (CHF per year, on the average W of (A), CHF 2.9 m)

| | Gain | Return part | Risk part | 95% interval (block bootstrap) | p-value | Newey-West t |
|---|---|---|---|---|---|---|
| (B) 100 − age vs (A) | −1,827 | −2,467 | +640 | [−6,972, +3,300] | 0.75 | −0.90 |
| (D) vs (A): "your job is a bond, take more risk" | **+11,589** | +14,244 | −2,655 | [+619, +23,867] | **0.02** | 2.46 |
| (C) vs (A): total wealth | **+12,072** | +14,556 | −2,484 | [+1,707, +23,935] | **0.01** | 2.60 |
| (C) vs (D): the hedge "avoid your own sector" | +483 | +312 | +171 | [−362, +1,461] | 0.15 | 0.56 |

### Readable metrics (proposal, section 4.1)

| | (A) | (B) | (D) | (C) |
|---|---|---|---|---|
| Correlation of the portfolio with the sector part of the career | 0.71 | 0.71 | 0.79 | **0.76** |
| CHF result in the sector's worst 10% months | −10,301 | −8,734 | −17,351 | **−16,772** |
| Total-wealth volatility | 4.79% | 4.66% | 5.29% | 5.26% |
| Portfolio volatility / max drawdown | 10.4% / −42% | 9.8% / −44% | 16.1% / −61% | 15.6% / −59% |
| Turnover per year / total costs (CHF) | 0.43 / 2,954 | 0.31 / 1,817 | 0.25 / 1,361 | 0.19 / 829 |
| Final portfolio (own path, CHF; savings CHF 323k in total) | 893k | 825k | 1.25 m | 1.29 m |

### Crises (portfolio return)

| | (A) | (B) | (D) | (C) |
|---|---|---|---|---|
| 2008 financial crisis | −27.4% | −28.3% | −44.5% | −41.6% |
| 2011 euro crisis | −3.1% | −7.3% | −15.3% | −13.5% |
| Jan 2015 CHF shock | −6.2% | −3.7% | −9.6% | −9.6% |
| COVID 2020 | −16.8% | −12.4% | −20.5% | −20.5% |
| 2022 rates and inflation | −18.0% | −14.7% | −19.6% | −19.6% |
| Credit Suisse 2023 | +1.4% | +1.4% | +0.4% | +0.4% |

**What the banker results say.**

- **The total-wealth view adds about CHF 12k a year, and it is significant.** Almost all of it comes from the first effect: the career is a large bond, so the portfolio can hold more equities.
- **The hedge points the right way, but it is small.** It lowers risk and loses less than (D) in the bank crises (2008, 2011, worst bank months). It is not significant at our cautious β_S = 0.15.
- **(C) and (D) lose more than (A) in crises,** because they hold more equities. That is the price of the first effect. We show it rather than hide it (proposal, section 4.1).

## Robustness (proposal, section 4.2)

### 1. Wrong beta

The decisions use β_S = 0.15. The career is then replayed with a "true" β_S from 0 to 2× that value.

| True β_S | 0 | 0.075 | 0.15 | 0.225 | 0.30 |
|---|---|---|---|---|---|
| Gain (C) vs (D), CHF/yr | 301 | 376 | 452 | 530 | 610 |
| Risk part | −11 | 73 | 158 | 244 | 331 |
| p-value | 0.24 | 0.19 | 0.14 | 0.11 | 0.08 |

- **The risk part is a straight line in the true β** (R² 0.9999), as the theory h(2β − h) says. The tests check that formula exactly.
- **The break-even is at a true β of 0.01.** Long-only caps the hedge at "zero banks", far below the career's exposure (CHF 0.5 m of banks). So the hedge is almost never too large: robust to overestimating β_S, as the proposal expected.
- **At true β = 0 the gain is still CHF 301, all return part.** European banks underperformed from 2005 to 2026, so avoiding them was also luck.

### 2. Job loss

Twelve months without salary, savings or pension credits when European banks fall 20%, 30% or 40% below their peak. After the new job, the peak starts again.

- **Gains.** (C) vs (A) moves between CHF 8,984 and 10,134 per year. The hedge, (C) vs (D), stays at CHF 393 to 478 per year, with a risk part of CHF 163 to 172.
- **Job losses.** 7 at the 20% threshold, 5 at 30%, 4 at 40% (2008, 2011, 2016, 2020 at 40%).
- **The portfolio after a job loss.** From each job loss to the end of the unemployment, every portfolio gained. The job is lost after the crash, and markets rebounded. (C) gained the most.

### 3. Prior (Roberto's point)

The global market portfolio is compared with a Swiss pension-fund portfolio (Pictet LPP-40, mapped as in Roberto's stage 3.3).

- **With the Swiss prior, the hedge loses CHF 952 a year.** The return part is −1,334 and the risk part +383.
- **Why.** The Swiss prior gives banks a positive weight, so (D) holds 7.8% banks on average and (C) still 4.1%. The return part depends on which assets boomed.
- **The risk part stays positive.**

### 4. Hedge map: when does the hedge matter? (from Roberto's version)

Starting portfolio (CHF 100k to 3 m) × β_S (0.15, 0.30, 0.60). Here the adviser knows the true β.

- **The risk part is positive in all 12 cases.** It grows with both the portfolio and β_S: from CHF 158 a year (100k, 0.15) to CHF 52,632 (3 m, 0.60).
- **Total-wealth volatility falls in every case.** For example, from 12.6% to 10.6% for the largest client.
- **The return part is often negative.** To hedge a strong bank link, (C) also cuts equities, which boomed from 2005 to 2026. So only 3 of the 12 totals are positive, and none is significant.

### 5. Every persona

| Persona | (D) vs (A) | (C) vs (A) | Hedge (C) vs (D) | Return part | Risk part |
|---|---|---|---|---|---|
| Banker | 11,589 | 12,072 | 483 | 312 | 171 |
| Software engineer | 13,289 | 11,429 | −1,861 | −2,301 | 440 |
| Dentist | 19,248 | 18,929 | −319 | −347 | 27 |
| Engineer in industry | 10,724 | 10,467 | −257 | −281 | 24 |
| Shop owner | 8,870 | 8,970 | 99 | −5 | 104 |
| Construction worker | 8,916 | 8,899 | −17 | −58 | 41 |
| Energy employee | 12,322 | 10,925 | −1,397 | −1,901 | 504 |
| Civil servant | 11,684 | 11,702 | 18 | −5 | 23 |

(D) vs (A) is significant for everybody (p ≤ 0.02).

## The message for the presentation

1. **The total-wealth view works, for every client.** It adds about CHF 9k to 19k a year of certainty equivalent, and it is significant.
2. **The hedge does what it is built for, in every test.** Its risk part is positive for every persona, every job-loss threshold, both priors and every client of the hedge map.
3. **Its return part depends on history.** It costs money when the client's sector boomed (tech) and earns money when the sector crashed (banks).
4. **At cautious betas the hedge is small, and long-only keeps it safe:** its break-even true β is about 0.01.

## Changes from Roberto's version

The headline is the same: (C) vs (A) CHF 12,072 here vs 12,084 in his, (D) vs (A) 11,589 vs 11,599, (C) vs (D) 483 vs 455. Kept from his version: the fair comparison with the same wealth shares, the risk/return split, the job-loss thresholds and the hedge map. Changes:

1. **H between birthdays** is interpolated, so it does not jump once a year.
2. **The pension accumulates month by month,** and it gets no credits during unemployment.
3. **Job loss.** H falls by the unpaid year and is back to the model after the new job (the year is lost once). The bank peak restarts after the new job, so 2011, 2016 and 2020 can also cost the job.
4. **One reference (A) for every comparison,** so the gains add up. Return and risk parts are shown for every table.
5. **Prior sensitivity** (global vs Swiss), and the backtest of every persona.

## Open points

1. **Savings rate 10% of gross salary,** and a constant CHF 15k of deferred shares: both are assumptions (sliders in the app).
2. **One historical path.** The certainty equivalent uses realised returns, so it carries the 2005–2026 equity boom. The bootstrap intervals and the risk/return split show this.
3. **The job-loss rule** triggers on bank drawdowns. At 20% it fires 7 times in 21 years, so 20% is probably too sensitive for a sector with 28% volatility.
4. **No turnover penalty** (optional in the proposal): turnover is already low (0.2 to 0.4 a year).
