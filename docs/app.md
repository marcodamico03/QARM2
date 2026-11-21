# The app (step 5)

- Code: `app.py` (the page), `src/dashboard.py` (the numbers, from steps 1 to 4), `src/theme.py` (the look),
  `src/jobs.py` (the backtest in the background), `.streamlit/config.toml` (dark theme, no Deploy button,
  auto reload).
- Settings of the page (job menu, input limits, texts): section `app` of `config/params.yaml`.
- Tests: `notebooks/test_app.py`. It also runs the app headless for every persona.

```bash
pip install -r requirements.txt
streamlit run app.py
python notebooks/test_app.py
```

## One page

**Side panel: the client.** Job, age, salary, savings; pension fund (estimated, or from the certificate);
locked employer shares (banking); how much the pay follows the sector (not at all, a little, a lot). Under
"Advanced": wages and interest rates, the rules (short selling, borrowing, cap per sector or stock) and a
market view (Black-Litterman).

**The answer, on one screen.**

| Part | What the client sees |
|---|---|
| Headline | "Invest your CHF 100'000 all in shares." and why, in one sentence |
| Wealth bar | Everything the client owns: career, pension, savings, with the total (CHF 3.44 m for the banker) |
| Your portfolio | Our plan and typical advice as two bars, the holdings in CHF, three reasons (the job works like a bond, the pay already depends on the sector, the value in CHF a year) |
| Since 2005 | The backtest for this client: our plan against typical advice, final value, gain in CHF a year with its p-value, worst fall, and the result with the Swiss pension-fund prior (p = 0.06 for the banker) |
| What your career is worth | Each remaining year's salary and its value today; the bars add up to the career value (CHF 3.29 m) |
| As you get older | The plan solved again every 2 years until retirement: shares, then bonds near retirement |

**Details (tabs).** Why this plan (three strategies, the two effects, demand with no rule, expected returns),
Risk (share of risk of total wealth; inside the portfolio, LN5), Rules (cost of each rule from the Lagrange
multipliers), Backtest (four strategies, gains with 95% intervals, crises, numbers), Robustness (wrong beta,
hedge map, job loss, prior), Assumptions.

## Behaviour

- **Everything updates by itself.** No button. The optimizer answers in about 0.1 s, the plan by age in 0.7 s.
- **The backtest runs by itself.** When the app opens and every time the client changes, the backtest of that
  client starts in the background: the page appears at once, and the backtest shows up after a few seconds
  (about 30 seconds for the robustness tests). Results are kept in `data/cache/` (not in git), so a client seen
  before is instant. `python -m src.precompute` can compute every persona in advance, but it is not needed.
- **No Deploy button**: `toolbarMode = "minimal"`. **Auto reload** when a file is saved: `runOnSave = true`.

## Idea of the design

- **Dark and quiet**, like the evening view of a banking app: graphite panels, one font (Manrope).
- **Muted colours with one meaning each**: indigo the career, sage the pension, champagne the savings and our
  plan, steel blue bonds, rose the client's own sector; typical advice is always grey.
- **The answer first**: one sentence, then four panels that prove it; every detail is one click away in the tabs.

## Demo for the class (3 minutes)

1. **Banking, 35.** "Invest your CHF 100'000 all in shares": the job works like CHF 2.63 m of Swiss bonds,
   the savings are 2.9% of what he owns. The career panel explains the CHF 3.29 m.
2. **Since 2005.** CHF 1.29 m against CHF 893k, CHF 12k a year better (p = 0.01); with the Swiss prior p = 0.06.
3. **Live change.** Age 60, savings CHF 1 m: bonds come in (8%). Then Dentist: nothing to offset.
4. **Tabs.** Rules (the cost of each rule) and Robustness.
