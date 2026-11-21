# The app (step 5)

- Entry: `app.py`. Pages: `views/plan.py` ("Your plan") and `views/analyst.py` ("Analyst view").
- Numbers: `src/dashboard.py` (from steps 1 to 4). Look: `src/theme.py` and `.streamlit/config.toml`.
- Slow results: `src/jobs.py` runs the backtest in the background; `src/precompute.py` computes it in advance.
- Settings of the pages (job menu, input limits, texts): section `app` of `config/params.yaml`.
- Tests: `notebooks/test_app.py`. It also runs both pages headless for every persona.

```bash
pip install -r requirements.txt
python -m src.precompute
streamlit run app.py
python notebooks/test_app.py
```

## Two pages

**Your plan** is the app as a client would use it, published online. One page, read from top to bottom, no jargon:

| Part | What the client sees |
|---|---|
| About you | Job, age, salary, savings; pension (estimated or from the certificate); locked employer shares; how much the pay follows the sector (not at all, a little, a lot) |
| The advice | One sentence ("100% shares. No bonds."), the portfolio as one band in CHF, typical advice below it, and three reasons: the job works like a bond, the job already depends on the sector, what the plan is worth in CHF a year |
| What you really own | Total wealth as one band, and the career year by year: each year's salary and what it is worth today. The dark bars add up to the career value (CHF 3.29 m for the banker) |
| As you get older | The plan solved again every 2 years until retirement, with the wealth expected then: shares, then bonds come in near retirement |
| Would it have worked? | The backtest 2005–2026 for this client, our plan against typical advice; the final value, the gain in CHF a year with its p-value, and the same with the Swiss pension-fund prior (p = 0.06 for the banker, as Roberto asked to say) |
| How we work this out | The method and the rules in plain words; the analyst view has the rest |

**Analyst view** is the previous app, for the class and the adviser: side panel with every setting (it starts from the client of "Your plan"), six tabs (wealth, portfolio, cost of the rules, risk, backtest with robustness, assumptions).

## Behaviour

- **Everything updates by itself.** No button: the optimizer answers in about 0.1 s, the plan by age in about 0.7 s.
- **The backtest runs in the background.** The page appears at once with a waiting bar; the page checks every second and shows the backtest when it is ready (a few seconds; about 30 seconds for the robustness tests of the analyst view). The client can keep changing the inputs meanwhile. Results are kept in `data/cache/` (not in git) and are instant the next time. `python -m src.precompute` computes every persona in advance; `--robustness` adds the robustness tests (about 5 minutes).
- **No "Deploy" button.** `toolbarMode = "minimal"` in `.streamlit/config.toml` hides the button Streamlit shows to publish the app on its cloud, and the developer menu.
- **Auto rerun.** `runOnSave = true`: saving a file reloads the app.

## Idea of the design

- **Bands.** Total wealth and every portfolio are drawn as one rounded band: career in plum, pension in green, portfolio and shares in saffron, bonds in blue, the client's sector in red.
- **Sentences first, numbers second.** Each part opens with one sentence that is the answer ("You own CHF 3.44 m. 96% of it is your future salary."), then the chart that proves it.
- **Fonts.** Bricolage Grotesque for headlines and big numbers, Onest for the text.

## Demo for the class (3 minutes)

Before the class: `python -m src.precompute --robustness`.

1. **Your plan, banking, 35.** "100% shares. No bonds." The three reasons. Scroll: "You own CHF 3.44 m, 96% is your future salary", and the bars that explain the CHF 3.29 m.
2. **Live change.** Age 60, savings CHF 1 m: bonds come in (8%), the career is 33% of wealth. Then "Dentist": nothing to offset.
3. **Would it have worked?** Our plan ends at CHF 1.29 m against CHF 893k, worth CHF 12k a year (p = 0.01), less certain with the Swiss prior (p = 0.06).
4. **Analyst view.** The cost of each rule and the robustness tests.
