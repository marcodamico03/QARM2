"""
Client presets. Every number here is an assumption and is shown on the app's
'What we assume' page. Change them here and the whole project follows.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Persona:
    name: str
    age: int
    retirement_age: int
    salary: float               # gross annual salary today, CHF
    salary_growth: float        # per year
    savings_rate: float         # share of salary added to the portfolio each year
    portfolio: float            # free financial portfolio today, CHF
    pension: float              # 2nd-pillar account today, CHF
    beta_B: float               # how bond-like the career is (units of the CHF bond index)
    beta_S: float               # exposure of the career to the client's sector
    sector_asset: str = "eu_banks"
    employer_asset: str | None = None
    employer_lockup: float = 0.0  # minimum weight in employer stock (deferred shares)

    def with_(self, **changes) -> "Persona":
        """Copy with some fields changed, e.g. BANKER.with_(beta_S=0.6) for sliders."""
        return replace(self, **changes)


# Core client of the project: the Zurich banker
# From the proposal: age 35, CHF 100k portfolio, career worth > CHF 2m, beta_B around 3,
# conservative beta_S. Other values are our assumptions.
BANKER = Persona(
    name="Zurich banker",
    age=35,
    retirement_age=65,
    salary=150_000,
    salary_growth=0.02,
    savings_rate=0.15,
    portfolio=100_000,
    pension=120_000,
    beta_B=3.0,
    beta_S=0.3,
    sector_asset="eu_banks",
    employer_asset="ubs",
    employer_lockup=0.10,
)

# Comparison client: same life, but a job with no stock-market exposure
CIVIL_SERVANT = BANKER.with_(name="Civil servant", salary=120_000, beta_S=0.0,
                             employer_asset=None, employer_lockup=0.0)

# Layer 3 client: about to retire, almost no human capital left
RETIREE = BANKER.with_(name="64-year-old about to retire", age=64, salary=160_000,
                       portfolio=400_000, pension=900_000, employer_asset=None,
                       employer_lockup=0.0)

PERSONAS = {p.name: p for p in (BANKER, CIVIL_SERVANT, RETIREE)}
