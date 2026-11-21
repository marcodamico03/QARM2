"""Look of the app (step 5): colours, fonts, CSS and small HTML pieces.

The idea: the client's whole balance sheet as one band of colour, like the colours of
Swiss banknotes (plum for the career, green for the pension, saffron for the portfolio),
on a cool, quiet background. Rounded shapes, no shadows, one display typeface for the
numbers (Bricolage Grotesque) and one text typeface (Onest).
No Streamlit here: the functions return strings and Plotly settings.
"""
import html

import numpy as np

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#221F33"          # text: deep indigo, not black
MUTED = "#6E6A84"        # secondary text
LINE = "#DEDCEA"         # hairlines and empty tracks
PAGE = "#F3F2F8"         # page background, cool lavender grey
PANEL = "#FFFFFF"        # plates
PLUM = "#5B2A86"         # the career (human capital)
GREEN = "#2F8F5B"        # the pension
SAFFRON = "#F0A202"      # the portfolio
RED = "#D64545"          # the client's own sector, losses
BLUE = "#2D6CB5"         # bonds

ASSET_COLORS = {
    "EQ_WORLD": "#F0A202", "EQ_CH": "#E5702A", "BANKS_EU": "#D64545", "BOND_CHF": "#2D6CB5",
    "BOND_GLB_H": "#86AEDD", "REIT_CH": "#2F8F5B", "GOLD": "#C9A227", "EMPLOYER": "#8A5A44",
    "CASH": "#C9C7D6",
}
SECTOR_COLOR = "#D64545"   # any SEC_... column: the client's own sector

SHORT_NAMES = {
    "EQ_WORLD": "World equities", "EQ_CH": "Swiss equities", "BANKS_EU": "European banks",
    "BOND_CHF": "Swiss gov. bonds", "BOND_GLB_H": "Global bonds, hedged", "REIT_CH": "Swiss real estate",
    "GOLD": "Gold", "EMPLOYER": "Employer shares", "CASH": "Cash", "CAREER": "Your career",
    "PENSION": "Your pension",
}

FONTS = "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Onest:wght@400;500;600&display=swap"
DISPLAY = "'Bricolage Grotesque', 'Onest', system-ui, sans-serif"
TEXT = "'Onest', system-ui, -apple-system, sans-serif"


def asset_name(asset, params=None):
    """Short readable name of a column (sector menu names come from params)."""
    if asset in SHORT_NAMES:
        return SHORT_NAMES[asset]
    if params is not None and asset in params["data"].get("sectors", {}):
        return params["data"]["sectors"][asset]["label"].split(" (")[0]
    return asset


def asset_color(asset):
    return ASSET_COLORS.get(asset, SECTOR_COLOR if str(asset).startswith("SEC_") else "#9C98B3")


# ---------------------------------------------------------------------------
# Number formats
# ---------------------------------------------------------------------------

def chf(x, short=True):
    """CHF amount: 3.44 m, 493k, 1,250."""
    x = float(x)
    sign = "−" if x < 0 else ""
    a = abs(x)
    if short and a >= 999_500:
        return f"{sign}CHF {a / 1e6:.2f} m"
    if short and a >= 10_000:
        return f"{sign}CHF {a / 1e3:.0f}k"
    return f"{sign}CHF {a:,.0f}".replace(",", "'")


def pct(x, digits=0):
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-digits - 2):
        x = 0.0
    return f"{x:.{digits}%}".replace("-", "−")


def esc(text):
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def css():
    text = f"""<style>
@import url('{FONTS}');
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --page:{PAGE}; --panel:{PANEL};
        --plum:{PLUM}; --green:{GREEN}; --saffron:{SAFFRON}; --red:{RED}; --blue:{BLUE}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--page); color: var(--ink);
        font-family: {TEXT}; }}
/* text font everywhere except icon spans (Material icons use their own font) */
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea, .stApp table,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-baseweb="tab"],
[data-baseweb="select"] div {{ font-family: {TEXT}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ font-family: {DISPLAY} !important; color: var(--ink); letter-spacing: -0.015em; }}

/* sidebar: the client card */
[data-testid="stSidebar"] {{ background: var(--panel); border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 18px;
        background: var(--page); }}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{ border: none; }}
[data-testid="stSidebar"] summary {{ font-weight: 600; }}

/* tabs as a pill switch: new Streamlit (react-aria, data-testid stTab) and old (baseweb) */
.stTabs {{ margin-top: 1.6rem; }}
.stTabs [role="tablist"] {{ gap: 4px; background: #E7E5F0; padding: 5px; border-radius: 999px;
        width: fit-content; max-width: 100%; overflow-x: auto; }}
.stTabs [role="tablist"]::after {{ display: none; }}
.stTabs [role="tab"] {{ height: 38px; padding: 0 18px; border-radius: 999px; background: transparent;
        color: var(--muted); font-weight: 500; white-space: nowrap; }}
.stTabs [role="tab"][data-selected], .stTabs [role="tab"][aria-selected="true"] {{ background: var(--panel) !important;
        color: var(--ink) !important; }}
.stTabs [role="tab"][data-hovered] {{ color: var(--ink); }}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-testid="stTabPanel"], .stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.6rem; }}

/* widgets */
.stButton > button {{ border-radius: 999px; border: 1px solid var(--line); background: var(--panel);
        color: var(--ink); font-weight: 600; padding: .45rem 1.2rem; }}
.stButton > button:hover {{ border-color: var(--plum); color: var(--plum); }}
[data-testid="stExpander"] {{ border-radius: 20px; border: 1px solid var(--line); background: var(--panel); }}
div[data-baseweb="select"] > div {{ border-radius: 14px; }}
[data-testid="stPlotlyChart"] {{ background: var(--panel); border-radius: 26px; padding: 14px 10px 4px; }}

/* hero */
.btp-who {{ color: var(--muted); font-size: 1.02rem; margin: 0 0 .2rem; }}
.btp-hero {{ color: var(--ink); font-family: {DISPLAY}; font-weight: 800; font-size: clamp(2.1rem, 4.4vw, 3.6rem);
        line-height: 1.02; letter-spacing: -0.03em; margin: 0 0 1.4rem; max-width: 20ch; }}
.btp-hero .soft {{ display: block; color: var(--muted); font-weight: 400; font-size: .55em; letter-spacing: -0.01em;
        margin-top: .5rem; }}
.btp-ribbon {{ display: flex; height: 96px; border-radius: 999px; overflow: hidden; background: var(--line);
        gap: 4px; }}
.btp-ribbon > div {{ display: flex; align-items: center; padding: 0 26px; color: #fff; min-width: 0; }}
.btp-ribbon .career {{ background: var(--plum); }}
.btp-ribbon .pension {{ background: var(--green); padding: 0; }}
.btp-ribbon .portfolio {{ background: var(--saffron); padding: 0; }}
.btp-ribbon b {{ font-family: {DISPLAY}; font-size: 1.7rem; font-weight: 600; margin-right: .6rem; }}
.btp-legend {{ display: flex; flex-wrap: wrap; gap: 10px 26px; margin: .9rem 0 0 .3rem; color: var(--muted); }}
.btp-legend span b {{ color: var(--ink); font-weight: 600; }}
.btp-dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 7px;
        vertical-align: -1px; }}

/* plates */
.btp-plate {{ background: var(--panel); border-radius: 30px; padding: 26px 30px; height: 100%; }}
.btp-plate.tint-plum {{ background: #EDE6F5; }}
.btp-plate.tint-red {{ background: #FBE9E7; }}
.btp-plate.tint-blue {{ background: #E6EEF8; }}
.btp-plate.tint-green {{ background: #E5F2EA; }}
.btp-big {{ font-family: {DISPLAY}; font-weight: 800; font-size: clamp(1.7rem, 2.6vw, 2.6rem); line-height: 1;
        letter-spacing: -0.03em; white-space: nowrap; }}
.btp-plate p {{ margin: .55rem 0 0; color: var(--ink); line-height: 1.45; max-width: 46ch; }}
.btp-plate .note {{ color: var(--muted); font-size: .92rem; }}
.btp-section {{ font-family: {DISPLAY}; font-weight: 600; font-size: 1.45rem; margin: 2.2rem 0 .3rem;
        letter-spacing: -0.02em; }}
.btp-lead {{ color: var(--muted); margin: 0 0 1rem; max-width: 70ch; line-height: 1.5; }}

/* allocation pills */
.btp-alloc {{ margin: 0 0 18px; }}
.btp-alloc .head {{ display: flex; justify-content: space-between; align-items: baseline; margin: 0 0 7px 4px; }}
.btp-alloc .head b {{ font-weight: 600; }}
.btp-alloc .head span {{ color: var(--muted); font-size: .92rem; }}
.btp-bar {{ display: flex; height: 46px; border-radius: 999px; overflow: hidden; gap: 3px; background: var(--line); }}
.btp-bar > div {{ display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 600;
        font-size: .86rem; white-space: nowrap; overflow: hidden; }}
.btp-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 6px; }}
.btp-chip {{ background: var(--panel); border-radius: 999px; padding: 5px 13px 5px 10px; font-size: .88rem; }}

/* price tags */
.btp-tags {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; }}
.btp-tag {{ background: var(--panel); border-radius: 24px 24px 24px 8px; padding: 20px 22px; position: relative; }}
.btp-tag .price {{ font-family: {DISPLAY}; font-weight: 800; font-size: 2rem; letter-spacing: -0.02em; }}
.btp-tag .per {{ color: var(--muted); font-size: .9rem; }}
.btp-tag .what {{ margin-top: .55rem; line-height: 1.4; }}
.btp-tag .hole {{ position: absolute; top: 18px; right: 20px; width: 12px; height: 12px; border-radius: 50%;
        background: var(--page); }}

/* tables */
.btp-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--panel);
        border-radius: 22px; overflow: hidden; font-size: .93rem; }}
.btp-table th {{ text-align: right; font-weight: 600; color: var(--muted); padding: 12px 14px;
        border-bottom: 1px solid var(--line); }}
.btp-table th:first-child, .btp-table td:first-child {{ text-align: left; }}
.btp-table td {{ text-align: right; padding: 10px 14px; border-bottom: 1px solid #F0EFF5;
        font-variant-numeric: tabular-nums; }}
.btp-table th.txt, .btp-table td.txt {{ text-align: left; }}
.btp-table tr:last-child td {{ border-bottom: none; }}
.btp-pos {{ color: {GREEN}; font-weight: 600; }}
.btp-neg {{ color: {RED}; font-weight: 600; }}
.btp-quiet {{ color: var(--muted); }}

/* small facts row */
.btp-facts {{ display: flex; flex-wrap: wrap; gap: 10px; margin: .6rem 0 0; }}
.btp-fact {{ background: var(--panel); border-radius: 16px; padding: 10px 16px; }}
.btp-fact b {{ font-family: {DISPLAY}; font-weight: 600; font-size: 1.15rem; display: block; }}
.btp-fact span {{ color: var(--muted); font-size: .86rem; }}


/* ---------- page "Your plan" ---------- */
.btp-brand {{ font-family: {DISPLAY}; font-weight: 800; font-size: 1.35rem; letter-spacing: -0.02em;
        margin: 0 0 1.4rem; display: flex; align-items: center; gap: .55rem; }}
.btp-brand i {{ width: 26px; height: 26px; border-radius: 50%; display: inline-block;
        background: conic-gradient(var(--plum) 0 62%, var(--green) 62% 80%, var(--saffron) 80% 100%); }}
.st-key-about {{ background: var(--panel); border-radius: 30px; padding: 22px 26px 10px; }}
.st-key-about [data-testid="stWidgetLabel"] p {{ color: var(--muted); font-size: .9rem; }}
.btp-kicker {{ color: var(--muted); font-size: 1.05rem; margin: 2.6rem 0 .3rem; }}
.btp-verdict {{ font-family: {DISPLAY}; font-weight: 800; font-size: clamp(2.4rem, 6vw, 4.6rem); line-height: .98;
        letter-spacing: -0.035em; margin: 0 0 .7rem; }}
.btp-sub {{ font-size: 1.15rem; line-height: 1.45; max-width: 58ch; margin: 0 0 1.6rem; }}
.btp-band {{ display: flex; height: 72px; border-radius: 999px; overflow: hidden; gap: 4px; background: var(--line); }}
.btp-band > div {{ display: flex; align-items: center; padding: 0 22px; font-weight: 600; white-space: nowrap;
        overflow: hidden; min-width: 0; }}
.btp-band.thin {{ height: 22px; gap: 3px; }}
.btp-band.thin > div {{ padding: 0; }}
.btp-bandrow {{ display: grid; grid-template-columns: 130px 1fr; align-items: center; gap: 14px; margin: 10px 0 0; }}
.btp-bandrow span {{ color: var(--muted); font-size: .92rem; }}
.btp-hold {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 6px 34px;
        margin: 1.3rem 0 0; }}
.btp-hold > div {{ display: grid; grid-template-columns: 12px 1fr auto; gap: 12px; align-items: baseline;
        padding: 9px 0; border-bottom: 1px solid var(--line); }}
.btp-hold b {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.btp-hold small {{ display: block; color: var(--muted); font-size: .84rem; }}
.btp-reasons {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px;
        margin: 2rem 0 0; }}
.btp-reason {{ border-radius: 28px; padding: 24px 26px; }}
.btp-reason h4 {{ font-family: {DISPLAY} !important; font-weight: 600; font-size: 1.3rem; line-height: 1.15;
        margin: 0 0 .55rem; padding: 0; letter-spacing: -0.02em; }}
.btp-reason p {{ margin: 0; line-height: 1.5; }}
.btp-h2 {{ font-family: {DISPLAY}; font-weight: 800; font-size: clamp(1.8rem, 3.6vw, 2.7rem); line-height: 1.04;
        letter-spacing: -0.03em; margin: 4.2rem 0 .7rem; max-width: 24ch; }}
.btp-p {{ font-size: 1.06rem; line-height: 1.55; max-width: 64ch; margin: 0 0 1.3rem; }}
.btp-chart-title {{ font-weight: 600; margin: 1.2rem 0 .1rem 4px; }}
.btp-chart-note {{ color: var(--muted); font-size: .92rem; margin: 0 0 .4rem 4px; }}
.btp-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin: 1rem 0 0; }}
.btp-tile {{ background: var(--panel); border-radius: 28px; padding: 22px 24px; }}
.btp-tile .v {{ font-family: {DISPLAY}; font-weight: 800; font-size: clamp(1.6rem, 2.4vw, 2.2rem); letter-spacing: -0.03em;
        white-space: nowrap; }}
.btp-tile p {{ margin: .45rem 0 0; line-height: 1.45; }}
.btp-tile .q {{ color: var(--muted); font-size: .9rem; }}
.btp-wait {{ background: var(--panel); border-radius: 28px; padding: 28px; }}
.btp-wait p {{ margin: 0 0 14px; color: var(--muted); }}
.btp-wait .track {{ height: 10px; border-radius: 999px; background: var(--line); overflow: hidden; }}
.btp-wait .track i {{ display: block; height: 100%; width: 35%; border-radius: 999px; background: var(--plum); }}
.btp-foot {{ color: var(--muted); font-size: .86rem; margin: 4rem 0 0; max-width: 80ch; line-height: 1.5; }}
@media (max-width: 760px) {{
  .btp-band {{ height: 54px; }} .btp-bandrow {{ grid-template-columns: 1fr; gap: 4px; }}
  .btp-ribbon {{ height: 70px; }} .btp-ribbon b {{ font-size: 1.2rem; }} .btp-ribbon > div {{ padding: 0 14px; }}
  .btp-plate {{ padding: 20px; border-radius: 24px; }}
}}
@media (prefers-reduced-motion: no-preference) {{
  .btp-ribbon > div {{ animation: grow .9s cubic-bezier(.2,.8,.2,1) both; transform-origin: left; }}
  @keyframes grow {{ from {{ transform: scaleX(.2); opacity: 0; }} to {{ transform: scaleX(1); opacity: 1; }} }}
  .btp-wait .track i {{ animation: slide 1.4s ease-in-out infinite; }}
  @keyframes slide {{ from {{ transform: translateX(-100%); }} to {{ transform: translateX(290%); }} }}
}}
</style>"""
    # no blank line: Markdown would end the HTML block there
    return "\n".join(line for line in text.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------

def ribbon(sheet):
    """The hero: total wealth as one band, career / pension / portfolio in proportion."""
    W = sheet["W"]
    parts = [("career", sheet["H"], "Your career"), ("pension", sheet["P"], "Your pension"),
             ("portfolio", sheet["F"], "Your portfolio")]
    cells = []
    for cls, value, label in parts:
        share = value / W if W > 0 else 0
        if share <= 0:
            continue
        inner = f"<b>{chf(value)}</b>{esc(label)}" if cls == "career" and share > 0.35 else ""
        cells.append(f'<div class="{cls}" style="flex:{max(share, 0.012):.5f} 1 0" title="{esc(label)}: '
                     f'{chf(value)} ({pct(share, 1)})">{inner}</div>')
    legend = "".join(
        f'<span><i class="btp-dot" style="background:{c}"></i>{esc(label)} <b>{chf(v)}</b> ({pct(v / W, 1)})</span>'
        for c, v, label in ((PLUM, sheet["H"], "Career"), (GREEN, sheet["P"], "Pension"),
                            (SAFFRON, sheet["F"], "Portfolio")) if v > 0)
    return f'<div class="btp-ribbon">{"".join(cells)}</div><div class="btp-legend">{legend}</div>'


def plate(big, text, note=None, tint=None, color=None):
    style = f' style="color:{color}"' if color else ""
    cls = f"btp-plate tint-{tint}" if tint else "btp-plate"
    note_html = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="{cls}"><div class="btp-big"{style}>{big}</div><p>{text}</p>{note_html}</div>'


def section(title, lead=None):
    lead_html = f'<p class="btp-lead">{lead}</p>' if lead else ""
    return f'<div class="btp-section">{esc(title)}</div>{lead_html}'


def facts(items):
    return '<div class="btp-facts">' + "".join(
        f'<div class="btp-fact"><b>{value}</b><span>{esc(label)}</span></div>' for value, label in items) + "</div>"


def allocation(title, weights, params=None, note=None):
    """One portfolio as a rounded bar: one segment per holding, the label inside when it fits."""
    segments = []
    for asset, w in weights.items():
        if w < 0.0005:
            continue
        label = f"{asset_name(asset, params)} {pct(w)}" if w >= 0.16 else (pct(w) if w >= 0.055 else "")
        text_color = INK if asset in ("CASH", "GOLD", "BOND_GLB_H", "EQ_WORLD") else "#fff"
        segments.append(f'<div style="flex:{w:.5f} 1 0;background:{asset_color(asset)};color:{text_color}" '
                        f'title="{esc(asset_name(asset, params))}: {pct(w, 1)}">{esc(label)}</div>')
    note_html = f"<span>{note}</span>" if note else ""
    return (f'<div class="btp-alloc"><div class="head"><b>{esc(title)}</b>{note_html}</div>'
            f'<div class="btp-bar">{"".join(segments)}</div></div>')


def chips(assets, params=None):
    return '<div class="btp-chips">' + "".join(
        f'<span class="btp-chip"><i class="btp-dot" style="background:{asset_color(a)}"></i>'
        f'{esc(asset_name(a, params))}</span>' for a in assets) + "</div>"


def price_tags(tags):
    """tags: list of (price text, per text, what text, colour)."""
    return '<div class="btp-tags">' + "".join(
        f'<div class="btp-tag"><i class="hole"></i><div class="price" style="color:{color}">{price}</div>'
        f'<div class="per">{esc(per)}</div><div class="what">{what}</div></div>'
        for price, per, what, color in tags) + "</div>"


def table(df, formats=None, signed=(), index=True):
    """A DataFrame as a rounded HTML table. formats: {column: function}; signed: columns coloured
    green when positive and red when negative."""
    formats = formats or {}
    cols = list(df.columns)
    numeric = {c: all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                      for v in df[c]) for c in cols}
    head = ("<th></th>" if index else "") + "".join(
        f'<th{"" if numeric[c] else " class=txt"}>{esc(c)}</th>' for c in cols)
    rows = []
    for idx, row in df.iterrows():
        cells = [f"<td>{esc(idx)}</td>"] if index else []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) and np.isnan(v):
                text, cls = "", ""
            else:
                text = formats[c](v) if c in formats else esc(v)
                cls = "" if numeric[c] else " class=txt"
                if c in signed and isinstance(v, (float, int, np.floating)):
                    cls = ' class="btp-pos"' if v > 0 else (' class="btp-neg"' if v < 0 else "")
            cells.append(f"<td{cls}>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="btp-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# Families of the page "Your plan" (src/dashboard.py, GROUPS)
GROUP_COLORS = {"Shares": SAFFRON, "Bonds": BLUE, "Real estate and gold": "#7FB08F", "Cash": "#C9C7D6"}


def text_on(color):
    """Ink on light colours, white on dark ones."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return INK if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#fff"


def brand():
    return '<div class="btp-brand"><i></i>Beyond the Portfolio</div>'


def band(weights, params=None, thin=False, min_weight=0.0005):
    """A portfolio as one rounded band, one segment per holding (the label inside when it fits)."""
    cells = []
    for asset, w in weights.items():
        if w < min_weight:
            continue
        color = asset_color(asset)
        label = "" if thin else (f"{asset_name(asset, params)} {pct(w)}" if w >= 0.22 else
                                 (pct(w) if w >= 0.07 else ""))
        cells.append(f'<div style="flex:{w:.5f} 1 0;background:{color};color:{text_on(color)}" '
                     f'title="{esc(asset_name(asset, params))}: {pct(w, 1)}">{esc(label)}</div>')
    return f'<div class="btp-band{" thin" if thin else ""}">{"".join(cells)}</div>'


def band_row(label, weights, params=None, min_weight=0.0005):
    return (f'<div class="btp-bandrow"><span>{esc(label)}</span>'
            f'{band(weights, params, thin=True, min_weight=min_weight)}</div>')


def holdings(rows):
    """rows: list of (colour, name, note or None, amount text, share text)."""
    return '<div class="btp-hold">' + "".join(
        f'<div><i class="btp-dot" style="background:{c}"></i><span>{esc(name)}'
        f'{f"<small>{esc(note)}</small>" if note else ""}</span><b>{amount} <span class="btp-quiet">{share}</span></b></div>'
        for c, name, note, amount, share in rows) + "</div>"


def reasons(cards):
    """cards: list of (title, text, tint): the 'why' of the advice."""
    tints = {"blue": "#E6EEF8", "red": "#FBE9E7", "plum": "#EDE6F5", "green": "#E5F2EA", "white": PANEL}
    return '<div class="btp-reasons">' + "".join(
        f'<div class="btp-reason" style="background:{tints[t]}"><h4>{esc(title)}</h4><p>{text}</p></div>'
        for title, text, t in cards) + "</div>"


def tiles(items):
    """items: list of (value, text, quiet note or None, colour)."""
    return '<div class="btp-tiles">' + "".join(
        f'<div class="btp-tile"><div class="v" style="color:{color}">{value}</div><p>{text}</p>'
        f'{f"<p class=q>{note}</p>" if note else ""}</div>' for value, text, note, color in items) + "</div>"


def chart_title(title, note=None):
    note_html = f'<p class="btp-chart-note">{esc(note)}</p>' if note else ""
    return f'<div class="btp-chart-title">{esc(title)}</div>{note_html}'


def waiting(text):
    return f'<div class="btp-wait"><p>{esc(text)}</p><div class="track"><i></i></div></div>'


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

def plot_layout(height=380, **extra):
    """Common layout for every chart: transparent, the app's fonts, light grid, legend on top."""
    layout = dict(
        height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=TEXT, color=INK, size=13),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family=TEXT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=LINE, tickcolor=LINE),
        yaxis=dict(gridcolor="#ECEAF3", zeroline=False, linecolor="rgba(0,0,0,0)"),
    )
    layout.update(extra)
    return layout
