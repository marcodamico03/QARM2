"""Look of app_v3: light and dark, in the spirit of Apple's design (system font, quiet cards,
one accent colour, size-specific letter spacing). The page switches between the two with a
toggle; Streamlit's own widgets follow because the theme is set with streamlit_theme().

Colours keep one meaning everywhere:
    our plan, savings, shares   blue (the accent)
    typical advice              grey
    bonds                       teal         real assets   orange
    career                      indigo       pension       green
    your own industry, losses   red
No Streamlit here: the functions return strings and Plotly settings.
"""
import html

from src.theme import chf, pct        # the number formats of the project (CHF 1'250, CHF 3.29 m, 2.9%)

LIGHT = {
    "mode": "light", "bg": "#F5F5F7", "card": "#FFFFFF", "raised": "#F2F2F7", "line": "#E5E5EA",
    "text": "#1D1D1F", "muted": "#6E6E73", "faint": "#AEAEB2",
    "ours": "#0071E3", "typical": "#8E8E93", "bonds": "#30B0C7", "real": "#FF9500", "cash": "#C7C7CC",
    "career": "#5856D6", "pension": "#34C759", "industry": "#FF3B30", "good": "#248A3D", "bad": "#D70015",
    "band": "rgba(0,0,0,0.05)", "shadow": "0 1px 2px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.05)",
}
DARK = {
    "mode": "dark", "bg": "#000000", "card": "#1C1C1E", "raised": "#2C2C2E", "line": "#38383A",
    "text": "#F5F5F7", "muted": "#98989D", "faint": "#636366",
    "ours": "#0A84FF", "typical": "#8E8E93", "bonds": "#64D2FF", "real": "#FF9F0A", "cash": "#48484A",
    "career": "#5E5CE6", "pension": "#30D158", "industry": "#FF453A", "good": "#30D158", "bad": "#FF6961",
    "band": "rgba(255,255,255,0.06)", "shadow": "none",
}
GROUPS = {"equities": "Shares", "real": "Real assets", "bonds": "Bonds", "cash": "Cash"}
GROUP_KEYS = {"equities": "ours", "real": "real", "bonds": "bonds", "cash": "cash"}
FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', system-ui, sans-serif"


def palette(dark):
    return DARK if dark else LIGHT


def streamlit_theme(c):
    """Streamlit theme options for the palette (set with streamlit.config.set_option)."""
    return {"base": c["mode"], "primaryColor": c["ours"], "backgroundColor": c["bg"],
            "secondaryBackgroundColor": c["raised"], "textColor": c["text"]}


def esc(text):
    return html.escape(str(text))


def group_color(c, group):
    return c[GROUP_KEYS[group]]


# ---------------------------------------------------------------------------
# CSS (one string, no blank line: a blank line would end the HTML block in Markdown)
# ---------------------------------------------------------------------------

def css(c):
    rules = f"""
<style>
html, body, .stApp :is(div, p, span, label, input, textarea, button, li, td, th, summary, h1, h2, h3):not([data-testid="stIconMaterial"]) {{ font-family: {FONT} !important; }}
.stApp {{ background: {c['bg']}; color: {c['text']}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {c['card']}; border-right: 1px solid {c['line']}; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ font-size: 0.8rem; color: {c['muted']}; letter-spacing: 0.01em; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }}
[class*="st-key-card"] {{ background: {c['card']}; border: 1px solid {c['line']}; border-radius: 18px;
  padding: 1.3rem 1.4rem 1.1rem; box-shadow: {c['shadow']}; }}
.v3-brand {{ display: flex; align-items: center; gap: 0.5rem; font-weight: 600; font-size: 0.95rem; white-space: nowrap;
  letter-spacing: -0.01em; color: {c['text']}; margin: 0.2rem 0 0.4rem; }}
.v3-dot {{ width: 14px; height: 14px; border-radius: 50%; background: conic-gradient({c['career']} 0 70%, {c['pension']} 0 85%, {c['ours']} 0); }}
.st-key-theme_switch {{ display: flex; justify-content: flex-end; }}
.st-key-theme_switch label p {{ color: {c['muted']}; font-size: 0.85rem; }}
.v3-kicker {{ font-size: 0.82rem; font-weight: 500; color: {c['muted']}; letter-spacing: 0.01em; margin-bottom: 0.35rem; }}
.v3-title {{ font-size: clamp(1.7rem, 3.2vw, 2.55rem); line-height: 1.08; font-weight: 700; letter-spacing: -0.025em;
  color: {c['text']}; margin: 0 0 0.6rem; }}
.v3-title em {{ font-style: normal; color: {c['ours']}; }}
.stApp .v3-lede {{ font-size: 1.05rem !important; line-height: 1.5 !important; color: {c['muted']}; max-width: 54rem; margin: 0 0 1.4rem !important; }}
.v3-lede b {{ color: {c['text']}; font-weight: 600; }}
.v3-step {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: {c['ours']}; margin-bottom: 0.25rem; }}
.v3-h {{ font-size: 1.22rem; line-height: 1.2; font-weight: 650; letter-spacing: -0.015em; color: {c['text']}; margin: 0 0 0.35rem; }}
.stApp .v3-p {{ font-size: 0.92rem !important; line-height: 1.5 !important; color: {c['muted']}; margin: 0 0 0.8rem !important; }}
.v3-p b {{ color: {c['text']}; font-weight: 600; }}
.stApp .v3-note {{ font-size: 0.78rem !important; line-height: 1.45 !important; color: {c['faint']}; margin: 0.5rem 0 0 !important; }}
.v3-row {{ display: grid; grid-template-columns: 6.5rem minmax(0, 1fr); align-items: center; gap: 0.8rem; margin: 0.55rem 0; }}
.v3-row-label {{ font-size: 0.85rem; font-weight: 600; color: {c['text']}; }}
.v3-row-label span {{ display: block; font-weight: 400; font-size: 0.75rem; color: {c['muted']}; }}
.v3-bar {{ display: flex; height: 34px; border-radius: 9px; overflow: hidden; background: {c['raised']}; }}
.v3-seg {{ display: flex; align-items: center; justify-content: center; font-size: 0.78rem; font-weight: 600;
  color: #fff; white-space: nowrap; overflow: hidden; min-width: 0; font-variant-numeric: tabular-nums; }}
.v3-legend {{ display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; margin: 0.7rem 0 0.2rem; font-size: 0.8rem; color: {c['muted']}; }}
.v3-legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 0.4rem; vertical-align: 0; }}
.v3-list {{ margin: 0.9rem 0 0; border-top: 1px solid {c['line']}; }}
.v3-li {{ display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 0.9rem; padding: 0.55rem 0;
  border-bottom: 1px solid {c['line']}; font-size: 0.88rem; color: {c['text']}; align-items: baseline; }}
.v3-li span {{ color: {c['muted']}; font-variant-numeric: tabular-nums; }}
.v3-li b {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.v3-why {{ margin: 1rem 0 0.2rem; display: grid; gap: 0.6rem; }}
.stApp .v3-why > div {{ display: grid; grid-template-columns: 3px minmax(0, 1fr); gap: 0.75rem; font-size: 0.86rem;
  line-height: 1.45; color: {c['muted']}; }}
.v3-why i {{ border-radius: 3px; }}
.v3-why b {{ color: {c['text']}; font-weight: 600; }}
.v3-big {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; color: {c['text']}; font-variant-numeric: tabular-nums; }}
.v3-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: 0.7rem; margin: 0.3rem 0 0.4rem; }}
.v3-tile {{ background: {c['raised']}; border-radius: 13px; padding: 0.8rem 0.9rem; min-width: 0; }}
.v3-tile-label {{ font-size: 0.76rem; color: {c['muted']}; line-height: 1.3; margin-bottom: 0.45rem; }}
.v3-tile-value {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.015em; color: {c['text']}; font-variant-numeric: tabular-nums; }}
.v3-tile-sub {{ font-size: 0.76rem; color: {c['muted']}; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }}
.v3-change {{ display: grid; grid-template-columns: 1.6rem minmax(0, 1fr) auto; gap: 0.75rem; align-items: start;
  padding: 0.75rem 0; border-top: 1px solid {c['line']}; }}
.v3-num {{ width: 1.6rem; height: 1.6rem; border-radius: 50%; background: {c['raised']}; color: {c['text']};
  font-size: 0.8rem; font-weight: 700; display: flex; align-items: center; justify-content: center; }}
.v3-change-title {{ font-size: 0.95rem; font-weight: 600; color: {c['text']}; }}
.v3-change-text {{ font-size: 0.84rem; line-height: 1.45; color: {c['muted']}; margin-top: 0.15rem; }}
.v3-change-value {{ text-align: right; font-size: 1.02rem; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.v3-pill {{ display: inline-block; margin-top: 0.3rem; padding: 0.12rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }}
.v3-pill-good {{ background: color-mix(in srgb, {c['good']} 14%, transparent); color: {c['good']}; }}
.v3-pill-weak {{ background: {c['raised']}; color: {c['muted']}; }}
.v3-wait {{ padding: 2.2rem 0; text-align: center; color: {c['muted']}; font-size: 0.9rem; }}
.v3-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 0.3rem 0 1.2rem; }}
.v3-table th {{ text-align: right; font-weight: 600; color: {c['muted']}; font-size: 0.75rem; padding: 0.45rem 0.6rem;
  border-bottom: 1px solid {c['line']}; }}
.v3-table th:first-child, .v3-table td:first-child {{ text-align: left; padding-left: 0; }}
.v3-table td {{ text-align: right; padding: 0.45rem 0.6rem; border-bottom: 1px solid {c['line']}; color: {c['text']};
  font-variant-numeric: tabular-nums; }}
.v3-sub {{ font-size: 0.95rem; font-weight: 650; color: {c['text']}; margin: 0.6rem 0 0.2rem; }}
.stApp .v3-foot {{ font-size: 0.76rem !important; color: {c['faint']}; margin-top: 2rem; line-height: 1.5 !important; }}
.stApp .v3-side-note {{ font-size: 0.76rem !important; color: {c['faint']}; line-height: 1.45 !important; margin-top: 0.8rem; }}
[data-testid="stExpander"] details {{ border: 1px solid {c['line']}; border-radius: 18px; background: {c['card']}; }}
[data-testid="stExpander"] summary p {{ font-weight: 600; font-size: 1rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>"""
    return "\n".join(line for line in rules.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------

def brand(c):
    return f'<div class="v3-brand"><span class="v3-dot"></span>Beyond the Portfolio</div>'


def heading(step, title, text=None):
    out = f'<div class="v3-step">{esc(step)}</div><div class="v3-h">{esc(title)}</div>'
    return out + (f'<p class="v3-p">{text}</p>' if text else "")


def bar(c, label, sub, parts, show_values=True):
    """One horizontal bar: parts = [(share, colour, text)], shares add up to about 1."""
    segs = "".join(f'<div class="v3-seg" style="flex:{max(s, 0):.6f};background:{col}">'
                   f'{esc(t) if show_values and s >= 0.07 else ""}</div>' for s, col, t in parts if s > 0.0005)
    if not label:                                          # a bar alone, full width
        return f'<div class="v3-bar">{segs}</div>'
    return (f'<div class="v3-row"><div class="v3-row-label">{esc(label)}<span>{esc(sub)}</span></div>'
            f'<div class="v3-bar">{segs}</div></div>')


def legend(items):
    return '<div class="v3-legend">' + "".join(f'<span><i style="background:{col}"></i>{esc(t)}</span>'
                                               for col, t in items) + "</div>"


def rows(items):
    """[(name, detail, value)] as a quiet list."""
    return '<div class="v3-list">' + "".join(f'<div class="v3-li">{name}<span>{esc(detail)}</span><b>{esc(value)}</b></div>'
                                             for name, detail, value in items) + "</div>"


def why(items):
    """The reasons behind the plan, one per line: [(colour, html text)] (bold lead, then the reason)."""
    return '<div class="v3-why">' + "".join(f'<div><i style="background:{col}"></i><span>{text}</span></div>'
                                            for col, text in items) + "</div>"


def tiles(items):
    """[(label, value, sub)]"""
    return '<div class="v3-tiles">' + "".join(
        f'<div class="v3-tile"><div class="v3-tile-label">{esc(l)}</div><div class="v3-tile-value">{v}</div>'
        f'<div class="v3-tile-sub">{s}</div></div>' for l, v, s in items) + "</div>"


def change(number, title, text, value, value_color, reliable, chance):
    pill = (f'<span class="v3-pill v3-pill-good">Reliable: it helped in {chance}</span>' if reliable
            else f'<span class="v3-pill v3-pill-weak">Not proven: it helped in {chance}</span>')
    return (f'<div class="v3-change"><div class="v3-num">{number}</div><div><div class="v3-change-title">{esc(title)}</div>'
            f'<div class="v3-change-text">{text}</div>{pill}</div>'
            f'<div class="v3-change-value" style="color:{value_color}">{value}</div></div>')


def waiting(text):
    return f'<div class="v3-wait">{esc(text)}</div>'


def table(df, formats=None):
    """A DataFrame as a quiet HTML table; formats: {column: function}."""
    formats = formats or {}
    head = "<tr><th></th>" + "".join(f"<th>{esc(col)}</th>" for col in df.columns) + "</tr>"
    body = ""
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{esc(formats.get(col, str)(v))}</td>" for col, v in row.items())
        body += f"<tr><td>{esc(idx)}</td>{cells}</tr>"
    return f'<table class="v3-table">{head}{body}</table>'


def signed_chf(x):
    return ("+" if x >= 0 else "") + chf(x, short=False)


def chance(p):
    """p-value of the block bootstrap (share of replayed histories where the change did not help)
    in plain words: the share where it helped."""
    helped = 1 - p
    return "more than 99% of replayed histories" if helped > 0.99 else f"{helped:.0%} of replayed histories"


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

def plot_layout(c, height, **extra):
    layout = dict(height=height, margin=dict(l=4, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=c["muted"], size=12),
                  showlegend=False, hoverlabel=dict(bgcolor=c["card"], bordercolor=c["line"],
                                                    font=dict(color=c["text"], family=FONT)),
                  xaxis=dict(showgrid=False, zeroline=False, linecolor=c["line"], tickcolor=c["line"]),
                  yaxis=dict(showgrid=True, gridcolor=c["line"], zeroline=False))
    layout.update(extra)
    return layout


__all__ = ["chf", "pct", "palette", "streamlit_theme", "css", "brand", "heading", "bar", "legend", "rows",
           "tiles", "why", "change", "waiting", "table", "signed_chf", "chance", "plot_layout", "GROUPS", "group_color"]
