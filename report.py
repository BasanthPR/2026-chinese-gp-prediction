"""
2026 Chinese Grand Prix — F1 Prediction Report
Author: Basanth
LinkedIn-ready, 10-page professional report
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Layout constants ───────────────────────────────────────────────────────────
W, H         = letter                   # 612 × 792
LM           = 43                       # left margin
RM           = W - 43                   # right margin
CW           = RM - LM                  # content width = 526
HEADER_H     = 40
FOOTER_Y     = 52
Y_START      = H - HEADER_H - 18       # first content y = 734

# ── Colours ───────────────────────────────────────────────────────────────────
RED          = colors.HexColor("#CC0000")
DARK         = colors.HexColor("#1A1A1A")
MIDGREY      = colors.HexColor("#555555")
LIGHTGREY    = colors.HexColor("#888888")
ROW_ALT      = colors.HexColor("#F5F5F5")
ROW_WHITE    = colors.white
RULE_GREY    = colors.HexColor("#DDDDDD")
ACCENT_BLUE  = colors.HexColor("#003366")
GOLD         = colors.HexColor("#D4A017")

# ── Team colours ──────────────────────────────────────────────────────────────
TEAM_HEX = {
    "mercedes":     "#00D2BE", "ferrari":      "#DC0000",
    "mclaren":      "#FF8700", "red_bull":     "#0600EF",
    "alpine":       "#0090FF", "haas":         "#888888",
    "racing_bulls": "#1E3A5F", "williams":     "#005AFF",
    "aston_martin": "#006F62", "audi":         "#BB002B",
    "cadillac":     "#7B7B7B",
}

OUT  = Path(__file__).parent / "output"
IMGS = {
    "chart1": OUT / "1_stacked_win_probabilities.png",
    "chart2": OUT / "2_sprint_gp_correlation.png",
    "chart3": OUT / "3_monte_carlo_distribution.png",
    "chart4": OUT / "4_shap_heatmap.png",
    "chart5": OUT / "5_tire_strategy.png",
    "chart6": OUT / "6_scenario_matrix.png",
    "chart7": OUT / "7_championship_standings.png",
}

# ── Raw data ──────────────────────────────────────────────────────────────────
RESULTS = [
    # (rank, name, team, grid, sprint, xgb, mc, bayes, ensemble, ci)
    (1,  "George Russell",    "mercedes",     2,  1,  32.7, 18.0, 48.4, 28.6, "[39.8%, 56.0%]"),
    (2,  "Lewis Hamilton",    "ferrari",      3,  3,  31.6, 11.8, 17.7, 21.2, "[11.8%, 24.1%]"),
    (3,  "Kimi Antonelli",    "mercedes",     1,  5,  18.4,  7.2, 20.6, 12.8, "[14.2%, 27.2%]"),
    (4,  "Charles Leclerc",   "ferrari",      4,  2,   5.9, 19.2,  7.1, 12.5, "[3.5%, 11.7%]"),
    (5,  "Oscar Piastri",     "mclaren",      5,  6,   7.9,  8.7,  3.2,  7.1, "[0.9%, 6.6%]"),
    (6,  "Lando Norris",      "mclaren",      6,  4,   0.7, 13.6,  2.0,  7.0, "[0.4%, 4.7%]"),
    (7,  "Liam Lawson",       "racing_bulls", 12, 7,   0.0,  5.4,  0.1,  2.4, "—"),
    (8,  "Max Verstappen",    "red_bull",     8,  9,   1.0,  4.1,  0.4,  2.4, "[0.0%, 1.6%]"),
    (9,  "Ollie Bearman",     "haas",         10, 8,   0.0,  4.7,  0.1,  2.1, "—"),
    (10, "Esteban Ocon",      "haas",         13, 10,  0.0,  2.3,  0.0,  1.0, "—"),
]

SPRINT_RESULTS = [
    (1,  "George Russell",   "Mercedes",     "+0.000s",   "Wire-to-wire winner"),
    (2,  "Charles Leclerc",  "Ferrari",      "+0.674s",   "Charged from P6 — fastest lap"),
    (3,  "Lewis Hamilton",   "Ferrari",      "+2.554s",   "Briefly led lap 3"),
    (4,  "Lando Norris",     "McLaren",      "+4.433s",   "Solid, unflustered"),
    (5,  "Kimi Antonelli",   "Mercedes",     "+5.688s",   "+10s penalty, true pace was P2"),
    (6,  "Oscar Piastri",    "McLaren",      "+6.809s",   "Consistent"),
    (7,  "Liam Lawson",      "Racing Bulls", "+10.900s",  "Best of the rest"),
    (8,  "Ollie Bearman",    "Haas",         "+11.271s",  "Strong sprint debut"),
    (9,  "Max Verstappen",   "Red Bull",     "+11.619s",  "Fell to P16, recovered"),
    (10, "Esteban Ocon",     "Haas",         "+13.887s",  "Clean race"),
    ("DNF", "Nico Hülkenberg", "Audi",       "—",         "Triggered safety car"),
    ("DNF", "Valtteri Bottas", "Cadillac",   "—",         "Mechanical"),
    ("DNF", "Arvid Lindblad", "Racing Bulls","—",         "Incident"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ── Drawing helpers ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def header_band(c, page_num, title, subtitle=""):
    """Red top band with page number."""
    c.setFillColor(RED)
    c.rect(0, H - HEADER_H, W, HEADER_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(LM, H - 27, title)
    if subtitle:
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#FFAAAA"))
        c.drawString(LM, H - 39, subtitle)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 8)
    c.drawRightString(RM, H - 27, f"PAGE {page_num} / 10")
    c.drawRightString(RM, H - 39, "2026 CHINESE GRAND PRIX  ·  F1 PREDICTION SYSTEM V2")


def footer(c):
    """Footer rule + attribution."""
    c.setStrokeColor(RULE_GREY)
    c.setLineWidth(0.5)
    c.line(LM, FOOTER_Y - 4, RM, FOOTER_Y - 4)
    c.setFillColor(LIGHTGREY)
    c.setFont("Helvetica", 7)
    c.drawString(LM, FOOTER_Y - 16, "Built by Basanth  ·  github.com/BasanthPR/2026-chinese-gp-prediction  ·  XGBoost + Monte Carlo + Bayesian Ensemble")
    c.drawRightString(RM, FOOTER_Y - 16, "Shanghai International Circuit  ·  Round 2  ·  March 23 2026")


def section_heading(c, y, text, size=11):
    """Red heading with underline rule. Returns y after heading."""
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", size)
    c.drawString(LM, y, text.upper())
    c.setStrokeColor(RED)
    c.setLineWidth(0.8)
    c.line(LM, y - 3, RM, y - 3)
    return y - 16


def sub_heading(c, y, text, size=9.5):
    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", size)
    c.drawString(LM, y, text)
    return y - 13


def body_text(c, y, text, size=8.5, color=DARK, width=CW, indent=0):
    """Simple single-line body text. Returns y after line."""
    c.setFillColor(color)
    c.setFont("Helvetica", size)
    c.drawString(LM + indent, y, text)
    return y - (size + 3)


def wrapped_text(c, y, text, size=8.5, color=DARK, width=CW, line_h=12, indent=0):
    """Word-wrap text to content width. Returns final y."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    c.setFillColor(color)
    c.setFont("Helvetica", size)
    words = text.split()
    line, lines = [], []
    for w in words:
        test = " ".join(line + [w])
        if stringWidth(test, "Helvetica", size) <= width - indent:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        c.drawString(LM + indent, y, ln)
        y -= line_h
    return y


def bold_wrapped(c, y, label, text, size=8.5, width=CW, line_h=12):
    """Bold label + normal text, wrapped."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", size)
    c.drawString(LM, y, label)
    lw = stringWidth(label, "Helvetica-Bold", size) + 4
    c.setFont("Helvetica", size)
    # Continue on same line then wrap remainder
    words = text.split()
    line = []
    first = True
    for w in words:
        test = " ".join(line + [w])
        avail = width - (lw if first else 0)
        if stringWidth(test, "Helvetica", size) <= avail:
            line.append(w)
        else:
            segment = " ".join(line)
            if first:
                c.drawString(LM + lw, y, segment)
                first = False
            else:
                c.drawString(LM, y, segment)
            y -= line_h
            line = [w]
    if line:
        segment = " ".join(line)
        if first:
            c.drawString(LM + lw, y, segment)
        else:
            c.drawString(LM, y, segment)
    return y - line_h


def rule(c, y, color=RULE_GREY, width=0.5):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(LM, y, RM, y)
    return y - 6


def place_image(c, path, y, w=None, h=None, center=True):
    """Place image scaled to content width. Returns y below image."""
    if not Path(path).exists():
        return y - 20
    w = w or CW
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    scaled_h = h or (w * ih / iw)
    x = LM if not center else LM + (CW - w) / 2
    c.drawImage(str(path), x, y - scaled_h, width=w, height=scaled_h,
                preserveAspectRatio=True, mask="auto")
    return y - scaled_h - 6


def color_dot(c, x, y, hex_color, r=4):
    c.setFillColor(colors.HexColor(hex_color))
    c.circle(x, y + 2, r, fill=1, stroke=0)


def draw_table(c, y, data, col_widths, row_height=14, header_bg=RED,
               alt=True, font_size=8, header_font_size=8):
    """Draw a table manually. data[0] = header row. Returns y below table."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    x0 = LM
    for ri, row in enumerate(data):
        if y - row_height < FOOTER_Y:
            break
        # Row background
        if ri == 0:
            c.setFillColor(header_bg)
        elif alt and ri % 2 == 0:
            c.setFillColor(ROW_ALT)
        else:
            c.setFillColor(ROW_WHITE)
        c.rect(x0, y - row_height, sum(col_widths), row_height, fill=1, stroke=0)

        # Cell text
        cx = x0
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            cell_str = str(cell)
            if ri == 0:
                c.setFillColor(colors.white)
                c.setFont("Helvetica-Bold", header_font_size)
            else:
                c.setFillColor(DARK)
                c.setFont("Helvetica", font_size)
            c.drawString(cx + 4, y - row_height + 4, cell_str)
            cx += cw

        # Row border
        c.setStrokeColor(RULE_GREY)
        c.setLineWidth(0.3)
        c.rect(x0, y - row_height, sum(col_widths), row_height, fill=0, stroke=1)
        y -= row_height
    return y - 4


def winner_box(c, y, name, team, prob, ci):
    """Big predicted winner highlight box."""
    box_h = 52
    c.setFillColor(RED)
    c.roundRect(LM, y - box_h, CW, box_h, 6, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 8)
    c.drawString(LM + 16, y - 13, "PREDICTED RACE WINNER")
    c.setFont("Helvetica-Bold", 20)
    c.drawString(LM + 16, y - 32, name)
    c.setFont("Helvetica-Bold", 13)
    tc = colors.HexColor(TEAM_HEX.get(team, "#FFFFFF"))
    c.setFillColor(tc)
    c.drawString(LM + 16, y - 46, f"{prob}%  WIN PROBABILITY")
    c.setFillColor(colors.HexColor("#FFDDDD"))
    c.setFont("Helvetica", 8)
    c.drawRightString(RM - 16, y - 46, f"90% CI {ci}")
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 8)
    c.drawRightString(RM - 16, y - 13, "XGBoost 30%  ·  Monte Carlo 45%  ·  Bayesian 25%")
    return y - box_h - 10


def stat_box(c, x, y, w, h, label, value, sub="", bg=colors.HexColor("#F9F9F9")):
    c.setFillColor(bg)
    c.roundRect(x, y - h, w, h, 4, fill=1, stroke=0)
    c.setStrokeColor(RULE_GREY)
    c.setLineWidth(0.5)
    c.roundRect(x, y - h, w, h, 4, fill=0, stroke=1)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(x + w/2, y - h + 22, value)
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica", 7)
    c.drawCentredString(x + w/2, y - h + 13, label)
    if sub:
        c.setFillColor(LIGHTGREY)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(x + w/2, y - h + 5, sub)


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 1 — COVER ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page1(c):
    header_band(c, 1, "F1 RACE WINNER PREDICTION  ·  SYSTEM V2", "Sprint-Calibrated Ensemble  ·  2026 Season")

    y = Y_START

    # Title block
    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(LM, y, "2026 CHINESE GRAND PRIX")
    y -= 26
    c.setFont("Helvetica", 10)
    c.setFillColor(MIDGREY)
    c.drawString(LM, y, "Shanghai International Circuit  ·  Round 2  ·  March 23, 2026  ·  Sprint Weekend")
    y -= 20
    rule(c, y, RED, 1.5)
    y -= 10

    # Winner box
    y = winner_box(c, y, "George Russell", "mercedes", 28.6, "[39.8%, 56.0%]")

    # Stat boxes row
    bw, bh = 120, 44
    gap = (CW - 4 * bw) / 3
    stat_box(c, LM,                    y, bw, bh, "MODELS USED",    "3",      "XGB · MC · Bayesian")
    stat_box(c, LM + bw + gap,         y, bw, bh, "SIMULATIONS",    "10,000", "Monte Carlo races")
    stat_box(c, LM + 2*(bw+gap),       y, bw, bh, "TRAINING ROWS",  "1,600",  "Races 2010–2025")
    stat_box(c, LM + 3*(bw+gap),       y, bw, bh, "XGB CV AUC",     "0.911",  "GroupKFold ±0.036")
    y -= bh + 14

    # Top 10 table
    y = section_heading(c, y, "Top 10 Predicted Finish Order")

    header = ["#", "Driver", "Team", "Grid", "Sprint", "XGB%", "MC%", "Bayes%", "Ensemble", "90% CI"]
    col_w  = [18, 112, 78, 30, 38, 36, 36, 42, 50, 72]
    rows   = [header]
    for r in RESULTS:
        rk, nm, tm, gd, sp, xg, mc, by, en, ci = r
        rows.append([
            str(rk), nm, tm.replace("_"," ").title(),
            f"P{gd}", f"P{sp}", f"{xg:.1f}%", f"{mc:.1f}%",
            f"{by:.1f}%", f"{en:.1f}%", ci
        ])
    y = draw_table(c, y, rows, col_w, row_height=13, font_size=7.5, header_font_size=7.5)

    # Team colour legend
    y -= 6
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(MIDGREY)
    c.drawString(LM, y, "TEAM LEGEND:  ")
    x_leg = LM + 72
    for team_name, hex_c in list(TEAM_HEX.items())[:6]:
        color_dot(c, x_leg, y, hex_c, 4)
        c.setFillColor(MIDGREY)
        c.setFont("Helvetica", 7)
        label = team_name.replace("_"," ").title()
        c.drawString(x_leg + 8, y, label)
        x_leg += 68
    y -= 14

    # Note on model divergence
    c.setFillColor(colors.HexColor("#FFF3F3"))
    c.roundRect(LM, y - 26, CW, 26, 4, fill=1, stroke=0)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(LM + 8, y - 10, "KEY DIVERGENCE:")
    c.setFillColor(DARK)
    c.setFont("Helvetica", 8)
    c.drawString(LM + 98, y - 10, "Bayesian gives Russell 48.4% (pole win history + sprint winner boost). Monte Carlo gives Leclerc 19.2% — sprint fastest lap + race-craft.")
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica", 7.5)
    c.drawString(LM + 8, y - 21, "Ensemble blends both signals. Where models disagree sharply, confidence intervals widen — see page 8.")

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 2 — RACE CONTEXT + CHART 1 ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page2(c):
    header_band(c, 2, "RACE CONTEXT  ·  QUALIFYING GRID & CONDITIONS", "Shanghai International Circuit")
    y = Y_START

    # Two columns
    col_l = LM
    col_r = LM + CW // 2 + 8
    cw2   = CW // 2 - 8

    # Left column: qualifying grid
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(col_l, y, "GP QUALIFYING GRID")
    c.setStrokeColor(RED)
    c.setLineWidth(0.6)
    c.line(col_l, y - 3, col_l + cw2, y - 3)
    y_l = y - 16

    quali_grid = [
        ("P1", "Kimi Antonelli",  "mercedes",    "Youngest ever GP polesitter — age 19"),
        ("P2", "George Russell",  "mercedes",    "Q3 power issue — set single lap only"),
        ("P3", "Lewis Hamilton",  "ferrari",     "FP1 P6 → Quali P3: divergence = +3"),
        ("P4", "Charles Leclerc", "ferrari",     "Sprint P2, fastest lap"),
        ("P5", "Oscar Piastri",   "mclaren",     "Consistent McLaren front"),
        ("P6", "Lando Norris",    "mclaren",     "Sprint P4 — pace solid"),
        ("P7", "Pierre Gasly",    "alpine",      "Best of the midfield"),
        ("P8", "Max Verstappen",  "red_bull",    "Fell to P16 in sprint, recovered"),
        ("P9", "Isack Hadjar",    "red_bull",    "Rookie learning weekend"),
        ("P10","Ollie Bearman",   "haas",        "FP1 oversteer incidents"),
    ]
    for pos, name, team, note in quali_grid:
        hex_c = TEAM_HEX.get(team, "#888888")
        c.setFillColor(colors.HexColor(hex_c))
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_l, y_l, pos)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_l + 22, y_l, name)
        c.setFillColor(MIDGREY)
        c.setFont("Helvetica", 7)
        c.drawString(col_l + 22, y_l - 9, note)
        y_l -= 20

    # Williams/Aston note
    c.setFillColor(LIGHTGREY)
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(col_l, y_l, "Williams & Aston Martin eliminated in Q1")
    y_l -= 10

    # Right column: conditions
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(col_r, y, "RACE CONDITIONS — SUNDAY")
    c.setStrokeColor(RED)
    c.setLineWidth(0.6)
    c.line(col_r, y - 3, col_r + cw2, y - 3)
    y_r = y - 16

    conditions = [
        ("Air Temp",        "16–19°C"),
        ("Track Temp",      "~33°C"),
        ("Rain Probability","25% peak — modelled"),
        ("Wind",            "ENE 11 km/h, gusts to 36 km/h"),
        ("Cloud Cover",     "55%"),
        ("Safety Car Rate", "56% (sprint-adjusted)"),
        ("Circuit Length",  "5.451 km, 16 corners"),
        ("Race Distance",   "56 laps / 305 km"),
        ("DRS Zones",       "2 (main straight + T13–T14)"),
        ("Tire Compounds",  "Soft / Medium / Hard"),
    ]
    for label, val in conditions:
        c.setFillColor(MIDGREY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_r, y_r, label + ":")
        c.setFillColor(DARK)
        c.setFont("Helvetica", 8)
        c.drawString(col_r + 100, y_r, val)
        y_r -= 14

    # Separator rule
    y = min(y_l, y_r) - 10
    rule(c, y, RED, 0.8)
    y -= 8

    # Chart 1 — fill remaining space
    y = section_heading(c, y, "Ensemble Win Probability — All Drivers")
    img_h = max(160, y - FOOTER_Y - 28)
    y = place_image(c, IMGS["chart1"], y, w=CW, h=img_h)

    y -= 4
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, "Chart 1 — Win probability stacked by model contribution. Russell leads overall; Leclerc's Monte Carlo signal (19.2%) is the strongest non-Bayesian signal in the field.")

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 3 — HOW IT WORKS ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page3(c):
    header_band(c, 3, "HOW IT WORKS  ·  DATA PIPELINE & FEATURE ENGINEERING", "V2 Architecture")
    y = Y_START

    # Intro paragraph
    c.setFillColor(DARK)
    c.setFont("Helvetica", 8.5)
    intro = ("Think of predicting an F1 race like picking a horse race — but instead of gut feel, "
             "you feed every relevant signal into a machine that has watched 1,600 races. "
             "The model doesn't know what a tyre is, but it knows that when a driver who was "
             "fastest in practice qualifies lower than expected, they tend to race better than their grid slot suggests. "
             "That was the exact insight the Australia model missed on Lewis Hamilton — and V2 fixes it.")
    y = wrapped_text(c, y, intro, size=8.5, line_h=12)
    y -= 8

    # Data pipeline
    y = section_heading(c, y, "Data Pipeline")

    pipeline = [
        ("Jolpica-F1 API",   "Historical race & qualifying results, 2010–2025. 1,600 rows, cached locally as Parquet."),
        ("OpenF1 API",       "Real-time session data — lap times, pit stops, car telemetry, weather."),
        ("FastF1 Library",   "Official F1 data wrapper. Used for tire compounds, sector times, stint analysis."),
        ("Sprint Data",      "2026 China sprint result hardcoded as ground truth. This is the key V2 upgrade."),
    ]
    for src, desc in pipeline:
        c.setFillColor(RED)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(LM, y, f"  {src}:")
        c.setFillColor(DARK)
        c.setFont("Helvetica", 8)
        c.drawString(LM + 108, y, desc)
        y -= 12
    y -= 4

    # Feature tiers — visual boxes
    y = section_heading(c, y, "4-Tier Feature Engineering")

    tier_desc = [
        ("TIER 0", "SPRINT RACE", "#CC0000", "white",
         "The V2 breakthrough. Sprint finishing position, per-lap pace gap to Russell, positions gained, "
         "tyre stress flags, reliability flags. This tier carries the highest predictive weight because it's "
         "real race data from the same circuit, same weekend, same cars. The model knows Hamilton ran at "
         "P2-level pace in the sprint. It knows Leclerc charged from P6 to P2. It knows Antonelli's true "
         "pace was masked by a 10-second penalty."),
        ("TIER 1", "QUALIFYING & PRACTICE", "#003366", "white",
         "Grid position, qualifying gap to pole, FP1 pace delta. The key V2 addition: fp1_to_quali_divergence "
         "— the difference between FP1 classification and qualifying position. Hamilton: FP1 P6, Quali P3, "
         "divergence = +3. Large positive divergence means a driver improves toward race pace, a signal V1 "
         "completely missed."),
        ("TIER 2", "DRIVER & CONSTRUCTOR", "#1A1A1A", "white",
         "ELO ratings (updated through Australia + sprint), constructor strength (rolling), Shanghai circuit "
         "history (Hamilton: 6 wins, Verstappen: 1, Piastri: 1), teammate qualifying gap. Antonelli beat "
         "Russell by 0.222s in qualifying — encoded as a positive teammate gap signal."),
        ("TIER 3/4", "REGULATION ERA & FLAGS", "#555555", "white",
         "Year-1 regulation flag (3× DNF multiplier — V2 upgrade from 2×). Energy management advantage "
         "per constructor: Mercedes +0.0s, Ferrari −0.05s, McLaren −0.08s, Red Bull −0.12s on straights. "
         "Ferrari flip-flop wing uncertainty. Antonelli pressure factor (28% historical win rate for "
         "under-22 polesitters, 22% retirement rate). Hamilton energy depletion flag from sprint."),
    ]

    from reportlab.pdfbase.pdfmetrics import stringWidth
    box_h_tier = 86
    for tier, name, bg, fg, desc in tier_desc:
        if y - box_h_tier < FOOTER_Y:
            break
        # Background box
        c.setFillColor(colors.HexColor(bg))
        c.roundRect(LM, y - box_h_tier, CW, box_h_tier, 4, fill=1, stroke=0)
        # Left accent stripe
        c.setFillColor(RED if bg not in ("#CC0000",) else colors.HexColor("#990000"))
        c.rect(LM, y - box_h_tier, 4, box_h_tier, fill=1, stroke=0)
        # Tier badge
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(LM + 10, y - 13, tier)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(LM + 58, y - 13, name)
        # Description — wrap to 5 lines
        fg_c = colors.HexColor("#E8E8E8") if bg not in ("#F9F9F9", "white") else DARK
        c.setFillColor(fg_c)
        c.setFont("Helvetica", 7.8)
        words = desc.split()
        line, lines_t = [], []
        for w in words:
            test = " ".join(line + [w])
            if stringWidth(test, "Helvetica", 7.8) <= CW - 20:
                line.append(w)
            else:
                lines_t.append(" ".join(line))
                line = [w]
        if line:
            lines_t.append(" ".join(line))
        ty = y - 26
        for ln in lines_t[:5]:
            c.drawString(LM + 10, ty, ln)
            ty -= 11
        y -= box_h_tier + 5

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 4 — XGBOOST ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page4(c):
    header_band(c, 4, "MODEL 1  ·  XGBOOST CLASSIFIER", "Gradient-Boosted Trees — 1,600 Historical Races")
    y = Y_START

    y = section_heading(c, y, "What XGBoost Does Here")
    txt1 = ("XGBoost is like building 300 racing analysts, each slightly different, then asking them all to vote. "
            "Each analyst (decision tree) looks at a slice of the data and asks a series of yes/no questions: "
            "'Is the driver starting from the front row? Does their constructor win more than 80% of races? "
            "Did they qualify significantly better than their practice sessions suggested?' "
            "The final answer is a weighted vote across all 300 analysts.")
    y = wrapped_text(c, y, txt1, size=8.5, line_h=12)
    y -= 6
    txt2 = ("Training on 1,600 races from 2010 to 2025 taught it patterns that hold across eras: "
            "grid position is king, but a driver starting P3 who was fastest in practice is more likely to win than "
            "the raw grid position suggests. GroupKFold cross-validation means the model was tested on races it "
            "had never seen — a clean, honest measure of accuracy.")
    y = wrapped_text(c, y, txt2, size=8.5, line_h=12)
    y -= 8

    # Config box
    c.setFillColor(colors.HexColor("#F5F5F5"))
    c.roundRect(LM, y - 56, CW, 56, 4, fill=1, stroke=0)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(LM + 8, y - 12, "MODEL CONFIGURATION")
    configs = [
        ("n_estimators", "300"),   ("max_depth", "5"),
        ("learning_rate","0.05"),  ("subsample", "0.8"),
        ("colsample_bytree","0.8"),("scale_pos_weight","21  (1 winner per 22-car grid)"),
        ("eval_metric","AUC"),     ("CV AUC","0.911 ± 0.036  (GroupKFold, 5 folds)"),
    ]
    cx, cy2 = LM + 8, y - 24
    for i, (k, v) in enumerate(configs):
        col = LM + 8 + (i % 2) * (CW // 2)
        row_y = cy2 - (i // 2) * 11
        c.setFillColor(MIDGREY)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(col, row_y, k + ":")
        c.setFillColor(DARK)
        c.setFont("Helvetica", 7.5)
        c.drawString(col + 90, row_y, v)
    y -= 62

    y = section_heading(c, y, "XGBoost Raw Output — Top 10")
    xgb_header = ["#", "Driver", "Team", "Grid", "Sprint", "XGB Raw %", "Note"]
    xgb_cw     = [18, 118, 82, 28, 38, 56, 170]
    xgb_notes  = {
        "russell":   "Sprint winner + Bayesian grid boost",
        "hamilton":  "fp1_to_quali_divergence correction lifts this signal",
        "antonelli": "Pole position boosts ELO + grid features",
        "leclerc":   "Grid P4 limits XGB; MC compensates via race-craft",
        "piastri":   "Consistent mid-pack signal, no standout features",
        "norris":    "MC sees his race pace; XGB sees P6 grid — diverges",
    }
    xgb_rows = [xgb_header]
    for rk, nm, tm, gd, sp, xg, mc, by, en, ci in RESULTS:
        driver_id = nm.lower().split()[-1]
        note = xgb_notes.get(driver_id, "—")
        xgb_rows.append([str(rk), nm, tm.replace("_"," ").title(),
                         f"P{gd}", f"P{sp}", f"{xg:.1f}%", note])
    y = draw_table(c, y, xgb_rows, xgb_cw, row_height=13, font_size=7.5, header_font_size=7.5)
    y -= 8

    # Key insight box
    c.setFillColor(colors.HexColor("#FFF3F3"))
    c.roundRect(LM, y - 44, CW, 44, 4, fill=1, stroke=0)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(LM + 8, y - 13, "KEY INSIGHT — WHERE XGB DIVERGES FROM MC")
    c.setFillColor(DARK)
    c.setFont("Helvetica", 8)
    c.drawString(LM + 8, y - 25, "Norris: XGB gives 0.7% (sees P6 grid + modest historical win rate). Monte Carlo gives 13.6% (sprint pace calibration shows")
    c.drawString(LM + 8, y - 36, "strong race pace vs field). This is exactly why MC carries 45% weight — it sees pace signals the historical model cannot.")
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM + 8, y - 47, "Ensemble resolves this to 7.0% — a sensible middle ground that values both signals.")

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 5 — SHAP HEATMAP ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page5(c):
    header_band(c, 5, "FEATURE IMPORTANCE  ·  SHAP VALUE ANALYSIS", "What Actually Drives the Prediction")
    y = Y_START

    y = section_heading(c, y, "Reading the SHAP Heatmap")
    txt = ("SHAP — SHapley Additive exPlanations — tells us exactly how much each feature pushed a specific "
           "driver's win probability up or down. Think of it as a forensic breakdown: rather than just saying "
           "'Hamilton is rated at 21.2%', SHAP tells you how much of that came from his Shanghai circuit history, "
           "how much from his qualifying gap, and how much from the sprint race result. "
           "Green cells mean the feature helped. Red cells mean it hurt. The magnitude shows how much.")
    y = wrapped_text(c, y, txt, size=8.5, line_h=12)
    y -= 8

    # SHAP chart
    img_h = 190
    if y - img_h > FOOTER_Y + 30:
        y = place_image(c, IMGS["chart4"], y, w=CW, h=img_h)
    else:
        img_h = y - FOOTER_Y - 40
        y = place_image(c, IMGS["chart4"], y, w=CW, h=img_h)

    y -= 4
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, "Chart 4 — SHAP feature importance heatmap. Rows = feature tiers. Columns = top 10 drivers.")
    y -= 14

    y = section_heading(c, y, "Key Feature Findings")

    findings = [
        ("Sprint Tier 0 leads for frontrunners.",
         "For Russell, Hamilton, and Leclerc, the sprint finishing position and pace gap are the "
         "single strongest predictors — more impactful than grid position alone. The model learned "
         "this from 2021-2025: if you won or podiumed the sprint, you're significantly more likely "
         "to win on Sunday."),
        ("Circuit history is Hamilton's trump card.",
         "With 6 Shanghai wins, his Tier 2 circuit history SHAP value is the highest of any driver "
         "in the field. The Bayesian model in particular gives this heavy weight — it explains why "
         "Bayesian rates Hamilton at 17.7% despite his P3 grid."),
        ("Grid position still matters, but less on sprint weekends.",
         "In Australia (no sprint), grid position was the dominant Tier 1 signal. In China, "
         "sprint pace overrides it for the top 4. For P8+ drivers, grid is still decisive "
         "— hence the 5% cap applied post-ensemble."),
        ("Reliability risk hurts Russell and Antonelli most.",
         "Both have explicit reliability flags this weekend. The feature pushes their raw "
         "probabilities down before the ensemble even runs, then the post-ensemble multiplier "
         "(×0.92 each) applies a further haircut."),
    ]

    for title, detail in findings:
        if y < FOOTER_Y + 30:
            break
        c.setFillColor(RED)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(LM, y, f"  {title}")
        y -= 12
        y = wrapped_text(c, y, detail, size=8, line_h=11, indent=12)
        y -= 6

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 6 — MONTE CARLO ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page6(c):
    header_band(c, 6, "MODEL 2  ·  MONTE CARLO RACE SIMULATOR", "10,000 × 56-Lap Sprint-Calibrated Races")
    y = Y_START

    y = section_heading(c, y, "How the Simulator Works")
    txt1 = ("The Monte Carlo simulator is the closest thing to actually running the race 10,000 times. "
            "Each simulation plays out all 56 laps at Shanghai: drivers accelerate down the 1.175km back "
            "straight, battle through the Turn 1 complex that destroys front tyres, pit under safety cars, "
            "and occasionally retire with failures. No two simulations are identical — each draws from "
            "distributions of pace, degradation, safety car timing, and weather.")
    y = wrapped_text(c, y, txt1, size=8.5, line_h=12)
    y -= 6

    # Simulator components in two columns
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(LM, y, "V2 SIMULATOR COMPONENTS")
    c.setStrokeColor(RED); c.setLineWidth(0.6)
    c.line(LM, y - 3, RM, y - 3)
    y -= 14

    mc_comps = [
        ("Sprint-Calibrated Pace Matrix",
         "Russell = 0.0 baseline. Every driver's race pace derived from sprint lap deltas: Leclerc +0.04s/lap, "
         "Hamilton +0.13s, Norris +0.23s. Year-1 uncertainty adds ±0.30s/lap noise (V1 used ±0.20)."),
        ("Tyre Degradation Model",
         "Sprint confirmed Shanghai's front-left dominance. Soft: 15–18 lap window. Medium: 25–30. "
         "Hard: 35–45. Hamilton and aggressive starters get 15% higher FL degradation rate."),
        ("DNF Probability — 3× Multiplier",
         "Base rate 0.5%/lap × 3× Year-1 multiplier = 1.5%/lap. Russell: +0.5% additional. "
         "Antonelli: +0.3% additional. Per race this means ~20-25% DNF probability per car."),
        ("Safety Car Model",
         "56% race probability (historical 45% × 1.2 sprint multiplier). SC duration: U(3,6) laps. "
         "Leaders lose positional advantage. SC pit window triggered for nearby stop-window drivers."),
        ("Weather Model",
         "25% rain onset during race window. Rain lap sampled uniformly across laps 5–45. "
         "If rain triggers: Hamilton +45% pace boost, Antonelli −10%. SC probability jumps to 80%."),
        ("Energy Management",
         "Mercedes: 0.0 reference. Ferrari: −0.05s/lap on straights. McLaren: −0.08. "
         "Red Bull: −0.12. Shanghai's 1.175km back straight amplifies this gap significantly."),
    ]

    col_l_x = LM; col_r_x = LM + CW // 2 + 6
    cw2 = CW // 2 - 8
    y_l = y; y_r = y
    for i, (title, desc) in enumerate(mc_comps):
        col_x = col_l_x if i % 2 == 0 else col_r_x
        y_col = y_l if i % 2 == 0 else y_r
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_x, y_col, title)
        y_col -= 11
        # Wrap desc to column width
        from reportlab.pdfbase.pdfmetrics import stringWidth
        words = desc.split()
        line_t, lines_t = [], []
        for w in words:
            test_s = " ".join(line_t + [w])
            if stringWidth(test_s, "Helvetica", 7.5) <= cw2:
                line_t.append(w)
            else:
                lines_t.append(" ".join(line_t))
                line_t = [w]
        if line_t:
            lines_t.append(" ".join(line_t))
        c.setFillColor(MIDGREY)
        c.setFont("Helvetica", 7.5)
        for ln in lines_t[:3]:
            c.drawString(col_x, y_col, ln)
            y_col -= 10
        y_col -= 4
        if i % 2 == 0:
            y_l = y_col
        else:
            y_r = y_col

    y = min(y_l, y_r) - 6

    # MC Results table
    y = section_heading(c, y, "Monte Carlo Output — Top 10")
    mc_header = ["#", "Driver", "MC Win%", "MC Podium%", "MC DNF%", "Note"]
    mc_cw     = [18, 130, 60, 70, 55, 180]
    mc_data = [
        (1, "George Russell",   "18.0%", "52.1%", "18.3%", "Sprint winner — dominant dry race pace"),
        (2, "Charles Leclerc",  "19.2%", "48.3%", "12.1%", "Highest MC share — sprint race-craft signal"),
        (3, "Lando Norris",     "13.6%", "39.2%", "14.2%", "Sprint pace shows real speed vs XGB guess"),
        (4, "Lewis Hamilton",   "11.8%", "36.4%", "15.0%", "Hamilton tyre depletion → pace ceiling laps 30+"),
        (5, "Oscar Piastri",    " 8.7%", "30.1%", "13.8%", "Consistent, clean races"),
        (6, "Max Verstappen",   " 4.1%", "18.2%", "14.2%", "Red Bull energy deficit on straights"),
        (7, "Kimi Antonelli",   " 7.2%", "26.0%", "21.6%", "High DNF — reliability flag applied"),
        (8, "Ollie Bearman",    " 4.7%", "14.0%", "15.2%", "FP1 oversteer — tyre stress modelled"),
        (9, "Liam Lawson",      " 5.4%", "16.3%", "14.1%", "Sprint P7 — best of the rest"),
        (10,"Esteban Ocon",     " 2.3%", " 9.0%", "13.8%", "Haas midfield consistency"),
    ]
    mc_rows = [mc_header] + [[str(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in mc_data]
    y = draw_table(c, y, mc_rows, mc_cw, row_height=13, font_size=7.5, header_font_size=7.5)
    y -= 6
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, f"2,491 of 10,000 simulations (24.9%) included a rain onset event — matching the 25% forecast probability.")

    y -= 12
    # MC chart - compact
    img_h_mc = min(130, y - FOOTER_Y - 20)
    y = place_image(c, IMGS["chart3"], y, w=CW, h=img_h_mc)
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, "Chart 3 — Distribution of race winners across 10,000 Monte Carlo simulations.")

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 7 — SPRINT ANALYSIS + BAYESIAN ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page7(c):
    header_band(c, 7, "MODEL 3  ·  SPRINT CORRELATION & BAYESIAN INFERENCE", "Three-Stage Posterior Update")
    y = Y_START

    # Sprint-GP correlation chart (compact)
    y = section_heading(c, y, "Sprint Race → GP Outcome: What the Data Says")
    img_h2 = 145
    y = place_image(c, IMGS["chart2"], y, w=CW, h=img_h2)
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, "Chart 2 — Left: historical sprint-to-GP win correlation 2021–2025. Right: 2026 China sprint vs GP grid positions.")
    y -= 14

    txt_sprint = ("The left chart tells a clear story: sprint winners convert to GP wins at a 38% rate historically. "
                  "P2 finishers convert at 18%. By P5, you're down to 6%. The right scatter shows this weekend's "
                  "mismatch — Russell won the sprint from P1 SQ but starts P2 in the GP. Leclerc finished sprint P2 from "
                  "SQ P6 but qualifies GP P4. That gap between sprint performance and qualifying position is where the Bayesian model "
                  "finds its edge.")
    y = wrapped_text(c, y, txt_sprint, size=8.5, line_h=12)
    y -= 10

    # Bayesian model
    y = section_heading(c, y, "Bayesian Inference — 3-Stage Update")

    txt_bayes = ("Bayesian inference works like a detective building a case. You start with what you already know — "
                 "Hamilton has 6 Shanghai wins, making him a serious prior contender. Then you update that belief "
                 "as new evidence arrives. V2 adds a brand-new second stage: the sprint result itself. "
                 "A sprint win isn't just a points bonus. It's probabilistic evidence that tells you "
                 "something about Sunday.")
    y = wrapped_text(c, y, txt_bayes, size=8.5, line_h=12)
    y -= 6

    # Stage outputs table
    stage_header = ["Stage", "Evidence Used", "Russell", "Hamilton", "Antonelli", "Leclerc"]
    stage_cw = [60, 150, 60, 60, 60, 60]
    stage_rows = [
        stage_header,
        ["Stage 1\nPrior", "Shanghai wins 2010–2025\n+ constructor ELO pseudo-counts",
         "3.2%", "15.5%", "2.1%", "4.8%"],
        ["Stage 2\nSprint", "Sprint finishing position\nto GP win correlation (2021–25)",
         "33.5%", "24.4%", "8.1%", "12.4%"],
        ["Stage 3\nGrid", "Historical P(win|pole)\nat Shanghai: 46.7%",
         "48.4%", "17.7%", "20.7%", "7.1%"],
    ]
    y = draw_table(c, y, stage_rows, stage_cw, row_height=18, font_size=7.5, header_font_size=7.5)
    y -= 8

    # Annotation boxes
    annotations = [
        ("Hamilton's Prior Advantage",
         "15.5% prior (stage 1) reflects his 6 Shanghai wins — the strongest circuit-history prior in the field. "
         "After the sprint update, it drops to 24.4% — lower than Russell because Hamilton only finished sprint P3. "
         "After grid position (P3), the model settles at 17.7%."),
        ("Antonelli's Volatility",
         "Stage 1 prior is just 2.1% — no Shanghai history. Sprint update (P5 finish, masked by penalty) gives "
         "a modest 8.1%. But pole position (stage 3) rockets him to 20.7% via historical P(win|pole) at Shanghai = 46.7%. "
         "Under-22 pole variance (28% win, 22% retire) applies as a calibration."),
        ("Leclerc's Sprint Evidence",
         "Sprint P2 (from P6 grid) provides strong evidence — 15% GP win correlation. "
         "The Bayesian model sees 12.4% after sprint, then grid P4 moderates this to 7.1%. "
         "Monte Carlo's 19.2% comes from sprint pace. The tension between 7.1% and 19.2% "
         "is exactly what creates a wider confidence interval for Leclerc."),
    ]
    for title, detail in annotations:
        if y < FOOTER_Y + 28:
            break
        c.setFillColor(RED)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(LM, y, f"  {title}")
        y -= 11
        y = wrapped_text(c, y, detail, size=8, line_h=11, indent=10)
        y -= 5

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 8 — ENSEMBLE ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page8(c):
    header_band(c, 8, "THE ENSEMBLE  ·  WHERE MODELS MEET", "Blending Three Signals Into One Prediction")
    y = Y_START

    y = section_heading(c, y, "Why These Weights?")
    txt = ("No single model gets it right every time. XGBoost is brilliant at learning what historically predicts "
           "race wins, but it has never seen a 2026 car. The Monte Carlo simulator accounts for pace, strategy, "
           "weather, and luck — but its sprint calibration is only as good as 19 laps of data. Bayesian inference "
           "captures deep historical knowledge but can be slow to update when a new name (Antonelli) sits on pole. "
           "The ensemble is built to let each model speak where it's strongest.")
    y = wrapped_text(c, y, txt, size=8.5, line_h=12)
    y -= 8

    # Weights rationale box
    c.setFillColor(colors.HexColor("#F5F5F5"))
    c.roundRect(LM, y - 52, CW, 52, 4, fill=1, stroke=0)
    weights_data = [
        ("Monte Carlo", "45%", "#FFE66D", "Sprint-calibrated pace data makes this the most accurate signal of the three. "
                                           "We can observe what every car actually did over 19 race laps."),
        ("XGBoost",     "30%", "#4ECDC4", "Trained on 1,600 races. Reliable for historical patterns, "
                                           "grid position signals, and ELO-based driver quality."),
        ("Bayesian",    "25%", "#FF6B9D", "Strongest on drivers with Shanghai history (Hamilton). "
                                           "Provides a principled uncertainty framework via credible intervals."),
    ]
    wy = y - 12
    for model, pct, hex_c, desc in weights_data:
        c.setFillColor(colors.HexColor(hex_c))
        c.rect(LM + 6, wy - 4, 8, 8, fill=1, stroke=0)
        c.setFillColor(RED)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(LM + 20, wy, f"{model}  {pct}")
        c.setFillColor(MIDGREY)
        c.setFont("Helvetica", 7.5)
        c.drawString(LM + 20 + 90, wy, desc)
        wy -= 16
    y -= 58

    # Post-ensemble adjustments table
    y = section_heading(c, y, "Post-Ensemble Adjustments")
    adj_header = ["Driver", "Adjustment", "Reason", "Impact"]
    adj_cw = [110, 80, 190, 136]
    adj_rows = [
        adj_header,
        ["George Russell",   "× 0.92 reliability", "Q3 power/gear issue in qualifying",             "28.6% → effective 26.3% raw (before norm.)"],
        ["Kimi Antonelli",   "× 0.92 reliability", "Sprint collision penalty + crash tendency",      "12.8% → effective 11.8% raw"],
        ["Ferrari drivers",  "± 5% CI expansion",  "Flip-flop rear wing uncertainty (race spec?)",  "Wider confidence intervals only"],
        ["P8+ drivers",      "Capped at 5%",        "Sprint gain ≤ 3 positions override applies",    "Lawson, Bearman, Ocon etc. capped"],
        ["25% wet blend",    "P_final = 0.75×dry +", "25% rain probability during race window",      "Hamilton/Verstappen boosted in wet branch"],
        ["",                 "0.25×wet ensemble",    "",                                              ""],
    ]
    y = draw_table(c, y, adj_rows, adj_cw, row_height=14, font_size=7.5, header_font_size=7.5)
    y -= 8

    # Full ensemble table with per-model scores
    y = section_heading(c, y, "Full 22-Driver Ensemble — All Model Scores")
    ens_header = ["Rank", "Driver", "Grid", "Sprint", "XGB%", "MC%", "Bayes%", "Ensemble%", "90% CI"]
    ens_cw     = [28, 122, 28, 38, 38, 38, 42, 54, 72]
    all_22 = [
        (1,"George Russell",2,1,32.7,18.0,48.4,28.6,"[39.8–56.0%]"),
        (2,"Lewis Hamilton",3,3,31.6,11.8,17.7,21.2,"[11.8–24.1%]"),
        (3,"Kimi Antonelli",1,5,18.4,7.2,20.6,12.8,"[14.2–27.2%]"),
        (4,"Charles Leclerc",4,2,5.9,19.2,7.1,12.5,"[3.5–11.7%]"),
        (5,"Oscar Piastri",5,6,7.9,8.7,3.2,7.1,"[0.9–6.6%]"),
        (6,"Lando Norris",6,4,0.7,13.6,2.0,7.0,"[0.4–4.7%]"),
        (7,"Liam Lawson",12,7,0.0,5.4,0.1,2.4,"—"),
        (8,"Max Verstappen",8,9,1.0,4.1,0.4,2.4,"[0.0–1.6%]"),
        (9,"Ollie Bearman",10,8,0.0,4.7,0.1,2.1,"—"),
        (10,"Esteban Ocon",13,10,0.0,2.3,0.0,1.0,"—"),
        (11,"Isack Hadjar",9,11,1.6,0.9,0.1,0.9,"—"),
        (12,"Pierre Gasly",7,12,0.1,1.0,0.1,0.5,"—"),
        (13,"Carlos Sainz",16,13,0.0,0.8,0.0,0.4,"—"),
        (14,"Nico Hülkenberg",11,"DNF",0.0,0.7,0.0,0.3,"—"),
        (15,"Alex Albon",17,14,0.0,0.4,0.0,0.2,"—"),
        (16,"G. Bortoleto",14,15,0.0,0.4,0.0,0.2,"—"),
        (17,"Fernando Alonso",18,17,0.0,0.3,0.0,0.1,"—"),
        (18,"Valtteri Bottas",15,"DNF",0.0,0.3,0.0,0.1,"—"),
        (19,"Lance Stroll",19,16,0.0,0.2,0.0,0.1,"—"),
        (20,"Arvid Lindblad",20,"DNF",0.0,0.1,0.0,0.0,"—"),
        (21,"Jack Doohan",21,18,0.0,0.0,0.0,0.0,"—"),
        (22,"Oliver Crawford",22,19,0.0,0.0,0.0,0.0,"—"),
    ]
    ens_rows = [ens_header]
    for r in all_22:
        rk, nm, gd, sp, xg, mc, by, en, ci = r
        sp_str = f"P{sp}" if isinstance(sp, int) else sp
        ens_rows.append([str(rk), nm, f"P{gd}", sp_str,
                         f"{xg:.1f}%", f"{mc:.1f}%", f"{by:.1f}%", f"{en:.1f}%", ci])
    y = draw_table(c, y, ens_rows, ens_cw, row_height=12, font_size=7, header_font_size=7.5)

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 9 — TYRE STRATEGY + SCENARIO MATRIX ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page9(c):
    header_band(c, 9, "STRATEGY & SCENARIOS  ·  TYRES + RACE CONDITIONS", "Tyre Windows and Weather Scenarios")
    y = Y_START

    # Tyre strategy section
    y = section_heading(c, y, "Tyre Strategy Simulation — Top 5 Drivers")
    txt = ("Each of the 5 spaghetti traces represents a different race simulation. The wide spread "
           "shows strategic variance — safety car timing shifts optimal stop windows by ±4 laps. "
           "Hamilton's traces show noticeably steeper slopes from lap 20+ in many simulations, "
           "reflecting the modelled tyre degradation from his aggressive sprint driving style.")
    y = wrapped_text(c, y, txt, size=8.5, line_h=12)
    y -= 6
    img_h5 = 170
    y = place_image(c, IMGS["chart5"], y, w=CW, h=img_h5)
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, "Chart 5 — 100 sample tyre strategy simulations per driver. Red/yellow shaded bands = optimal pit stop windows (stop 1 / stop 2).")
    y -= 18

    # Scenario matrix
    y = section_heading(c, y, "Race Scenario Matrix — Dry/Wet × Safety Car/No Safety Car")
    txt2 = ("The 2×2 scenario matrix is one of the most useful outputs — it shows where probability "
            "mass shifts under different race conditions. Safety cars hurt the leader (Russell) "
            "and help mid-grid runners. Rain is Hamilton's and Verstappen's best friend, "
            "and a potential disaster for a 19-year-old leading on slicks for the first time.")
    y = wrapped_text(c, y, txt2, size=8.5, line_h=12)
    y -= 6
    img_h6 = min(190, y - FOOTER_Y - 24)
    y = place_image(c, IMGS["chart6"], y, w=CW, h=img_h6)
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, "Chart 6 — Scenario matrix: solid bars = no safety car, hatched = safety car deployed. Left = dry, right = wet conditions.")

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE 10 — CHAMPIONSHIP + SPRINT TABLE + V1→V2 + RISK ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def page10(c):
    header_band(c, 10, "CHAMPIONSHIP IMPACT  ·  SPRINT RESULTS  ·  RISK MATRIX", "Final Analysis")
    y = Y_START

    # Championship chart (compact)
    y = section_heading(c, y, "Championship Implications — After Predicted Result")
    img_h7 = 130
    y = place_image(c, IMGS["chart7"], y, w=CW, h=img_h7)
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(LM, y, "Chart 7 — Projected standings after China GP. Russell extends lead with predicted win. Hamilton closes gap via podium finish.")
    y -= 14

    # Sprint results + V1/V2 in two columns — sprint left, V1→V2 right
    col_l = LM
    col_r = LM + CW // 2 + 6
    cw2   = CW // 2 - 8          # 254pt each column

    # ── Left: sprint results ───────────────────────────────────────────────
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_l, y, "SPRINT RACE RESULTS  (19 LAPS)")
    c.setStrokeColor(RED); c.setLineWidth(0.6)
    c.line(col_l, y - 3, col_l + cw2, y - 3)
    y_l = y - 16

    # Sprint table draws directly at col_l — override draw_table x0
    spr_header = ["Pos", "Driver", "Gap", "Note"]
    spr_cw     = [22, 92, 50, 90]          # sum = 254
    spr_rows   = [spr_header]
    for pos, name, _team, gap, note in SPRINT_RESULTS[:11]:
        spr_rows.append([str(pos), name, gap, note])

    x0_save = LM
    # Draw sprint table with custom x origin
    x0 = col_l
    for ri, row in enumerate(spr_rows):
        rh = 11
        if y_l - rh < FOOTER_Y:
            break
        if ri == 0:
            c.setFillColor(RED)
        elif ri % 2 == 0:
            c.setFillColor(ROW_ALT)
        else:
            c.setFillColor(ROW_WHITE)
        c.rect(x0, y_l - rh, sum(spr_cw), rh, fill=1, stroke=0)
        cx = x0
        for ci2, (cell, cw_) in enumerate(zip(row, spr_cw)):
            if ri == 0:
                c.setFillColor(colors.white)
                c.setFont("Helvetica-Bold", 7)
            else:
                c.setFillColor(DARK)
                c.setFont("Helvetica", 7)
            c.drawString(cx + 3, y_l - rh + 3, str(cell))
            cx += cw_
        c.setStrokeColor(RULE_GREY); c.setLineWidth(0.3)
        c.rect(x0, y_l - rh, sum(spr_cw), rh, fill=0, stroke=1)
        y_l -= rh
    y_l -= 4

    # ── Right: V1 → V2 comparison table ───────────────────────────────────
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(col_r, y, "V1 (AUSTRALIA)  →  V2 (CHINA)")
    c.setStrokeColor(RED); c.setLineWidth(0.6)
    c.line(col_r, y - 3, col_r + cw2, y - 3)
    y_r = y - 16

    v2_header = ["Component", "V1", "V2 Change"]
    v2_cw     = [88, 50, 116]              # sum = 254
    v2_rows   = [
        v2_header,
        ["Sprint Tier 0",      "None",       "Added — highest weight tier"],
        ["DNF Multiplier",     "2×",         "Upgraded to 3× (Year-1 regs)"],
        ["FP Proxy",           "FP2 missing","FP1 lap delta hardcoded"],
        ["FP-Quali Divergence","Not encoded","fp1_to_quali_divergence"],
        ["Hamilton P(win)",    "2.5%",       "21.2% — V1 failure fixed"],
        ["MC Ensemble Weight", "35%",        "45% (sprint calibration)"],
        ["Bayesian Stages",    "1 (prior)",  "3 stages: prior+sprint+grid"],
        ["Weather Model",      "None",       "25% rain blend added"],
    ]

    x0 = col_r
    for ri, row in enumerate(v2_rows):
        rh = 12
        if y_r - rh < FOOTER_Y:
            break
        if ri == 0:
            c.setFillColor(colors.HexColor("#1A1A1A"))
        elif ri % 2 == 0:
            c.setFillColor(ROW_ALT)
        else:
            c.setFillColor(ROW_WHITE)
        c.rect(x0, y_r - rh, sum(v2_cw), rh, fill=1, stroke=0)
        cx = x0
        for ci2, (cell, cw_) in enumerate(zip(row, v2_cw)):
            if ri == 0:
                c.setFillColor(colors.white)
                c.setFont("Helvetica-Bold", 7)
            elif ci2 == 1:
                c.setFillColor(colors.HexColor("#993300"))
                c.setFont("Helvetica", 7)
            elif ci2 == 2:
                c.setFillColor(colors.HexColor("#005500"))
                c.setFont("Helvetica", 7)
            else:
                c.setFillColor(DARK)
                c.setFont("Helvetica-Bold", 7)
            c.drawString(cx + 3, y_r - rh + 3, str(cell))
            cx += cw_
        c.setStrokeColor(RULE_GREY); c.setLineWidth(0.3)
        c.rect(x0, y_r - rh, sum(v2_cw), rh, fill=0, stroke=1)
        y_r -= rh
    y_r -= 4

    y = min(y_l, y_r) - 8
    rule(c, y, RED, 0.8)
    y -= 8

    # Risk table
    y = section_heading(c, y, "Key Risk Factors")
    risk_header = ["Level", "Risk", "Probability", "Who It Affects", "Model Response"]
    risk_cw = [52, 120, 58, 108, 172]
    risk_rows = [
        risk_header,
        ["CRITICAL", "Antonelli rookie pressure",    "Always present", "Antonelli (P1)",      "22% retire rate prior; ×0.92 reliability penalty"],
        ["HIGH",     "Safety car deployment",        "56% probability", "All — hurts leaders","SC model: +25% probability for P5–P10 field"],
        ["HIGH",     "Rain onset during race",       "25% probability", "Hamilton +45%",       "Wet ensemble branch blended at 25% weight"],
        ["HIGH",     "Russell Q3 reliability",       "Unresolved",      "Russell",             "×0.92 post-ensemble penalty + DNF flag"],
        ["MEDIUM",   "Ferrari pit strategy risk",    "20% chance",      "Hamilton, Leclerc",   "Ferrari pit: 20% suboptimal stop in MC"],
        ["MEDIUM",   "Hamilton tyre management",     "Sprint-confirmed", "Hamilton (P3)",       "15% higher deg rate + pace ceiling laps 30+"],
        ["LOW",      "Wind gusts (36 km/h)",          "Forecast",        "Turn 1 field wide",  "Not modelled directly — risk flag only"],
    ]
    y = draw_table(c, y, risk_rows, risk_cw, row_height=12, font_size=7, header_font_size=7.5)
    y -= 8

    # Closing statement
    c.setFillColor(colors.HexColor("#F5F5F5"))
    c.roundRect(LM, y - 32, CW, 32, 4, fill=1, stroke=0)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(LM + 8, y - 13, "FINAL CALL:")
    c.setFillColor(DARK)
    c.setFont("Helvetica", 8.5)
    c.drawString(LM + 72, y - 13, "George Russell 28.6%  ·  Lewis Hamilton 21.2%  ·  Kimi Antonelli 12.8%  ·  Charles Leclerc 12.5%")
    c.setFillColor(MIDGREY)
    c.setFont("Helvetica", 8)
    c.drawString(LM + 8, y - 25, ("The race is Russell's to lose. Leclerc is the wildcard. If it rains, back Hamilton. "
                                   "If there's an early safety car, watch Lawson and Bearman."))

    footer(c)
    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_report():
    pdf_path = OUT / "china_gp_2026_prediction_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setTitle("2026 Chinese Grand Prix — F1 Prediction System V2")
    c.setAuthor("Basanth")
    c.setSubject("F1 Race Winner Prediction — Sprint-Calibrated Ensemble")

    print("Building 10-page report...")
    print("  Page 1 — Cover / Prediction Summary...")
    page1(c)
    print("  Page 2 — Race Context + Chart 1...")
    page2(c)
    print("  Page 3 — How It Works...")
    page3(c)
    print("  Page 4 — XGBoost...")
    page4(c)
    print("  Page 5 — SHAP Heatmap...")
    page5(c)
    print("  Page 6 — Monte Carlo...")
    page6(c)
    print("  Page 7 — Sprint + Bayesian...")
    page7(c)
    print("  Page 8 — Ensemble...")
    page8(c)
    print("  Page 9 — Tyre Strategy + Scenarios...")
    page9(c)
    print("  Page 10 — Championship + Sprint Table + V1→V2 + Risk...")
    page10(c)

    c.save()
    print(f"\n  Report saved: {pdf_path}")
    print(f"  File size: {pdf_path.stat().st_size / 1024:.1f} KB")
    return pdf_path


if __name__ == "__main__":
    build_report()
