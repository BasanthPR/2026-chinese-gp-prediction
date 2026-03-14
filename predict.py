"""
2026 Chinese Grand Prix — F1 Race Winner Prediction System V2
Main prediction runner: trains all models, generates ensemble, produces PDF report.

Usage:
    python predict.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   F1 Race Winner Prediction System V2 — 2026 Chinese Grand Prix  ║
║   Shanghai International Circuit | Round 2 | March 23, 2026      ║
║   Sprint-Calibrated Ensemble: XGBoost + Monte Carlo + Bayesian    ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run_full_pipeline():
    """Execute all 11 pipeline stages and produce outputs."""
    start_time = time.time()
    print(BANNER)

    # ── Stage 1: Data Ingestion ─────────────────────────────────────────────────
    print("=" * 65)
    print("STAGE 1/6 — Data Ingestion")
    print("=" * 65)
    from data.fetch_data import fetch_historical_results, fetch_all_sprint_history
    hist = fetch_historical_results(2010, 2025)
    sprint_hist = fetch_all_sprint_history(2021, 2025)
    print(f"  Historical races: {len(hist)} rows")
    print(f"  Sprint history:   {len(sprint_hist)} rows")

    # ── Stage 2: Feature Engineering ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 2/6 — Feature Engineering")
    print("=" * 65)
    from features.engineering import build_feature_matrix
    from features.sprint_features import get_full_sprint_feature_set
    from data.historical_data import GRID_2026

    feature_matrix = build_feature_matrix()
    sprint_features = get_full_sprint_feature_set()
    print(f"  Feature matrix: {feature_matrix.shape[0]} drivers × {feature_matrix.shape[1]} features")
    print(f"  Sprint features: {sprint_features.shape[0]} drivers × {sprint_features.shape[1]} features")

    # ── Stage 3: XGBoost ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 3/6 — XGBoost Classifier")
    print("=" * 65)
    from models.xgboost_model import run_xgboost
    xgb_probs = run_xgboost()

    # ── Stage 4: Bayesian Model ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 4/6 — Bayesian Inference (3-Stage Update)")
    print("=" * 65)
    from models.bayesian_model import run_bayesian_model
    bayes_probs, credible_intervals = run_bayesian_model()

    # ── Stage 5: Monte Carlo ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 5/6 — Monte Carlo Race Simulation (10,000 races)")
    print("=" * 65)
    from models.monte_carlo_sim import run_monte_carlo_model
    mc_probs = run_monte_carlo_model()

    # ── Stage 6: Ensemble ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 6/6 — Ensemble (XGB 30% | MC 45% | Bayes 25%)")
    print("=" * 65)
    from models.ensemble import run_ensemble
    ensemble_df = run_ensemble(xgb_probs, mc_probs, bayes_probs)

    # ── Visualizations ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("VISUALIZATIONS — Generating 7 charts")
    print("=" * 65)
    from visualizations.plots import generate_all_charts
    predicted_winner = ensemble_df.index[0]
    chart_paths = generate_all_charts(ensemble_df, predicted_winner)

    # ── Final Prediction Table ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL PREDICTION — 2026 Chinese Grand Prix")
    print("=" * 65)
    _print_prediction_table(ensemble_df, credible_intervals)

    # ── Key Risk Factors ────────────────────────────────────────────────────────
    _print_risk_factors(ensemble_df)

    # ── PDF Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PDF REPORT — Generating china_gp_2026_prediction.pdf")
    print("=" * 65)
    _generate_pdf_report(ensemble_df, credible_intervals, chart_paths)

    elapsed = time.time() - start_time
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs saved to: {OUTPUT_DIR.absolute()}")


def _print_prediction_table(ensemble_df: pd.DataFrame, ci_df: pd.DataFrame):
    """Print the ranked prediction table."""
    from data.historical_data import GRID_2026, CHINA_2026_QUALI
    from data.sprint_data import SPRINT_RESULT
    from features.config import DRIVER_DNF_MULTIPLIERS, BASE_DNF_RATE_PER_LAP, REG_YEAR1_MULTIPLIER

    key_risks = {
        "antonelli":  "Rookie pressure, collision risk (Young pole: 22% retire rate)",
        "russell":    "Q3 power/gear issue — reliability flag",
        "hamilton":   "Tyre depletion (killed left tyre in sprint)",
        "leclerc":    "Ferrari pit strategy risk (missed VSC in Australia)",
        "norris":     "McLaren pace deficit vs Mercedes on straights",
        "piastri":    "P5 — needs Turn 1 incident to contend",
        "verstappen": "P8 grid + Red Bull energy management deficit",
        "gasly":      "P7 — outside shot via SC/chaos only",
        "bearman":    "Oversteer issues in FP1, tyre stress flag",
        "hadjar":     "P9 — Red Bull B-team, limited upside",
    }

    header = (f"{'Rank':<5} {'Driver':<22} {'Team':<14} "
              f"{'Grid':>4} {'Sprint':>6} "
              f"{'P(Win)':>8} {'90% CI':>16} {'Key Risk':<45}")
    print(header)
    print("─" * len(header))

    for rank, (driver, row) in enumerate(ensemble_df.head(10).iterrows(), 1):
        prob = row["final_ensemble"]
        grid = int(CHINA_2026_QUALI.get(driver, {"grid": 22})["grid"])
        sprint_pos = SPRINT_RESULT.get(driver, {}).get("sprint_pos", 22)
        sprint_str = f"P{sprint_pos}" if sprint_pos != 99 else "DNF"

        if driver in ci_df.index:
            ci_lo = ci_df.loc[driver, "ci_lo_90"]
            ci_hi = ci_df.loc[driver, "ci_hi_90"]
            ci_str = f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]"
        else:
            ci_str = f"[{max(0,prob-0.06)*100:.1f}%, {min(1,prob+0.08)*100:.1f}%]"

        risk = key_risks.get(driver, "—")
        name = GRID_2026[driver][0]
        team = GRID_2026[driver][1].replace("_", " ").title()

        print(f"  {rank:<3} {name:<22} {team:<14} "
              f"P{grid:<3} {sprint_str:>6} "
              f"{prob*100:>7.1f}% {ci_str:>16} {risk:<45}")

    print()

    # Save to text file
    txt_path = OUTPUT_DIR / "prediction_output.txt"
    with open(txt_path, "w") as f:
        f.write("2026 CHINESE GRAND PRIX — F1 PREDICTION SYSTEM V2\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Rank':<5} {'Driver':<22} {'Team':<14} "
                f"{'Grid':>4} {'Sprint':>6} {'P(Win)':>8} {'90% CI':>18}\n")
        f.write("-" * 70 + "\n")
        for rank, (driver, row) in enumerate(ensemble_df.iterrows(), 1):
            prob = row["final_ensemble"]
            grid = int(CHINA_2026_QUALI.get(driver, {"grid": 22})["grid"])
            sprint_pos = SPRINT_RESULT.get(driver, {}).get("sprint_pos", 22)
            sprint_str = f"P{sprint_pos}" if sprint_pos != 99 else "DNF"
            if driver in ci_df.index:
                ci_lo = ci_df.loc[driver, "ci_lo_90"]
                ci_hi = ci_df.loc[driver, "ci_hi_90"]
                ci_str = f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]"
            else:
                ci_str = f"[{max(0,prob-0.06)*100:.1f}%, {min(1,prob+0.08)*100:.1f}%]"
            name = GRID_2026[driver][0]
            team = GRID_2026[driver][1].replace("_", " ").title()
            from data.sprint_data import SPRINT_RESULT as SR
            sprint_pos2 = SR.get(driver, {}).get("sprint_pos", 22)
            sprint_str2 = f"P{sprint_pos2}" if sprint_pos2 != 99 else "DNF"
            f.write(f"  {rank:<3} {name:<22} {team:<14} "
                    f"P{grid:<3} {sprint_str2:>6} {prob*100:>7.1f}% {ci_str:>18}\n")
    print(f"  Prediction table saved to {txt_path}")


def _print_risk_factors(ensemble_df: pd.DataFrame):
    """Print key risk factors for the race."""
    print("\n" + "=" * 65)
    print("KEY RISK FACTORS")
    print("=" * 65)
    risks = [
        ("CRITICAL", "Antonelli rookie pressure", "Youngest pole ever. Historical young-pole retire rate 22%. High variance outcome."),
        ("HIGH",     "Safety Car probability",    "56% adjusted rate (sprint confirmed). Could reshuffle entire race."),
        ("HIGH",     "Rain (25% probability)",     "Hamilton/Verstappen benefit most. Antonelli/rookies most exposed."),
        ("HIGH",     "Russell reliability",         "Q3 power/gear issue unresolved. Could re-emerge mid-race."),
        ("MEDIUM",   "Ferrari pit strategy",        "Historically suboptimal under VSC/SC. Australia miss repeated?"),
        ("MEDIUM",   "Hamilton tyre management",    "Killed left tyre in sprint. 56 laps requires very different approach."),
        ("MEDIUM",   "Energy management unknown",   "50/50 power split unproven in full race. Mercedes leads but not guaranteed."),
        ("LOW",      "Weather (wind 36km/h gusts)", "Hamilton flagged wind effects. Turn 1 could be chaotic."),
    ]
    for level, title, detail in risks:
        color_map = {"CRITICAL": "!!", "HIGH": " !", "MEDIUM": " ~", "LOW": "  "}
        marker = color_map.get(level, "  ")
        print(f"  {marker} [{level:<8}] {title:<30} {detail}")


def _generate_pdf_report(
    ensemble_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    chart_paths: list,
):
    """Generate PDF report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image as RLImage, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        print("  [WARN] ReportLab not installed — skipping PDF generation")
        print("  Install with: pip install reportlab")
        return

    from data.historical_data import GRID_2026, CHINA_2026_QUALI
    from data.sprint_data import SPRINT_RESULT

    pdf_path = OUTPUT_DIR / "china_gp_2026_prediction.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 fontSize=18, spaceAfter=6,
                                 textColor=colors.HexColor("#E10600"))
    h1_style = ParagraphStyle("h1", parent=styles["Heading1"],
                               fontSize=13, spaceBefore=12, spaceAfter=4,
                               textColor=colors.HexColor("#0D0D0D"))
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
                               fontSize=11, spaceBefore=8, spaceAfter=3)
    body = styles["BodyText"]
    body.fontSize = 9

    story = []

    # Cover
    story.append(Paragraph("F1 RACE WINNER PREDICTION SYSTEM V2", title_style))
    story.append(Paragraph("2026 Chinese Grand Prix — Shanghai International Circuit", h1_style))
    story.append(Paragraph("Round 2, Season 2026 | March 23, 2026 | Sprint Weekend", body))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#E10600")))
    story.append(Spacer(1, 0.3*cm))

    # Executive Summary
    story.append(Paragraph("1. EXECUTIVE SUMMARY", h1_style))
    predicted_winner = ensemble_df.index[0]
    winner_name = GRID_2026[predicted_winner][0]
    winner_prob = ensemble_df.loc[predicted_winner, "final_ensemble"]
    story.append(Paragraph(
        f"<b>Predicted Winner: {winner_name} ({winner_prob*100:.1f}%)</b>", h2_style))

    # Top 5 table
    table_data = [["Rank", "Driver", "Team", "Grid", "Sprint", "P(Win)", "90% CI"]]
    for rank, (driver, row) in enumerate(ensemble_df.head(5).iterrows(), 1):
        prob = row["final_ensemble"]
        grid = int(CHINA_2026_QUALI.get(driver, {"grid": 22})["grid"])
        sp = SPRINT_RESULT.get(driver, {}).get("sprint_pos", 22)
        sprint_str = f"P{sp}" if sp != 99 else "DNF"
        if driver in ci_df.index:
            ci_lo = ci_df.loc[driver, "ci_lo_90"]
            ci_hi = ci_df.loc[driver, "ci_hi_90"]
            ci_str = f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]"
        else:
            ci_str = f"[{max(0,prob-0.06)*100:.1f}%, {min(1,prob+0.08)*100:.1f}%]"
        name = GRID_2026[driver][0]
        team = GRID_2026[driver][1].replace("_", " ").title()
        table_data.append([str(rank), name, team, f"P{grid}", sprint_str,
                           f"{prob*100:.1f}%", ci_str])

    t = Table(table_data, colWidths=[1*cm, 4*cm, 3.5*cm, 1.5*cm, 1.5*cm, 2*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#E10600")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [colors.white, colors.HexColor("#F8F8F8")]),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # V1 vs V2 comparison
    story.append(Paragraph("2. MODEL ARCHITECTURE — V1 vs V2", h1_style))
    v_table_data = [
        ["Component",       "V1 (Australia)",              "V2 (China)",                   "Reason"],
        ["Sprint Features", "Not available",               "Tier 0 (highest weight)",      "Sprint weekend → real race data"],
        ["Reliability DNF", "2x multiplier",               "3x multiplier",                "V1 underweighted reg-change DNFs"],
        ["FP Proxy",        "FP2 long-run (missing)",      "FP1 lap delta (hardcoded)",    "China is sprint-only (no FP2)"],
        ["FP-Quali Div.",   "Not encoded",                 "fp1_to_quali_divergence",      "Fixed Hamilton undervaluation"],
        ["MC Ensemble Wt",  "35%",                         "45%",                          "Sprint calibration raises MC value"],
        ["Bayesian Prior",  "Historical only",             "3-stage (prior+sprint+grid)",  "Sprint is strong evidence"],
        ["Weather Model",   "Not included",                "25% rain blend",               "Shanghai rain window"],
        ["Grid Cap",        "Not included",                "P8+ capped at 5%",             "Prevents overrating back runners"],
    ]
    vt = Table(v_table_data, colWidths=[3.2*cm, 3.8*cm, 4.2*cm, 4.8*cm])
    vt.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 7.5),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [colors.white, colors.HexColor("#F0F4FF")]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(vt)
    story.append(Spacer(1, 0.4*cm))

    # Sprint Analysis
    story.append(Paragraph("3. SPRINT RACE ANALYSIS — What It Tells Us About Sunday", h1_style))
    sprint_points = [
        "<b>Russell (Sprint winner):</b> Dominant wire-to-wire. 75%+ MC win probability from P2 on GP grid. "
        "Only concern: Q3 gear issue. If that recurs, race ends early.",
        "<b>Leclerc (P2 from P6):</b> Best racer in the field on current form. Set fastest sprint lap. "
        "Ferrari start strategic liability — Australia showed they can miss critical pit windows.",
        "<b>Hamilton (P3, briefly led):</b> True race pace is P2. fp1_to_quali_divergence=+3 fixed the "
        "Australia V1 error. However: 'I killed my left tyre' in sprint. 56 laps will test "
        "whether he can manage differently.",
        "<b>Antonelli (P5 with 10s penalty — true pace P2):</b> Historic pole at age 19. "
        "Historical data: 28% win rate for under-22 polestarters, but 22% retire rate. "
        "Prone to incidents under pressure (Australia contact, China sprint penalty).",
        "<b>Verstappen (P9 from P8, fell to P16):</b> Red Bull clearly lacks energy management "
        "on Shanghai's back straight. Net negative sprint. Only threat: Safety Car chaos.",
    ]
    for point in sprint_points:
        story.append(Paragraph(f"• {point}", body))
        story.append(Spacer(1, 0.15*cm))

    story.append(PageBreak())

    # Visualizations — embed charts
    story.append(Paragraph("4. VISUALIZATIONS", h1_style))
    chart_titles = [
        "Win Probability by Model (Stacked Bar)",
        "Sprint-to-GP Correlation",
        "Monte Carlo Race Simulation (10,000 races)",
        "Feature Importance Heatmap (SHAP by Tier)",
        "Tire Strategy Simulation (100 sample races)",
        "Race Scenario Matrix (Dry/Wet × SC/No SC)",
        "Championship Standings Implications",
    ]

    for i, (path, chart_title) in enumerate(zip(chart_paths, chart_titles)):
        if Path(path).exists():
            story.append(Paragraph(f"Chart {i+1}: {chart_title}", h2_style))
            img = RLImage(str(path), width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # Risk Factors
    story.append(Paragraph("5. KEY RISK FACTORS", h1_style))
    risk_table = [
        ["Level",    "Risk",                      "Impact"],
        ["CRITICAL", "Antonelli rookie pressure",  "22% retire rate for under-22 polestarters. High variance."],
        ["HIGH",     "Safety Car (56% probability)","Bunches field. Russell advantage evaporates under SC."],
        ["HIGH",     "Rain (25% probability)",      "Hamilton +45% boost, Antonelli -10%. Race reshuffled."],
        ["HIGH",     "Russell Q3 reliability",      "Power/gear issue unresolved. Could end race."],
        ["MEDIUM",   "Ferrari pit strategy",        "Historically suboptimal (missed Australia VSC window)."],
        ["MEDIUM",   "Hamilton tyre management",    "Sprint tyre kill. Must drive differently over 56 laps."],
        ["LOW",      "Wind gusts (36 km/h)",         "Hamilton flagged. Could cause Turn 1 incidents."],
    ]
    rt = Table(risk_table, colWidths=[2.5*cm, 5*cm, 8.5*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#CC0000")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [colors.white, colors.HexColor("#FFF0F0")]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(rt)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "Generated by F1 Prediction System V2 | BasanthPR/2026-chinese-gp-prediction",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=7,
                       textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"  PDF report saved to {pdf_path}")


if __name__ == "__main__":
    run_full_pipeline()
