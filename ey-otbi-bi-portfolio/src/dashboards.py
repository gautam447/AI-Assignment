"""Generate OTBI-style HTML dashboards from the demo warehouse."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analytics import (
    aging_buckets,
    attrition_by_reason,
    headcount_by_bu,
    headcount_by_dept,
    kpis,
    spend_by_source,
    spend_by_supplier,
)

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def render_dashboard(
    title: str,
    project_code: str,
    subject_area: str,
    kpi_cards: list[tuple[str, str, str]],
    tables: list[tuple[str, list[dict[str, Any]]]],
    charts: list[tuple[str, str, list[str], list[float]]],
    talking_points: list[str],
) -> str:
    cards = "".join(
        f"<article class='kpi'><p class='lbl'>{lbl}</p><p class='val'>{val}</p><p class='sub'>{sub}</p></article>"
        for lbl, val, sub in kpi_cards
    )
    chart_blocks = []
    for heading, _kind, labels, values in charts:
        peak = max(values) if values else 1
        bars = []
        for label, value in zip(labels, values):
            width = 0 if peak == 0 else round(100 * float(value) / float(peak), 1)
            bars.append(
                "<div class='bar-row'><span class='bar-lbl'>"
                + html.escape(str(label))
                + f"</span><span class='bar-track'><span class='bar-fill' style='width:{width}%'></span></span>"
                f"<span class='bar-val'>{value:,.0f}</span></div>"
            )
        chart_blocks.append(
            "<section class='panel'><h2>"
            + html.escape(heading)
            + "</h2>"
            + "".join(bars)
            + "</section>"
        )
    table_html = []
    for heading, rows in tables:
        if not rows:
            continue
        headers = list(rows[0].keys())
        body = []
        for row in rows[:12]:
            body.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(row[h]))}</td>" for h in headers)
                + "</tr>"
            )
        table_html.append(
            "<section class='panel wide'><h2>"
            + heading
            + "</h2><table><thead><tr>"
            + "".join(f"<th>{h}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></section>"
        )
    tips = "".join(f"<li>{t}</li>" for t in talking_points)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root {{ --navy:#12263a; --blue:#1f4e79; --bg:#f3f6fa; --card:#fff; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--navy); }}
    header {{ background: linear-gradient(90deg, #0e2a47, #1f4e79); color:#fff; padding: 22px 32px; }}
    header p {{ margin: 6px 0 0; opacity: .85; font-size: 14px; }}
    nav {{ display:flex; gap:16px; padding: 10px 32px; background:#0b1c2e; }}
    nav a {{ color:#d6e4f0; text-decoration:none; font-size: 13px; }}
    main {{ padding: 24px 32px 48px; }}
    .kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
    .kpi {{ background: var(--card); padding: 16px; border-radius: 10px; box-shadow: 0 1px 3px rgba(18,38,58,.08); }}
    .lbl {{ margin:0; font-size:12px; color:#5b6770; text-transform: uppercase; letter-spacing:.04em; }}
    .val {{ margin: 6px 0 0; font-size: 28px; font-weight: 650; }}
    .sub {{ margin: 4px 0 0; font-size: 12px; color:#5b6770; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
    .panel {{ background: var(--card); padding: 16px; border-radius: 10px; box-shadow: 0 1px 3px rgba(18,38,58,.08); }}
    .wide {{ grid-column: 1 / -1; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #e4ebf2; text-align:left; }}
    th {{ color:#5b6770; font-weight: 600; }}
    .talk {{ margin-top: 16px; }}
    .bar-row {{ display:grid; grid-template-columns: 140px 1fr 72px; gap:8px; align-items:center; margin: 8px 0; font-size: 13px; }}
    .bar-track {{ background:#e4ebf2; height:10px; border-radius: 6px; overflow:hidden; }}
    .bar-fill {{ display:block; height:100%; background:#1f4e79; }}
    .bar-val {{ text-align:right; font-variant-numeric: tabular-nums; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>{project_code} · Subject area: {subject_area} · Generated {generated} · Demo warehouse, not a live Fusion instance</p>
  </header>
  <nav>
    <a href="index.html">Portfolio home</a>
    <a href="otbi_hcm_workforce.html">OTBI HCM</a>
    <a href="otbi_ap_aging.html">OTBI Payables</a>
    <a href="bip_payroll_register.html">BIP Payroll</a>
    <a href="bip_ap_invoice_register.html">BIP AP Register</a>
  </nav>
  <main>
    <section class="kpis">{cards}</section>
    <div class="grid">
      {''.join(chart_blocks)}
      {''.join(table_html)}
      <section class="panel talk wide">
        <h2>Interview talking points</h2>
        <ul>{tips}</ul>
      </section>
    </div>
  </main>
</body>
</html>
"""


def build_hcm(conn) -> str:
    kpi = kpis(conn)
    by_bu = headcount_by_bu(conn)
    by_dept = headcount_by_dept(conn)
    reasons = attrition_by_reason(conn)
    return render_dashboard(
        title="Workforce Headcount & Attrition",
        project_code="OTBI-HCM-01",
        subject_area="Workforce Management - Worker Assignment Real Time",
        kpi_cards=[
            ("Headcount", f"{kpi['headcount']:,}", "Active employee assignments as of date"),
            ("Hires (12m)", f"{kpi['hires_12m']:,}", "PER assignment HIRE actions"),
            ("Exits (12m)", f"{kpi['terms_12m']:,}", "TERMINATION actions"),
            ("Attrition", f"{kpi['attrition_pct']}%", "Exits / closing headcount"),
        ],
        tables=[
            ("Headcount by business unit", by_bu),
            ("Gender mix by department", by_dept),
        ],
        charts=[
            ("Headcount by BU", "bar", [r["business_unit"] for r in by_bu], [r["headcount"] for r in by_bu]),
            ("Exit reasons (12 months)", "doughnut", [r["reason"] for r in reasons], [r["exits"] for r in reasons]),
        ],
        talking_points=[
            "Default as-of date is sysdate; past/future uses SET VARIABLE PARAM_EFFECTIVE_DATE.",
            "Grain is assignment (PER_ALL_ASSIGNMENTS_M), not person — dual assignments would double-count if you sum headcount blindly.",
            "Data security follows HCM roles; a line manager only sees their hierarchy.",
            "Attrition is not a delivered measure — calculate exits from Work Relationship / action events, then divide by average or closing HC.",
            "Would not mix this subject area with Payables in one analysis; different Fusion apps.",
        ],
    )


def build_ap(conn) -> str:
    kpi = kpis(conn)
    aging = aging_buckets(conn)
    suppliers = spend_by_supplier(conn)
    sources = spend_by_source(conn)
    overdue_pct = 0.0
    if kpi["open_ap_amount"]:
        overdue_pct = round(100 * kpi["overdue_ap_amount"] / kpi["open_ap_amount"], 1)
    return render_dashboard(
        title="Payables Aging & Supplier Spend",
        project_code="OTBI-FIN-01",
        subject_area="Payables Invoices - Transactions Real Time",
        kpi_cards=[
            ("Open invoices", f"{kpi['open_invoices']:,}", "Unpaid + partial standard invoices"),
            ("Open AP", f"{kpi['open_ap_amount']:,.0f}", "Amount remaining"),
            ("Overdue AP", f"{kpi['overdue_ap_amount']:,.0f}", f"{overdue_pct}% of open AP"),
            ("Sources", str(len(sources)), "MANUAL / IDT / FBDI / SCANNED"),
        ],
        tables=[("Supplier spend", suppliers), ("Invoice source mix", sources)],
        charts=[
            ("Aging (amount remaining)", "bar", [r["bucket"] for r in aging], [r["amount_remaining"] for r in aging]),
            ("Invoiced by source", "bar", [r["invoice_source"] for r in sources], [r["invoiced"] for r in sources]),
        ],
        talking_points=[
            "Aging is a CASE on due date vs as-of date; keep it in the analysis so Finance can change buckets without a BIP change.",
            "Amount Remaining is the measure; Invoice Amount would overstate once partial payments exist.",
            "Validation Status and Approval Status are local dimensions — add a measure from the same subject area if you cross SA.",
            "OTBI is the ops dashboard; the pixel-perfect AP register for auditors is BIP.",
            "Schedule the analysis daily to a role, or burst a BIP sibling for BU controllers.",
            "Do not sum Amount Remaining across INR/GBP/USD. Prompt by BU/ledger or add a USD equivalent measure.",
        ],
    )


def build_index(kpi: dict[str, Any]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>EY interview portfolio – OTBI &amp; BI Publisher</title>
  <style>
    body {{ margin:0; font-family:"Segoe UI", Arial, sans-serif; background:#f3f6fa; color:#12263a; }}
    header {{ background:#0e2a47; color:#fff; padding:36px 32px; }}
    main {{ padding: 28px 32px 60px; max-width: 1100px; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; }}
    article {{ background:#fff; padding:18px; border-radius:10px; box-shadow:0 1px 3px rgba(18,38,58,.08); }}
    a.btn {{ display:inline-block; margin-top:10px; background:#1f4e79; color:#fff; padding:8px 12px; border-radius:6px; text-decoration:none; }}
    code {{ background:#e8eef5; padding:2px 6px; border-radius:4px; }}
    @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Cloud reporting portfolio</h1>
    <p>OTBI analyses + BI Publisher reports for an EY Oracle technical interview. Client: Aether Global Industries (demo).</p>
  </header>
  <main>
    <p>As of {kpi['as_of']}: headcount <strong>{kpi['headcount']}</strong>, attrition <strong>{kpi['attrition_pct']}%</strong>, open AP <strong>{kpi['open_ap_amount']:,.0f}</strong>, Jul net pay <strong>{kpi['jul_net_pay']:,.0f}</strong>.</p>
    <div class="grid">
      <article>
        <h2>OTBI-HCM-01 Workforce</h2>
        <p>Headcount, hires, exits, attrition by BU/department. Maps to Worker Assignment Real Time.</p>
        <a class="btn" href="otbi_hcm_workforce.html">Open dashboard</a>
      </article>
      <article>
        <h2>OTBI-FIN-01 Payables aging</h2>
        <p>Open AP, overdue, aging buckets, supplier spend, invoice source mix.</p>
        <a class="btn" href="otbi_ap_aging.html">Open dashboard</a>
      </article>
      <article>
        <h2>BIP-HCM-01 Payroll register</h2>
        <p>Pixel-perfect register, XML data model, email bursting by business unit.</p>
        <a class="btn" href="bip_payroll_register.html">Open report</a>
      </article>
      <article>
        <h2>BIP-FIN-01 AP invoice register</h2>
        <p>Auditor-facing invoice listing with approval and payment status.</p>
        <a class="btn" href="bip_ap_invoice_register.html">Open report</a>
      </article>
    </div>
    <p>Rebuild with <code>python3 src/build_portfolio.py</code>. Design docs live in <code>docs/</code>.</p>
  </main>
</body>
</html>
"""
