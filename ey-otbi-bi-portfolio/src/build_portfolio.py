#!/usr/bin/env python3
"""Rebuild the demo warehouse, OTBI dashboards, and BIP report pack."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from analytics import invoice_register, kpis, payroll_register  # noqa: E402
from bip_reports import (  # noqa: E402
    bursting_file,
    html_report,
    write_csv,
    write_interview_pack,
    write_xml_data_model,
)
from dashboards import build_ap, build_hcm, build_index  # noqa: E402
from generate_sample_data import connect, schema, seed  # noqa: E402

DB = ROOT / "data" / "aether_fusion_demo.db"
OUT = ROOT / "output"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = connect(DB)
    schema(conn)
    seed(conn, 47)
    conn.row_factory = sqlite3.Row

    kpi = kpis(conn)
    payroll = payroll_register(conn, "Jul-26")
    invoices = invoice_register(conn)

    (OUT / "index.html").write_text(build_index(kpi), encoding="utf-8")
    (OUT / "otbi_hcm_workforce.html").write_text(build_hcm(conn), encoding="utf-8")
    (OUT / "otbi_ap_aging.html").write_text(build_ap(conn), encoding="utf-8")

    write_xml_data_model(payroll, "PAYROLL_REGISTER", "G_EMPLOYEE", OUT / "bip_payroll_data_model.xml")
    write_csv(payroll, OUT / "bip_payroll_register.csv")
    bursting_file(payroll, "business_unit", "email", OUT / "bip_payroll_bursting.csv")
    html_report(
        "Monthly Payroll Register – Jul-26",
        "BIP-HCM-01 · Data model: custom SQL on pay run results · Burst key: Business Unit",
        payroll,
        OUT / "bip_payroll_register.html",
        money_cols={"earnings", "deductions", "employer_cost", "net_pay"},
    )
    html_report(
        "AP Invoice Register",
        "BIP-FIN-01 · Parameters: BU, date range, payment status · Output: Excel/PDF",
        invoices,
        OUT / "bip_ap_invoice_register.html",
        money_cols={"invoice_amount", "amount_paid", "amount_remaining"},
    )
    write_xml_data_model(invoices, "AP_INVOICE_REGISTER", "G_INVOICE", OUT / "bip_ap_invoice_data_model.xml")
    write_csv(invoices, OUT / "bip_ap_invoice_register.csv")
    write_interview_pack(kpi, OUT / "kpis.json")
    conn.close()
    print(f"Dashboards and reports written to {OUT}")
    for name, value in kpi.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
