from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from analytics import aging_buckets, kpis, payroll_register  # noqa: E402
from generate_sample_data import connect, schema, seed  # noqa: E402


def _warehouse(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "demo.db")
    schema(conn)
    seed(conn, 47)
    conn.row_factory = sqlite3.Row
    return conn


def test_headcount_and_attrition_are_coherent(tmp_path: Path) -> None:
    conn = _warehouse(tmp_path)
    kpi = kpis(conn)
    assert kpi["headcount"] > 0
    assert kpi["terms_12m"] >= 0
    assert kpi["hires_12m"] >= 0
    expected = round((kpi["terms_12m"] / kpi["headcount"]) * 100, 1)
    assert kpi["attrition_pct"] == expected
    active = conn.execute(
        """SELECT COUNT(*) FROM per_all_assignments_m
           WHERE assignment_status = 'ACTIVE' AND headcount = 1"""
    ).fetchone()[0]
    assert kpi["headcount"] == active
    conn.close()


def test_aging_sums_to_open_standard_ap(tmp_path: Path) -> None:
    conn = _warehouse(tmp_path)
    kpi = kpis(conn)
    buckets = aging_buckets(conn)
    total = round(sum(r["amount_remaining"] for r in buckets), 2)
    assert total == kpi["open_ap_amount"]
    assert {r["bucket"] for r in buckets} <= {"Current", "1-30", "31-60", "61-90", "90+"}
    conn.close()


def test_payroll_net_is_earnings_minus_deductions(tmp_path: Path) -> None:
    conn = _warehouse(tmp_path)
    rows = payroll_register(conn, "Jul-26")
    assert rows
    for row in rows:
        assert round(row["net_pay"], 2) == round(row["earnings"] - row["deductions"], 2)
    bus = {r["business_unit"] for r in rows}
    assert "Aether India BU" in bus
    conn.close()


def test_burst_keys_cover_business_units(tmp_path: Path) -> None:
    conn = _warehouse(tmp_path)
    rows = payroll_register(conn, "Jul-26")
    keys = {r["business_unit"] for r in rows}
    assert keys == {"Aether India BU", "Aether UK BU", "Aether US BU"}
    conn.close()
