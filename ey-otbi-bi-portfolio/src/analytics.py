"""KPI logic that mirrors OTBI analyses and BIP data models."""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

AS_OF = date(2026, 9, 14)


def q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return list(conn.execute(sql, params))


def kpis(conn: sqlite3.Connection, as_of: date = AS_OF) -> dict[str, Any]:
    as_of_s = as_of.isoformat()
    headcount = conn.execute(
        """
        SELECT COALESCE(SUM(headcount), 0)
        FROM per_all_assignments_m
        WHERE assignment_type = 'E'
          AND effective_latest_change = 'Y'
          AND ? BETWEEN effective_start_date AND effective_end_date
          AND assignment_status = 'ACTIVE'
        """,
        (as_of_s,),
    ).fetchone()[0]
    hires_12m = conn.execute(
        """
        SELECT COUNT(*)
        FROM per_assignment_actions
        WHERE action_type = 'HIRE'
          AND effective_date >= date(?, '-12 months')
          AND effective_date <= ?
        """,
        (as_of_s, as_of_s),
    ).fetchone()[0]
    terms_12m = conn.execute(
        """
        SELECT COUNT(*)
        FROM per_assignment_actions
        WHERE action_type = 'TERMINATION'
          AND effective_date >= date(?, '-12 months')
          AND effective_date <= ?
        """,
        (as_of_s, as_of_s),
    ).fetchone()[0]
    attrition = round((terms_12m / headcount) * 100, 1) if headcount else 0.0

    open_ap = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(amount_remaining), 0)
        FROM ap_invoices
        WHERE payment_status IN ('UNPAID', 'PARTIAL')
          AND invoice_type = 'STANDARD'
        """
    ).fetchone()
    overdue = conn.execute(
        """
        SELECT COALESCE(SUM(amount_remaining), 0)
        FROM ap_invoices
        WHERE payment_status IN ('UNPAID', 'PARTIAL')
          AND invoice_type = 'STANDARD'
          AND due_date < ?
        """,
        (as_of_s,),
    ).fetchone()[0]
    payroll_net = conn.execute(
        """
        SELECT COALESCE(SUM(CASE WHEN classification IN ('Earnings') THEN result_value
                                 WHEN classification = 'Deduction' THEN -result_value
                                 ELSE 0 END), 0)
        FROM pay_run_results r
        JOIN pay_payroll_actions a ON a.payroll_action_id = r.payroll_action_id
        WHERE a.period_name = 'Jul-26'
        """
    ).fetchone()[0]
    return {
        "as_of": as_of_s,
        "headcount": int(headcount),
        "hires_12m": int(hires_12m),
        "terms_12m": int(terms_12m),
        "attrition_pct": attrition,
        "open_invoices": int(open_ap[0]),
        "open_ap_amount": round(open_ap[1], 2),
        "overdue_ap_amount": round(overdue, 2),
        "jul_net_pay": round(payroll_net, 2),
    }


def headcount_by_bu(conn: sqlite3.Connection, as_of: date = AS_OF) -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT o.name AS business_unit,
               SUM(a.headcount) AS headcount,
               SUM(a.fte) AS fte,
               ROUND(AVG(a.annual_salary), 0) AS avg_salary
        FROM per_all_assignments_m a
        JOIN hr_all_organization_units o ON o.organization_id = a.business_unit_id
        WHERE a.assignment_type = 'E'
          AND a.effective_latest_change = 'Y'
          AND ? BETWEEN a.effective_start_date AND a.effective_end_date
          AND a.assignment_status = 'ACTIVE'
        GROUP BY o.name
        ORDER BY headcount DESC
        """,
        (as_of.isoformat(),),
    )
    return [dict(r) for r in rows]


def headcount_by_dept(conn: sqlite3.Connection, as_of: date = AS_OF) -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT d.name AS department,
               SUM(a.headcount) AS headcount,
               SUM(CASE WHEN p.gender = 'F' THEN a.headcount ELSE 0 END) AS female_hc,
               SUM(CASE WHEN p.gender = 'M' THEN a.headcount ELSE 0 END) AS male_hc
        FROM per_all_assignments_m a
        JOIN hr_all_organization_units d ON d.organization_id = a.department_id
        JOIN per_all_people_f p ON p.person_id = a.person_id
        WHERE a.assignment_type = 'E'
          AND a.effective_latest_change = 'Y'
          AND ? BETWEEN a.effective_start_date AND a.effective_end_date
          AND a.assignment_status = 'ACTIVE'
        GROUP BY d.name
        ORDER BY headcount DESC
        """,
        (as_of.isoformat(),),
    )
    return [dict(r) for r in rows]


def attrition_by_reason(conn: sqlite3.Connection, as_of: date = AS_OF) -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT action_reason AS reason, COUNT(*) AS exits
        FROM per_assignment_actions
        WHERE action_type = 'TERMINATION'
          AND effective_date >= date(?, '-12 months')
          AND effective_date <= ?
        GROUP BY action_reason
        ORDER BY exits DESC
        """,
        (as_of.isoformat(), as_of.isoformat()),
    )
    return [dict(r) for r in rows]


def aging_buckets(conn: sqlite3.Connection, as_of: date = AS_OF) -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT
          CASE
            WHEN julianday(?) - julianday(due_date) <= 0 THEN 'Current'
            WHEN julianday(?) - julianday(due_date) BETWEEN 1 AND 30 THEN '1-30'
            WHEN julianday(?) - julianday(due_date) BETWEEN 31 AND 60 THEN '31-60'
            WHEN julianday(?) - julianday(due_date) BETWEEN 61 AND 90 THEN '61-90'
            ELSE '90+'
          END AS bucket,
          COUNT(*) AS invoice_count,
          ROUND(SUM(amount_remaining), 2) AS amount_remaining
        FROM ap_invoices
        WHERE payment_status IN ('UNPAID', 'PARTIAL')
          AND invoice_type = 'STANDARD'
        GROUP BY 1
        """,
        (as_of.isoformat(),) * 4,
    )
    order = {"Current": 0, "1-30": 1, "31-60": 2, "61-90": 3, "90+": 4}
    data = [dict(r) for r in rows]
    data.sort(key=lambda r: order.get(r["bucket"], 9))
    return data


def spend_by_supplier(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT s.supplier_name,
               s.category,
               COUNT(*) AS invoice_count,
               ROUND(SUM(i.invoice_amount), 2) AS invoiced,
               ROUND(SUM(i.amount_remaining), 2) AS open_amount
        FROM ap_invoices i
        JOIN poz_suppliers s ON s.supplier_id = i.supplier_id
        WHERE i.invoice_type = 'STANDARD'
        GROUP BY s.supplier_name, s.category
        ORDER BY invoiced DESC
        """,
    )
    return [dict(r) for r in rows]


def spend_by_source(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT invoice_source,
               COUNT(*) AS invoice_count,
               ROUND(SUM(invoice_amount), 2) AS invoiced
        FROM ap_invoices
        GROUP BY invoice_source
        ORDER BY invoiced DESC
        """,
    )
    return [dict(r) for r in rows]


def payroll_register(conn: sqlite3.Connection, period: str = "Jul-26") -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT p.person_number,
               p.first_name || ' ' || p.last_name AS employee_name,
               p.email,
               bu.name AS business_unit,
               d.name AS department,
               a.payroll_name,
               a.period_name,
               a.pay_date,
               r.currency_code,
               ROUND(SUM(CASE WHEN r.classification = 'Earnings' THEN r.result_value ELSE 0 END), 2) AS earnings,
               ROUND(SUM(CASE WHEN r.classification = 'Deduction' THEN r.result_value ELSE 0 END), 2) AS deductions,
               ROUND(SUM(CASE WHEN r.classification = 'Employer Cost' THEN r.result_value ELSE 0 END), 2) AS employer_cost,
               ROUND(SUM(CASE WHEN r.classification = 'Earnings' THEN r.result_value
                              WHEN r.classification = 'Deduction' THEN -r.result_value
                              ELSE 0 END), 2) AS net_pay
        FROM pay_run_results r
        JOIN pay_payroll_actions a ON a.payroll_action_id = r.payroll_action_id
        JOIN per_all_people_f p ON p.person_id = r.person_id
        JOIN per_all_assignments_m asg ON asg.assignment_id = r.assignment_id
        JOIN hr_all_organization_units bu ON bu.organization_id = asg.business_unit_id
        JOIN hr_all_organization_units d ON d.organization_id = asg.department_id
        WHERE a.period_name = ?
        GROUP BY p.person_number, employee_name, p.email, bu.name, d.name,
                 a.payroll_name, a.period_name, a.pay_date, r.currency_code
        ORDER BY bu.name, employee_name
        """,
        (period,),
    )
    return [dict(r) for r in rows]


def invoice_register(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = q(
        conn,
        """
        SELECT i.invoice_num,
               i.invoice_date,
               i.due_date,
               s.supplier_name,
               bu.name AS business_unit,
               i.invoice_type,
               i.invoice_source,
               i.approval_status,
               i.payment_status,
               i.currency_code,
               i.invoice_amount,
               i.amount_paid,
               i.amount_remaining,
               CAST(julianday('2026-09-14') - julianday(i.due_date) AS INTEGER) AS days_past_due
        FROM ap_invoices i
        JOIN poz_suppliers s ON s.supplier_id = i.supplier_id
        JOIN hr_all_organization_units bu ON bu.organization_id = i.business_unit_id
        ORDER BY i.invoice_date DESC
        LIMIT 400
        """,
    )
    return [dict(r) for r in rows]
