#!/usr/bin/env python3
"""Generate a Fusion-like SQLite warehouse for OTBI and BI Publisher demos.

Tables are named after Oracle Fusion Cloud objects so interview discussion
maps 1:1 to subject areas and BIP data models.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "aether_fusion_demo.db"

BUSINESS_UNITS = [
    ("Aether India BU", "INDIA_LE", "INR"),
    ("Aether UK BU", "UK_LE", "GBP"),
    ("Aether US BU", "US_LE", "USD"),
]
DEPARTMENTS = [
    "Finance",
    "Human Resources",
    "Procurement",
    "Sales",
    "Information Technology",
    "Manufacturing",
    "Internal Audit",
]
LOCATIONS = [
    ("Gurgaon Hub", "IN"),
    ("Noida Campus", "IN"),
    ("London Canary Wharf", "GB"),
    ("New York Midtown", "US"),
]
JOBS = [
    ("Analyst", "IC1", 1.0),
    ("Senior Analyst", "IC2", 1.0),
    ("Consultant", "IC3", 1.0),
    ("Manager", "M1", 1.0),
    ("Senior Manager", "M2", 1.0),
    ("Director", "M3", 1.0),
]
GRADES = ["G1", "G2", "G3", "G4", "G5", "G6"]
SUPPLIERS = [
    ("Vantage IT Services", "IT"),
    ("Northwind Facilities", "Facilities"),
    ("Helios Logistics", "Logistics"),
    ("Pinnacle Staffing", "Contingent Labor"),
    ("Summit Professional", "Professional Services"),
    ("Aria Office Supplies", "Indirect"),
    ("Vertex Cloud Hosting", "SaaS"),
    ("Harbor Travel Desk", "T&E"),
]
INVOICE_SOURCES = ["MANUAL", "IDT", "FBDI", "SCANNED"]
PAY_ELEMENTS = [
    ("Basic Salary", "Earnings"),
    ("HRA", "Earnings"),
    ("Special Allowance", "Earnings"),
    ("Employer PF", "Employer Cost"),
    ("Income Tax", "Deduction"),
    ("Employee PF", "Deduction"),
]

FIRST = [
    "Aarav", "Ananya", "Rohan", "Priya", "Ishaan", "Meera", "Kabir", "Nisha",
    "James", "Olivia", "Noah", "Amelia", "Liam", "Sophia", "Ethan", "Mia",
]
LAST = [
    "Sharma", "Patel", "Nair", "Reddy", "Kapoor", "Iyer", "Khan", "Das",
    "Walker", "Bennett", "Hughes", "Foster", "Reed", "Coleman", "Hayes", "Cole",
]


def _date(iso: str) -> date:
    return date.fromisoformat(iso)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS pay_run_results;
        DROP TABLE IF EXISTS pay_payroll_actions;
        DROP TABLE IF EXISTS ap_invoice_payments;
        DROP TABLE IF EXISTS ap_invoice_lines;
        DROP TABLE IF EXISTS ap_invoices;
        DROP TABLE IF EXISTS poz_suppliers;
        DROP TABLE IF EXISTS per_assignment_actions;
        DROP TABLE IF EXISTS per_all_assignments_m;
        DROP TABLE IF EXISTS per_all_people_f;
        DROP TABLE IF EXISTS hr_all_organization_units;
        DROP TABLE IF EXISTS per_jobs;
        DROP TABLE IF EXISTS per_locations;

        CREATE TABLE hr_all_organization_units (
            organization_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            org_type TEXT NOT NULL,
            legal_entity TEXT,
            currency_code TEXT,
            country TEXT
        );

        CREATE TABLE per_locations (
            location_id INTEGER PRIMARY KEY,
            location_name TEXT NOT NULL,
            country TEXT NOT NULL
        );

        CREATE TABLE per_jobs (
            job_id INTEGER PRIMARY KEY,
            job_name TEXT NOT NULL,
            job_family TEXT NOT NULL,
            fte REAL NOT NULL
        );

        CREATE TABLE per_all_people_f (
            person_id INTEGER PRIMARY KEY,
            person_number TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            gender TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            original_hire_date TEXT NOT NULL,
            email TEXT NOT NULL
        );

        CREATE TABLE per_all_assignments_m (
            assignment_id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            assignment_number TEXT NOT NULL,
            assignment_type TEXT NOT NULL,
            assignment_status TEXT NOT NULL,
            effective_start_date TEXT NOT NULL,
            effective_end_date TEXT NOT NULL,
            effective_latest_change TEXT NOT NULL,
            business_unit_id INTEGER NOT NULL,
            department_id INTEGER NOT NULL,
            location_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            grade TEXT NOT NULL,
            fte REAL NOT NULL,
            headcount REAL NOT NULL,
            manager_id INTEGER,
            annual_salary REAL NOT NULL,
            currency_code TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES per_all_people_f(person_id)
        );

        CREATE TABLE per_assignment_actions (
            action_id INTEGER PRIMARY KEY,
            assignment_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_reason TEXT NOT NULL,
            effective_date TEXT NOT NULL
        );

        CREATE TABLE poz_suppliers (
            supplier_id INTEGER PRIMARY KEY,
            supplier_name TEXT NOT NULL,
            category TEXT NOT NULL,
            site TEXT NOT NULL,
            country TEXT NOT NULL
        );

        CREATE TABLE ap_invoices (
            invoice_id INTEGER PRIMARY KEY,
            invoice_num TEXT NOT NULL,
            invoice_date TEXT NOT NULL,
            gl_date TEXT NOT NULL,
            invoice_type TEXT NOT NULL,
            invoice_source TEXT NOT NULL,
            supplier_id INTEGER NOT NULL,
            business_unit_id INTEGER NOT NULL,
            invoice_amount REAL NOT NULL,
            amount_paid REAL NOT NULL,
            amount_remaining REAL NOT NULL,
            currency_code TEXT NOT NULL,
            due_date TEXT NOT NULL,
            approval_status TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            payment_status TEXT NOT NULL,
            terms_days INTEGER NOT NULL
        );

        CREATE TABLE ap_invoice_lines (
            line_id INTEGER PRIMARY KEY,
            invoice_id INTEGER NOT NULL,
            line_number INTEGER NOT NULL,
            line_type TEXT NOT NULL,
            amount REAL NOT NULL,
            cost_center TEXT NOT NULL
        );

        CREATE TABLE ap_invoice_payments (
            payment_id INTEGER PRIMARY KEY,
            invoice_id INTEGER NOT NULL,
            payment_date TEXT NOT NULL,
            payment_amount REAL NOT NULL,
            payment_method TEXT NOT NULL
        );

        CREATE TABLE pay_payroll_actions (
            payroll_action_id INTEGER PRIMARY KEY,
            payroll_name TEXT NOT NULL,
            period_name TEXT NOT NULL,
            pay_date TEXT NOT NULL,
            business_unit_id INTEGER NOT NULL
        );

        CREATE TABLE pay_run_results (
            run_result_id INTEGER PRIMARY KEY,
            payroll_action_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            assignment_id INTEGER NOT NULL,
            element_name TEXT NOT NULL,
            classification TEXT NOT NULL,
            result_value REAL NOT NULL,
            currency_code TEXT NOT NULL
        );
        """
    )


def seed(conn: sqlite3.Connection, seed_value: int = 47) -> None:
    rng = random.Random(seed_value)
    today = date(2026, 9, 14)

    bu_ids: dict[str, int] = {}
    dept_ids: dict[str, int] = {}
    org_id = 100
    for name, le, ccy in BUSINESS_UNITS:
        country = "IN" if "India" in name else ("GB" if "UK" in name else "US")
        conn.execute(
            "INSERT INTO hr_all_organization_units VALUES (?,?,?,?,?,?)",
            (org_id, name, "BU", le, ccy, country),
        )
        bu_ids[name] = org_id
        org_id += 1
    for dept in DEPARTMENTS:
        conn.execute(
            "INSERT INTO hr_all_organization_units VALUES (?,?,?,?,?,?)",
            (org_id, dept, "DEPARTMENT", None, None, None),
        )
        dept_ids[dept] = org_id
        org_id += 1

    loc_ids = []
    for i, (name, country) in enumerate(LOCATIONS, start=1):
        conn.execute("INSERT INTO per_locations VALUES (?,?,?)", (i, name, country))
        loc_ids.append(i)

    job_ids = []
    for i, (name, family, fte) in enumerate(JOBS, start=1):
        conn.execute("INSERT INTO per_jobs VALUES (?,?,?,?)", (i, name, family, fte))
        job_ids.append(i)

    people = []
    person_id = 1000
    for _ in range(420):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        gender = rng.choice(["F", "F", "M", "M", "X"])
        hire = today - timedelta(days=rng.randint(60, 4200))
        dob = hire - timedelta(days=rng.randint(22 * 365, 55 * 365))
        person_number = f"EMP{person_id}"
        email = f"{first.lower()}.{last.lower()}{person_id}@aether.example"
        conn.execute(
            "INSERT INTO per_all_people_f VALUES (?,?,?,?,?,?,?,?)",
            (
                person_id,
                person_number,
                first,
                last,
                gender,
                dob.isoformat(),
                hire.isoformat(),
                email,
            ),
        )
        people.append((person_id, hire))
        person_id += 1

    assignment_id = 5000
    assignments = []
    for person_id, hire in people:
        bu_name, _, ccy = rng.choice(BUSINESS_UNITS)
        dept = rng.choice(DEPARTMENTS)
        loc = rng.choice(loc_ids)
        job = rng.choice(job_ids)
        grade = rng.choice(GRADES)
        salary = round(rng.uniform(35000, 220000), 2)
        # ~12% terminated in the last 18 months
        terminated = rng.random() < 0.12
        term_date = None
        status = "ACTIVE"
        end_date = date(4712, 12, 31)
        if terminated:
            term_date = hire + timedelta(days=rng.randint(90, max(90, (today - hire).days)))
            if term_date > today:
                term_date = today - timedelta(days=rng.randint(5, 200))
            if term_date < hire:
                term_date = hire + timedelta(days=30)
            status = "INACTIVE"
            end_date = term_date
        conn.execute(
            """INSERT INTO per_all_assignments_m
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                assignment_id,
                person_id,
                f"E{assignment_id}",
                "E",
                status,
                hire.isoformat(),
                end_date.isoformat(),
                "Y",
                bu_ids[bu_name],
                dept_ids[dept],
                loc,
                job,
                grade,
                1.0,
                0.0 if terminated else 1.0,
                None,
                salary,
                ccy,
            ),
        )
        conn.execute(
            "INSERT INTO per_assignment_actions VALUES (?,?,?,?,?,?)",
            (
                assignment_id,
                assignment_id,
                person_id,
                "HIRE",
                "New Hire",
                hire.isoformat(),
            ),
        )
        if terminated and term_date:
            conn.execute(
                "INSERT INTO per_assignment_actions VALUES (?,?,?,?,?,?)",
                (
                    assignment_id + 20000,
                    assignment_id,
                    person_id,
                    "TERMINATION",
                    rng.choice(["Resignation", "Mutual Separation", "End of Contract", "Performance"]),
                    term_date.isoformat(),
                ),
            )
        assignments.append((assignment_id, person_id, bu_ids[bu_name], status, ccy, salary))
        assignment_id += 1

    supplier_rows = []
    for i, (name, category) in enumerate(SUPPLIERS, start=1):
        country = rng.choice(["IN", "GB", "US"])
        conn.execute(
            "INSERT INTO poz_suppliers VALUES (?,?,?,?,?)",
            (i, name, category, f"{name.split()[0]}-HQ", country),
        )
        supplier_rows.append(i)

    invoice_id = 90000
    line_id = 1
    payment_id = 1
    for n in range(1800):
        inv_date = today - timedelta(days=rng.randint(1, 400))
        terms = rng.choice([15, 30, 45, 60])
        due = inv_date + timedelta(days=terms)
        amount = round(rng.uniform(250, 185000), 2)
        bu_name, _, ccy = rng.choice(BUSINESS_UNITS)
        source = rng.choices(INVOICE_SOURCES, weights=[40, 15, 25, 20])[0]
        inv_type = rng.choices(["STANDARD", "CREDIT", "PREPAYMENT"], weights=[88, 8, 4])[0]
        if inv_type == "CREDIT":
            amount = -abs(amount)
        approval = rng.choices(["APPROVED", "INITIATED", "REJECTED"], weights=[86, 10, 4])[0]
        validation = rng.choices(["VALIDATED", "UNVALIDATED", "NEEDS REVALIDATION"], weights=[90, 7, 3])[0]
        # Payment behavior: older approved invoices more likely paid
        age = (today - due).days
        if approval != "APPROVED" or inv_type == "PREPAYMENT":
            paid = 0.0
            pay_status = "UNPAID"
        elif age < -10 and rng.random() < 0.35:
            paid = 0.0
            pay_status = "UNPAID"
        elif rng.random() < 0.55:
            paid = amount
            pay_status = "PAID"
        elif rng.random() < 0.5:
            paid = round(amount * rng.uniform(0.3, 0.7), 2)
            pay_status = "PARTIAL"
        else:
            paid = 0.0
            pay_status = "UNPAID"
        remaining = round(amount - paid, 2)
        conn.execute(
            """INSERT INTO ap_invoices VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                invoice_id,
                f"INV-{invoice_id}",
                inv_date.isoformat(),
                inv_date.isoformat(),
                inv_type,
                source,
                rng.choice(supplier_rows),
                bu_ids[bu_name],
                amount,
                paid,
                remaining,
                ccy,
                due.isoformat(),
                approval,
                validation,
                pay_status,
                terms,
            ),
        )
        cc = rng.choice(["FIN-100", "IT-200", "PRC-300", "OPS-400", "SAL-500"])
        conn.execute(
            "INSERT INTO ap_invoice_lines VALUES (?,?,?,?,?,?)",
            (line_id, invoice_id, 1, "ITEM", amount, cc),
        )
        line_id += 1
        if paid:
            pay_date = inv_date + timedelta(days=rng.randint(5, terms + 20))
            if pay_date > today:
                pay_date = today
            conn.execute(
                "INSERT INTO ap_invoice_payments VALUES (?,?,?,?,?)",
                (
                    payment_id,
                    invoice_id,
                    pay_date.isoformat(),
                    paid,
                    rng.choice(["EFT", "CHECK", "WIRE"]),
                ),
            )
            payment_id += 1
        invoice_id += 1

    active = [a for a in assignments if a[3] == "ACTIVE"]
    periods = [("May-26", date(2026, 5, 31)), ("Jun-26", date(2026, 6, 30)), ("Jul-26", date(2026, 7, 31))]
    payroll_action_id = 1
    run_result_id = 1
    for period_name, pay_date in periods:
        for bu_name, _, ccy in BUSINESS_UNITS:
            bu_id = bu_ids[bu_name]
            conn.execute(
                "INSERT INTO pay_payroll_actions VALUES (?,?,?,?,?)",
                (payroll_action_id, f"{bu_name} Monthly", period_name, pay_date.isoformat(), bu_id),
            )
            for asg_id, person_id, asg_bu, status, asg_ccy, salary in active:
                if asg_bu != bu_id:
                    continue
                monthly = salary / 12
                splits = {
                    "Basic Salary": monthly * 0.5,
                    "HRA": monthly * 0.2,
                    "Special Allowance": monthly * 0.3,
                    "Employer PF": monthly * 0.12,
                    "Income Tax": monthly * 0.18,
                    "Employee PF": monthly * 0.12,
                }
                for element, classification in PAY_ELEMENTS:
                    conn.execute(
                        "INSERT INTO pay_run_results VALUES (?,?,?,?,?,?,?,?)",
                        (
                            run_result_id,
                            payroll_action_id,
                            person_id,
                            asg_id,
                            element,
                            classification,
                            round(splits[element], 2),
                            asg_ccy,
                        ),
                    )
                    run_result_id += 1
            payroll_action_id += 1

    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Aether Fusion demo warehouse")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    conn = connect(args.db)
    schema(conn)
    seed(conn, args.seed)
    people = conn.execute("SELECT COUNT(*) FROM per_all_people_f").fetchone()[0]
    invoices = conn.execute("SELECT COUNT(*) FROM ap_invoices").fetchone()[0]
    pays = conn.execute("SELECT COUNT(*) FROM pay_run_results").fetchone()[0]
    conn.close()
    print(f"Wrote {args.db}")
    print(f"People={people} Invoices={invoices} Payroll results={pays}")
    print(f"Generated at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
