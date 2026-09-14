# Oracle Cloud OTBI & BI Publisher – EY interview portfolio

Four reporting objects you can demo without a Fusion login: two **OTBI-style dashboards** and two **BI Publisher-style reports**. Data lives in SQLite tables named after Fusion objects. Design docs use the same subject areas, bursting keys, and security language EY Oracle roles expect.

## Projects

| Code | Type | What you demo |
| --- | --- | --- |
| [OTBI-HCM-01](docs/OTBI-HCM-01_workforce.md) | OTBI | Workforce headcount, hires, exits, attrition |
| [OTBI-FIN-01](docs/OTBI-FIN-01_payables_aging.md) | OTBI | AP aging buckets, overdue, supplier spend |
| [BIP-HCM-01](docs/BIP-HCM-01_payroll_register.md) | BIP | Payroll register + email bursting by BU |
| [BIP-FIN-01](docs/BIP-FIN-01_ap_invoice_register.md) | BIP | AP invoice register (audit pack) |

Interview script: [docs/INTERVIEW_TALK_TRACK.md](docs/INTERVIEW_TALK_TRACK.md)  
Architecture: [docs/architecture.md](docs/architecture.md)

## Run (Python 3.10+, stdlib only)

```bash
cd ey-otbi-bi-portfolio
python3 src/build_portfolio.py
python3 src/serve.py
```

Open http://127.0.0.1:8765/index.html

Tests:

```bash
python3 -m pytest tests/test_kpis.py -q
```

`pytest` is the only extra dependency (`pip install -r requirements.txt`).

## What to say while you click

1. **OTBI HCM** – assignment grain, as-of date, attrition is calculated from exit actions, not a delivered measure.  
2. **OTBI Payables** – Amount Remaining, credits off the aging view, invoice source mix for shared services.  
3. **BIP Payroll** – XML data model, RTF stand-in, bursting CSV with KEY = Business Unit.  
4. **BIP AP Register** – parameters, PDF/Excel from one model, OTBI stays for daily ops.

This is a **design + working demo**, not a live Fusion catalog. In an instance you would create the analyses and data models under `/Shared Folders/Custom/...` using the SQL and docs in this folder.
