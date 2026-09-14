# EY Oracle interview – OTBI & BIP talk track

Use the HTML dashboards as the demo. Use this page when they ask theory.

## 30-second intro

“I built four reporting objects the way we would on Fusion: two OTBI dashboards (HCM workforce and AP aging) and two BIP reports (payroll register with bursting, AP invoice register). I can walk subject area, grain, prompts, security, and why I did not use BIP for the dashboards.”

## OTBI vs BIP (they will ask)

| | OTBI | BIP |
| --- | --- | --- |
| Data | Subject areas on live tables | Data model (SQL, HCM extracts, CSV, HTTP) |
| Layout | Views / dashboards | RTF, eText, Excel, PDF |
| Users | Analysts | Operations, legal, integrations |
| Bursting | Catalog schedule, limited | First-class KEY / email / FTP |
| Cross-SA | Same Cloud app, common dims, measure from each | Any tables you are allowed to query |

## Subject areas used in this portfolio

- **Workforce Management - Worker Assignment Real Time** — assignment grain, default as-of sysdate, `PARAM_EFFECTIVE_DATE` for history. Not a full assignment-change history (use Assignment Event SA).
- **Payables Invoices - Transactions Real Time** — invoice grain, Amount Remaining for aging.

## Build OTBI (steps)

Reports and Analytics → Create Analysis → subject area → Criteria columns → filters → Results views → Prompts → save under `/Shared Folders/Custom` → Dashboard → permissions → test with a restricted user.

## Build BIP (steps)

Catalog → Data Model → SQL dataset → parameters + LOVs → get XML → Word template with repeating groups → upload RTF → create report → bursting query → ESS / schedule → unit test XML vs UI.

## Security

OTBI inherits Fusion data security. BIP SQL can **bypass** UI security if the BI report duty is too broad — call that out; use roles, data filters, and do not expose PII in bursting emails.

## Performance

- OTBI: fewer columns, filter early, avoid cross-SA unless required, no “starts with %”.
- BIP: binds on indexed dates, burst large outputs, don’t `SELECT *`.

## Likely case questions

**“Headcount does not match Core HR.”**  
Check as-of date, assignment type E vs C, latest change flag, primary vs all assignments, future-dated rows, security profile.

**“Aging does not match AP trial balance.”**  
Credits included, prepaid invoices, accounted vs entered amount, BU vs ledger, as-of vs run date, unvalidated invoices.

**“Payroll PDF wrong net pay.”**  
Element classification mapping, voided runs, multiple assignments, currency, template summing hidden rows.

**“Can OTBI email a PDF per supplier?”**  
Poor fit. BIP bursting.

**“Can I join workers to AP invoices in one OTBI analysis?”**  
Not cleanly across HCM and ERP apps. Warehouse / FAW / BIP SQL if tables are in the same reporting DB and security allows.

## Project stories (STAR)

**HCM OTBI:** HR needed attrition they could slice by BU. I used assignment real time for HC and action events for exits, documented as-of date, tested manager security.

**Payables OTBI:** Shared services wanted source mix after an OCR rollout. Aging on amount remaining, credits excluded.

**Payroll BIP:** Controllers needed one PDF each. Data model at employee grain, bursting KEY = BU, ESS after payroll confirm.

**AP BIP:** Audit pack with parameters and Excel + PDF from one data model.

## Questions to ask them

- OTBI/BIP only, or OIC + HDL + Fast Formula as well?
- HCM, ERP, or both?
- Instance access in the first weeks, or design-only?
