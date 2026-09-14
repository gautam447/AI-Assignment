# BIP-HCM-01 – Monthly Payroll Register (with bursting)

**Tool:** Oracle BI Publisher  
**Catalog:** `/Shared Folders/Custom/HCM/Payroll/`  
**Consumers:** Payroll operations, BU finance controllers  

## Why BIP, not OTBI

Payslips and registers need a fixed layout, totals, statutory labels, and one PDF per BU emailed on pay confirmation. OTBI cannot burst pixel-perfect PDFs with an RTF template.

## Objects

| Object | Name | Purpose |
| --- | --- | --- |
| Data model | `PayrollRegisterDM` | SQL dataset + bursting dataset |
| Parameter | `P_PERIOD` | e.g. Jul-26 |
| Parameter | `P_BU` | menu, default ALL |
| Template | `PAYROLL_REGISTER.rtf` | Logo, repeating group `G_EMPLOYEE`, totals |
| Report | `PayrollRegister.xdo` | PDF + Excel |
| Bursting | Email by `BUSINESS_UNIT` | KEY = BU name |

## Data model notes

- Parent query aggregates run results to employee grain (one XML node per person).
- Earnings / Deductions / Employer Cost / Net Pay are conditional sums on classification — same pattern as Fusion element classifications.
- Do not join OTBI subject areas here; BIP SQL uses payroll result tables (demo: `pay_run_results`).
- LOV for `P_BU`: `SELECT name FROM hr_all_organization_units WHERE org_type = 'BU'`.

## Bursting

KEY in the bursting query must equal a repeating group value in the XML (`BUSINESS_UNIT`). Channel EMAIL, `PARAMETER1` = recipient, `PARAMETER2` = subject, `PARAMETER3` = body. Save output true so payroll can re-download from catalog history.

In this demo, `output/bip_payroll_bursting.csv` is the bursting result set and `output/bip_payroll_data_model.xml` is the XML a template would bind to.

## ESS / payroll flow

1. Payroll run completes  
2. Custom ESS job or payroll flow task submits the BIP report  
3. Burst PDFs land in email + catalog  
4. Failure: BIP job status + bursting log; re-run for one KEY if a controller bounced  

## Test cases

| ID | Check |
| --- | --- |
| PR-01 | Net pay = earnings − deductions |
| PR-02 | Inactive assignments have no Jul-26 rows |
| PR-03 | Burst file has one row per BU |
| PR-04 | Parameter ALL returns every BU |

SQL: `sql/bip_payroll_register.sql`. Preview: `output/bip_payroll_register.html`.
