# OTBI-HCM-01 – Workforce Headcount & Attrition

**Client:** Aether Global Industries (demo)  
**Tool:** Oracle Transactional Business Intelligence  
**Subject area:** Workforce Management - Worker Assignment Real Time  
**Companion events:** Workforce Management - Work Relationship Real Time / assignment actions  

## Business requirement

HR leadership needs a daily as-of-date view of:

- Active headcount and FTE by business unit, department, location, job, grade
- Rolling 12-month hires and exits
- Attrition % and exit reason mix
- Gender mix for DEI reporting

They rejected a BIP extract because they want prompts, drill, and a dashboard they can change without an RTF cycle.

## Why OTBI, not BIP

| Need | Choice |
| --- | --- |
| Interactive filters, pivot, dashboard | OTBI |
| Pixel-perfect PDF emailed to 40 payroll partners | BIP |
| Cross-module worker + invoice | Not OTBI — use FAW / warehouse |

## Analysis design

**Folder:** `/Shared Folders/Custom/HCM/Workforce/OTBI-HCM-01 Headcount Attrition`

**Criteria (columns):**

- Worker Assignment Details: Assignment Status, Assignment Type, Headcount, FTE
- Business Unit, Department, Location, Job, Grade
- Worker: Person Number, Gender (hidden in dashboard, used in view)

**Filters:**

- Assignment Type is equal to `E`
- Assignment Status is equal to `Active`
- Headcount is greater than `0`

**Prompt:** `P_AS_OF` calendar prompt. Advanced SQL prefix:

```sql
SET VARIABLE PARAM_EFFECTIVE_DATE='@{P_AS_OF}{@{system.currentTime}[yyyy-MM-dd]}';
```

**Calculated column – Attrition %** (dashboard narrative / Excel export):

`Exits 12m / Closing headcount * 100`

Exits come from a second analysis (union or dashboard with two analyses). Do not invent attrition inside Worker Assignment Real Time — that SA does not store assignment history.

## Dashboard

1. KPI row: Headcount, FTE, Hires 12m, Exits 12m, Attrition %
2. Bar: Headcount by BU
3. Pivot: Department x Gender
4. Pie: Exit reason
5. Table: drill to person number (action link to worker quick report)

## Security

- Duty: `ORA_PER_REST_SERVICE_ACCESS_WORKFORCE_INFORMATION` plus custom reporting duty
- Data: same as HCM (area of responsibility / person security profile)
- Catalog: grant to `Aether HR Analyst` role only; no open to authenticated user

## Build steps in Fusion (say this in the interview)

1. Reports and Analytics → Create → Analysis → choose subject area  
2. Criteria → folders → add columns → filters  
3. Results → table / pivot / graph views  
4. Prompts tab → effective date  
5. Dashboard → columns → dashboard prompts → save to Custom folder  
6. Catalog permissions → test with HR Analyst and with a restricted line manager  
7. Schedule if they want a Monday 6am snapshot to email

## Test cases

| ID | Scenario | Expected |
| --- | --- | --- |
| HC-01 | As-of today | Active assignments only, latest change Y |
| HC-02 | As-of 12 months ago | Historical assignment via PARAM_EFFECTIVE_DATE |
| HC-03 | Manager login | Only subordinates |
| HC-04 | Contingent workers | Excluded (type C) unless stakeholder asks |

## Demo warehouse

Run `python3 src/build_portfolio.py` and open `output/otbi_hcm_workforce.html`. SQL twin: `sql/otbi_hcm_workforce.sql`.
