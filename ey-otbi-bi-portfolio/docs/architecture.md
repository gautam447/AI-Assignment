# Architecture – OTBI vs BIP vs this demo

```text
                    ┌──────────────────────────┐
                    │  Oracle Fusion Cloud     │
                    │  HCM / ERP transactions  │
                    └────────────┬─────────────┘
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
    OTBI subject areas     BIP data models        FAW / warehouse
    (real-time analyses)   (SQL + template)     (cross-pillar analytics)
           │                     │
           ▼                     ▼
    Dashboards, prompts    PDF / Excel / XML
    ad-hoc, drill          bursting, ESS jobs
```

## When to pick what (interview one-liner)

- **OTBI:** business user, real-time, dashboard, subject-area columns, same security as the app.
- **BIP:** pixel-perfect, bursting, XML for integrations, payroll/legal layouts, SQL data model.
- **FAW / ADW:** historical trend, mixed HCM+ERP grains, heavy ETL — not this portfolio.
- **Smart View / FRS:** GL and financial statements; not transactional AP/HCM ops.

## Demo mapping

This repo does **not** log into Fusion. It uses SQLite tables named after Fusion objects, SQL that matches OTBI/BIP patterns, and HTML that stands in for dashboards and RTF output so you can walk the design live.

| Fusion object | Demo table |
| --- | --- |
| PER_ALL_PEOPLE_F | per_all_people_f |
| PER_ALL_ASSIGNMENTS_M | per_all_assignments_m |
| AP_INVOICES_ALL | ap_invoices |
| POZ_SUPPLIERS | poz_suppliers |
| Payroll run results | pay_run_results |

## Delivery folder convention (say this)

`/Shared Folders/Custom/<Pillar>/<Module>/<ProjectCode> <Name>`  
Never edit `/Shared Folders/Human Capital Management` delivered content — copy/customize.
