# BIP-FIN-01 – AP Invoice Register

**Tool:** BI Publisher  
**Use:** Month-end audit pack, shared-services quality review  

## Requirement

Finance wants a filterable invoice listing (PDF and Excel) with supplier, BU, source, approval, validation, payment status, and remaining amount. Daily ops stay on OTBI-FIN-01; this report is the signed month-end artifact.

## Parameters

| Parameter | Type | Default |
| --- | --- | --- |
| P_START_DATE | Date | First day of prior period |
| P_END_DATE | Date | Period end |
| P_BU | Menu | ALL |
| P_STATUS | Menu | ALL |

## Layout

RTF header: client logo, report name, parameter values, run date/time, requested by. Table repeating group `G_INVOICE`. Footer: page x of y, confidential. Excel template for the same data model so AP can pivot without a second query.

## Performance

- Filter on `invoice_date` / `gl_date` with bind parameters (no `TRUNC` on the column if an index exists in Fusion).
- Avoid `SELECT *`.
- If volume > 200k rows, split by BU bursting or schedule off-peak.

## Demo

`sql/bip_ap_invoice_register.sql`  
`output/bip_ap_invoice_register.html`  
`output/bip_ap_invoice_data_model.xml`
