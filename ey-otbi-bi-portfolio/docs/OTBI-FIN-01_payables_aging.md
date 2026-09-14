# OTBI-FIN-01 – Payables Aging & Supplier Spend

**Tool:** OTBI  
**Subject area:** Payables Invoices - Transactions Real Time  
**Optional second SA:** Payables Payments - Disbursements Real Time (only if you add a measure from each)

## Business requirement

AP and procurement want:

- Open AP and overdue AP by BU
- Aging buckets Current / 1-30 / 31-60 / 61-90 / 90+
- Top suppliers by invoiced and by open amount
- Channel mix (Manual, FBDI, IDT, Scanned) for a shared-services quality review

## Analysis design

**Folder:** `/Shared Folders/Custom/Financials/Payables/OTBI-FIN-01 Aging Spend`

**Columns:**

- Invoice Number, Invoice Date, GL Date, Due Date
- Supplier Name, Supplier Category
- Business Unit, Invoice Source, Invoice Type
- Approval Status, Validation Status, Payment Status
- Invoice Amount, Amount Paid, Amount Remaining

**Filter:** Invoice Type = Standard; Payment Status in Unpaid, Partial.

**Aging bucket** (column formula):

```
CASE
  WHEN TIMESTAMPDIFF(SQL_TSI_DAY, "Due Date", CURRENT_DATE) <= 0 THEN 'Current'
  WHEN TIMESTAMPDIFF(SQL_TSI_DAY, "Due Date", CURRENT_DATE) <= 30 THEN '1-30'
  WHEN TIMESTAMPDIFF(SQL_TSI_DAY, "Due Date", CURRENT_DATE) <= 60 THEN '31-60'
  WHEN TIMESTAMPDIFF(SQL_TSI_DAY, "Due Date", CURRENT_DATE) <= 90 THEN '61-90'
  ELSE '90+'
END
```

**Prompts:** Business Unit (choice list), GL Date between, Supplier Name (search).

## Design decisions to defend

- Use **Amount Remaining**, not Invoice Amount, once partial payments exist.
- Credit memos kept off the aging dashboard (separate analysis) so buckets stay intuitive.
- OTBI hits live tables — no heavy cross-SA joins; supplier master is already in this SA.
- Pixel-perfect month-end pack for audit is BIP-FIN-01, not this dashboard.
- Never add Amount Remaining across mixed currencies; prompt by BU or use a reporting currency.

## Security

Payables invoice data security + `AP_INQUIRY` style privileges. Catalog to AP Analyst and BU Controller. Hide supplier bank attributes.

## Demo

`output/otbi_ap_aging.html` and `sql/otbi_ap_aging.sql`.
