-- OTBI-FIN-01 equivalent SQL
-- Subject area: Payables Invoices - Transactions Real Time
-- Measures: Invoice Amount, Amount Paid, Amount Remaining
-- Typical filters: Invoice Type = Standard, BU prompt, GL Date between

SELECT
    CASE
        WHEN JULIANDAY('2026-09-14') - JULIANDAY(i.due_date) <= 0 THEN 'Current'
        WHEN JULIANDAY('2026-09-14') - JULIANDAY(i.due_date) BETWEEN 1 AND 30 THEN '1-30'
        WHEN JULIANDAY('2026-09-14') - JULIANDAY(i.due_date) BETWEEN 31 AND 60 THEN '31-60'
        WHEN JULIANDAY('2026-09-14') - JULIANDAY(i.due_date) BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END                              AS "Aging Bucket",
    bu.name                          AS "Business Unit",
    s.supplier_name                  AS "Supplier Name",
    i.invoice_source                 AS "Invoice Source",
    COUNT(*)                         AS "Invoice Count",
    ROUND(SUM(i.invoice_amount), 2)  AS "Invoice Amount",
    ROUND(SUM(i.amount_paid), 2)     AS "Amount Paid",
    ROUND(SUM(i.amount_remaining), 2) AS "Amount Remaining"
FROM ap_invoices i
JOIN poz_suppliers s
  ON s.supplier_id = i.supplier_id
JOIN hr_all_organization_units bu
  ON bu.organization_id = i.business_unit_id
WHERE i.invoice_type = 'STANDARD'
  AND i.payment_status IN ('UNPAID', 'PARTIAL')
GROUP BY 1, bu.name, s.supplier_name, i.invoice_source
ORDER BY "Amount Remaining" DESC;
