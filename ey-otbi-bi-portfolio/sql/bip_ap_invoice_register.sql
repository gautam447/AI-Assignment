-- BIP-FIN-01 AP Invoice Register data model
-- Catalog: /Shared Folders/Custom/Financials/Payables/APInvoiceRegisterDM
-- Parameters: P_START_DATE, P_END_DATE, P_BU, P_STATUS

SELECT
    i.invoice_num            AS INVOICE_NUM,
    i.invoice_date           AS INVOICE_DATE,
    i.due_date               AS DUE_DATE,
    s.supplier_name          AS SUPPLIER_NAME,
    bu.name                  AS BUSINESS_UNIT,
    i.invoice_type           AS INVOICE_TYPE,
    i.invoice_source         AS INVOICE_SOURCE,
    i.approval_status        AS APPROVAL_STATUS,
    i.validation_status      AS VALIDATION_STATUS,
    i.payment_status         AS PAYMENT_STATUS,
    i.currency_code          AS CURRENCY_CODE,
    i.invoice_amount         AS INVOICE_AMOUNT,
    i.amount_paid            AS AMOUNT_PAID,
    i.amount_remaining       AS AMOUNT_REMAINING
FROM ap_invoices i
JOIN poz_suppliers s
  ON s.supplier_id = i.supplier_id
JOIN hr_all_organization_units bu
  ON bu.organization_id = i.business_unit_id
WHERE i.invoice_date BETWEEN :P_START_DATE AND :P_END_DATE
  AND (bu.name = :P_BU OR :P_BU = 'ALL')
  AND (i.payment_status = :P_STATUS OR :P_STATUS = 'ALL')
ORDER BY i.invoice_date DESC, i.invoice_num;
