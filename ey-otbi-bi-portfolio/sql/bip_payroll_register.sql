-- BIP-HCM-01 data model (SQL dataset)
-- Catalog: /Shared Folders/Custom/HCM/Payroll/PayrollRegisterDM
-- Parameters: P_PERIOD (string), P_BU (menu from BU query)
-- Bursting dataset below.

SELECT
    p.person_number                          AS PERSON_NUMBER,
    p.first_name || ' ' || p.last_name       AS EMPLOYEE_NAME,
    p.email                                  AS EMAIL,
    bu.name                                  AS BUSINESS_UNIT,
    d.name                                   AS DEPARTMENT,
    a.payroll_name                           AS PAYROLL_NAME,
    a.period_name                            AS PERIOD_NAME,
    a.pay_date                               AS PAY_DATE,
    r.currency_code                          AS CURRENCY_CODE,
    SUM(CASE WHEN r.classification = 'Earnings' THEN r.result_value ELSE 0 END)      AS EARNINGS,
    SUM(CASE WHEN r.classification = 'Deduction' THEN r.result_value ELSE 0 END)     AS DEDUCTIONS,
    SUM(CASE WHEN r.classification = 'Employer Cost' THEN r.result_value ELSE 0 END) AS EMPLOYER_COST,
    SUM(CASE WHEN r.classification = 'Earnings' THEN r.result_value
             WHEN r.classification = 'Deduction' THEN -r.result_value
             ELSE 0 END)                     AS NET_PAY
FROM pay_run_results r
JOIN pay_payroll_actions a
  ON a.payroll_action_id = r.payroll_action_id
JOIN per_all_people_f p
  ON p.person_id = r.person_id
JOIN per_all_assignments_m asg
  ON asg.assignment_id = r.assignment_id
JOIN hr_all_organization_units bu
  ON bu.organization_id = asg.business_unit_id
JOIN hr_all_organization_units d
  ON d.organization_id = asg.department_id
WHERE a.period_name = :P_PERIOD
  AND (bu.name = :P_BU OR :P_BU = 'ALL')
GROUP BY
    p.person_number, p.first_name, p.last_name, p.email,
    bu.name, d.name, a.payroll_name, a.period_name, a.pay_date, r.currency_code
ORDER BY bu.name, EMPLOYEE_NAME;

-- Bursting query (delivered BIP bursting schema)
-- KEY must match a repeating group in the XML (BUSINESS_UNIT)
SELECT
    bu.name                                              AS "KEY",
    'PAYROLL_REGISTER'                                   AS "TEMPLATE",
    'en-US'                                              AS "LOCALE",
    'PDF'                                                AS "OUTPUT_FORMAT",
    'true'                                               AS "SAVE_OUTPUT",
    'Payroll_Register_' || REPLACE(bu.name, ' ', '_')    AS "OUTPUT_NAME",
    'EMAIL'                                              AS "DEL_CHANNEL",
    'controller.' || LOWER(REPLACE(bu.name, ' ', '.')) || '@aether.example' AS "PARAMETER1",
    'Payroll register ' || :P_PERIOD                     AS "PARAMETER2",
    'Please find attached the payroll register.'         AS "PARAMETER3"
FROM hr_all_organization_units bu
WHERE bu.org_type = 'BU';
