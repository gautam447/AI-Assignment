-- OTBI-HCM-01 equivalent SQL
-- Subject area: Workforce Management - Worker Assignment Real Time
-- In Fusion you would pick columns in Criteria, not write SQL.
-- Advanced tab can add:
--   SET VARIABLE PARAM_EFFECTIVE_DATE='@{P_AS_OF}{2026-09-14}';

SELECT
    bu.name                          AS "Business Unit",
    dept.name                        AS "Department",
    loc.location_name                AS "Location",
    job.job_name                     AS "Job",
    asg.grade                        AS "Grade",
    SUM(asg.headcount)               AS "Headcount",
    SUM(asg.fte)                     AS "FTE",
    ROUND(AVG(asg.annual_salary), 0) AS "Average Annual Salary"
FROM per_all_assignments_m asg
JOIN per_all_people_f p
  ON p.person_id = asg.person_id
JOIN hr_all_organization_units bu
  ON bu.organization_id = asg.business_unit_id
JOIN hr_all_organization_units dept
  ON dept.organization_id = asg.department_id
JOIN per_locations loc
  ON loc.location_id = asg.location_id
JOIN per_jobs job
  ON job.job_id = asg.job_id
WHERE asg.assignment_type = 'E'
  AND asg.effective_latest_change = 'Y'
  AND DATE('2026-09-14') BETWEEN asg.effective_start_date AND asg.effective_end_date
  AND asg.assignment_status = 'ACTIVE'
GROUP BY
    bu.name, dept.name, loc.location_name, job.job_name, asg.grade
ORDER BY "Headcount" DESC;

-- Attrition companion query (Work Relationship / Assignment Action events)
SELECT
    action_reason AS "Exit Reason",
    COUNT(*)      AS "Exits 12m"
FROM per_assignment_actions
WHERE action_type = 'TERMINATION'
  AND effective_date >= DATE('2026-09-14', '-12 months')
  AND effective_date <= DATE('2026-09-14')
GROUP BY action_reason;
