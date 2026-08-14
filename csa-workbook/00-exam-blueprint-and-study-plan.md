# Chapter 00 — Exam Blueprint and Study Plan

## What CSA certifies

The Certified System Administrator exam confirms you can **configure, implement, and maintain** a ServiceNow instance. Passing means you can:

- Navigate Next Experience and find applications/modules quickly
- Configure lists, forms, tasks, notifications, and analytics
- Deliver self-service with Knowledge, Catalog, Workflow Studio, and Virtual Agent
- Manage the data model, access control, imports, CMDB/CSDM, and instance security
- Move configuration with update sets and apply basic client/server scripting safely

CSA is the foundation credential. Most CIS and CAD paths expect this knowledge.

## Exam structure

| Item | Detail |
| --- | --- |
| Questions | 60 |
| Time | 90 minutes (~90 seconds per question) |
| Types | Single-answer multiple choice; multi-select (“select 2/3”) |
| Scoring | 1 point per question; no partial credit |
| Delivery | Pearson VUE test center or OnVUE online proctor |
| Registration window | 90 days after you pay to schedule and sit |
| Result | Immediate conditional pass/fail plus domain percentages |
| Maintenance | Annual delta exam + CMP fee |

Read multi-select stems slowly. If it says “select 3,” select exactly three. Missing one correct option zeros the whole item.

## Official domain map (January 2026)

| # | Domain | Weight | Subtopics you must cover |
| --- | ---: | ---: | --- |
| 1 | Platform Overview and Navigation | 7% | Platform overview; capabilities and services; the instance; Next Experience Unified Navigation |
| 2 | Instance Configuration | 10% | Installing applications and plugins; personalizing/customizing; common user interfaces |
| 3 | Configuring Applications for Collaboration | 20% | Lists, filters, tags; list/form anatomy; form configuration; templates and save options; advanced form config; task management; Visual Task Boards; visualizations, dashboards, Platform Analytics; notifications |
| 4 | Self-Service and Automation | 20% | Knowledge Management; Service Catalog; Workflow Studio; Virtual Agent |
| 5 | Database Management and Platform Security | **30%** | Data schema; application/access control; importing data; CMDB and CSDM; Security Center; Shared Responsibility Model |
| 6 | Data Migration and Integration | 13% | UI Policies; Business Rules; system update sets; scripting in ServiceNow |

**Study math:** Domain 5 alone is almost one-third of the exam. Domains 3 + 4 + 5 = **70%**. Do not spend equal time on all six chapters.

## Recommended experience

ServiceNow lists:

- Database and system-administration concepts
- Access to admin applications and modules
- Familiarity with IT help-desk / Incident, Problem, Change
- Three to six months using or maintaining an instance

If you do not have job time, a **Personal Developer Instance** plus the labs in Chapter 08 is the substitute.

## Official prep (use these first)

1. **Welcome to ServiceNow** (Now Learning)
2. **ServiceNow Administration Fundamentals** (Now Learning)
3. Product docs for your exam release
4. Optional: MeasureUp official practice exams (see ServiceNow KB0013408)

Community notes, Udemy banks, and this workbook are secondary. The exam is written from official training and docs.

## Sample items from the official blueprint

These are published by ServiceNow. Study the *reason*, not just the letter.

1. Which application is available to **all users**?  
   **Self Service** — users with no role still get Self-Service. Change, Incident, and Facilities require roles.

2. Which module lists tasks assigned to a user’s **group** but not yet to a person?  
   **My Groups Work** — not “My Teams Work/Tasks.”

3. What is a **transform map**?  
   A map of Import Set fields to fields on an existing target table.

4. Multiple Choice, Single Line Text, and Select Box are **variable types** (Catalog).

5. Scripting language on the platform is **JavaScript**.

## How questions are actually written

Most items are scenario-based:

- “A fulfiller cannot see a field. What should you check first?”
- “You need data to be mandatory even from imports and web services. What do you use?”
- “Which save option creates a new record and leaves the form open?”

Memorizing definitions is not enough. You must know **where it runs** (client vs server), **what it captures** (update set vs data), and **what happens if it fails** (ACL deny, coalesce miss, unpublished knowledge article).

## Suggested study sequence

| Phase | Focus | Outcome |
| --- | --- | --- |
| 1 | Chapter 00 + PDI setup + Chapter 01–02 | You can navigate, impersonate, brand, and manage users/groups/roles |
| 2 | Chapter 05 (full) | You can draw table relationships, write ACL evaluation order, import with coalesce, explain CMDB/CSDM/Security Center |
| 3 | Chapter 04 | You can build a catalog item, knowledge article, flow, and describe Virtual Agent |
| 4 | Chapter 03 | You can configure lists/forms, tasks, VTBs, reports, notifications |
| 5 | Chapter 06 | You can choose UI Policy vs Data Policy vs Business Rule vs Client Script; move an update set |
| 6 | Chapters 07–10 | Comparisons, labs catch-up, timed practice, glossary drill |

## PDI rules that save time

- Request a PDI at [developer.servicenow.com](https://developer.servicenow.com).
- Inactive PDIs hibernate and can be reclaimed. Log in regularly.
- If you break the instance, reset it. Rebuilding is useful revision.
- Always work in a **named update set**, never the Default update set, when you want to keep work.
- Use `tablename.list` in the filter navigator (for example `incident.list`, `sys_user.list`).
- Impersonate after every ACL, catalog, or knowledge change.

## Exam-day method

1. First pass: answer everything you know in under 45 minutes. Flag the rest.
2. Multi-select: cover the stem, pick answers from memory, then uncover options.
3. When two answers look right, pick the **most specific platform feature** (UI Policy before Client Script; Data Policy when server-side integrity is required; Flow Designer before custom Business Rule if no-code suffices).
4. Do not leave blanks. There is no penalty for guessing.

## What this workbook does not do

- It does not reproduce live exam items.
- It does not replace Now Learning labs.
- It does not cover CIS-ITSM process depth (CAB, CMDB Health KPIs, etc.) beyond CSA scope.

Next: [Chapter 01 — Platform Overview and Navigation](01-platform-overview-and-navigation.md).
