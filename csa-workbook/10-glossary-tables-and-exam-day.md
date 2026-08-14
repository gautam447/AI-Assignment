# Chapter 10 — Glossary, Tables, and Exam Day

## Navigator shortcuts

| Type | Opens |
| --- | --- |
| `incident.list` | Incident list |
| `sys_user.list` | Users |
| `sys_user_group.list` | Groups |
| `sys_user_role.list` | Roles |
| `sys_properties.list` | Properties |
| `sys_script.list` | Business Rules |
| `sys_script_client.list` | Client Scripts |
| `sys_ui_policy.list` | UI Policies |
| `sys_security_acl.list` | ACLs |
| `sys_update_set.list` | Update sets |
| `kb_knowledge.list` | Articles |
| `sc_cat_item.list` | Catalog items |
| `cmdb_ci.list` | CIs |
| `cmdb_rel_ci.list` | CI relationships |
| `task_sla.list` | Task SLA instances |
| `cache.do` | Flush cache |

---

## Core tables (database names)

| Label | Name |
| --- | --- |
| User | `sys_user` |
| Group | `sys_user_group` |
| Role | `sys_user_role` |
| Task | `task` |
| Incident | `incident` |
| Problem | `problem` |
| Change Request | `change_request` |
| Request | `sc_request` |
| Requested Item | `sc_req_item` |
| Catalog Task | `sc_task` |
| Catalog Item | `sc_cat_item` |
| Knowledge | `kb_knowledge` |
| Knowledge Base | `kb_knowledge_base` |
| CI | `cmdb_ci` |
| CI Relationship | `cmdb_rel_ci` |
| Dictionary | `sys_dictionary` |
| Business Rule | `sys_script` |
| Client Script | `sys_script_client` |
| Update Set | `sys_update_set` |
| Customer Update | `sys_update_xml` |
| Notification | `sysevent_email_action` |
| Import Set | `sys_import_set` |
| Transform Map | `sys_transform_map` |

---

## Common roles

| Role | Use |
| --- | --- |
| `admin` | Instance admin |
| `security_admin` | Elevated ACL / high security |
| `itil` | Fulfiller |
| `snc_internal` / `snc_external` | Explicit role model |
| `approver_user` | Approvals without full itil |
| `knowledge` / `knowledge_admin` | KM |
| `catalog_admin` | Catalog |
| `import_admin` | Import Sets |
| `report_user` / `report_publisher` | Reports |
| `personalize_list` / `personalize_form` | Layouts |
| `flow_designer` / `action_designer` / `flow_operator` | Workflow Studio |
| `delegated_developer` (pattern) | Studio without full admin |

---

## Glossary

**ACL** — Access Control List rule; authorization on table/field/operation.

**aPaaS** — Application Platform as a Service.

**Application** — Navigator grouping and/or scoped app.

**Application Service** — Deployed instance of an application (CSDM; operational).

**Assignment rule** — Auto-sets assignment group/user from conditions.

**Async Business Rule** — Runs after the transaction via queue.

**Banner frame** — Top chrome: logo, search, user menu.

**Base table** — Table others extend (`task`, `cmdb`).

**Breadcrumb** — Clickable filter trail.

**Business Application** — Logical app in CSDM Design domain.

**Business Rule** — Server script/logic on database operations.

**Catalog item** — Orderable offering producing REQ/RITM.

**Client Script** — Browser JavaScript on a form/list.

**Clone** — Copy one instance over another (usually prod → subprod).

**Coalesce** — Transform-map match key for update vs insert.

**Configuration Item (CI)** — Record in the CMDB representing a managed thing.

**CMDB** — Configuration Management Database.

**CSDM** — Common Service Data Model.

**Content frame** — Main work area.

**Contextual security** — ACLs at each object level (table/field).

**Customization** — Instance-wide config (usually update-set captured).

**Data Policy** — Server mandatory/read-only.

**Delegated Development** — Constrained Studio access.

**Dependency View** — Visual map of CI relationships.

**Dictionary** — Field/table metadata.

**Dictionary override** — Child table overrides parent field settings.

**Discovery** — Bottom-up CI population via MID Server.

**Display Business Rule** — Server logic at form load; fills `g_scratchpad`.

**Dot-walking** — Following reference fields in queries and forms.

**Elevated role** — Session privilege (e.g. `security_admin`).

**Encoded query** — `active=true^priority=1` style filter string.

**Event** — Named signal (`incident.inserted`) that notifications/script actions listen to.

**Filter navigator** — Search box in the application navigator.

**Flow Designer** — No-code flows (part of Workflow Studio).

**Foundation data** — Users, groups, locations, companies (CSDM Foundation).

**GlideRecord** — Server API for table operations.

**g_form / g_user** — Client APIs.

**Guided Setup** — Checklist to configure a plugin/app.

**High Security** — Plugin/settings: default deny, elevation for ACLs.

**Impersonate** — Act as another user for testing.

**Import Set** — Staging pipeline for inbound data.

**Inactivity monitor** — Alerts when a task is untouched.

**Inbound email action** — Create/update records from received mail.

**Insert and Stay** — Save as a **new** record; keep form open.

**Instance** — A customer’s dedicated ServiceNow environment.

**IRE** — Identification and Reconciliation Engine (CMDB identity).

**Knowledge base** — Container of articles with user criteria.

**List editing** — Edit a cell in a list without opening the form.

**MID Server** — Management, Instrumentation, and Discovery; on-network worker.

**Module** — Item under an application menu.

**Multi-instance** — Isolated customer environments (ServiceNow architecture).

**My Groups Work** — Group-assigned, typically unassigned tasks.

**My Work** — Tasks assigned to me.

**Next Experience** — Current Unified Navigation UI.

**Notification** — Outbound message (email/SMS/push) from an event.

**Now Mobile** — Employee mobile app (iOS/Android).

**Now Support (HI)** — Portal for instance operations with ServiceNow.

**OLA** — Operational Level Agreement (internal).

**Order guide** — One request spanning multiple items.

**PDI** — Personal Developer Instance.

**Personalization** — Per-user UI changes.

**Platform Analytics** — Visualizations, dashboards, KPI signals.

**Playbook / PAD** — Process Automation Designer guided process.

**Plugin** — Activatable ServiceNow package.

**Preview (update set)** — Validate a retrieved set before commit.

**Process Automation Designer** — Cross-enterprise playbooks.

**Record Producer** — Catalog form that inserts a target-table record.

**Reference field** — Stores sys_id of a record on another table.

**Related list** — Child/related records on a form.

**REQ / RITM / SCTASK** — Request, Requested Item, Catalog Task.

**Role** — Bundle of permissions; can contain other roles.

**Schema Map** — Diagram of table relationships.

**Scope** — Application boundary for files and APIs.

**Security Center** — Instance security posture dashboard.

**Service Mapping** — Top-down mapping of a service to CIs.

**Shared Responsibility Model** — Split of security duties between ServiceNow and customer.

**SLA** — Service Level Agreement; start/pause/stop on `task_sla`.

**Spoke** — Bundle of Flow actions (often an integration).

**sys_id** — 32-character unique record ID.

**Tag** — Label on records for ad-hoc grouping.

**Task** — Base table for work records.

**Transform Map** — Maps import staging fields to a target table.

**UC** — Underpinning Contract (supplier).

**UI Action** — Button, link, or context-menu item (client and/or server).

**UI Policy** — Client mandatory/visible/read-only rules.

**Update set** — Container of customer updates (configuration).

**User criteria** — Audience rules for catalog and knowledge.

**User Presence** — Shows who is on a record.

**Variable / variable set** — Catalog questions; reusable bundles.

**View** — Named list/form layout.

**Virtual Agent** — Conversational self-service.

**Visual Task Board** — Kanban board (guided/flexible/freeform).

**Workflow Editor** — Legacy graphical workflow.

**Workflow Studio** — Current automation workspace (flows, playbooks).

**Work notes** — Internal task journal.

---

## Troubleshooting order (steal this in scenarios)

**Cannot see module:** module role → application menu role → impersonation sanity.

**Cannot see record:** ACL table read → before-query BR → domain separation.

**Cannot see field:** view → UI Policy → field ACL → dictionary.

**Notification missing:** event fired? condition? recipient active/email? instance sending mail? subscription? weight?

**Catalog item invisible:** active? category? user criteria? impersonate ESS user?

**Import duplicates:** coalesce missing or not unique.

**Update set missing a change:** wrong current set; object is data; workflow not published.

**Form slow:** client GlideRecord / too many onChange scripts / related lists.

---

## Official sample items (keep cold)

1. Available to all users → **Self-Service**  
2. Group work unassigned → **My Groups Work**  
3. Transform map → **Import Set fields to existing table**  
4. Multiple Choice / Single Line Text / Select Box → **Variable types**  
5. Scripting language → **JavaScript**

---

## Exam-day checklist

**Week before**

- [ ] Finish Domain 5 labs 10–14  
- [ ] Recite Chapter 07 tables out loud  
- [ ] Two timed 60-question sessions (use Chapter 09 twice, plus Now Learning quizzes)  
- [ ] Confirm exam **release** (docs filter) matches what you studied  
- [ ] Pearson/OnVUE system check if remote  

**Night before**

- [ ] Stop cramming new products  
- [ ] Sleep  
- [ ] Prepare ID, allowed drinks, clear desk (OnVUE)  

**During**

- [ ] 90 minutes / 60 questions  
- [ ] Multi-select: count exactly N answers  
- [ ] Flag and move; second pass for ACL/coalesce/save-button items  
- [ ] Prefer no-code (UI Policy, Flow, Data Policy) over script when both could work  
- [ ] Prefer security objects (ACL) over UI Policy when the stem is about access  

**After you pass**

- Credly badge  
- Annual **delta** exam + CMP fee to stay current  
- Next path: CIS-ITSM, CAD, or CSDM/Data Foundations depending on role  

**If you do not pass**

- Read domain percentages on the report  
- Restudy only weak domains + PDI labs  
- Observe the official retake wait and fee rules in the Credentialing Program Guide  

---

## Study resources (official first)

1. Now Learning: Welcome to ServiceNow  
2. Now Learning: ServiceNow Administration Fundamentals  
3. [CSA Mainline Exam Blueprint KB0011554](https://learning.servicenow.com/lxp/en/credentials/certified-system-administrator-mainline-exam-blueprint?id=kb_article_view&sysparm_article=KB0011554)  
4. Product documentation for your exam release  
5. MeasureUp official practice (KB0013408)  
6. PDI: [developer.servicenow.com](https://developer.servicenow.com)  
7. This workbook: Chapters 00–09  

Unofficial dumps are a policy and learning risk. They also go stale when the blueprint changes (as it did in January 2026).

---

You now have the full CSA working ebook. Start with a PDI and [Lab 0](08-hands-on-lab-workbook.md). Good luck.
