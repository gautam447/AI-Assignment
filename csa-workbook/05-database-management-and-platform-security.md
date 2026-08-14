# Chapter 05 — Database Management and Platform Security (30%)

**Blueprint subtopics**

1. Data Schema  
2. Application / Access Control  
3. Importing Data  
4. CMDB and CSDM  
5. Security Center  
6. Shared Responsibility Model  

This is the largest domain. If you can only master one chapter, master this one.

---

## 1. Data Schema

Everything in ServiceNow is a **record in a table**. Forms, lists, CMDB, users, even Business Rules are rows.

### Tables

| Kind | Meaning | Example |
| --- | --- | --- |
| **Base table** | Extended by others; often not used directly | `task`, `cmdb` |
| **Core / parent** | Out-of-box important tables | `incident`, `sys_user` |
| **Child / extended** | Inherits parent’s fields | `incident` extends `task` |
| **Custom** | You created it | Global: `u_asset_loan`; scoped: `x_intd_loan` |

Creating a table: **All > System Definition > Tables** (or Tables & Columns). Creating a table typically also creates:

- A menu module (list)
- Default form and list
- ACLs (depending on how you create it)
- An application file

**Number Maintenance** sets prefixes (`INC`, `PRB`, `u_loan`).

### Table names vs labels

- Label: Incident  
- Name: `incident`  
- Custom global prefix: `u_`  
- Scoped prefix: `x_<vendor>_<app>_`

Exam items often ask the **database name** (`sys_user`, `sys_user_group`, `task_sla`, `cmdb_ci`).

### Extension (parent–child)

Extension is an **is-a** relationship.

- Child **inherits** fields, some business logic, and can override via **dictionary override**
- Child has its **own** table name and extra fields
- `sys_class_name` stores the actual class of a row in a hierarchy (critical in CMDB)

If you add a field to `task`, it appears on Incident, Problem, Change, unless you should have added it only on the child.

### Relationship types

| Type | How it is built | UI |
| --- | --- | --- |
| **Extension** | Table’s “Extends table” | Hierarchy / schema map |
| **One-to-many** | Reference field on the *many* side | Reference field + related list on the one side |
| **Many-to-many** | M2M table with two references | Related lists on both sides |
| **Defined relationship** | Relationship record | Related list |

**One-to-many example:** many Incidents reference one User (`caller_id` → `sys_user`). On the User form, a related list can show that user’s incidents.

**Many-to-many example:** Incidents ↔ CIs via `task_ci` (Affected CIs).

### sys_id

Every record has a **sys_id**: 32-character globally unique identifier (GUID). Display numbers (`INC0001234`) can change format; **sys_id** is the real primary key. Copy sys_id from form context menu.

### Dictionary

**All > System Definition > Dictionary** (`sys_dictionary`) describes every field:

- Type, max length
- Mandatory, read-only, default
- Reference table
- Attributes
- Display field (which field is the display value for references)

**Tables & Columns** is a friendlier explorer. **Schema Map** visualizes relationships (roles `admin` or `personalize_dictionary`).

### Dictionary override

On a child table, override parent field settings:

- Default value
- Mandatory / read-only
- Reference qualifier
- Display value
- Dependent fields

Without override, changing `task.short_description` affects every child.

### Dot-walking

Follow reference fields: `caller_id.department.name`. In condition builders a **green +** expands the reference. Used in filters, notifications, GlideRecord, export, and access rules.

### Field types (schema view)

You already saw UI types in Domain 3. Schema extras:

- **Document ID** — reference to any table
- **Conditions** — encoded query builder stored as text
- **Script** — JavaScript
- **Password2** — encrypted password
- **Wiki / HTML** — rich text
- **Currency** — money with currency code

### Schema Map

Open from Tables & Columns → Schema Map. Shows:

- Tables this table extends
- Tables that extend it
- Reference relationships

Use it when a question asks “how do you **visualize** table relationships?”

### Application scope and the data model

- Global scope: classic customizations, `u_` tables
- Application scope: files belong to an app; other apps need **cross-scope** privileges
- Scope **protects** tables from random global Business Rules

CSA: scope **encapsulates** an application’s files and data access.

### Deleting vs deactivating

Deleting a table drops data. Deactivating a field or making it inactive hides it from new use. Same idea as users: prefer **inactive** for people; be extremely careful deleting tables in production.

---

## 2. Application / Access Control

Security is **contextual**: every table and field can have its own rules. Default on modern instances is **deny unless allowed**.

### Layers of “why can’t I see it?”

1. Not logged in / locked out / inactive  
2. **Application menu** role  
3. **Module** role  
4. **Before Query** Business Rule (silently filters rows)  
5. **ACL** table read  
6. **ACL** field read (field empty or asterisks)  
7. UI Policy / Client Script hiding a field (still in XML if they know how — not security)  
8. User criteria (catalog/knowledge)  
9. Domain separation (licensed; know it exists for MSP-style isolation)

UI Policy is **not** security. ACL is.

### ACL record (`sys_security_acl`)

An Access Control List rule answers: **who can perform an operation on this object?**

**Operations (CRUD + extras):**

- Create
- Read
- Write (update)
- Delete
- execute (scripts, some UI)
- edit_ci_relations, list_edit, report_on, etc.

**Object names:**

| ACL name | Controls |
| --- | --- |
| `incident` or `incident.None` | **Row / table** access (the record as a whole) |
| `incident.*` | **All fields** not covered by a more specific field ACL |
| `incident.short_description` | That **one field** |

### The three requirements (AND)

To pass **one** ACL the user must satisfy **all** configured parts:

1. **Roles** listed on the ACL (user has at least one, via group inheritance)
2. **Condition** (field filters) evaluates true
3. **Script** returns true (`answer = true`)

If a part is empty, it is not tested. Empty role + empty condition + empty script means the ACL is wide open for that operation (dangerous).

### Evaluation order (memorize and be able to explain)

ServiceNow evaluates ACLs from **specific to general** in a processing order. Practical rules used on CSA:

1. User must pass **table-level** (row) ACL for the operation **first**. If table read fails, field ACLs never grant the record.
2. Then **field-level** ACLs. If any field ACL exists, the user must pass the ACL that applies to that field.
3. Matching uses the **most specific** object name: `table.field` beats `table.*` beats other wildcards.
4. If **multiple ACLs** match the same object+operation, the user must typically **pass one** of them (OR between matching rules at the same level — historically “pass any matching ACL”). Training wording: you must meet the permissions of the applicable rules; when several rules exist, passing a matching rule can grant access.
5. If **no ACL** exists for a table in a deny-by-default / High Security world, access is **denied**.
6. `admin` often bypasses via `admin overrides` on the ACL unless that checkbox is cleared.

**Do not confuse:**

- Multiple **parts of one ACL** = AND (roles AND condition AND script)
- Multiple **ACL records** at the same point = usually OR (any one grants)

Write this on your scratch paper in the exam.

### Wildcard `*` vs `.None`

- `table.None` — table/row
- `table.*` — fields in general
- `*` — extremely broad; avoid in custom work

A user can **read the record** but not **a field** if table read passes and field read fails. The field appears blank or is omitted.

### Creating ACLs

- Elevate to **`security_admin`**
- All > System Security > Access Control (ACL)
- Or table record → Access Controls related list

High Security plugin makes elevation mandatory. Session only.

### Roles and groups again

Preferred model: **roles on groups, users in groups**. ACL scripts can also check `gs.hasRole('itil')` but roles on the ACL record are clearer.

Role inheritance: a parent role contains child roles; assigning the parent grants children.

### Authentication vs authorization

| Authentication | Authorization |
| --- | --- |
| Who are you? | What may you do? |
| Local, LDAP, SSO, MFA, CAC/PIV | Roles, ACLs, user criteria, query BRs |

### Elevated privilege

`security_admin` is an **elevated** role: not constantly on, even for admins. Prevents accidental ACL edits.

### Domain separation (awareness only)

Splits data by domain (MSP). Not a CSA implementation topic, but “data isolation between customers on one instance” may point to domain separation.

### Application access (scoped)

On a table in a scoped app, **Application Access** settings control whether *other applications* can CRUD that table, not whether a *user* can. User access is still ACLs.

---

## 3. Importing Data

Use **Import Sets** to load external data into ServiceNow tables **with transformation**, not by typing into a list.

### Pipeline

```
File / Data Source  →  Import Set table (staging, u_*)  →  Transform Map  →  Target table
```

1. **Load Data** (All > System Import Sets > Load Data)  
   Excel, CSV, XML (and data sources: JDBC, LDAP, HTTP, etc.)
2. Platform creates or reuses a **staging table**
3. You define a **Transform Map**
4. **Run Transform**
5. Check **Transform History**, import set rows (inserted/updated/skipped/error)

Roles: `admin`, `import_admin`.

### Transform Map

Official definition: a map that determines relationships between fields in an **Import Set** and fields in an **existing table**.

Pieces:

| Piece | Purpose |
| --- | --- |
| Field maps | Source field → target field |
| **Coalesce** | Match existing records to **update** instead of insert |
| Automatic Mapping | Maps when **names match** |
| Mapping Assist | Drag when names differ |
| Transform scripts | onStart, onBefore, onAfter, onComplete, onForeignInsert, source script |
| Choice action | create / ignore / reject for choice values |

One source field can map to multiple targets. Scripts can set fields that are not in the file.

### Coalesce (exam favorite)

Mark one or more mapped fields as **coalesce**:

| Situation | Result |
| --- | --- |
| Match found on coalesce field(s) | **Update** that target record |
| No match | **Insert** new target record |
| Multiple coalesce fields | All must match (AND) |
| Coalesce on empty | Dangerous — can smash the wrong row; avoid |

If multiple target rows match, behavior is “first found” style — design unique coalesce keys (email, employee number, serial number, sys_id).

**No coalesce** = always insert = duplicates.

### Data sources vs one-off files

| One-off Load Data | Data Source |
| --- | --- |
| Manual Excel/CSV | Repeatable JDBC, LDAP, file, REST |
| Ad-hoc cleanup | Scheduled imports |

LDAP import typically loads users into `sys_user` via transform maps provided or cloned from templates.

### Robust Import Set Transformer / RTE

Newer instances have **Robust Transform Engine** / ETL-style import. CSA still tests classic Import Set + Transform Map language. If both appear, classic transform map matches the official sample item.

### What import is not

- **Update sets** move **configuration**, not bulk user data
- **xml import** of records is a different admin tool (small sets, captured XML)
- **Clone** copies a whole instance
- **CMDB Identification and Reconciliation** (IRE) is the right engine for Discovery/Service Graph; import sets can still feed CMDB but IRE is preferred for CI identity

### Data Policy during import

UI Policies do **not** apply to import set transforms. **Data Policies** do (server). If a field must be mandatory even when loaded from Excel, use Data Policy or dictionary mandatory.

---

## 4. CMDB and CSDM

### What a CMDB is

The **Configuration Management Database** stores **Configuration Items (CIs)** — the things you manage — and **relationships** between them.

CIs can be:

- **Tangible**: servers, laptops, routers, UPSs  
- **Intangible**: business services, application services, databases, certificates, documents  

Benefits (classic exam list):

- Impact analysis for Change (“what breaks if this server reboots?”)
- Incident/Problem linkage to failing CIs
- Identifying problem trends on the same CI
- Locating failed changes
- Understanding service health

### Key tables

| Table | Name | Role |
| --- | --- | --- |
| `cmdb` | CMDB (base) | Root of CI hierarchy |
| `cmdb_ci` | Configuration Item | Base CI class used operationally |
| `cmdb_rel_ci` | CI Relationship | Parent/child (or other types) between two CIs |
| `cmdb_ci_service` | Service | Service CIs |
| `cmdb_ci_linux_server` etc. | Classes | Specific CI types |

Relationships have a **type** (Depends on, Runs on, Virtualized by, Used by, …). **Suggested Relationships** constrain which types make sense between two classes.

**CI Class Manager**: All > Configuration > CI Class Manager — the modern UI to manage classes, identification rules, reconciliation, and hierarchy.

**Dependency Views**: map a CI and its related CIs visually (impact/hierarchy). Path often Dependency Views / related link on a CI.

### How the CMDB gets populated

| Source | Style |
| --- | --- |
| **Discovery** | Bottom-up: finds devices/apps via MID Server, probes, sensors |
| **Service Mapping** | Top-down: maps a service entry point down to infrastructure |
| **Service Graph Connectors** | Third-party sources through IRE |
| **Manual / import** | Spreadsheets, forms |
| **Asset** processes | Hardware lifecycle may create CIs |

**MID Server** = Management, Instrumentation, and Discovery. A Windows/Linux service **inside** the customer network that talks out to the instance. Required for Discovery, many LDAP/JDBC imports, Orchestration.

**Identification and Reconciliation Engine (IRE)** prevents duplicate CIs using identification rules (e.g. serial number + correlation id).

### CSDM — Common Service Data Model

CSDM is **not a product you install**. It is ServiceNow’s **standard data model and best-practice map** for how CIs and related records should be used across the platform (ITSM, ITOM, SPM, APM, CSM, …).

It gives:

- Standard **terms and tables**
- Standard **relationships**
- A staged implementation (crawl, walk, run, fly in older docs)

#### Conceptual domains (learn both “classic 4+foundation” and that CSDM has grown)

Common CSA-level picture:

| Domain | What lives there | Operational? |
| --- | --- | --- |
| **Foundation** | Users, groups, locations, companies, products, contracts — mostly **not** CMDB relationships | Referential data |
| **Design** | Business Application, Business Capability — planning/architecture | Generally **not** selected as incident “what broke” infrastructure |
| **Build** | SDLC / DevOps digital product build objects | Not operational CIs |
| **Manage Technical / Technology Management Services** | Application Service (deployed instance), Technical Service, infrastructure CIs from Discovery | **Yes** — ITOM |
| **Sell / Consume** | Business Service, Service Offering, request catalog of what the business sells/consumes | Portfolio + catalog |
| **Manage Portfolio** | Portfolio layer over services; service owners | Management view |

Older training used: Foundation, Design, Manage Technical Services, Sell/Consume, Manage Portfolio. Newer CSDM 5 adds Ideation & Strategy, Design & Planning, Build & Integration, Service Delivery, Service Consumption, Manage Portfolios. For CSA, remember:

- Foundation data is **referenced**, usually not related via `cmdb_rel_ci`
- **Business Application** (design) vs **Application Service** (deployed instance you actually incident against)
- **Business Service** / offerings are what you sell or consume
- **Request catalog is not a CMDB table**
- CSDM tables are mostly **OOB CMDB classes** used in a standard way

#### Why CSDM matters on CSA

Questions will ask:

- Where should a **Business Application** live? (Design)
- What does **Discovery** populate? (technology / infrastructure CIs)
- What standard provides **common vocabulary** for CMDB? (**CSDM**)
- Impact of a server outage on a **business service** (follow relationships / dependency view)

Do not invent custom service tables when an OOB CSDM class exists.

### CMDB Health (awareness)

OOB dashboards/tests: completeness, correctness, compliance. Quick start tests exist for CMDB content. Deep health KPIs are more CIS / CSDM exam than CSA, but “how do you test CMDB content?” can be **quick start tests**.

---

## 5. Security Center

**Security Center** is the instance dashboard for **platform security posture** (not SecOps / SIR, and not ITOM Security Incident).

Think of it as the admin’s **security health of the Now Platform itself**:

- Highlights risky properties and configurations
- Tracks security-related **checks / scores** (hardening)
- Surfaces recommendations (password policies, default accounts, encryption, high security, etc.)
- Helps `admin` / security teams see whether the instance follows ServiceNow hardening guidance

CSA expectations:

- Know **what it is** and **who uses it** (admins / security admins)
- Know it is about **instance security posture**, not the CMDB of firewalls
- Know it ties to the **Shared Responsibility Model** (your side of the work)

If a question says “where do you review instance security configuration findings in one place?” → Security Center.

(Exact widgets change by release. Do not memorize pixel layouts.)

---

## 6. Shared Responsibility Model

ServiceNow is SaaS/aPaaS. Security duties are **split**.

| ServiceNow typically owns | Customer typically owns |
| --- | --- |
| Physical data centers, hardware | Who you grant `admin` / `security_admin` |
| Hypervisor / platform availability | User lifecycle, roles, groups, ACLs |
| Core product patching of the cloud | How you configure ACLs, public pages, integrations |
| Network of the service | MID Server hardening **in your network** |
| Encryption at rest capabilities of the service | What data you put in the instance; classifications; encryption configurations you enable |
| Platform monitoring of the service | Access reviews, audit of *your* customizations |
| Providing Security Center / hardening guidance | Acting on Security Center findings |
| Instance isolation (multi-instance) | Sub-prod data (clone data may be production PII — you must scrub) |

**You cannot outsource:**

- Bad ACLs that expose HR data
- Catalog items available to the world
- Guest / public pages you published
- Storing secrets in scripts
- Not reviewing admin list

**They cannot expect you to:**

- Patch the physical host
- Design ServiceNow’s multi-instance isolation
- Operate the global data center

Exam scenarios: “Who is responsible for configuring ACLs?” → **customer**. “Who is responsible for physical security of the data center?” → **ServiceNow**.

---

## Extra security topics that show up under this domain

### High Security plugin

- Default deny
- Elevation required for ACL edits
- Security properties (e.g. escape XML, use stronger cookies)

### Encryption (awareness)

- Column encryption / platform encryption options
- Certificates
- HTTPS is standard for the instance URL

### Access to email

System address filters: which inbound senders are accepted.

### Contextual security

Training phrase: security can apply at **every level of the object hierarchy** (table and field). That phrase = ACLs.

---

## High-yield facts

- Tables extend; fields live in the dictionary; sys_id is 32 characters.
- Relationships: extend, one-to-many (reference), many-to-many.
- Schema Map visualizes table relationships.
- ACL parts AND; matching ACL records at the same level typically OR.
- Table ACL first, then field ACL; `table.field` > `table.*`.
- Elevate `security_admin` to edit ACLs.
- Import: Load → staging table → Transform Map → target; **coalesce** = update vs insert.
- Transform map = Import Set fields to existing table fields.
- CMDB: `cmdb`, `cmdb_ci`, `cmdb_rel_ci`; Dependency Views; CI Class Manager.
- Discovery bottom-up; Service Mapping top-down; MID Server in your network.
- CSDM = standard model/vocabulary; Foundation is referential; catalog is not a CMDB table.
- Security Center = instance security posture.
- Shared Responsibility: ServiceNow secures the cloud; you secure **your configuration and users**.

## Domain 5 self-check

- [ ] Create `u_csa_widget` extending nothing; add a reference to User; show related list on User.
- [ ] Open Schema Map for `incident`.
- [ ] Dictionary override `short_description` mandatory on a child only.
- [ ] Create an ACL that lets a custom role read `u_csa_widget` and nothing else; impersonate.
- [ ] Import a 5-row CSV of users with coalesce on email; run twice; prove no duplicates.
- [ ] Open a CI, view Dependency View, find `cmdb_rel_ci`.
- [ ] Write five ServiceNow-owned vs customer-owned security duties.
- [ ] Open Security Center and read one recommendation.

Next: [Chapter 06 — Data Migration and Integration](06-data-migration-and-integration.md).
