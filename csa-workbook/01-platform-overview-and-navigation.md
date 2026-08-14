# Chapter 01 — Platform Overview and Navigation (7%)

**Blueprint subtopics**

1. ServiceNow Platform overview
2. Platform capabilities and services
3. The ServiceNow Instance
4. Next Experience Unified Navigation

This domain is small but it supplies the vocabulary used everywhere else. One pass with a PDI is enough if you can name every chrome element and every way a user can enter the platform.

---

## 1. ServiceNow Platform overview

### What ServiceNow is

ServiceNow is a cloud **application Platform-as-a-Service (aPaaS)**. Customers do not install a database and app server in their own data center for the core product. ServiceNow hosts the instance; you configure applications on top.

The platform is used to run **workflows** across:

| Workflow type | Typical products |
| --- | --- |
| IT | ITSM, ITOM, ITAM, SecOps |
| Employee | HRSD, Workplace, Legal, etc. |
| Customer | CSM, Field Service |
| Creator / builder | App Engine, custom scoped apps |

CSA does not require product-specialist depth. You must know that the **same platform** (tables, forms, ACLs, Flow Designer, CMDB) underpins those workflows.

### Multi-instance architecture

ServiceNow uses a **multi-instance** architecture, not a classic multi-tenant shared-database model.

- Each customer gets **dedicated application and database** processes (an *instance*).
- Instances are isolated. One customer’s outage or customization does not sit in another customer’s database.
- A typical customer landscape is **at least two instances**: production plus one or more sub-production (development, test, UAT).
- Upgrades, cloning, and plugin activation are instance-level operations.

**Clone** copies production data and configuration into a sub-production instance so you can test with realistic data. Cloning **overwrites** the target instance.

### Now Support (HI)

Instance operations that you cannot do inside the instance itself go through **Now Support** (historically the HI — Hosted Instance — portal):

- Request an upgrade or plugin that requires ServiceNow action
- Manage instance version / patch
- File platform support cases
- Request clones (depending on contract and process)

Admins still do day-to-day configuration *inside* the instance.

### Personal Developer Instance (PDI)

A PDI is a free, time-limited instance for learning. It includes demo data (users like Abel Tuter, incidents, catalog items). It is the correct place to break things. It is **not** a production-like customer landscape (no HI clone process, different plugin entitlements).

---

## 2. Platform capabilities and services

Know the *purpose* of the core platform capabilities. You will see them named as distractors.

| Capability | What it is | CSA angle |
| --- | --- | --- |
| ITSM | Incident, Problem, Change, Request, Knowledge | Task table, assignment, SLAs |
| CMDB | Configuration items and relationships | Domain 5 |
| Service Catalog | Ordering goods and services | Domain 4 |
| Knowledge | Self-service articles | Domain 4 |
| Now Intelligence / Predictive Intelligence | ML suggestions (category, assignment) | Know it exists; assignment-rule best practices still apply |
| Performance Analytics / Platform Analytics | KPIs, dashboards, visualizations | Domain 3 |
| Flow / Workflow Studio | No-code automation | Domain 4 |
| Virtual Agent | Conversational self-service | Domain 4 |
| Integration Hub / spokes | Reusable integration actions | Mentioned with Flow Designer |
| MID Server | Management, Instrumentation, and Discovery — on-prem probe | Discovery and import/LDAP context |
| Discovery / Service Mapping | Populate and map CIs | CMDB |
| Service Portal / Employee Center | End-user web UI | Common interfaces |
| Now Mobile | iOS and Android apps | Common interfaces |
| Workspaces (configurable / UI Builder) | Agent/fulfiller experience | Next Experience |

### Who uses the platform

| Persona | Typical interface | Typical roles |
| --- | --- | --- |
| End user / employee | Service Portal, Employee Center, Virtual Agent, Now Mobile | Often **no role**, or `snc_internal` only |
| Fulfiller / technician | Next Experience lists/forms, Agent Workspace | `itil` |
| Approver | Email, portal, native UI | Sometimes no fulfiller role |
| Admin | Native UI, Studio, Security Center | `admin` |
| Security admin | ACLs, High Security | `security_admin` (elevated) |
| Developer | Studio, App Engine | `admin` or delegated-dev roles |

**Self-Service is available to all users.** That official sample question is testing this exact idea. Incident, Change, and many backend applications require roles.

---

## 3. The ServiceNow Instance

### URL and naming

An instance has a unique name, for example `https://dev12345.service-now.com`. Production, test, and dev have **different URLs**. Never develop in production.

### What “the instance” contains

- Applications and modules (menus)
- Tables and records
- Users, groups, roles
- System properties
- Plugins (activated features)
- Customizations captured in update sets / apps

### Logging in

Authentication methods you should be able to name:

| Method | Where credentials live |
| --- | --- |
| Local database | `sys_user` password in ServiceNow |
| LDAP / Active Directory | Corporate directory; often via MID Server |
| SSO (SAML, etc.) | Identity provider (Okta, Azure AD, etc.) |
| Multi-factor | Extra factor on top of SSO or local |
| Certificate (CAC / PIV) | Smart-card environments |

CSA will not ask you to configure SAML XML. It will ask **which method** is being used or what happens if LDAP is down (local admin break-glass accounts matter).

### First login and onboarding

New users may see an **onboarding** experience / welcome content. Admins can set a welcome message and branding. After login, the **homepage or dashboard** (or workspace landing) loads in the content frame.

### User menu actions (memorize)

From the user avatar / profile menu you typically can:

- **Profile** — user record settings
- **Impersonate User** — act as another user (admin). Critical for testing ACLs, catalog, and portal
- **Elevate role** — turn on `security_admin` for the **current session** only (needed to create/edit ACLs when High Security is on)
- **Logout**
- System settings / preferences (often also via gear icon)

Impersonation does **not** require the target user’s password. Always **unimpersonate** before making admin changes.

### System Settings vs instance branding

| | System Settings (gear) | Instance branding / system properties |
| --- | --- | --- |
| Scope | **This user** (personalization) | **All users** (customization) |
| Examples | Compact UI, date format, language, accessibility, notifications | Banner logo, system name, colors, default theme |
| Captured in update set? | No (user preference) | Often yes (sys properties, themes) |

Exam language: **personalization** = per user; **customization** = instance-wide configuration.

---

## 4. Next Experience Unified Navigation

Next Experience (introduced around San Diego, now the default) is the current native UI. CSA still expects you to know the three classic regions because they still exist conceptually:

```
+------------------+----------------------------------------+
| BANNER FRAME     | logo, global search, help, user menu,  |
|                  | contextual app pill, notifications     |
+------------------+----------------------------------------+
| APP NAVIGATOR    | CONTENT FRAME                          |
| All / Favorites  | lists, forms, dashboards, workspaces,  |
| History / Filter | portals-in-iframe, Studio, etc.        |
+------------------+----------------------------------------+
```

### Banner frame

Typical elements:

- **Logo** — click returns to the home landing page
- **Global search** — searches multiple tables (can use Boolean operators in ALL CAPS)
- **Help**
- **User presence / notifications / inbox** (release-dependent)
- **User menu**
- **Contextual app pill** — which application you are in

**Connect Chat** historically appeared in the banner as real-time messaging. Know it as the legacy collaboration tool; many orgs now use Teams/Slack + Virtual Agent.

### Application navigator

Tabs/sections:

| Area | Purpose |
| --- | --- |
| **All** | Full menu of applications and modules |
| **Favorites** | Starred apps/modules, plus dragged filters and records |
| **History** | Recently viewed items (default ~30). Does not include every UI page |
| **Workspaces** | Configurable workspaces (Next Experience) |
| **Filter navigator** | Type-ahead search of the menu |

**Application** = grouping in the menu (for example Incident).  
**Module** = an item under it (for example Create New, All, Open).

Path notation you will see in training: **All > Incident > Create New**.

### Navigation filter tricks

| Type | Result |
| --- | --- |
| `incident` | Filters menu to matching apps/modules |
| `incident.list` | Opens the Incident list |
| `incident.FORM` / `.do` patterns | Direct record/form operations |
| `cache.do` | Flushes instance cache (admin). Use after property changes that seem “stuck” |
| `sys_properties.list` | System properties |
| `sys_user.list` | Users |
| `*.config` on some tables | Table config |

### Favorites

Ways to favorite:

- Click the **star** next to an application or module
- Drag a **breadcrumb / filter** to Favorites
- Drag a **record** to Favorites

Favorites are per user (personalization).

### History

Fastest way to go back to a recently opened list or record. History is **not** a complete audit log. Transaction logs and record history (audit) are different features.

### Content frame

This is where work happens:

- Homepages (legacy; often deprecated via property)
- Dashboards
- Lists
- Forms
- Visual Task Boards
- Reports
- Workspaces can also take over the whole window

### Global search and wildcards

- `*` is the wildcard in many list searches
- List header search: “starts with” vs contains depends on field type and wildcards
- Encoded queries sit behind breadcrumbs

---

## Other interfaces (still Domain 1/2 overlap)

Users do **not** only use Unified Navigation.

| Interface | Audience | Notes |
| --- | --- | --- |
| **Now Platform UI / Next Experience** | Fulfillers, admins | Primary admin interface |
| **Service Portal** | End users | Widget-based portal (AngularJS era) |
| **Employee Center / Employee Center Pro** | Employees | Modern portal family |
| **Now Mobile** | iOS and Android | Requests, knowledge, tasks, approvals |
| **Now Support app** | Customers contacting ServiceNow | Not your company portal |
| **Agent / Configurable Workspace** | Agents | UI Builder; record pages, lists, related |
| **Virtual Agent** | Chat users | Web, mobile, Teams, Slack |

If a question says “users without roles should request a laptop,” the answer is almost always **Service Catalog via Self-Service / Portal**, not the backend Incident application.

---

## Collaboration presence

**User Presence** shows who is viewing or editing a record, reducing collisions. Combined with the **activity stream** (work notes vs additional comments) this is the native collaboration model on task records.

- **Additional comments** — visible to the caller / end user
- **Work notes** — fulfiller-only (typically)

---

## Guided Setup

Many applications ship a **Guided Setup** (or Success / Now Create style checklists):

- Empty circle = not started
- Partial / complete indicators track tasks

Know it as the **recommended path to configure a plugin** after activation, not as a coding tool.

---

## High-yield facts to memorize

- ServiceNow is **aPaaS** with **multi-instance** architecture.
- Three UI regions: **banner, navigator, content frame**.
- **Self-Service** is for everyone; **itil** is the standard fulfiller role.
- **Impersonate** tests as another user; **Elevate** enables `security_admin` this session.
- **Personalization** ≠ **customization**.
- History tab = recent navigation, not CMDB history and not audit.
- Logo click = home.
- `cache.do` flushes cache.
- Now Mobile = iOS and Android.
- Founder trivia sometimes appears in unofficial quizzes (Fred Luddy); do not prioritize trivia over architecture.

## Domain 1 self-check

- [ ] Sketch the Next Experience layout from memory.
- [ ] Name four ways a user can interact with ServiceNow (UI, portal, mobile, VA, workspace).
- [ ] Explain multi-instance vs multi-tenant in one sentence.
- [ ] Impersonate Abel Tuter on a PDI and note which modules disappear.
- [ ] Create a favorite from a filtered list breadcrumb.

Next: [Chapter 02 — Instance Configuration](02-instance-configuration.md).
