# Chapter 02 — Instance Configuration (10%)

**Blueprint subtopics**

1. Installing applications and plugins
2. Personalizing / customizing the instance
3. Common user interfaces in the Platform

This domain is about making the instance *yours*: plugins, branding, properties, users, and the UIs people actually open.

---

## 1. Installing applications and plugins

### Plugins

A **plugin** is a bundled set of tables, modules, properties, and logic that ServiceNow can activate.

| Action | Who | Notes |
| --- | --- | --- |
| Activate a plugin | `admin` (many) | **All > System Definition > Plugins** |
| Request a plugin | Now Support | Some paid/store items need entitlement |
| Batch install | CI/CD / App repo | Install multiple apps/plugins together |

After activation:

- New **applications and modules** appear in the navigator
- New **tables** may be created
- **Guided Setup** may appear
- Some plugins are **irreversible** or hard to undo — use sub-production first

Demo data is often a **separate plugin** (for example ITSM demo data). PDIs usually already have demo data; customer instances often should **not**.

### Applications vs plugins vs Store apps

| Term | Meaning |
| --- | --- |
| Application | Navigator grouping and/or scoped app (`sys_app`) |
| Plugin | Activatable ServiceNow package |
| Store app | App Store / storehouse install; versioned |
| Custom app | Built in Studio; scoped (`x_company_...`) |

**Delegated Development** lets non-admins develop in Studio with constrained permissions. Admins assign app-specific developer rights rather than full `admin`.

### Studio and App Engine

CSA does not require you to build a full scoped app, but you should know:

- **Studio** is the IDE for application files
- Scoped apps isolate tables (`x_` prefix) and APIs
- Global custom tables historically use `u_` prefix
- Application **scope** restricts which files a developer can change

### Continuous Integration / Continuous Delivery

Modern app deployment uses **source control** (Git) and **CI/CD spokes**, not only update sets. For CSA:

- Update sets still matter for **global / classic** customization (Domain 6)
- Store apps and scoped apps can install as **versions**
- Batch install can push several apps at once

---

## 2. Personalizing and customizing the instance

### Branding

Typical branding tasks (All > System Properties / Basic Configuration UI16 / Next Experience themes):

- Banner **logo** / header image
- **Browser tab title** / page title
- System **name**
- Header and accent **colors**
- Default **theme** (Next Experience Polaris themes)

These are **customizations** (instance-wide). They belong in an update set or theme app so they promote DEV → TEST → PROD.

### System properties

Properties (`sys_properties`) are key-value switches.

Examples worth knowing:

| Property (concept) | Effect |
| --- | --- |
| Homepage deprecation | `glide.ui.homepage.enable` / `com.glideapp.home.deprecate_homepages` style flags push users to **dashboards** |
| CSS / UI properties | Legacy UI16 look |
| Email sending | Whether the instance sends mail (often off on sub-prod) |
| Link unfurling for VA | `com.glide.cs.enable_link_unfurling` |

Always search properties in `sys_properties.list`. Changing a property can require `cache.do` or a new session.

### Localization

To support multiple languages:

- Activate the **localization / I18N** plugins
- Translated text lives in language files (`sys_ui_message`, field labels, etc.)
- Users pick **language** in system settings (personalization)

### Users, groups, and roles

This is core admin work. Many Domain 2 and Domain 5 questions overlap here.

#### Users — `sys_user`

A user is a record that can log in.

Important fields:

- User ID, First/Last name, Email
- Active
- Locked out
- Password (local auth)
- Time zone, date format, language
- Department, location, manager (foundation data)
- Roles (related list) — prefer **not** to assign here directly

**Never delete** users who created records if you can avoid it; **deactivate** (`active = false`) instead so history remains.

#### Groups — `sys_user_group`

A group is a set of users who share work and roles.

- Groups can contain **subgroups**
- Roles assigned to a group are inherited by members
- Assignment rules and catalog fulfillment target **groups**
- A user can belong to many groups

#### Roles — `sys_user_role`

A role is a bundle of permissions. Roles can **contain other roles** (inheritance).

| Role | Typical meaning |
| --- | --- |
| `admin` | Almost everything; still elevate for some security operations |
| `itil` | Fulfiller: incidents, problems, changes, catalog tasks |
| `snc_internal` | Logged-in internal user (explicit roles era) |
| `snc_external` | External/customer user |
| `approver_user` | Approve without full itil |
| `knowledge` / `knowledge_admin` | Knowledge contribute / admin |
| `catalog_admin` | Catalog administration |
| `report_user` / `report_publisher` | Run vs publish reports |
| `personalize_list` / `personalize_form` | Configure list/form layouts |
| `import_admin` | Import sets |
| `security_admin` | ACL maintenance (elevated) |
| `flow_designer` | Create flows |

**Best practice (memorize):**

1. Create **roles** for a function.
2. Assign roles to **groups**.
3. Put **users in groups**.
4. Do not assign roles directly to users except break-glass admins.

This makes joiner/mover/leaver processes manageable.

#### Role inheritance example

If group **Service Desk** has role `itil`, and `itil` contains other roles, every member gets `itil` **and** contained roles. Removing the user from the group removes those inherited roles (unless another group still grants them).

### User administration tasks

| Task | Module |
| --- | --- |
| Create user | All > User Administration > Users |
| Create group | Groups |
| Create role | Roles |
| Impersonate | User menu |
| Password reset | User record / IdP |
| Locked out | Clear locked out after failed logins |

### Company and foundation data

Configure early because everything references it:

- Companies
- Departments
- Locations
- Buildings / cost centers (as needed)

CSDM calls this **Foundation** data. It is usually **not** CMDB CIs.

---

## 3. Common user interfaces in the Platform

### Native lists and forms

Default fulfiller UI. Anatomy is Domain 3; here you only need to know they are the **standard** interface and are **view-based** (Default, Self Service, Workspace views can differ).

### Homepages vs dashboards

| Homepages | Dashboards |
| --- | --- |
| Legacy | Current |
| Often deprecated by property | Responsive; Platform Analytics |
| Gauges | Widgets / data visualizations |
| Per-user homepage | Shareable, role-based |

If asked how to stop users creating homepages, the answer is the **deprecate homepages** system property, not deleting the Reports module.

### Service Portal

- Built from **pages, containers, widgets**
- End users search knowledge, chat with Virtual Agent, order from catalog, check request status
- Branding via portal record (logo, theme, CSS)
- “Can We Help You?” style catalog categories often hold **incident record producers**

### Employee Center

Successor experience for employees; catalog + knowledge + topics. For CSA, treat it as the **modern self-service portal**.

### Workspaces

- **Configurable Workspace** built with **UI Builder**
- Used by agents (CSM, ITSM workspace)
- Lists, record pages, related items, landing pages
- Next Experience visual design

You do not need to master UI Builder for CSA. Know that workspaces are a **primary interface** alongside Unified Navigation.

### Mobile

| App | Purpose |
| --- | --- |
| Now Mobile | Employee: requests, knowledge, tasks |
| Mobile Agent (where licensed) | Fulfiller on the go |
| Now Support | Talking to ServiceNow, not your users |

**Mobile Studio** (or mobile app config) is used to create/modify mobile applets. Now Mobile supports **iOS and Android**.

### Notifications UI

Users manage **notification subscriptions** in System Settings. This is personalization. The notification **definitions** themselves are customizations (Domain 3).

---

## Personalization vs customization (exam favorite)

| | Personalization | Customization |
| --- | --- | --- |
| Who | One user | The instance / many users |
| Examples | List column order (personal), favorites, theme density, language | Form layout for a view, business rules, catalog items, branding |
| Gear icon list layout | Personal (green dot on gear) | Configure > List Layout is for everyone on that view |
| Update set | Not captured | Captured (most config) |

**Configure** (right-click header) vs **Personalize** (gear on list):

- Gear = my columns
- Configure > List Layout / Form Layout = everyone using that **view**

---

## Application menus and modules

Admins can:

- Create **applications** (menu headers)
- Create **modules** (links: list of records, new record, URL, separator, run a script, etc.)
- Restrict modules with **roles**
- Order modules with an order number

If a user cannot see a module, check **role on the module**, **role on the application menu**, and **ACLs on the table** — three different layers.

---

## Number maintenance

Record numbers (`INC0000001`, `CHG0000042`) are configured in **System Definition > Number Maintenance**:

- Prefix
- Number of digits
- Starting number

This is instance configuration, not scripting.

---

## High Security Settings

When the High Security plugin is active (normal on modern instances):

- Creating or editing **ACLs** requires elevating to `security_admin`
- Default deny behaviors are stricter
- Security properties appear under System Security

Details of ACL evaluation are Domain 5. Here, remember: **elevate roles is a session-only privilege**.

---

## High-yield facts

- Activate plugins in sub-production first; some cannot be cleanly reversed.
- Branding and properties = customization; gear settings = personalization.
- Roles → Groups → Users.
- Deactivate users; do not delete them if they own history.
- Homepages are legacy; dashboards are current.
- Module visibility ≠ table ACL, but both can hide data.
- `admin` is not always enough; ACL work needs **elevated** `security_admin`.
- Now Mobile = iOS + Android.
- Localization needs the language plugins plus user language setting.

## Domain 2 self-check

- [ ] Activate a harmless plugin (or open Guided Setup for one already active).
- [ ] Change banner logo on PDI, then revert.
- [ ] Create group **CSA Study**, role `u_csa_reader`, add a user, impersonate.
- [ ] Hide a module from `itil` and confirm with impersonation.
- [ ] Distinguish personal list layout vs Configure > List Layout.

Next: [Chapter 03 — Configuring Applications for Collaboration](03-configuring-applications-for-collaboration.md).
