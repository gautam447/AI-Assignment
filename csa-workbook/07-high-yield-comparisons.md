# Chapter 07 — High-Yield Comparisons

If two answers look right, this chapter is how you separate them. Memorize the left-column names and one sentence each.

---

## Personalization vs customization

| | Personalization | Customization |
| --- | --- | --- |
| Who | One user | Instance / many users |
| Examples | Favorites, gear list columns, language, notification subscriptions | Form layout for a view, BR, catalog item, banner logo |
| Update set | No | Yes (typically) |

---

## Configure vs Personalize (lists/forms)

| Configure > List/Form Layout | Gear / Personalize List |
| --- | --- |
| Changes the **view** for everyone who uses it | Changes **my** columns |
| Needs `personalize_list` / `personalize_form` / admin | Ordinary user |
| Green dot? No | Green dot on gear |

---

## Submit vs Update vs Save vs Insert vs Insert and Stay

| Action | New sys_id? | Form closes? |
| --- | --- | --- |
| Submit | Yes (new record) | Yes |
| Update | No | Yes |
| Save | No (or first insert of new) | **No** |
| Insert | **Yes** (clone of current values) | Yes |
| Insert and Stay | **Yes** | **No** |

---

## My Work vs My Groups Work

| My Work | My Groups Work |
| --- | --- |
| Assigned **to me** | Assigned to **my group**, usually unassigned to a person |

---

## Additional comments vs work notes

| Additional comments | Work notes |
| --- | --- |
| Caller / customer visible | Fulfiller internal |
| Often emails the caller | Often emails the assignment group |

---

## SLA vs OLA vs UC vs inactivity monitor

| SLA | Customer-facing clock |
| --- | --- |
| OLA | Internal team clock |
| UC | Supplier clock |
| Inactivity monitor | “Nobody touched this” alert, not a contract |

SLA definition: **Start / Pause / Stop**. Instances live on `task_sla`.

---

## VTB types

| Guided | Lanes = a field (State); drag updates the field |
| --- | --- |
| Flexible | Field-based plus extra lanes |
| Freeform | You invent lanes; personal/ad-hoc |

---

## Report Share vs Publish vs Schedule

| Share | Give to users/groups/roles (recommended) |
| --- | --- |
| Publish | Public-ish URL; needs publisher rights |
| Schedule | Recurring email of the report |
| Add to dashboard | Widget on a dashboard |

Report Designer tabs: **Data → Type → Configure → Style**.

---

## Knowledge article vs catalog item vs record producer vs order guide vs content item

| Knowledge | Instructions; no fulfillment chain |
| --- | --- |
| Catalog item | REQ → RITM → SCTASK |
| Record producer | Creates a row on a **target table** (often Incident) |
| Order guide | One form, **many** items / RITMs |
| Content item | Link/knowledge; not an order |

---

## User criteria vs ACL vs module role

| Module role | Sees the **menu item** |
| --- | --- |
| ACL | Sees/changes **records and fields** |
| User criteria | Sees **catalog/knowledge** objects |

All three can apply at once.

---

## Flow vs Workflow Editor vs Business Rule vs PAD playbook

| Flow / Workflow Studio | Modern automation; **can wait** |
| --- | --- |
| Workflow Editor | Legacy graph; still on old catalog items |
| Business Rule | Server, same transaction; **cannot wait** |
| PAD playbook | Cross-enterprise / agent-guided stages |

---

## UI Policy vs Data Policy vs Client Script vs ACL vs Dictionary

| UI Policy | Client UX: mandatory / visible / read-only |
| --- | --- |
| Data Policy | Server enforcement of mandatory / read-only (imports, API) |
| Client Script | Client JavaScript (onLoad/Change/Submit/CellEdit) |
| Dictionary | Always-on field definition |
| ACL | **Security** (CRUD); not cosmetics |

---

## Business Rule timings

| before | Validate / set fields / abort / query filter |
| --- | --- |
| after | Other records, events; keep short |
| async | Slow work, out of band |
| display | `g_scratchpad` for the form |

---

## Client API vs server API

| Client | `g_form`, `g_user`, `g_scratchpad` |
| --- | --- |
| Server | `current`, `previous`, `gs`, `GlideRecord` |

Language: **JavaScript**. Not Java. Not Jelly.

---

## Update set vs Import Set vs Clone vs App install

| Update set | Move **configuration** DEV→TEST→PROD |
| --- | --- |
| Import Set | Move **data** from files/sources via transform map |
| Clone | Copy **whole instance** (usually PROD → subprod) |
| Scoped app install | Versioned application files |
| XML record export | Small data sets, not a release process |

---

## Coalesce vs no coalesce

| Coalesce on key | Match → **update**; no match → insert |
| --- | --- |
| No coalesce | Always **insert** (duplicates) |

---

## Table ACL vs field ACL

| `incident` / `.None` | May I access the **row**? |
| --- | --- |
| `incident.*` | Fields in general |
| `incident.priority` | That field |

Fail table read → no field can save you. Parts of **one** ACL are **AND**. Multiple ACLs at the same spot are typically **OR**.

---

## Authentication vs authorization vs elevation

| Authentication | Local / LDAP / SSO / MFA / cert |
| --- | --- |
| Authorization | Roles, ACLs, criteria |
| Elevation | `security_admin` **this session** to edit ACLs |

---

## CMDB vs Asset vs CSDM vs Discovery vs Service Mapping

| CMDB | Operational CIs + relationships (`cmdb_ci`, `cmdb_rel_ci`) |
| --- | --- |
| Asset | Financial/lifecycle (often `alm_*`); related but not the same |
| CSDM | **Model and vocabulary** for how to use CMDB classes |
| Discovery | Bottom-up find devices (MID Server) |
| Service Mapping | Top-down map a service |

Foundation data (users, locations) is mostly **not** related with `cmdb_rel_ci`.

Business Application (Design) ≠ Application Service (deployed instance).

---

## ServiceNow vs customer (Shared Responsibility)

| ServiceNow | Data centers, platform software, instance isolation, cloud ops |
| --- | --- |
| Customer | Users, groups, roles, ACLs, what data you store, MID Servers, acting on Security Center |

---

## Homepages vs dashboards

| Homepages | Legacy; often deprecated by property |
| --- | --- |
| Dashboards | Current; reports + PA visualizations |

---

## Impersonate vs login as vs elevate

| Impersonate | Test as that user; no password |
| --- | --- |
| Login as them | You do not have their password; don’t |
| Elevate | Extra privilege on **your** session |

---

## Next: labs

Practice every pair you are shaky on in [Chapter 08](08-hands-on-lab-workbook.md).
