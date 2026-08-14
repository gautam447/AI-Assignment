# Chapter 06 — Data Migration and Integration (13%)

**Blueprint subtopics**

1. UI Policies  
2. Business Rules  
3. System update sets  
4. Scripting in ServiceNow  

The name “Data Migration and Integration” is broader than the subtopics. On the January 2026 blueprint this domain is really **how configuration and logic move and run**. Import Sets themselves sit in Domain 5; here you master **client vs server logic** and **update sets**.

---

## 1. UI Policies

A **UI Policy** is a no-code rule that changes a **form in the browser**.

Path: table form context menu → Configure → UI Policies, or All > System UI > UI Policies (`sys_ui_policy`).

### What a UI Policy can do (via UI Policy Actions)

- **Mandatory** true/false
- **Visible** true/false
- **Read-only** true/false

It does this **when a condition is true** (and can reverse when false if “Reverse if false” is checked — usually leave it on).

### Where it runs

- **Client** (browser) on forms that load the policy
- Optional: **Global** vs specific view
- Catalog has a sibling: **Catalog UI Policy**

UI Policies do **not** protect data on import, mobile in all cases, web services, or list edit unless other features help. They are **UX**, not security.

### UI Policy vs Dictionary vs Data Policy vs Client Script

| Need | Tool |
| --- | --- |
| Always mandatory, all records, all input paths | Dictionary mandatory or **Data Policy** |
| Mandatory only when Priority is 1 on the form | **UI Policy** |
| Hide a field with script / set a value on change | Client Script (or UI Policy + script if needed) |
| Prevent unauthorized read | **ACL** |

Prefer UI Policy over Client Script when you only need mandatory/visible/read-only. ServiceNow training is explicit: **low code first**.

### Order

UI Policies have an **order** number. Higher order runs later. If two policies fight (one clears mandatory, one sets it), order and reverse-if-false explain the result.

### On load

Run on form load and when conditions change. There is no onSubmit UI Policy; use Client Script onSubmit for “block save.”

---

## 2. Business Rules

A **Business Rule (BR)** is **server-side** JavaScript (or low-code) that runs when a record is accessed in the database.

Table: `sys_script`. Path: All > System Definition > Business Rules.

### When they run (timing)

| When | In the same database transaction? | Typical use |
| --- | --- | --- |
| **before** | Yes, before the insert/update/delete is committed | Validate, abort, set fields (`current.priority = 1`), modify query |
| **after** | Yes, after commit of this record | Update **other** records, fire events, but keep it light |
| **async** | **No** — queued | Email-ish work, integrations, anything slow |
| **display** | On form load, server → client | Put data in `g_scratchpad` for Client Scripts |

**Query** is a special **before** BR on read: `current.addQuery(...)` filters rows. This can hide records like an ACL and is easy to forget when troubleshooting.

### Insert / Update / Delete / Query checkboxes

A BR is bound to operations. “When to run” conditions (Filter Conditions) plus Advanced script.

### `current` and `previous`

- `current` — the record being processed  
- `previous` — the record before this update (update only)

`current.setAbortAction(true)` stops a before BR insert/update.

### Business Rule vs Flow vs Workflow

| | Business Rule | Flow |
| --- | --- | --- |
| Wait for approval 2 days | No | Yes |
| Set a field in the same save | Yes (before) | Possible but BR is tighter |
| Debug | Scripts debugger / log | Execution details |
| Admin who cannot script | Harder | Better |

If the exam says “automatically set assignment group **when the record is saved**, using server logic,” BR is correct. If it says “wait for approval then create a catalog task,” Flow is correct.

### Data Policies (bridge topic)

**Data Policy** (`sys_data_policy2`) enforces mandatory and read-only on the **server** (and can convert to UI Policy for the client). Use when values must be enforced for:

- Import Sets
- Web services
- List editor
- Record producers / API

UI Policy alone will **not** catch those paths.

---

## 3. System update sets

An **update set** is a container that records **configuration changes** so you can move them to another instance.

Path: All > System Update Sets > Local Update Sets (`sys_update_set`).

### What is captured

Generally **customizations** (customer updates) such as:

- Table/field dictionary changes
- Form/list layouts, views
- Business Rules, Client Scripts, UI Policies, UI Actions
- Modules, applications
- Catalog items, variables (as application files)
- ACLs, properties (most)
- Reports (as config)

Each captured file is a **Customer Update** (`sys_update_xml`) with a name like `sys_script_abc123`.

### What is typically NOT captured

**Data** — incidents, users, CMDB CIs, knowledge article *content* in many cases, attachments, scheduled execution logs.

Some things need **careful** handling:

- **Choices** may be in the update set
- **Demo data** is not an update set job
- **Groups/users** are data (sometimes exported separately)
- **Homepages** historically needed extra steps
- **Workflows** must be **published**; checkout state matters
- **Update set collisions** when two sets touch one record

To move **data**, use Import Sets, XML export of records, clone, or packaged datasets — not a normal update set.

### Lifecycle (DEV → TEST → PROD)

On **source** (DEV):

1. Create a **named** update set (never pile work into **Default**)
2. Make it **current**
3. Do the work
4. Mark the set **Complete**
5. **Export to XML** or use **Update Sources** / retrieve

On **target** (TEST/PROD):

1. **Retrieved Update Sets** (or import XML)
2. **Preview**
3. Fix **errors** (missing precedes, collisions)
4. **Commit**
5. Test

**Preview** is mandatory. Commit without preview is how you break production.

### Default update set

Every instance has Default. It catches changes when you forgot to switch sets. **Do not** migrate Default as your release vehicle. Move accidentally captured updates to a named set (`Move to Update Set`) before they escape.

### Merge / batch

You can **merge** update sets or use **batch** / parent sets so many completed sets preview as one release. Use this for a sprint’s worth of config.

### Backout

Committed update sets can sometimes be **backed out**, restoring previous customer update versions. This is not a substitute for source control or a clone. Treat backout as emergency, not process.

### Update sources

A production instance can **pull** from DEV if an update source is defined (credentials + URL). Retrieve → Preview → Commit. XML file transfer is equally common in CSA training.

### Scoped apps vs update sets

- Classic global work: **update sets**
- App Engine scoped apps: **publish app / install version / Git**
- Mixed: you may still see app files in update sets

CSA questions still center on **update set preview/commit** and **what is captured**.

### Customer updates list

If a change is missing from the set, look at **Customer Updates** related list. If empty, you were not in the right current set, or the object is not captured (data).

---

## 4. Scripting in ServiceNow

Official sample: the language is **JavaScript**.

You do **not** need to write complex algorithms for CSA. You **do** need to know **where code runs**, **which API**, and **which script type**.

### Client vs server (draw this from memory)

```
BROWSER (client)                         SERVER (instance)
----------------                         -----------------
Client Scripts                           Business Rules
UI Policies (+ scripts)                  Script Includes
Catalog Client Scripts                   Scheduled Jobs / Script Actions
some UI Actions                          many UI Actions
g_form, g_user, g_scratchpad             current, previous, gs, GlideRecord
DOM (avoid)                              GlideSystem, GlideDateTime
```

**GlideRecord in a Client Script** is a bad practice: it is a **synchronous round trip** (or uses the old `GlideRecord` client stub). Use **GlideAjax** + Script Include instead. Exam: client GlideRecord = performance problem.

### Client Scripts (`sys_script_client`)

Types:

| Type | When | Typical use |
| --- | --- | --- |
| **onLoad** | Form opens | Set display values, g_scratchpad consumption |
| **onChange** | A field changes | Hide, auto-fill, cascade |
| **onSubmit** | User submits | Validate; `return false` blocks save |
| **onCellEdit** | List editor | Validate list edits |

APIs:

- `g_form.getValue('priority')`, `setValue`, `setMandatory`, `setDisplay`, `setReadOnly`, `showFieldMsg`
- `g_user.userName`, `hasRole('itil')`
- `g_scratchpad.foo` (filled by display BR)

Client Scripts are **per table and view** (can be global). They are customization, captured in update sets.

### UI Actions (scripting angle)

May be client, server, or both. A client UI Action can call `gsftSubmit` to continue server-side. Know they are **buttons/links**, not the same as UI Policies.

### Script Includes

Reusable **server** functions. Can be:

- Called from BRs, flows, scheduled jobs
- **Client callable** (GlideAjax) — checkbox + `this.object.prototype`

This is the correct pattern for “get data from server while the user is on the form.”

### GlideRecord (server)

```javascript
var gr = new GlideRecord('incident');
gr.addQuery('priority', 1);
gr.addActiveQuery();
gr.query();
while (gr.next()) {
  gs.info(gr.number);
}
```

Know: `addQuery`, `query`, `next`, `insert`, `update`, `deleteRecord`, `get(sys_id)`.

**GlideSystem** `gs`: `gs.info`, `gs.error`, `gs.hasRole`, `gs.getUserID`, `gs.eventQueue`.

### Events and Script Actions

`gs.eventQueue('incident.updated', current, parm1, parm2)` fires events that **notifications** and **Script Actions** listen to. Notifications in Domain 3 hook these events.

### Other script types (recognize names)

| Type | Side | Use |
| --- | --- | --- |
| Scheduled Job / Scheduled Script Execution | Server | Cron-like |
| Script Action | Server | Runs on event |
| Fix Script | Server | One-off data fix (not for CSA build) |
| Background Script | Server | Admin scratch pad |
| Transform Map script | Server | Imports |
| ACL script | Server | `answer = true/false` |
| Inbound email script | Server | Parse mail |
| Notification mail script | Server | `${mail_script:...}` |

### Jelly, Angular, UI Builder

Legacy CMS/UI pages used **Jelly**. Service Portal used **AngularJS**. Workspaces use **UI Builder**. CSA answer for “what language do you script the platform in?” is still **JavaScript**, not Java, not Jelly (Jelly is a distractor from the official sample).

### Debugging

| Tool | Use |
| --- | --- |
| JavaScript Log and Field Watcher | Client |
| `gs.info` / System Logs > All | Server |
| Script Debugger / Session Debug | Server step-through |
| Debug Security / Debug ACLs | Why ACL failed |
| `cache.do` | Stale scripts/properties |
| Flow execution details | Flows, not BR |

---

## Integration awareness (domain name)

Without turning this into CIS-Integration:

- **REST / SOAP** outbound via Integration Hub spokes or REST Message
- **Inbound REST** Table API / Import Set API
- **LDAP** for users
- **MID Server** for inside-the-firewall endpoints
- **Email** as a primitive integration
- **Update sets / apps** move *config*, not the integration’s *data*

If asked “how do you load a spreadsheet of users?” the answer is **Import Set** (Domain 5), not an update set.

---

## Decision tree: which logic object?

```
Need to wait / approval / multi-step?     → Flow (Workflow Studio)
Need mandatory/visible/read-only on form? → UI Policy
Need that enforcement on import/API too?  → Data Policy
Need to set fields or abort on save?      → before Business Rule
Need to talk to other records after save? → after BR or Flow
Need slow / external call?                → async BR or Flow + spoke
Need browser validation / UX script?      → Client Script
Need reusable server function / Ajax?     → Script Include
Need to stop unauthorized access?         → ACL (not UI Policy)
Need to move the config to TEST?          → Update set / app install
Need to move 10,000 user rows?            → Import Set
```

---

## High-yield facts

- UI Policy = client mandatory/visible/read-only.
- Data Policy = server (imports/API too); can mirror to UI.
- Business Rule timings: before, after, async, display (+ query).
- Flows wait; BRs do not.
- Update sets capture **configuration**, not operational data.
- Never develop in Default update set; Preview then Commit.
- Scripting language = **JavaScript**.
- `g_form` / `g_user` = client; `current` / `gs` / `GlideRecord` = server.
- Display BR + `g_scratchpad` + onLoad Client Script is the standard handshake.
- Jelly is not the CSA scripting answer.

## Domain 6 self-check

- [ ] UI Policy: Priority 1 makes Description mandatory; impersonate to test.
- [ ] Data Policy on the same field; import a row without description and watch it fail.
- [ ] before BR: set assignment group from category.
- [ ] Create named update set, add the BR, complete, export XML, inspect customer updates.
- [ ] onChange Client Script vs UI Policy: know when each is enough.
- [ ] Explain display BR → g_scratchpad → Client Script in one paragraph.

Next: [Chapter 07 — High-yield comparisons](07-high-yield-comparisons.md).
