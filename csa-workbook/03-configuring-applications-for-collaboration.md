# Chapter 03 — Configuring Applications for Collaboration (20%)

**Blueprint subtopics**

1. Lists, Filters, and Tags  
2. List and Form anatomy  
3. Form Configuration  
4. Form templates and saving options  
5. Advanced Form Configuration  
6. Task Management  
7. Visual Task Boards (VTBs)  
8. Visualizations, Dashboards, and Platform Analytics  
9. Notifications  

This is one of the two 20% domains. Expect “what happens when you click X” questions.

---

## 1. Lists, Filters, and Tags

### What a list is

A list is a table of records. Each **row** is a record. Each **column** is a field.

### List anatomy

| Part | Function |
| --- | --- |
| Title bar | Table name, count, search, refresh, view name, list actions |
| Breadcrumbs | Current filter, clickable to peel off conditions |
| Column headers | Sort, filter, column context menu |
| Row checkboxes | Bulk actions |
| Fields | Display values; some allow list editing |
| Related lists | Appear on **forms**, not on the main list (do not confuse) |

### Context menus (right-click)

There are several, and exams love naming them:

| Menu | Where you click | Typical actions |
| --- | --- | --- |
| List title / header menu | List title | View, filters, refresh, create favorite |
| List column context menu | Column header | Sort, group, bar/pie chart, export, configure list |
| List field context menu | A cell | Show matching, filter out, copy URL, assign tag |
| Record / row context menu | A row | Open, copy, assign |
| Form context menu | Form header | Configure, copy sys_id, templates, save options |

### Filters and breadcrumbs

A condition has three parts:

1. **Field**
2. **Operator** (is, is not, contains, greater than, is empty, …)
3. **Value**

The breadcrumb trail (for example `Active = true > Priority = 1`) is the filter. Click an earlier crumb to drop later conditions.

Ways to build filters:

- Filter icon / condition builder
- Click a cell → Show Matching / Filter Out
- Natural Language Query (where enabled): type English, platform converts to conditions
- Encoded query (advanced): `active=true^priority=1`

**Save filter:** name it and set visibility:

- Me
- Group
- Everyone (requires rights)

Dragging a breadcrumb to **Favorites** saves that filter as a shortcut.

### List editing

Double-click a field (when list editing is enabled and the field is not read-only) to edit **inline** without opening the form. Not all field types support this.

### Personal vs configured list layout

| Gear (Personalize List) | Configure > List Layout |
| --- | --- |
| Per user | Per view, for all users of that view |
| Green dot on gear when personalized | No green dot |
| Reset to default restores view layout | Changes the view itself |
| Role often not required | `personalize_list` / `admin` |

### Tags

Tags are labels you attach to records (and sometimes pages) to group work without changing the data model.

- Personal tags vs shared/assignable tags (depending on configuration)
- Filter lists by tag
- Useful for ad-hoc collaboration (“release-week”, “needs-CAB”)

Do not confuse tags with **categories** (a real field) or **user criteria**.

### List search wildcards

Common list search behaviors:

- `text` often means starts-with on some fields
- `*text` contains
- `%` / `*` usage can vary by field type

If a question mentions searching “anywhere in the field,” pick the **wildcard** answer.

### Export and quick reports

From a list column context menu you can:

- Export (Excel, CSV, XML, JSON, PDF — formats vary)
- Create a bar or pie chart from that column (quick report)
- Configure list

---

## 2. List and Form anatomy

### Form anatomy

| Part | Function |
| --- | --- |
| Form header | Number, header buttons (Save, Update, …), more actions |
| Form sections | Groups of fields (for example Notes, Related Records) |
| Fields | Data controls |
| Embedded lists | Occasional in-form lists |
| Related lists | Records pointing to this one (e.g. Affected CIs) |
| Activity / activity stream | History of comments, work notes, field changes |
| Form annotation | Instructional text on the form |
| Formatter | UI widget (activity stream, process flow, CI relations) |
| UI actions | Buttons, links, context menu items |

### Field types you must recognize

| Type | Behavior |
| --- | --- |
| String / Wide Text | Text; max length in dictionary |
| Integer / Decimal / Float | Numbers |
| Boolean | True/false (checkbox) |
| Choice / Multiple Choice | Dropdown / radio from choice list |
| Date / Date-Time | Calendar |
| Reference | Points to a record on another table; lookup icon |
| List (glide_list) | Multiple references |
| Journal / Journal Input | Additional comments, work notes |
| HTML | Knowledge bodies, emails |
| Duration | SLA-style times |
| Image / File attachment | Binary (attachments also via paperclip) |
| URL | Link |
| Color | Color picker |
| Translated text | I18N |

**Reference fields** show a display value (usually Name or Number) but store a **sys_id**. The lookup icon opens a popup list. **Dot-walking** follows reference arrows (Incident.Caller.Department).

### Activity stream

On task forms, under Notes:

- **Additional comments** — customer-visible
- **Work notes** — internal
- Field history if auditing is on

**User Presence** may show who else has the record open.

---

## 3. Form Configuration

### Views

A **view** is a named layout of the same table (Default, Self Service, Workspace, Mobile, ess, …). Changing Default does **not** change Self Service.

To switch views: context menu on list or form → View.

### Form Layout vs Form Designer

| Form Layout | Form Designer |
| --- | --- |
| Classic slushbucket | Visual canvas |
| Sections as lists of fields | Drag fields, sections, annotations |
| Configure > Form Layout | Configure > Form Design |
| Same result: `sys_ui_section` / form records | Same result |

Field Navigator in Form Designer: existing fields vs creating a **new field** (which also creates a dictionary entry — that is a data model change, Domain 5).

Field controls in designer: **handle** (move), **remove** (from layout, not from dictionary), **edit** (dictionary/label).

Roles: `admin`, `personalize_form`.

### Related lists

Configure > Related Lists adds/removes tabs at the bottom of the form. A related list is usually:

- Records whose reference field points here (one-to-many)
- Or an M2M table
- Or a defined relationship

Example: Incident form → Related lists: Affected CIs, Child Incidents, SLAs, Catalog Tasks (depending on table).

### Form sections

Use sections to group related fields and to keep forms usable. Tabs vs stacked sections is a form layout option.

---

## 4. Form templates and saving options

### Save / Submit / Update / Insert (memorize this table)

Assume you are on a form:

| Action | New record? | Closes form? | Typical when visible |
| --- | --- | --- | --- |
| **Submit** | Creates | Yes | Inserting a new record |
| **Update** | Updates existing | Yes | Existing record |
| **Save** | Inserts or updates | **No** (stays open) | Header button / context |
| **Insert** | Creates a **copy as new** (new sys_id) using current field values | Yes | Context menu on existing |
| **Insert and Stay** | Creates a copy as new | **No** | Context menu |
| **Copy** (on some tables e.g. Incident) | Table-specific copy logic | Varies | UI action |

**Insert** is not the same as **Submit**. Insert from an *existing* record clones values into a new row.

**Update** is not **Save**. Update closes; Save stays.

These four words appear constantly on CSA.

### Form templates

Templates pre-fill fields (category, assignment group, default text).

- Template bar on the form
- Apply a template when creating similar records
- Toggle template bar from form context menu
- Creating templates is a privilege (`template_editor` / admin patterns)

Templates are a productivity feature, not a security boundary. ACLs still apply.

---

## 5. Advanced Form Configuration

### UI actions

Buttons, links, and menu items (`sys_ui_action`). Can run:

- **Client** (JavaScript in browser, `g_form`)
- **Server** (GlideRecord / `current`)
- Both (client first, then server)

Show on form button, list banner, list context, related list, etc. Conditions and roles control visibility.

### Form annotations and formatters

- Annotations: static help text
- Formatters: activity stream, process flow formatter (states as a graphic), CI relations, calendar

### Dictionary extras that affect forms

- **Mandatory**, **Read only**, **Default value** on the dictionary = always, all UI (and more)
- **UI Policy** = conditional on the form (client)
- **Data Policy** = mandatory/read-only even for imports (server)
- **ACL** = security; can make a field appear empty or read-only
- **Client Script** = custom logic

If a field is mandatory for everyone always, dictionary or data policy is cleaner than a UI policy.

### Dictionary override

On an **extended** table, override parent field behavior (mandatory, default, reference qualifier, display) without changing the parent. Example: `short_description` on Incident vs Task.

### Reference qualifiers

Limit which records appear in a reference lookup (simple encoded query, or advanced script). Example: Assignment group only groups of type `itil`.

### Calculated / function fields

Some fields compute values (concatenation, arithmetic). Know they exist; do not use them as a substitute for Business Rules on complex logic.

### Views for different audiences

- Self Service view: few fields for the caller
- Default view: fulfiller
- Workspace view: agent workspace

A user “missing a field” may simply be on the **wrong view**, not an ACL problem. Check view first in troubleshooting order if they can see the record.

---

## 6. Task Management

### The Task table

`task` is a **base table**. Many process tables **extend** it:

- `incident`
- `problem`
- `change_request`
- `sc_request`, `sc_req_item`, `sc_task`
- `kb_submission` (in some versions)
- Visual Task Board cards can map to tasks

Because they extend Task, they share:

- Number, short description, description
- State / active
- Assignment group, assigned to
- Priority, impact, urgency
- Approval fields
- Work notes / additional comments
- SLAs (via Task SLA)

**Do not customize Task lightly** — dictionary changes cascade to children unless overridden.

### My Work vs My Groups Work

| Module | Meaning |
| --- | --- |
| **My Work** | Active tasks **assigned to me** |
| **My Groups Work** | Active tasks assigned to **one of my groups** and usually **not yet assigned to a person** |

Official sample: My Groups Work is the correct name. “My Teams Work” is a distractor.

Service Desk > My Work can include incidents, requests, changes, VTB tasks, etc. (any task type the module queries).

### Assignment

Ways a task gets an assignee:

1. Manual
2. **Assignment rules** (`sysrule_assignment`) — conditions → group and/or user  
   Path: All > System Policy > Rules > Assignment
3. **Assignment lookup rules** — historically Incident-specific lookup table
4. **Predictive Intelligence** — ML suggests category/assignee
5. Data lookup definitions
6. Flow / Business Rule (custom)

Best-practice patterns (from training):

- Assign a **group** from Category/Subcategory
- Assign a **user** only when the rule is certain
- Combine group + user from category, subcategory, or short description with PI

### Inactivity monitors

Notify (or escalate) when a task sits untouched for a defined time. Different from SLA, which measures **elapsed contractual time**.

### SLAs, OLAs, UCs (Service Level Management)

| Type | Between |
| --- | --- |
| **SLA** | Provider and customer |
| **OLA** | Internal teams |
| **UC** (Underpinning Contract) | Provider and supplier |

SLA definition lifecycle on a task:

1. **Start** condition (attach)
2. **Pause** (e.g. Pending Customer)
3. **Stop** (resolved / complete)

Table: `task_sla` (each attached SLA instance).

SLA engine properties: Service Level Management > Properties.

States like **breached** vs **achieved** appear on the Task SLA related list. CSA expects you to know attach/pause/stop, not every retroactive rebuild option.

### Approvals

Approvals (`sysapproval_approver`) attach to tasks. Users approve via:

- Email links
- Native list
- Portal / Virtual Agent
- Workspace

Flows and Workflow Studio typically drive multi-step approvals now.

### Work notes vs additional comments

| Additional comments | Work notes |
| --- | --- |
| Caller / customer can see | Fulfillers |
| Triggers customer-facing notifications often | Internal notifications |

Watch notification conditions: “comment becomes not empty” vs work notes.

---

## 7. Visual Task Boards (VTBs)

VTBs are Kanban-style boards for collaboration.

### Types

| Type | Lanes come from | Use when |
| --- | --- | --- |
| **Guided** | A field’s choices (e.g. State) | Process must follow that field; dragging updates the record |
| **Flexible** | A field, but you can add extra lanes | Mostly field-driven with extras |
| **Freeform** | You invent lanes | Personal organization; cards may be VTB tasks, not necessarily incidents |

### Capabilities

- Drag and drop to update fields (guided/flexible)
- See bottlenecks (cards piling in a lane)
- Members collaborate on one board
- Activity in one place

**Owner** of the board can edit board structure (lanes, members, sorting). Members work cards.

VTBs are **not** a replacement for Process Automation Designer playbooks, but exam questions treat them as the visual task collaboration tool.

---

## 8. Visualizations, Dashboards, and Platform Analytics

Reporting has been rebranded toward **Platform Analytics** (data visualizations + dashboards). CSA still uses classic **Report Designer** language.

### Opening reporting

- All > Reports > View / Run
- List column context menu → Bar Chart / Pie Chart
- Add to dashboard from a report

### Report Designer tabs

| Tab | You define |
| --- | --- |
| **Data** | Name, source type (table vs data source), table |
| **Type** | Visualization |
| **Configure** | Group by, stacked by, aggregation (count, average, sum, std dev), filters |
| **Style** | Colors, titles, display options |

### Chart families (recognize names)

| Family | Examples |
| --- | --- |
| Time series | Line, column, area, spline, step line |
| Bars | Bar, horizontal bar, histogram, Pareto |
| Pies | Pie, donut, semi-donut |
| Scores | Single score, speedometer, dial |
| Multidimensional | Heatmap, bubble, multi-level pivot |

### Sharing a report

Default: only the **creator** sees it.

| Action | Meaning |
| --- | --- |
| **Share** | Grant to users/groups/roles — **recommended** |
| Publish | Makes a public URL (broader; use carefully) |
| Schedule | Email PDF/Excel on a cadence |
| Add to dashboard | Places widget on a dashboard |
| Export PDF | One-off |

Role **report_publisher** is associated with publishing. Many users can **run** reports they can access without publishing.

### Dashboards

- Collection of widgets (reports, PA visualizations, content)
- Responsive layouts
- Share with roles/groups
- Homepage replacement

Platform Analytics adds:

- **Data visualizations** (newer designer)
- **KPI Signals** — detect meaningful variation vs noise (**anti-signal** = process in control)

You do not need PA admin depth (breakdown sources, indicator sources) for CSA. Know dashboards vs reports vs homepages.

### Quick charts from lists

Right-click a column → Pie Chart / Bar Chart uses **that field as group-by**. Fastest way to visualize the current filter.

---

## 9. Notifications

Notifications tell humans that something happened.

Path: **All > System Notification > Email > Notifications** (`sysevent_email_action`).

### How they fire

Typically:

1. A **system event** fires (for example `incident.updated`, or a custom event from a Business Rule / Flow)
2. Notification **When to send** conditions match
3. **Who will receive** is resolved (and subscriptions / devices apply)
4. **What will it contain** is rendered (subject, body, mail script, SMS text)

### Notification record tabs (classic)

| Tab | Content |
| --- | --- |
| When to send | Table, inserted/updated, conditions, event name, weight |
| Who will receive | Users, groups, fields (Assigned to, Caller), CC/BCC, event parm users |
| What will it contain | Subject, message HTML, SMS, meeting invitation, include attachments |

### Channels

- Email
- SMS
- Push (mobile)
- Meeting invitation (calendar)
- Digest (bundled summary)

### Subscriptions

Users can **subscribe/unsubscribe** to many notifications in System Settings. Admins can mark a notification as **forced** (cannot unsubscribe) for legal/critical mail.

### Inbound email

The instance can **receive** mail:

| Feature | Purpose |
| --- | --- |
| Inbound email action | Reply-to updates a record, or creates an incident from a new mail |
| Email address filters | Allowed senders |
| Watermark / record number in subject | Matches reply to the task |

Inbound action types conceptually: create/update a record, reply, interact with agent capacity in some products.

If email create is enabled, a user can open an incident **without a portal** by mailing the instance.

### Weight and waterfalls

If multiple notifications could send, **weight** can stop lower-weight ones. Know that conflicting notifications are controlled; don’t memorize every number.

---

## Troubleshooting collaboration issues (exam scenarios)

| Symptom | Check in order |
| --- | --- |
| User cannot see a field | View → UI Policy → ACL → Dictionary |
| User cannot see a record | Table ACL / query BR / before-query |
| User cannot see a module | Application/module role |
| Notification not received | Event fired? Condition? Recipient active? Email sending on? Subscription? Spam? |
| Assignment blank | Assignment rule order/condition; group empty; user not in group |
| SLA not attaching | Start condition, table, retroactive start, SLA engine |

---

## High-yield facts

- Filter = Field + Operator + Value; breadcrumbs display it.
- Gear = personal list; Configure = everyone on the view.
- Submit closes (new); Update closes (existing); Save stays; Insert copies to a new record.
- My Groups Work ≠ My Work.
- Task is the parent table for Incident/Problem/Change/Request tasks.
- VTB types: Guided, Flexible, Freeform.
- Report Designer: Data, Type, Configure, Style.
- Share is the recommended way to give others a report.
- Notifications: When, Who, What; usually event-driven.
- Comments vs work notes is a visibility and notification boundary.

## Domain 3 self-check

- [ ] Save a filter visible to a group and favorite the breadcrumb.
- [ ] Personalize columns, then reset; contrast with Configure > List Layout.
- [ ] Use Insert and Stay on an incident and confirm a new number.
- [ ] Create a Guided VTB on Incident State.
- [ ] Build a bar chart, Share it with `itil`, add it to a dashboard.
- [ ] Clone an existing notification and send it to yourself on incident insert.

Next: [Chapter 04 — Self-Service and Automation](04-self-service-and-automation.md).
