# Chapter 04 — Self-Service and Automation (20%)

**Blueprint subtopics**

1. Knowledge Management  
2. Service Catalog  
3. Workflow Studio  
4. Virtual Agent  

This domain is about letting employees help themselves and automating what happens after they click Submit.

---

## 1. Knowledge Management

Knowledge Management (KM) is the platform’s system of record for **articles** that answer questions and deflect incidents.

### Core records

| Record | Table (typical) | Purpose |
| --- | --- | --- |
| Knowledge Base | `kb_knowledge_base` | Container with owners and criteria |
| Category | `kb_category` | Hierarchy inside a base |
| Article | `kb_knowledge` | The document |
| User Criteria | `user_criteria` | Who can read or contribute |
| Feedback / rating | related | Helpfulness |
| Submission | kb submission flows | End-user suggests an article |

### Knowledge bases

A base has:

- **Manager / owner**
- **Can Read** user criteria (who sees articles)
- **Can Contribute** user criteria (who can author)
- Workflow for publish
- Sometimes a unique number prefix

Multiple bases are normal: IT, HR, Legal, Customer. Do **not** dump every article into one base if audiences differ.

### User criteria

User criteria match users by:

- Users, groups, roles
- Company, department, location
- Advanced script

Used heavily by **Knowledge** and **Catalog**. If an article or item is invisible, user criteria is the first catalog/knowledge-specific check after roles.

**Can Read** vs **Can Contribute** are separate. A fulfiller might contribute to IT but only read HR.

### Article lifecycle

Typical states:

1. **Draft** — authoring
2. **Review** — assigned reviewers / knowledge workflow
3. **Published** — visible to readers who pass criteria
4. **Retired** — hidden from search; kept for history
5. **Outdated** / revision — new version may replace

Versioning: publishing often creates a **new version**; previous published versions can remain in history. Users search **published** articles only.

### Authoring features

- HTML editor, images, attachments
- Templates for consistent structure
- Import Word (in some versions)
- Related articles / kb_knowledge related lists
- Valid to / valid from dates
- Display with / without rating

### How users find knowledge

- Global search
- Portal / Employee Center search
- Contextual search on Incident (suggested articles while typing short description)
- Virtual Agent “search knowledge” topics
- Knowledge blocks (reusable chunks) in newer KM

### Social and quality

- Helpful yes/no, comments, flags
- View counts
- Expired articles (review date)
- Knowledge owners use analytics to see deflection

### Knowledge vs catalog vs incident

| Need | Use |
| --- | --- |
| How do I reset VPN? | Knowledge article |
| I want a laptop | Catalog item |
| VPN is down for me right now | Incident (often a record producer) |

A **record producer** can still show **knowledge** results before submit (deflection).

---

## 2. Service Catalog

The Service Catalog is the **ordering system** for goods and services (and often for “report an issue” record producers).

### Hierarchy of catalog records (memorize)

When a user orders a **catalog item** (not a record producer):

```
REQ   sc_request        The overall request (shopping cart)
 └── RITM  sc_req_item  One per catalog item line
      └── SCTASK sc_task  Fulfillment tasks (0..n)
```

- One cart checkout → one **REQ**
- Two items in the cart → two **RITMs** under that REQ
- Each RITM may spawn several **SCTASks** (hardware team, software team, …)

**Record producers** skip this hierarchy: they create a record directly on a target table (often `incident`). No REQ/RITM unless you build that yourself.

### Catalog structure

| Object | Purpose |
| --- | --- |
| Catalog | Top-level catalog (Service Catalog, Technical Catalog, …) |
| Category | Folder of items (Can We Help You?, Hardware, …) |
| Catalog Item | Orderable offering |
| Variable | Question on the item form |
| Variable set | Reusable bundle of variables (e.g. Requested for + Location) |
| UI Policy / Catalog Client Script | Catalog-form behavior |
| Flow / Workflow | Fulfillment automation |
| User criteria | Who can see the catalog, category, or item |

### Catalog item types

| Type | What it does |
| --- | --- |
| **Catalog Item** | Standard order → REQ/RITM/(SCTASK) |
| **Record Producer** | Creates a record on any table (Incident, HR case, …) |
| **Order Guide** | One form that orders **multiple items** in a controlled set (onboarding: laptop + phone + access) |
| **Content Item** | Link to knowledge, URL, or other content — not an order |
| **Wizard** (legacy) | Stepped experience |

Official sample: Multiple Choice, Single Line Text, Select Box = **variable types**, not order guides.

### Variable types (high frequency)

| Variable | Looks like |
| --- | --- |
| Single Line Text | Short string |
| Multi Line Text | Paragraph |
| Multiple Choice | Radio buttons |
| Select Box | Dropdown |
| Checkbox | Boolean |
| Yes / No | Boolean-style |
| Date / Date-Time | Calendar |
| Reference | Lookup to a table |
| Lookup Select Box | Choices from a table |
| Requested For | User picker (common) |
| Attachment | File |
| Numeric Scale / Wide Text / HTML / Label / Container / Break | Layout and extras |
| List Collector | Multi-select from a table |
| Masked | Hidden input (PIN) |

**Variable sets** = reuse. Change once, all items that include the set update.

Containers, begin/end splits, and UI policies control **hiding** variables based on other answers (software name only if “Software” selected).

### Prices, quantities, and carts

Items can have price, recurring price, quantity. Cart and two-step checkout vs one-step order now (configure catalog settings). CSA expects existence of cart/order now, not every property name.

### Fulfillment

After submit:

1. Flow or workflow on the **item** or on `sc_req_item` runs
2. Approvals may pause the RITM
3. Catalog tasks (`sc_task`) assign to groups
4. When tasks complete and flow ends, RITM and REQ close

If no flow is attached, you get a request record that **nobody fulfills** unless a default rule exists. Always attach a flow in labs.

### Catalog UI Policies and Catalog Client Scripts

Same idea as form UI Policy / Client Script, but they run on the **catalog item form** (portal and native catalog). They do not automatically apply to the generated RITM form unless you also configure that.

### Record producer mapping

Producer variables map to target fields (Short description ← description variable). Unmapped variables can land in the description or a scratchpad. Know that **mapping** is how a producer fills Incident fields.

### Order guides

- Rule base: if location = Remote, include VPN item
- Cascading variables: requested-for flows into child items
- Produces multiple RITMs under one REQ

### Who sees what

Visibility stack:

1. Catalog published / active
2. Category active
3. Item active + available for
4. **User criteria** (not available for / available for)
5. Portal widgets search context

Impersonate an end user with **no itil role** when testing catalog.

---

## 3. Workflow Studio

**Workflow Studio** is the current umbrella for process automation on the Now Platform. For CSA you should treat it as:

| Tool | Job |
| --- | --- |
| **Flow Designer** | Record/date/application-triggered flows, subflows, actions |
| **Process Automation Designer (PAD)** | Cross-enterprise **playbooks** (stages, lanes, human tasks) |
| **Playbooks** | Guided runtime experience for agents |
| **Workflow Editor** | **Legacy** graph workflows (`wf_workflow`) — still exists on older items |

Blueprint wording is **Workflow Studio**, not only “Flow Designer.” If an answer choice says Workflow Studio and the question is about modern no-code automation, that is usually the current name.

### Flow Designer concepts

A **flow** =

1. **Trigger**
2. **Actions** (and logic: if, for each, wait, do in parallel)
3. Optional **error handling**
4. Data pills (outputs of earlier steps)

#### Triggers

| Trigger type | Example |
| --- | --- |
| Record | Incident created, RITM created, field updated |
| Date | Daily, weekly, specific datetime |
| Application | Service Catalog, SLA Task, Inbound email, MetricBase |
| Scheduled | Repeat |

Service Catalog trigger is how you bind a flow to an item instead of a legacy workflow.

#### Actions and spokes

An **action** is a reusable step. A **spoke** is a bundle of actions (ServiceNow Core, Slack, Teams, Jira, REST, …).

Core actions you should be able to name:

- Create Record / Update Record / Delete Record / Look Up Records
- Ask for Approval
- Send Email / Send Notification
- Create Catalog Task
- Log
- Call Subflow
- Wait for condition / Wait for duration

**Integration Hub** spokes require licenses for many third parties; CSA only needs the idea that spokes **encapsulate integrations** so you do not write SOAP from scratch.

#### Subflows

Reusable sequences **without** their own trigger. Called from flows. Pass inputs/outputs. Use them like functions.

#### Roles

| Role | Can |
| --- | --- |
| `flow_designer` | Create/edit flows |
| `action_designer` | Create/edit actions |
| `flow_operator` | Run / view execution details (operations) |

Executions show each step’s status — primary debugging tool. If a catalog item “does nothing,” open **Today’s Executions** / flow context.

#### When to use Flow Designer vs Business Rule vs Workflow Editor

| Use | Why |
| --- | --- |
| Flow Designer / Workflow Studio | New automation; readable; spokes; ServiceNow recommendation |
| Business Rule | Tight data-layer logic, query rules, cannot wait, must run in the same transaction |
| Workflow Editor | Maintain existing legacy workflows; avoid for new design if Flow can do it |

Flows can **wait** (approval, timer). Business Rules should **not** wait. That single distinction answers many items.

### Process Automation Designer / playbooks

PAD designs a **playbook**: stages that agents follow (qualify, investigate, resolve) with activities that may call flows. Used in CSM/ITSM workspaces.

CSA: know that **cross-enterprise** or **playbook-style** automation is PAD inside Workflow Studio, while a single “when record inserted, create task” is a Flow.

---

## 4. Virtual Agent

Virtual Agent (VA) is the conversational assistant for self-service.

### What it can do

- Answer FAQs (often by searching **Knowledge**)
- Guided troubleshooting
- Run **catalog** orders / record producers
- Check ticket status
- Authenticate the user and act as them
- **Issue Auto Resolution** (IAR): intercept or resolve incidents conversationally where licensed
- Handoff to a live agent

### Where it appears

- Service Portal / Employee Center widget
- Now Mobile
- Microsoft Teams, Slack, Google Chat / Workplace (integrations)
- Web chat widget

### Building blocks (conceptual)

| Piece | Purpose |
| --- | --- |
| Topics | A conversation script (password reset, order item) |
| Topic blocks | Reusable conversation fragments |
| NLU / Generative AI | Intent matching (Now Assist in newer releases) |
| Integrations | Catalog, knowledge, records |
| Setup topics | Greeting, fallback, live agent |

You are **not** expected to author a full NLU model for CSA. You **are** expected to know VA consumes **Knowledge + Catalog + system records**, and that it is a self-service channel equal to portal search.

### Link unfurling

VA can show previews of links (images, videos). Disable with a system property such as `com.glide.cs.enable_link_unfurling = false`. This exact property appears in community study lists.

### Live agent handoff

When NLU confidence is low or the user asks for a human, VA transfers to a chat queue (Agent Chat). That is still self-service automation plus collaboration.

---

## Putting the self-service stack together

A well-designed employee experience:

1. User asks VA or searches portal  
2. **Knowledge** article solves it → no ticket  
3. If they need something done → **Catalog item** or **record producer**  
4. **Flow** approves, creates SCTASKs, notifies  
5. User tracks REQ/RITM in portal  
6. Fulfillers work tasks in Next Experience / workspace  

CSA scenarios often ask which **layer** to use. Pick the least-ticket, most-standard option that fits.

---

## High-yield facts

- Knowledge visibility = published + user criteria.
- Catalog order chain: **REQ → RITM → SCTASK**.
- Record producer ≠ catalog item (no request chain by default).
- Order guide = many items, one request.
- Variables = questions; variable sets = reuse.
- Workflow Studio = Flow Designer + PAD/playbooks (+ legacy workflow access).
- Flows can wait; Business Rules cannot.
- Catalog items need a flow/workflow or they will not fulfill themselves.
- Virtual Agent sits on knowledge + catalog + chat channels (portal, mobile, Teams, Slack).
- Test everything by **impersonating a user with no itil role**.

## Domain 4 self-check

- [ ] Create a knowledge base, article, publish, impersonate, search portal.
- [ ] Create a catalog item with a variable set, attach a flow that creates an SCTASK and asks approval.
- [ ] Create a record producer that opens an Incident.
- [ ] Sketch REQ/RITM/SCTASK from a two-item cart.
- [ ] Open Workflow Studio and identify trigger vs action vs subflow.
- [ ] Find the Virtual Agent designer and list two OOB topics.

Next: [Chapter 05 — Database Management and Platform Security](05-database-management-and-platform-security.md).
