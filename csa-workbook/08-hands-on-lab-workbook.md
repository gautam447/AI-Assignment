# Chapter 08 — Hands-On Lab Workbook

Work these on a [Personal Developer Instance](https://developer.servicenow.com). Check off as you go. Each lab maps to exam skills, not to a live test.

Create a named update set first: `CSA Workbook Labs`.

---

## Lab 0 — Orientation (Domain 1)

1. Log in as admin. Identify banner, filter navigator, content frame.
2. Type `incident.list` in the filter navigator.
3. Star the Incident application. Drag the Open incidents breadcrumb to Favorites.
4. Open History and return to a record without searching.
5. Impersonate **Abel Tuter**. Note missing modules. Unimpersonate.
6. Open System Settings (gear) and toggle compact UI. Confirm it is **only you**.
7. Open user menu and find **Elevate Roles** (do not leave it elevated).

**Pass:** You can impersonate and you know which chrome is personal vs global.

---

## Lab 1 — Users, groups, roles (Domain 2)

1. Create role `u_csa_lab`.
2. Create group **CSA Lab Fulfillers** with that role plus `itil`.
3. Create user `csa.learner` (active, with email). Add to the group — **do not** add roles on the user record.
4. Impersonate `csa.learner`. Confirm Incident list is visible.
5. Remove the user from the group, impersonate again, confirm Incident is gone.

**Pass:** You can explain role inheritance via groups.

---

## Lab 2 — Branding and plugins (Domain 2)

1. Change the instance banner logo or title. Reload.
2. Open Plugins and find one **already active** (do not activate random paid plugins). Open its Guided Setup if present.
3. Open `sys_properties.list` and search `glide.ui`. Read, do not randomly edit production properties. On a PDI, you may inspect homepage deprecation properties.

**Pass:** You can say personalization ≠ banner logo.

---

## Lab 3 — Lists and filters (Domain 3)

1. On Incidents, build filter: Active is true AND Priority is 1.
2. Save the filter for **Group** CSA Lab Fulfillers.
3. Favorite the breadcrumb.
4. Gear-personalize columns (add Category). Confirm green dot. Reset to default.
5. Configure > List Layout on the Default view: add a column, impersonate `csa.learner`, confirm they see it.
6. Right-click a column → Bar Chart. Save a report.

**Pass:** You can distinguish gear vs Configure.

---

## Lab 4 — Forms, templates, save options (Domain 3)

1. Configure Form Design on Incident: add a section **CSA Lab** with a new field `u_lab_flag` (true/false).
2. Create a form template that sets Category and that flag.
3. Open an incident. Use **Save** (stays). Use **Update** (closes).
4. Use **Insert and Stay**. Confirm a **new number**.
5. Configure Related Lists: ensure Task SLAs is visible.

**Pass:** You can recite Submit/Update/Save/Insert.

---

## Lab 5 — Tasks, assignment, VTB, SLA (Domain 3)

1. Open Service Desk > **My Groups Work** as an `itil` user in a group that owns incidents.
2. Create an assignment rule: Category = Software → a software group. Test with a new incident.
3. Create a **Guided** VTB on Incident by State. Drag a card; confirm State changed.
4. Open an SLA definition (OOB Priority 1). Create a P1 incident and inspect `task_sla` related list (start/pause/stop conceptually).

**Pass:** You know My Groups Work and VTB types.

---

## Lab 6 — Notifications (Domain 3)

1. Copy an OOB incident notification into your update set.
2. Set **Who** to your admin email (PDI email may not send — still configure).
3. Identify When / Who / What tabs.
4. Open System Settings > Notifications as Abel Tuter and find subscriptions.

**Pass:** You can name the three notification tabs.

---

## Lab 7 — Knowledge (Domain 4)

1. Create knowledge base **CSA Lab KB**.
2. Add user criteria: Can Read = `snc_internal` or users without role (as PDI allows); Can Contribute = `itil`.
3. Draft an article “How to request a laptop,” publish it.
4. Impersonate Abel Tuter (ESS). Search and open it.
5. Retire the article; confirm it disappears from search.

**Pass:** Unpublished/retired ≠ searchable.

---

## Lab 8 — Catalog + flow (Domain 4)

1. Create catalog item **CSA Lab Laptop**.
2. Add variables: Single Line Text (justification), Select Box (standard/power), Reference (requested_for → User).
3. Create a variable set **Requested For Set** and add it to a second item to prove reuse.
4. In Workflow Studio / Flow Designer, create a flow:
   - Trigger: Service Catalog
   - Ask for Approval from manager
   - Create Catalog Task for an assignment group
5. Attach the flow to the item.
6. Impersonate Abel, order the item.
7. As admin, inspect **REQ → RITM → SCTASK**.
8. Create a **Record Producer** “Report Wi-Fi issue” targeting Incident. Submit; confirm **no REQ**.

**Pass:** You can draw REQ/RITM/SCTASK and contrast a producer.

---

## Lab 9 — Virtual Agent awareness (Domain 4)

1. Open Virtual Agent designer / topics (plugin must be active on PDI).
2. List two OOB topics (e.g. check ticket status, search knowledge).
3. If portal chat is enabled, send “I need a laptop” and see whether it offers catalog/knowledge.

**Pass:** You can explain VA uses knowledge + catalog.

---

## Lab 10 — Data schema (Domain 5)

1. Create table `u_lab_loan` (label Lab Loan).
2. Fields: borrower (ref User), ci (ref Configuration Item), due_date (Date), state (choices).
3. Open Schema Map. Find the reference to `sys_user`.
4. Add a related list of Lab Loans on the User form.
5. Create a dictionary override on a **child** of Task (only if you are comfortable reversing it) **or** read an existing override and write down what it changes.

**Pass:** You can explain extension vs reference.

---

## Lab 11 — ACLs (Domain 5)

1. Elevate `security_admin`.
2. On `u_lab_loan`, create ACLs:
   - Read/write for `u_csa_lab`
   - No access for users with only `itil` if you removed default ACLs carefully (PDI table creator may have already added ACLs — **read them first**)
3. Impersonate `csa.learner` (has `u_csa_lab`) — can open list.
4. Impersonate a user with only `itil` — confirm deny or limited.
5. Add a **field ACL** hiding `due_date` from `itil`. Confirm the rest of the form loads.

**Pass:** You can state table-then-field evaluation.

---

## Lab 12 — Import Sets (Domain 5)

1. Create a CSV:

```text
email,first_name,last_name
loan.a@example.com,Loan,Alpha
loan.b@example.com,Loan,Beta
```

2. Load Data → create import set table.
3. Transform Map to `sys_user`. Map email, first_name, last_name.
4. Coalesce on **email**.
5. Transform. Confirm two users.
6. Change Alpha’s last name in CSV, reload, transform again. Confirm **update**, not a third user.

**Pass:** You can define coalesce in one sentence.

---

## Lab 13 — CMDB / CSDM (Domain 5)

1. Open CI Class Manager. Find `cmdb_ci_computer` in the tree.
2. Open any server CI. Related list **CI Relationships**. Open Dependency View if available.
3. Find a **Business Application** vs an **Application Service** (labels vary). Write which CSDM domain each belongs to.
4. Open Configuration > Relationships > Suggested Relationships and read one pair.
5. Locate MID Server module (may be empty on PDI). Write why Discovery needs it.

**Pass:** You can name `cmdb_ci` and `cmdb_rel_ci`.

---

## Lab 14 — Security Center and shared responsibility (Domain 5)

1. Open **Security Center** (navigator search).
2. Read one finding or check. Write whether fixing it is **your** job or ServiceNow’s.
3. Write five bullets: ServiceNow-owned vs customer-owned.

**Pass:** You do not confuse Security Center with Security Incident Response.

---

## Lab 15 — UI Policy, Data Policy, Business Rule, Client Script (Domain 6)

On `u_lab_loan` or Incident (prefer custom table):

1. **UI Policy:** if state is Overdue, make due_date mandatory and visible.
2. **Client Script onChange:** when borrower changes, `g_form.showFieldMsg` a note.
3. **before Business Rule:** if due_date is in the past, set state to Overdue.
4. **Data Policy:** borrower mandatory. Try an import without borrower — should fail/enforce.
5. Optional **display BR:** `g_scratchpad.hello = 'world';` and onLoad Client Script `alert` or field message.

**Pass:** You can pick the right tool from Chapter 07’s decision tree.

---

## Lab 16 — Update sets (Domain 6)

1. Confirm all lab config is in `CSA Workbook Labs`.
2. Move any Default-set captures into your set.
3. Mark Complete. Export to XML. Open the XML and find a `sys_script` or `sys_ui_policy` customer update.
4. (Optional) Retrieve into another PDI / the same instance as retrieved — Preview. Read any preview errors.

**Pass:** You never plan to migrate Default.

---

## Lab 17 — Mini capstone

Build an employee path:

1. Knowledge article: how to request a loaner laptop.  
2. Catalog item with flow (approval + SCTASK).  
3. Record producer for broken loaner (Incident) with CI field.  
4. Notification to assignment group on SCTASK insert.  
5. Report + dashboard of open RITMs.  
6. ACL so only CSA Lab Fulfillers see `u_lab_loan`.  
7. Impersonate end user **and** fulfiller.

If you can demo this in 15 minutes, you are operationally ready for CSA scenarios.

---

Next: [Chapter 09 — Practice questions](09-practice-questions-and-explanations.md).
