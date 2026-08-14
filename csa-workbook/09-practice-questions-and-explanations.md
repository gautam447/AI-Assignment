# Chapter 09 — Practice Questions and Explanations

Original practice items written to the **January 2026 blueprint**. They are **not** live exam questions.

How to use:

- Timed set: questions 1–30 in 45 minutes, then 31–50 in 45 minutes.
- Multi-select: no partial credit in real life — grade the same way.
- After each miss, read the matching domain chapter, then redo the lab.

---

## Questions

### 1. (Domain 1)

ServiceNow’s cloud architecture is best described as:

- A. Multi-tenant shared database for all customers  
- B. Multi-instance, with dedicated application and database processes per customer  
- C. On-premises only aPaaS  
- D. A single global production instance  

### 2. (Domain 1)

Which application is available to users even when they have **no** fulfiller role?

- A. Change  
- B. Incident  
- C. Facilities  
- D. Self-Service  

### 3. (Domain 1)

What are the three primary regions of the Next Experience / platform UI?

- A. Banner frame, application navigator, content frame  
- B. Homepage, CMDB, catalog  
- C. MID Server, instance, HI portal  
- D. List, form, related list only  

### 4. (Domain 1)

An admin wants to test catalog visibility as an employee. The correct tool is:

- A. Elevate roles  
- B. Impersonate User  
- C. Delegate Development  
- D. Clone the instance  

### 5. (Domain 2)

Best practice for assigning permissions:

- A. Assign roles directly to every user  
- B. Assign roles to groups; add users to groups  
- C. Put ACLs on the user record  
- D. Give everyone `admin` and lock with UI Policies  

### 6. (Domain 2)

Changing **your** list columns with the gear icon is:

- A. A customization captured in an update set  
- B. A personalization for your user  
- C. A Data Policy  
- D. An ACL  

### 7. (Domain 2)

Which role is typically elevated **for the current session** before creating ACL records on a High Security instance?

- A. `itil`  
- B. `approver_user`  
- C. `security_admin`  
- D. `snc_internal`  

### 8. (Domain 2)

Homepages are being retired in favor of:

- A. Visual Task Boards  
- B. Update sets  
- C. Dashboards / Platform Analytics  
- D. MID Servers  

### 9. (Domain 3)

A filter condition is composed of:

- A. Role, group, ACL  
- B. Field, operator, value  
- C. Trigger, action, spoke  
- D. REQ, RITM, SCTASK  

### 10. (Domain 3)

Which save action creates a **new** record from the current form values and **leaves the form open**?

- A. Submit  
- B. Update  
- C. Insert  
- D. Insert and Stay  

### 11. (Domain 3)

Which module shows tasks assigned to the user’s **groups** but not yet to a person?

- A. My Teams Work  
- B. My Groups Work  
- C. My Groups Tasks  
- D. My Teams Tasks  

### 12. (Domain 3) — select 2

Which statements about Visual Task Boards are true?

- A. Guided boards use a field such as State for lanes  
- B. Freeform boards always update Incident State when you drag a card  
- C. Owners can manage board structure and members  
- D. VTBs replace the need for ACLs  

### 13. (Domain 3)

The recommended way to give colleagues access to a report is:

- A. Publish to the internet  
- B. Share with users, groups, or roles  
- C. Email the sys_id  
- D. Put the report in an update set only  

### 14. (Domain 3)

Notifications are typically triggered by:

- A. Schema maps  
- B. System events (plus conditions)  
- C. Coalesce fields  
- D. MID Server heartbeats only  

### 15. (Domain 3)

Additional comments on a task are:

- A. Visible to the caller / customer  
- B. Fulfiller-only  
- C. The same as work notes  
- D. Stored only in the CMDB  

### 16. (Domain 4)

A published knowledge article is still invisible to an employee. What should you check first among KM-specific controls?

- A. Transform map coalesce  
- B. User criteria on the knowledge base / article  
- C. Report Publisher role  
- D. Default update set  

### 17. (Domain 4)

A user checks out two catalog items in one cart. What is created?

- A. Two REQs and no RITMs  
- B. One REQ and two RITMs (each may have SCTASKs)  
- C. Only two Incidents  
- D. One SCTASK and no request  

### 18. (Domain 4)

Multiple Choice, Single Line Text, and Select Box are:

- A. Order Guides  
- B. Request Types  
- C. Variable Types  
- D. Related Lists  

### 19. (Domain 4)

A Record Producer is different from a Catalog Item because it:

- A. Always creates REQ/RITM/SCTASK  
- B. Creates a record on a **target table** such as Incident  
- C. Cannot use variables  
- D. Is only for CMDB inserts  

### 20. (Domain 4)

An order guide is used to:

- A. Merge update sets  
- B. Order multiple related catalog items in one request  
- C. Discover servers  
- D. Elevate roles  

### 21. (Domain 4)

Which automation can **wait** days for an approval without blocking a database transaction?

- A. before Business Rule  
- B. UI Policy  
- C. Flow in Workflow Studio / Flow Designer  
- D. Dictionary override  

### 22. (Domain 4)

Virtual Agent commonly uses which content sources?

- A. Only MID Server logs  
- B. Knowledge, Service Catalog, and system records  
- C. Only Security Center findings  
- D. Only update set XML  

### 23. (Domain 4)

Workflow Studio includes:

- A. Flow Designer and Process Automation Designer / playbooks  
- B. Only Import Sets  
- C. Only CI Class Manager  
- D. Only Now Support HI cases  

### 24. (Domain 5)

The unique 32-character identifier of a record is:

- A. Number  
- B. sys_id  
- C. Display value  
- D. Tag  

### 25. (Domain 5)

Incident extends Task. This relationship is:

- A. Many-to-many  
- B. Extension (parent–child)  
- C. A transform map  
- D. An order guide  

### 26. (Domain 5)

A reference field on Incident pointing to User is an example of:

- A. Extension  
- B. One-to-many (many incidents, one user)  
- C. Domain separation  
- D. A coalesce key only  

### 27. (Domain 5)

Which tool visualizes how tables relate (extend and reference)?

- A. Visual Task Board  
- B. Schema Map  
- C. Banner frame  
- D. Virtual Agent  

### 28. (Domain 5)

A dictionary override is used to:

- A. Change a parent field’s behavior on a **child** table only  
- B. Clone production  
- C. Publish a report  
- D. Create a MID Server  

### 29. (Domain 5)

For a **single** ACL, roles, condition, and script are combined with:

- A. OR (any one is enough)  
- B. AND (all configured parts must pass)  
- C. XOR  
- D. They are ignored if the user is `itil`  

### 30. (Domain 5)

A user passes table-level **read** on Incident but fails `incident.work_notes` read. What happens?

- A. The whole incident list is empty  
- B. They can open the record but not read that field  
- C. The instance logs them out  
- D. UI Policy grants the field anyway  

### 31. (Domain 5)

If no ACL grants access and High Security default-deny applies, the result is:

- A. Access allowed  
- B. Access denied  
- C. Access allowed to `snc_external` only  
- D. Access allowed on weekends  

### 32. (Domain 5)

A transform map is:

- A. A map of Import Set fields to fields on an existing target table  
- B. A map of incidents to knowledge  
- C. A CSDM domain  
- D. A type of VTB  

### 33. (Domain 5)

You import users twice. You want the second load to **update** existing people. You should:

- A. Remove all ACLs  
- B. Coalesce on a unique key such as email  
- C. Use Submit instead of Update  
- D. Put the CSV in an update set  

### 34. (Domain 5) — select 2

Which are CMDB core tables?

- A. `cmdb_ci`  
- B. `cmdb_rel_ci`  
- C. `sc_request`  
- D. `sysevent_email_action`  

### 35. (Domain 5)

CSDM is:

- A. A mandatory paid plugin that replaces ITSM  
- B. A common data model and best-practice vocabulary for platform data, especially CMDB services  
- C. The name of the MID Server service  
- D. A type of inbound email action  

### 36. (Domain 5)

Discovery typically needs a **MID Server** because:

- A. Browsers cannot render lists  
- B. Probes must run **inside** the customer network and talk out to the instance  
- C. Update sets cannot be previewed otherwise  
- D. Virtual Agent cannot search knowledge otherwise  

### 37. (Domain 5)

Who is responsible for designing ACLs that protect HR data in the instance?

- A. ServiceNow data center staff only  
- B. The customer / instance administrators  
- C. Pearson VUE  
- D. The MID Server vendor only  

### 38. (Domain 5)

Security Center is used to:

- A. Replace the CMDB  
- B. Review **instance security posture** and hardening findings  
- C. Design catalog variables  
- D. Merge update sets  

### 39. (Domain 6)

You need Description mandatory **only when Priority is 1**, on the form. Best tool:

- A. ACL execute  
- B. UI Policy  
- C. Clone  
- D. Order guide  

### 40. (Domain 6)

You need Description mandatory even when records are loaded by **Import Set**. Best tool:

- A. UI Policy only  
- B. Data Policy (or dictionary mandatory)  
- C. Client Script onLoad  
- D. VTB lane  

### 41. (Domain 6)

Which Business Rule timing is used to pass data to the client via `g_scratchpad`?

- A. async  
- B. display  
- C. after only  
- D. onCellEdit  

### 42. (Domain 6)

Which statement is true?

- A. Business Rules can wait two days for approval  
- B. Flows can wait for approval; Business Rules should not  
- C. UI Policies run on the server during import  
- D. Client Scripts use `GlideRecord` as best practice  

### 43. (Domain 6)

The scripting language on the Now Platform is:

- A. Java  
- B. AngularJS only  
- C. JavaScript  
- D. Jelly  

### 44. (Domain 6)

Update sets are designed to move:

- A. Incident data from last year  
- B. Configuration / customizations between instances  
- C. MID Server binaries  
- D. Physical data center assignments  

### 45. (Domain 6)

Correct promotion order for an update set on the target instance:

- A. Commit, then preview  
- B. Preview, resolve errors, then commit  
- C. Delete Default, then clone  
- D. Elevate, then impersonate  

### 46. (Domain 6)

You should develop in:

- A. The Default update set and migrate it weekly  
- B. A named update set that you complete and preview  
- C. Production, then clone back  
- D. Security Center  

### 47. (Domain 3/6)

`g_form` and `g_user` are available in:

- A. before Business Rules  
- B. Client Scripts and client UI Actions  
- C. Transform Map onStart only  
- D. MID Server probes  

### 48. (Domain 5/6)

UI Policy hiding a field is:

- A. Equivalent to a field ACL  
- B. Not a security control  
- C. The only way to stop web services from writing the field  
- D. Captured only as data, not in update sets  

### 49. (Domain 4/5)

The request catalog (Service Catalog) is:

- A. A CSDM Foundation CMDB relationship table  
- B. Not a CMDB table; it is the ordering experience that may **reference** services  
- C. The same as `cmdb_rel_ci`  
- D. Only available to `security_admin`  

### 50. (Domain 2/5)

A user cannot see the Incident **module** but can open an incident URL. Likely cause:

- A. Table ACL deny  
- B. Application or module role missing  
- C. Coalesce failed  
- D. Knowledge article retired  

---

## Answer key

| # | Answer | Why |
| ---: | --- | --- |
| 1 | **B** | Multi-instance architecture, not a shared multi-tenant DB. |
| 2 | **D** | Official sample: Self-Service is for all users. |
| 3 | **A** | Banner, navigator, content frame. |
| 4 | **B** | Impersonate tests as that user. Elevate is for `security_admin`. |
| 5 | **B** | Roles on groups; users in groups. |
| 6 | **B** | Gear = personalization. |
| 7 | **C** | Elevate `security_admin` to edit ACLs. |
| 8 | **C** | Dashboards replace homepages. |
| 9 | **B** | Field + operator + value; shown as breadcrumbs. |
| 10 | **D** | Insert and Stay = new record, form stays. |
| 11 | **B** | Official sample: My Groups Work. |
| 12 | **A, C** | Guided uses a field; owners manage the board. Freeform does not reliably update Incident State. VTBs are not security. |
| 13 | **B** | Share is the recommended action. |
| 14 | **B** | Events + When/Who/What. |
| 15 | **A** | Comments = customer; work notes = internal. |
| 16 | **B** | Can Read user criteria (and published state). |
| 17 | **B** | One REQ per checkout; RITM per item. |
| 18 | **C** | Official sample: variable types. |
| 19 | **B** | Producer writes the target table; no REQ chain by default. |
| 20 | **B** | Multiple items, one request. |
| 21 | **C** | Flows wait; BRs do not. |
| 22 | **B** | Knowledge, catalog, records; plus chat channels. |
| 23 | **A** | Workflow Studio umbrella. |
| 24 | **B** | sys_id is 32-char GUID. |
| 25 | **B** | Table extension. |
| 26 | **B** | Reference = one-to-many from the parent’s view. |
| 27 | **B** | Schema Map. |
| 28 | **A** | Child-only field behavior. |
| 29 | **B** | Parts of one ACL are AND. |
| 30 | **B** | Table pass, field fail → field hidden/blank. |
| 31 | **B** | Default deny. |
| 32 | **A** | Official sample definition. |
| 33 | **B** | Coalesce updates matches. |
| 34 | **A, B** | CI and relationship tables. `sc_request` is catalog. |
| 35 | **B** | Model/vocabulary, not a random plugin replacement of ITSM. |
| 36 | **B** | MID = in-network worker for Discovery/integrations. |
| 37 | **B** | Shared Responsibility: customer owns configuration/ACLs. |
| 38 | **B** | Instance security posture dashboard. |
| 39 | **B** | Conditional form behavior = UI Policy. |
| 40 | **B** | Server-side enforcement for imports. |
| 41 | **B** | display BR fills `g_scratchpad`. |
| 42 | **B** | Wait = Flow. |
| 43 | **C** | Official sample: JavaScript. |
| 44 | **B** | Config, not operational data. |
| 45 | **B** | Preview then commit. |
| 46 | **B** | Named sets; never migrate Default as process. |
| 47 | **B** | Client APIs. |
| 48 | **B** | UX only; use ACL/Data Policy for real control. |
| 49 | **B** | Catalog is not a CMDB table. |
| 50 | **B** | Module visibility ≠ table ACL. If the URL works, table read passed. |

---

## Scoring guide

| Score | Meaning |
| ---: | --- |
| 45–50 | Sit when official training is also complete |
| 38–44 | Review missed domains; redo those labs |
| 30–37 | Domain 5 + comparisons chapter for a full pass |
| < 30 | Restart from Chapters 00–02 with a PDI, then 05 |

If you miss **multi-select** items, practice covering the stem and listing answers before looking at options.

Next: [Chapter 10 — Glossary, tables, and exam day](10-glossary-tables-and-exam-day.md).
