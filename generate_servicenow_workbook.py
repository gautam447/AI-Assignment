#!/usr/bin/env python3
"""Generate a comprehensive ServiceNow ITSM Workbook PDF."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = "/opt/cursor/artifacts/ServiceNow_ITSM_Complete_Workbook.pdf"
WORKSPACE_COPY = "/workspace/ServiceNow_ITSM_Complete_Workbook.pdf"


class WorkbookPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "ServiceNow ITSM Complete Workbook", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, num, title):
        self.add_page()
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(0, 82, 147)
        self.cell(0, 12, f"Chapter {num}: {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)
        self.set_draw_color(0, 82, 147)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def subsection_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, f"- {text}")
        self.ln(1)

    def note_box(self, text):
        self.set_fill_color(235, 245, 255)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 82, 147)
        self.cell(0, 7, "NOTE", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(3)

    def tip_box(self, text):
        self.set_fill_color(240, 255, 240)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 120, 60)
        self.cell(0, 7, "TIP", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(3)

    def exercise_box(self, title, items):
        self.set_fill_color(255, 248, 230)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(180, 100, 0)
        self.cell(0, 7, f"EXERCISE: {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        for i, item in enumerate(items, 1):
            self.set_x(self.l_margin)
            self.multi_cell(0, 5, f"{i}. {item}", fill=True)
        self.ln(3)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 82, 147)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True)
        self.ln()

    def table_row(self, cols, widths, fill=False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        if fill:
            self.set_fill_color(245, 245, 245)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, fill=fill)
        self.ln()


def build_workbook():
    pdf = WorkbookPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 15, 15)

    # ---- COVER PAGE ----
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 82, 147)
    pdf.cell(0, 15, "ServiceNow ITSM", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "Complete Workbook", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Comprehensive Guide to IT Service Management", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, "on the ServiceNow Platform", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Covers: Incident | Problem | Change | Request | Knowledge", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "CMDB | SLA | Catalog | Discovery | ITOM | Administration", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "Workflows | Integrations | Reporting | Best Practices", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 7, "Edition: 2026 | For Training & Certification Preparation", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ---- TABLE OF CONTENTS ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 82, 147)
    pdf.cell(0, 12, "Table of Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    toc = [
        ("Chapter 1", "Introduction to ServiceNow & ITSM"),
        ("Chapter 2", "Platform Architecture & Navigation"),
        ("Chapter 3", "Incident Management"),
        ("Chapter 4", "Problem Management"),
        ("Chapter 5", "Change Management"),
        ("Chapter 6", "Service Catalog & Request Management"),
        ("Chapter 7", "Knowledge Management"),
        ("Chapter 8", "Service Level Management (SLA)"),
        ("Chapter 9", "Configuration Management (CMDB)"),
        ("Chapter 10", "Asset Management & Hardware Asset Management"),
        ("Chapter 11", "Discovery & Service Mapping"),
        ("Chapter 12", "Release Management"),
        ("Chapter 13", "Event Management & IT Operations"),
        ("Chapter 14", "Administration: Users, Groups, Roles & ACLs"),
        ("Chapter 15", "Workflows, Flow Designer & Automation"),
        ("Chapter 16", "Reporting, Dashboards & Performance Analytics"),
        ("Chapter 17", "Integrations & Data Import"),
        ("Chapter 18", "Virtual Agent, Now Assist & Self-Service"),
        ("Chapter 19", "ITIL Alignment & Best Practices"),
        ("Chapter 20", "Certification Prep & Practice Questions"),
        ("Appendix A", "Glossary of Key Terms"),
        ("Appendix B", "Useful Tables & Reference Data"),
    ]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)
    for ch, title in toc:
        pdf.cell(35, 7, ch)
        pdf.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ========== CHAPTER 1 ==========
    pdf.chapter_title(1, "Introduction to ServiceNow & ITSM")
    pdf.section_title("1.1 What is ServiceNow?")
    pdf.body_text(
        "ServiceNow is a cloud-based platform that provides enterprise service management capabilities. "
        "Originally focused on IT Service Management (ITSM), it has evolved into a multi-domain platform "
        "supporting HR, Customer Service, Security Operations, and custom applications. ServiceNow follows "
        "a single-instance, multi-tenant architecture delivered as Software-as-a-Service (SaaS)."
    )
    pdf.subsection_title("Key Platform Characteristics")
    pdf.bullet("Single data model: All applications share one underlying database and architecture.")
    pdf.bullet("Configuration over customization: Prefer out-of-box features, configure via UI before coding.")
    pdf.bullet("Now Platform: The foundation layer providing tables, forms, lists, workflows, and security.")
    pdf.bullet("Scoped applications: ITSM, ITOM, CSM, HRSD are packaged as separate scoped apps.")
    pdf.bullet("Instance-based: Each customer gets a dedicated logical instance (e.g., dev, test, prod).")

    pdf.section_title("1.2 What is ITSM?")
    pdf.body_text(
        "IT Service Management (ITSM) refers to the activities, policies, and processes an organization "
        "uses to design, deliver, manage, and improve IT services aligned with business needs. ITSM is "
        "most commonly aligned with ITIL (Information Technology Infrastructure Library) best practices."
    )
    pdf.subsection_title("Core ITSM Processes in ServiceNow")
    widths = [55, 130]
    pdf.table_header(["Process", "Purpose"], widths)
    rows = [
        ("Incident Management", "Restore normal service operation ASAP"),
        ("Problem Management", "Identify and eliminate root causes of incidents"),
        ("Change Management", "Control and track changes to IT infrastructure"),
        ("Request Management", "Fulfill standard user service requests"),
        ("Knowledge Management", "Capture, share, and reuse organizational knowledge"),
        ("Service Level Management", "Define, monitor, and report on service quality"),
        ("Configuration Management", "Maintain accurate CI and relationship data (CMDB)"),
    ]
    for i, row in enumerate(rows):
        pdf.table_row(row, widths, fill=(i % 2 == 0))

    pdf.section_title("1.3 ITIL 4 vs ServiceNow")
    pdf.body_text(
        "ITIL 4 introduces the Service Value System (SVS) and four dimensions of service management. "
        "ServiceNow ITSM applications map closely to ITIL practices but use ServiceNow-specific terminology "
        "and workflows. Understanding both frameworks helps during implementation and certification exams."
    )
    pdf.note_box(
        "ServiceNow certifications (CSA, CIS-ITSM, CAD) test practical platform knowledge. "
        "This workbook aligns content with common exam topics while focusing on real-world application."
    )
    pdf.exercise_box("Chapter 1 Review", [
        "Log into a ServiceNow Personal Developer Instance (PDI) at developer.servicenow.com.",
        "Navigate the Application Navigator and identify ITSM applications.",
        "List five ITSM processes and describe the purpose of each in your own words.",
        "Explore the ServiceNow Store and identify three ITSM-related scoped applications.",
    ])

    # ========== CHAPTER 2 ==========
    pdf.chapter_title(2, "Platform Architecture & Navigation")
    pdf.section_title("2.1 Instance Architecture")
    pdf.body_text(
        "A ServiceNow instance consists of databases, application servers, and a web tier. "
        "Data is stored in tables (similar to database tables), and the platform provides "
        "a metadata-driven UI layer that renders forms, lists, and related lists dynamically."
    )
    pdf.subsection_title("Core Platform Concepts")
    pdf.bullet("Tables: Store records (e.g., incident, change_request, cmdb_ci). Each table extends a parent table.")
    pdf.bullet("Records: Individual rows in a table, identified by sys_id (32-character GUID).")
    pdf.bullet("Fields/Columns: Attributes on a table (short text, reference, choice list, etc.).")
    pdf.bullet("Forms: UI for viewing/editing a single record.")
    pdf.bullet("Lists: Tabular views of multiple records with filtering and sorting.")
    pdf.bullet("Related Lists: Child records linked via reference fields on the parent form.")
    pdf.bullet("Modules: Navigation entries in the Application Navigator pointing to lists, forms, or URLs.")
    pdf.bullet("Applications: Logical groupings of modules (e.g., Incident Management under ITSM).")

    pdf.section_title("2.2 Table Inheritance & the Task Table")
    pdf.body_text(
        "ServiceNow uses table extension (class hierarchy). The task table is the base for most "
        "ITSM process tables. When you extend a table, child tables inherit all fields from parent tables."
    )
    pdf.subsection_title("Key Task Table Extensions")
    pdf.table_header(["Table", "Label", "Key Additional Fields"], [45, 45, 100])
    task_rows = [
        ("incident", "Incident", "caller_id, category, subcategory, impact, urgency"),
        ("problem", "Problem", "problem_state, known_error, workaround"),
        ("change_request", "Change Request", "type, risk, approval, cab_date"),
        ("sc_task", "Catalog Task", "request_item, sc_catalog"),
        ("sc_req_item", "Requested Item", "cat_item, quantity, price"),
    ]
    for i, row in enumerate(task_rows):
        pdf.table_row(row, [45, 45, 100], fill=(i % 2 == 0))

    pdf.section_title("2.3 Navigation Essentials")
    pdf.bullet("Application Navigator (left panel): Browse applications and modules.")
    pdf.bullet("Filter Navigator: Type to search modules, records, or favorites.")
    pdf.bullet("Favorites: Star modules for quick access.")
    pdf.bullet("History: Recently viewed records (click clock icon).")
    pdf.bullet("UI16 vs Next Experience: Classic UI vs modern Workspace/Next Experience UI.")
    pdf.bullet("List controls: Filter, group, sort, export, and personalize list layouts.")

    pdf.section_title("2.4 Personal Developer Instance (PDI)")
    pdf.body_text(
        "ServiceNow provides free Personal Developer Instances for learning. PDIs include ITSM, "
        "ITOM, and other applications. Instances hibernate after inactivity and reset periodically. "
        "Register at developer.servicenow.com to obtain yours."
    )
    pdf.tip_box(
        "Always practice in a PDI or sub-production instance. Never experiment with workflows, "
        "scripts, or ACL changes directly in production without change control."
    )

    # ========== CHAPTER 3 ==========
    pdf.chapter_title(3, "Incident Management")
    pdf.section_title("3.1 Purpose & Scope")
    pdf.body_text(
        "Incident Management restores normal service operation as quickly as possible and minimizes "
        "business impact. An incident is an unplanned interruption or reduction in quality of an IT service."
    )
    pdf.subsection_title("Incident Lifecycle States")
    pdf.table_header(["State", "Description", "Typical Actions"], [30, 70, 90])
    inc_states = [
        ("1 - New", "Incident logged, not yet assigned", "Auto-assignment rules may fire"),
        ("2 - In Progress", "Actively being worked", "Diagnosis, communication, escalation"),
        ("3 - On Hold", "Waiting on caller or vendor", "Set on-hold reason, set follow-up"),
        ("6 - Resolved", "Fix applied, pending caller confirmation", "Resolution code & notes required"),
        ("7 - Closed", "Confirmed resolved or auto-closed", "Survey may be sent, SLA stops"),
        ("8 - Canceled", "Duplicate or invalid", "Link to parent incident"),
    ]
    for i, row in enumerate(inc_states):
        pdf.table_row(row, [30, 70, 90], fill=(i % 2 == 0))

    pdf.section_title("3.2 Priority Matrix")
    pdf.body_text(
        "Priority is typically calculated from Impact and Urgency using a lookup matrix "
        "(data lookup: dl_u_priority). Impact = effect on business; Urgency = time sensitivity."
    )
    pdf.subsection_title("Impact & Urgency Values")
    pdf.bullet("1 - High: Critical business function affected / immediate action needed")
    pdf.bullet("2 - Medium: Degraded service / action needed soon")
    pdf.bullet("3 - Low: Minor inconvenience / can wait")

    pdf.section_title("3.3 Assignment & Routing")
    pdf.body_text(
        "Incidents are assigned to assignment groups and individual assignees. Assignment rules, "
        "routing configurations, and predictive intelligence can automate routing based on category, "
        "CI, location, or language."
    )
    pdf.subsection_title("Key Assignment Concepts")
    pdf.bullet("Assignment Group: Team responsible (e.g., Network, Service Desk Tier 1).")
    pdf.bullet("Assignee: Individual within the group working the ticket.")
    pdf.bullet("Skills-based routing: Match incidents to agents with required skills.")
    pdf.bullet("On-call schedules: Escalate to on-call engineer based on rota.")
    pdf.bullet("Work queues: Agent Workspace unified queue for incidents, cases, etc.")

    pdf.section_title("3.4 Major Incident Management")
    pdf.body_text(
        "Major Incident Management handles high-impact incidents affecting many users or critical services. "
        "ServiceNow provides Major Incident Management application with dedicated workflows, "
        "war room coordination, stakeholder communication, and post-incident review."
    )
    pdf.subsection_title("Major Incident Process Steps")
    pdf.bullet("Detection and declaration of major incident")
    pdf.bullet("Formation of major incident team (MIT)")
    pdf.bullet("Communication to stakeholders via email, SMS, status page")
    pdf.bullet("Parallel technical investigation and workaround")
    pdf.bullet("Post-incident review (PIR) and problem record creation")

    pdf.section_title("3.5 Incident Relationships")
    pdf.bullet("Parent/Child: Group related incidents under one parent")
    pdf.bullet("Problem: Link to underlying problem record")
    pdf.bullet("Change: Link to change that caused or fixes incident")
    pdf.bullet("Configuration Item: Affected CI in CMDB")
    pdf.bullet("Knowledge Article: Attach relevant KB articles")

    pdf.section_title("3.6 Agent Workspace for Incidents")
    pdf.body_text(
        "Agent Workspace (ITSM Workspace) provides a modern, unified interface for agents. "
        "Features include split-view record layout, related records panel, chat/collaboration, "
        "Knowledge sidebar, and embedded analytics."
    )
    pdf.exercise_box("Incident Management Lab", [
        "Create an incident as an end user via the Service Portal or Employee Center.",
        "As an agent, pick up the incident from the queue and set it to In Progress.",
        "Set Impact=1, Urgency=1 and verify Priority becomes 1 (Critical).",
        "Place the incident On Hold with reason 'Awaiting Caller'.",
        "Resolve with resolution code 'Solved (Permanently)' and close the incident.",
        "Create a second incident and relate it as a child of the first.",
    ])

    # ========== CHAPTER 4 ==========
    pdf.chapter_title(4, "Problem Management")
    pdf.section_title("4.1 Purpose & Scope")
    pdf.body_text(
        "Problem Management identifies the root cause of incidents and prevents recurrence. "
        "Unlike incidents (restore service fast), problems focus on diagnosis and permanent resolution. "
        "Problems may be reactive (from incidents) or proactive (trend analysis)."
    )
    pdf.subsection_title("Problem vs Incident")
    pdf.table_header(["Aspect", "Incident", "Problem"], [40, 75, 75])
    comp_rows = [
        ("Goal", "Restore service quickly", "Find and eliminate root cause"),
        ("User impact", "Service is broken now", "May have no immediate impact"),
        ("Timing", "Immediate response", "Investigation over time"),
        ("Output", "Resolution/workaround", "Root cause, permanent fix, known error"),
    ]
    for i, row in enumerate(comp_rows):
        pdf.table_row(row, [40, 75, 75], fill=(i % 2 == 0))

    pdf.section_title("4.2 Problem Lifecycle")
    pdf.bullet("New: Problem identified and logged")
    pdf.bullet("Assess: Initial assessment and categorization")
    pdf.bullet("Root Cause Analysis: Investigation using techniques (5 Whys, Kepner-Tregoe, etc.)")
    pdf.bullet("Fix in Progress: Permanent fix being developed/tested")
    pdf.bullet("Resolved: Root cause addressed, fix deployed")
    pdf.bullet("Closed: Verified no recurrence")

    pdf.section_title("4.3 Known Error Database (KEDB)")
    pdf.body_text(
        "When root cause is identified but permanent fix is not yet available, the problem becomes "
        "a Known Error with a documented workaround. Known errors link to knowledge articles and "
        "help agents resolve related incidents faster."
    )
    pdf.section_title("4.4 Problem Tasks")
    pdf.body_text(
        "Problem tasks break investigation into trackable work items (e.g., gather logs, test hypothesis). "
        "They follow the task table model and can be assigned to different teams."
    )
    pdf.exercise_box("Problem Management Lab", [
        "Create a problem from a resolved incident using the 'Create Problem' related link.",
        "Document root cause analysis notes in the problem record.",
        "Mark the problem as a Known Error with a workaround description.",
        "Create a problem task assigned to the Network team.",
        "Link multiple incidents to the same problem and verify the relationship.",
    ])

    # ========== CHAPTER 5 ==========
    pdf.chapter_title(5, "Change Management")
    pdf.section_title("5.1 Purpose & Scope")
    pdf.body_text(
        "Change Management ensures standardized methods and procedures for efficient handling of "
        "all changes to IT infrastructure. It balances the need for change with the risk of disruption."
    )
    pdf.subsection_title("Change Types")
    pdf.table_header(["Type", "Risk", "Approval", "Example"], [35, 25, 50, 80])
    chg_types = [
        ("Standard", "Pre-authorized, low risk", "Pre-approved template", "Password reset procedure"),
        ("Normal", "Variable risk", "CAB or peer approval", "Server OS patch"),
        ("Emergency", "High urgency", "ECAB post-approval", "Critical security hotfix"),
    ]
    for i, row in enumerate(chg_types):
        pdf.table_row(row, [35, 25, 50, 80], fill=(i % 2 == 0))

    pdf.section_title("5.2 Change Lifecycle States")
    pdf.bullet("-5 New: Change request created")
    pdf.bullet("-4 Assess: Risk and impact assessment")
    pdf.bullet("-3 Authorize: Awaiting approval")
    pdf.bullet("-2 Scheduled: Approved, scheduled for implementation")
    pdf.bullet("-1 Implement: Change being deployed")
    pdf.bullet("0 Review: Post-implementation review")
    pdf.bullet("3 Closed: Change complete")
    pdf.bullet("4 Canceled: Change canceled")

    pdf.section_title("5.3 Risk Assessment")
    pdf.body_text(
        "Change Risk is calculated from Risk Impact and Risk Likelihood. The Change Management "
        "Risk Assessment module provides questionnaires to score changes objectively. "
        "High-risk changes require additional scrutiny and CAB review."
    )
    pdf.section_title("5.4 Change Advisory Board (CAB)")
    pdf.body_text(
        "The CAB reviews normal changes, assesses risk, and approves or rejects them. "
        "Emergency CAB (ECAB) handles urgent changes. CAB meetings can be managed in ServiceNow "
        "with agenda items, attendees, and decision tracking."
    )
    pdf.subsection_title("Change Models (Modern Approach)")
    pdf.body_text(
        "Change Models replace legacy workflows with Flow Designer-based models. Each model defines "
        "states, transitions, approvals, and automations. Out-of-box models include Normal, Standard, "
        "Emergency, and templates for cloud and application changes."
    )
    pdf.section_title("5.5 Change Tasks")
    pdf.bullet("Implementation tasks: Steps to deploy the change")
    pdf.bullet("Backout tasks: Rollback plan if change fails")
    pdf.bullet("Test tasks: Validation after implementation")
    pdf.bullet("Each task can have different assignment groups and schedules")

    pdf.exercise_box("Change Management Lab", [
        "Create a Normal change request with planned start/end dates.",
        "Complete the risk assessment questionnaire and observe risk calculation.",
        "Submit for approval and approve as CAB manager.",
        "Move through Implement and Review states to Closed.",
        "Create a Standard change from a pre-approved template.",
        "Create an Emergency change and note the expedited workflow.",
    ])

    # ========== CHAPTER 6 ==========
    pdf.chapter_title(6, "Service Catalog & Request Management")
    pdf.section_title("6.1 Service Catalog Overview")
    pdf.body_text(
        "The Service Catalog provides a structured way for users to request IT services, hardware, "
        "software, and access. Catalog items are organized into categories and can include "
        "multi-step order guides, variable forms, and approval workflows."
    )
    pdf.subsection_title("Key Catalog Tables")
    pdf.table_header(["Table", "Purpose"], [55, 135])
    cat_tables = [
        ("sc_catalog", "Catalog container (e.g., Service Catalog, HR Catalog)"),
        ("sc_category", "Grouping of catalog items"),
        ("sc_cat_item", "Individual catalog item / service offering"),
        ("sc_cat_item_guide", "Order guide (bundle of items)"),
        ("item_option_new", "Variables/questions on catalog items"),
        ("sc_request", "Header record for a user's request"),
        ("sc_req_item", "Individual line item (RITM) within a request"),
        ("sc_task", "Fulfillment tasks for requested items"),
    ]
    for i, row in enumerate(cat_tables):
        pdf.table_row(row, [55, 135], fill=(i % 2 == 0))

    pdf.section_title("6.2 Catalog Item Design")
    pdf.subsection_title("Variables (Item Options)")
    pdf.bullet("Single Line Text, Multi Line Text, Select Box, CheckBox, Date, Reference, etc.")
    pdf.bullet("Variable sets: Reusable groups of variables across items")
    pdf.bullet("Catalog client scripts: Dynamic behavior on the catalog form")
    pdf.bullet("UI policies: Show/hide/mandatory fields based on conditions")

    pdf.subsection_title("Fulfillment Methods")
    pdf.bullet("Workflow/Flow: Automated fulfillment process")
    pdf.bullet("Execution plan: Sequential or parallel task creation")
    pdf.bullet("Direct delivery: No fulfillment (informational items)")

    pdf.section_title("6.3 Request Lifecycle")
    pdf.body_text(
        "User submits request -> sc_request created -> sc_req_item(s) generated -> approvals fire -> "
        "fulfillment tasks created -> items delivered -> request closed."
    )
    pdf.section_title("6.4 Service Portal & Employee Center")
    pdf.body_text(
        "Service Portal (legacy) and Employee Center (Next Experience) are self-service frontends. "
        "Users browse the catalog, submit requests, track status, and access knowledge without "
        "logging into the agent interface."
    )
    pdf.section_title("6.5 Record Producers")
    pdf.body_text(
        "Record producers are catalog items that create records in backend tables (e.g., incident, "
        "change) instead of standard request items. They provide a user-friendly form for "
        "structured record creation."
    )
    pdf.exercise_box("Service Catalog Lab", [
        "Browse the Service Catalog and order a standard item (e.g., laptop request).",
        "Track the request through approval and fulfillment stages.",
        "Create a custom catalog item with at least three variables.",
        "Add an approval workflow for manager approval.",
        "Create a record producer that generates an incident from a portal form.",
    ])

    # ========== CHAPTER 7 ==========
    pdf.chapter_title(7, "Knowledge Management")
    pdf.section_title("7.1 Purpose & Knowledge Base Structure")
    pdf.body_text(
        "Knowledge Management captures, organizes, and shares information to enable self-service "
        "and improve agent efficiency. Articles are organized in Knowledge Bases and Categories."
    )
    pdf.subsection_title("Article Lifecycle")
    pdf.bullet("Draft: Author writing content")
    pdf.bullet("Review: Subject matter expert review")
    pdf.bullet("Published: Available to target audience")
    pdf.bullet("Retired/Outdated: No longer recommended")

    pdf.section_title("7.2 Article Types & Properties")
    pdf.bullet("Knowledge Base: Top-level container with access controls")
    pdf.bullet("Category: Sub-grouping within a KB")
    pdf.bullet("Article template: Standard sections (Symptoms, Cause, Resolution)")
    pdf.bullet("Effective dating: Publish/validity date ranges")
    pdf.bullet("Versioning: Track article revisions")
    pdf.bullet("Translation: Multi-language knowledge support")

    pdf.section_title("7.3 Knowledge-Centered Service (KCS)")
    pdf.body_text(
        "KCS integrates knowledge creation into the workflow. Agents capture knowledge while solving "
        "issues. KCS practices include search before create, reuse, and flag/ fix. ServiceNow "
        "supports KCS with article linking, flagging, and quality metrics."
    )
    pdf.section_title("7.4 Knowledge in Agent Workspace")
    pdf.bullet("Knowledge sidebar: Suggested articles based on incident text")
    pdf.bullet("Attach article to incident for caller reference")
    pdf.bullet("Create article from resolution notes")
    pdf.bullet("Knowledge feedback and ratings")

    pdf.exercise_box("Knowledge Management Lab", [
        "Create a knowledge article using the standard IT template.",
        "Submit for review and publish to the IT knowledge base.",
        "Search for the article from an incident and attach it.",
        "Flag an article as outdated and create a revised version.",
    ])

    # ========== CHAPTER 8 ==========
    pdf.chapter_title(8, "Service Level Management (SLA)")
    pdf.section_title("8.1 SLA Fundamentals")
    pdf.body_text(
        "Service Level Agreements define expected service quality. ServiceNow SLA engine tracks "
        "task SLA records against defined schedules and conditions, providing breach warnings "
        "and escalation."
    )
    pdf.subsection_title("Key SLA Components")
    pdf.bullet("SLA Definition: Name, table, duration, schedule, conditions")
    pdf.bullet("Task SLA: Running instance on a specific task record")
    pdf.bullet("Schedule: Business hours calendar (excludes weekends/holidays)")
    pdf.bullet("Start condition: When SLA timer begins")
    pdf.bullet("Stop condition: When SLA timer stops (met/breached)")
    pdf.bullet("Pause condition: When timer pauses (e.g., On Hold)")

    pdf.section_title("8.2 SLA Stages")
    pdf.table_header(["Stage", "Meaning"], [50, 140])
    sla_stages = [
        ("In Progress", "SLA actively running"),
        ("Paused", "Timer paused per pause condition"),
        ("Completed", "Met target before breach"),
        ("Breached", "Target time exceeded"),
        ("Cancelled", "SLA no longer applicable"),
    ]
    for i, row in enumerate(sla_stages):
        pdf.table_row(row, [50, 140], fill=(i % 2 == 0))

    pdf.section_title("8.3 SLA Breakdowns & Reporting")
    pdf.body_text(
        "SLA breakdown defines milestone percentages (e.g., 50%, 75%, 90%) for warnings. "
        "Business Rules and Notifications alert assignees and managers before breach. "
        "SLA dashboards show compliance percentages by team, priority, and category."
    )
    pdf.section_title("8.4 OLA & UC")
    pdf.bullet("OLA (Operational Level Agreement): Internal team-to-team commitments")
    pdf.bullet("UC (Underpinning Contract): External vendor/supplier agreements")
    pdf.bullet("Both use the same SLA engine with different definitions")

    pdf.exercise_box("SLA Lab", [
        "Review existing SLA definitions for incidents (e.g., P1 Response 15 min).",
        "Create an incident and observe Task SLA record creation.",
        "Put incident On Hold and verify SLA pauses.",
        "Resolve before breach and confirm SLA stage = Completed.",
    ])

    # ========== CHAPTER 9 ==========
    pdf.chapter_title(9, "Configuration Management (CMDB)")
    pdf.section_title("9.1 CMDB Purpose")
    pdf.body_text(
        "The Configuration Management Database (CMDB) stores Configuration Items (CIs) and their "
        "relationships. It provides a single source of truth for IT infrastructure, enabling impact "
        "analysis, change planning, and incident correlation."
    )
    pdf.subsection_title("CI Classes (cmdb_ci hierarchy)")
    pdf.bullet("cmdb_ci: Base CI class")
    pdf.bullet("cmdb_ci_computer: Servers, desktops, laptops")
    pdf.bullet("cmdb_ci_server: Server-specific attributes")
    pdf.bullet("cmdb_ci_network_adapter, cmdb_ci_ip_address: Network components")
    pdf.bullet("cmdb_ci_appl: Application CIs")
    pdf.bullet("cmdb_ci_service: Business services")
    pdf.bullet("cmdb_ci_database, cmdb_ci_storage: Data infrastructure")

    pdf.section_title("9.2 Relationships")
    pdf.body_text(
        "Relationships connect CIs (e.g., Runs on, Depends on, Contains, Connects to). "
        "Relationship rules define valid connections between CI classes. The CI Relationship "
        "Builder visualizes dependency maps."
    )
    pdf.section_title("9.3 CMDB Health & Data Quality")
    pdf.subsection_title("CMDB Health Dashboard KPIs")
    pdf.bullet("Completeness: Required fields populated")
    pdf.bullet("Correctness: Valid field values")
    pdf.bullet("Compliance: Adherence to standards")
    pdf.bullet("Duplicate CIs: Identification and remediation")

    pdf.section_title("9.4 Identification & Reconciliation Engine (IRE)")
    pdf.body_text(
        "IRE prevents duplicate CIs by matching incoming data against existing records using "
        "identification rules and reconciliation rules. Data sources (Discovery, Import Sets, "
        "manual entry) feed into IRE for deduplication."
    )
    pdf.section_title("9.5 Service Definitions")
    pdf.body_text(
        "Business services map technical CIs to business capabilities. Service owners define "
        "service offerings, SLAs, and supporting infrastructure through the service modeling process."
    )
    pdf.exercise_box("CMDB Lab", [
        "Create a server CI manually with key attributes (name, IP, OS).",
        "Create a business service and relate the server with 'Runs on'.",
        "Use CI Relationship Builder to visualize dependencies.",
        "Run CMDB Health dashboard and review completeness scores.",
    ])

    # ========== CHAPTER 10 ==========
    pdf.chapter_title(10, "Asset Management & Hardware Asset Management")
    pdf.section_title("10.1 Asset vs Configuration Item")
    pdf.table_header(["Aspect", "Asset", "Configuration Item (CI)"], [40, 75, 75])
    asset_ci = [
        ("Focus", "Financial/lifecycle tracking", "Technical/service configuration"),
        ("Table", "alm_asset", "cmdb_ci (linked via asset field)"),
        ("Ownership", "Finance/procurement", "IT operations"),
        ("Lifecycle", "Acquire, deploy, retire, dispose", "Install, configure, change, decommission"),
    ]
    for i, row in enumerate(asset_ci):
        pdf.table_row(row, [40, 75, 75], fill=(i % 2 == 0))

    pdf.section_title("10.2 Asset Lifecycle States")
    pdf.bullet("In Stock -> In Transit -> In Use -> In Maintenance -> Retired -> Consumed")
    pdf.bullet("Asset records track cost, PO number, warranty, lease, depreciation")
    pdf.bullet("Stockrooms and transfer orders manage inventory movement")

    pdf.section_title("10.3 Software Asset Management (SAM)")
    pdf.body_text(
        "SAM tracks software licenses, entitlements, and compliance. It integrates with Discovery "
        "to detect installed software and reconcile against purchased licenses. "
        "Key tables: sam_sw_install, sam_sw_product, alm_license."
    )
    pdf.section_title("10.4 Hardware Asset Management (HAM)")
    pdf.body_text(
        "HAM extends asset management for hardware procurement, stockroom management, "
        "asset reservation, and disposal workflows. It integrates with the Service Catalog "
        "for hardware request fulfillment."
    )

    # ========== CHAPTER 11 ==========
    pdf.chapter_title(11, "Discovery & Service Mapping")
    pdf.section_title("11.1 ServiceNow Discovery")
    pdf.body_text(
        "Discovery automatically identifies IP-enabled devices and software in your network, "
        "populating the CMDB without manual entry. It uses probes and sensors sent from "
        "MID Server to scan target devices."
    )
    pdf.subsection_title("Discovery Components")
    pdf.bullet("MID Server: On-premise agent that runs probes within customer firewall")
    pdf.bullet("Discovery Schedule: Defines IP ranges, credentials, and timing")
    pdf.bullet("Probes: Scripts that query devices (SNMP, SSH, WMI, PowerShell)")
    pdf.bullet("Sensors: Process probe results and create/update CIs")
    pdf.bullet("Credentials: Stored securely for authenticated scanning")

    pdf.section_title("11.2 Discovery Phases")
    pdf.bullet("1. Port Scan: Identify live IPs and open ports")
    pdf.bullet("2. Classification: Determine device type")
    pdf.bullet("3. Identification: Match to CI class")
    pdf.bullet("4. Exploration: Deep scan for details (OS, apps, services)")
    pdf.bullet("5. CI Creation/Update: Populate CMDB via IRE")

    pdf.section_title("11.3 Service Mapping")
    pdf.body_text(
        "Service Mapping builds dynamic maps of application services by identifying traffic "
        "patterns and dependencies. It uses entry points (load balancers, URLs) and discovery "
        "patterns to map multi-tier applications automatically."
    )
    pdf.section_title("11.4 Cloud Discovery")
    pdf.body_text(
        "Cloud Discovery extends scanning to AWS, Azure, and GCP resources. "
        "Service accounts and cloud APIs populate cloud CIs (cmdb_ci_vm_instance, etc.)."
    )

    # ========== CHAPTER 12 ==========
    pdf.chapter_title(12, "Release Management")
    pdf.section_title("12.1 Purpose")
    pdf.body_text(
        "Release Management plans, schedules, and controls the rollout of releases (collections of "
        "changes) to production. It coordinates development, testing, and deployment across teams."
    )
    pdf.subsection_title("Release Record Structure")
    pdf.bullet("Release phase: Design, Build, Test, Deploy, Review")
    pdf.bullet("Release tasks: Work items assigned to teams")
    pdf.bullet("Related changes: Changes bundled in the release")
    pdf.bullet("Environment management: Dev, Test, Staging, Production")

    pdf.section_title("12.2 DevOps Integration")
    pdf.body_text(
        "ServiceNow DevOps integrates with CI/CD tools (Jenkins, GitLab, Azure DevOps) to "
        "track pipeline executions, correlate deployments with changes, and automate "
        "change creation from pipeline events."
    )

    # ========== CHAPTER 13 ==========
    pdf.chapter_title(13, "Event Management & IT Operations")
    pdf.section_title("13.1 Event Management")
    pdf.body_text(
        "Event Management correlates monitoring alerts from multiple sources (Nagios, SolarWinds, "
        "Datadog, etc.) into actionable incidents. It reduces alert noise through deduplication, "
        "correlation rules, and CI binding."
    )
    pdf.subsection_title("Alert Lifecycle")
    pdf.bullet("Alert received via connector or email")
    pdf.bullet("Binding to CI in CMDB")
    pdf.bullet("Correlation into alert group")
    pdf.bullet("Automation: Create incident or auto-resolve")
    pdf.bullet("Remediation: Run remediation workflow")

    pdf.section_title("13.2 ITOM Visibility")
    pdf.body_text(
        "IT Operations Management (ITOM) includes Discovery, Service Mapping, Event Management, "
        "Cloud Management, and Operational Intelligence. It provides visibility into infrastructure "
        "health and automates operational responses."
    )
    pdf.section_title("13.3 MetricBase & Operational Intelligence")
    pdf.body_text(
        "MetricBase stores time-series metrics from monitoring tools. Operational Intelligence "
        "applies anomaly detection and forecasting to predict issues before they impact users."
    )

    # ========== CHAPTER 14 ==========
    pdf.chapter_title(14, "Administration: Users, Groups, Roles & ACLs")
    pdf.section_title("14.1 User Management")
    pdf.body_text(
        "Users (sys_user table) represent people who access the instance. Each user has a user ID, "
        "roles, groups, departments, and preferences."
    )
    pdf.subsection_title("User Record Key Fields")
    pdf.bullet("User ID: Login identifier")
    pdf.bullet("Roles: Permissions assigned directly or via group")
    pdf.bullet("Groups: Team membership for assignment and access")
    pdf.bullet("Department, Location, Manager: Organizational context")
    pdf.bullet("Active: Controls login ability")

    pdf.section_title("14.2 Groups & Roles")
    pdf.body_text(
        "Groups (sys_user_group) organize users into teams. Roles (sys_user_role) define permissions. "
        "Assign roles to groups to grant permissions to all members. Common ITSM roles:"
    )
    pdf.table_header(["Role", "Capability"], [55, 135])
    roles = [
        ("itil", "Basic ITIL user - create/read incidents, requests"),
        ("itil_admin", "Full ITSM administration"),
        ("sn_incident_write", "Create and update incidents"),
        ("sn_change_write", "Create and update changes"),
        ("catalog_admin", "Manage service catalog"),
        ("admin", "Full system administration (use sparingly)"),
    ]
    for i, row in enumerate(roles):
        pdf.table_row(row, [55, 135], fill=(i % 2 == 0))

    pdf.section_title("14.3 Access Control Lists (ACLs)")
    pdf.body_text(
        "ACLs enforce record-level and field-level security. Each ACL specifies operation (read, write, "
        "create, delete), table/field, and conditions/roles required. ACLs evaluate in order; "
        "deny takes precedence."
    )
    pdf.subsection_title("ACL Types")
    pdf.bullet("Table ACL: Applies to entire table")
    pdf.bullet("Field ACL: Applies to specific field")
    pdf.bullet("Conditional ACL: Uses script or condition for dynamic evaluation")

    pdf.section_title("14.4 Data Policies & UI Policies")
    pdf.bullet("Data Policy: Server-side mandatory/read-only enforcement")
    pdf.bullet("UI Policy: Client-side show/hide/mandatory on forms")
    pdf.bullet("Client Scripts: JavaScript on form load, change, submit")

    pdf.exercise_box("Administration Lab", [
        "Create a new user and assign the itil role.",
        "Create an assignment group and add the user as a member.",
        "Review ACLs on the incident table (incident.acl.*).",
        "Create a UI policy to hide 'Resolution notes' until state is Resolved.",
    ])

    # ========== CHAPTER 15 ==========
    pdf.chapter_title(15, "Workflows, Flow Designer & Automation")
    pdf.section_title("15.1 Workflow vs Flow Designer")
    pdf.table_header(["Feature", "Workflow (Legacy)", "Flow Designer"], [40, 75, 75])
    wf_fd = [
        ("Editor", "Graphical Workflow Editor", "Flow Designer (low-code)"),
        ("Status", "Legacy, still supported", "Recommended for new automations"),
        ("Use cases", "Complex branching", "Approvals, notifications, integrations"),
        ("Subflows", "Subflows", "Reusable subflows"),
    ]
    for i, row in enumerate(wf_fd):
        pdf.table_row(row, [40, 75, 75], fill=(i % 2 == 0))

    pdf.section_title("15.2 Flow Designer Components")
    pdf.bullet("Trigger: Record created/updated, scheduled, inbound webhook")
    pdf.bullet("Actions: Built-in operations (create record, update, notify, ask for approval)")
    pdf.bullet("Conditions: If/else branching")
    pdf.bullet("Subflows: Reusable flow segments")
    pdf.bullet("Spokes: Integration actions (Slack, Teams, Jira, etc.)")

    pdf.section_title("15.3 Business Rules")
    pdf.body_text(
        "Business Rules run server-side JavaScript on database operations (before, after, async, display). "
        "Use for field calculations, validations, and related record updates. "
        "Always consider performance impact on bulk operations."
    )
    pdf.section_title("15.4 Notification & Email")
    pdf.body_text(
        "Notifications define email/SMS templates triggered by events. Use notification conditions "
        "and recipient lists (users, groups, fields). Email layouts support Jelly templating and "
        "dynamic content from triggering records."
    )

    # ========== CHAPTER 16 ==========
    pdf.chapter_title(16, "Reporting, Dashboards & Performance Analytics")
    pdf.section_title("16.1 Reporting")
    pdf.body_text(
        "ServiceNow Reports visualize data from any table. Report types include bar, pie, line, "
        "list, pivot, heatmap, and SQL-based reports (admin only)."
    )
    pdf.subsection_title("Report Design Steps")
    pdf.bullet("1. Select source table (e.g., incident)")
    pdf.bullet("2. Choose report type")
    pdf.bullet("3. Configure grouping, stacking, aggregation")
    pdf.bullet("4. Apply filters (query conditions)")
    pdf.bullet("5. Set roles and share with users/groups")

    pdf.section_title("16.2 Dashboards")
    pdf.body_text(
        "Dashboards combine multiple reports and performance analytics widgets on one page. "
        "Role-based dashboards serve different audiences (executive, manager, agent). "
        "Homepages can be set per role or user."
    )
    pdf.section_title("16.3 Performance Analytics")
    pdf.body_text(
        "Performance Analytics (PA) provides advanced analytics with indicators, breakdowns, "
        "targets, and time-series trending. It enables data-driven service improvement through "
        "KPI tracking and threshold alerting."
    )
    pdf.subsection_title("PA Key Concepts")
    pdf.bullet("Indicator: Metric definition (e.g., Incident Resolution Time)")
    pdf.bullet("Breakdown: Dimension for slicing data (by priority, team, category)")
    pdf.bullet("Target: Desired KPI value")
    pdf.bullet("Collection job: Scheduled data collection for indicators")

    # ========== CHAPTER 17 ==========
    pdf.chapter_title(17, "Integrations & Data Import")
    pdf.section_title("17.1 Integration Methods")
    pdf.table_header(["Method", "Use Case"], [55, 135])
    integ = [
        ("REST API (Table API)", "CRUD operations on records from external systems"),
        ("Import Sets", "Bulk data load from CSV, Excel, or JDBC"),
        ("Transform Maps", "Map import set rows to target tables"),
        ("IntegrationHub Spokes", "Pre-built connectors (Jira, Azure, SAP)"),
        ("MID Server", "On-premise connectivity for Discovery and integrations"),
        ("Event Rules", "Inbound events from monitoring tools"),
        ("Outbound Web Services", "SOAP/REST calls from ServiceNow to external APIs"),
    ]
    for i, row in enumerate(integ):
        pdf.table_row(row, [55, 135], fill=(i % 2 == 0))

    pdf.section_title("17.2 REST API Essentials")
    pdf.body_text(
        "The Table API (api/now/table/{tableName}) supports GET, POST, PUT, PATCH, DELETE. "
        "Authentication via Basic Auth, OAuth, or API keys. Use sysparm_query for encoded queries, "
        "sysparm_fields to limit returned fields, and sysparm_limit for pagination."
    )
    pdf.subsection_title("Example API Call")
    pdf.set_font("Courier", "", 9)
    pdf.multi_cell(0, 5, "GET /api/now/table/incident?sysparm_query=active=true^priority=1\n"
                  "Authorization: Basic {base64(user:password)}\n"
                  "Accept: application/json")
    pdf.ln(3)

    pdf.section_title("17.3 Import Sets & Transform Maps")
    pdf.body_text(
        "Import Sets load data into staging tables (u_import_*). Transform Maps define field mapping, "
        "coalesce fields (match existing records), and run transform scripts. "
        "Use for migration, bulk updates, and scheduled data sync."
    )

    # ========== CHAPTER 18 ==========
    pdf.chapter_title(18, "Virtual Agent, Now Assist & Self-Service")
    pdf.section_title("18.1 Virtual Agent")
    pdf.body_text(
        "Virtual Agent provides conversational AI for self-service. Users interact via chat to "
        "get answers, submit requests, and check ticket status. Topics define conversation flows "
        "with intent recognition and entity extraction."
    )
    pdf.subsection_title("Virtual Agent Components")
    pdf.bullet("Topics: Conversation flows triggered by user phrases")
    pdf.bullet("Entities: Data extracted from user input (date, email, choice)")
    pdf.bullet("NLU model: Natural language understanding for intent matching")
    pdf.bullet("Channels: Portal, Teams, Slack, mobile")

    pdf.section_title("18.2 Now Assist (Generative AI)")
    pdf.body_text(
        "Now Assist brings generative AI to ITSM: auto-summarize incidents, suggest resolutions, "
        "generate knowledge articles, and assist agents with contextual recommendations. "
        "It integrates with the Now Platform AI capabilities."
    )
    pdf.section_title("18.3 Employee Center")
    pdf.body_text(
        "Employee Center is the Next Experience self-service portal. It provides unified access "
        "to catalog, knowledge, cases, and Virtual Agent in a modern, mobile-friendly interface."
    )

    # ========== CHAPTER 19 ==========
    pdf.chapter_title(19, "ITIL Alignment & Best Practices")
    pdf.section_title("19.1 Implementation Best Practices")
    pdf.bullet("Start with out-of-box processes; customize incrementally")
    pdf.bullet("Define clear roles and responsibilities before go-live")
    pdf.bullet("Establish data governance for CMDB and catalog")
    pdf.bullet("Use sub-production instances (dev, test, UAT) for changes")
    pdf.bullet("Document configuration decisions in implementation wiki")
    pdf.bullet("Train agents and end users before launch")
    pdf.bullet("Measure KPIs from day one with PA and dashboards")

    pdf.section_title("19.2 Common Pitfalls to Avoid")
    pdf.bullet("Over-customizing before understanding OOB capabilities")
    pdf.bullet("Skipping CMDB data quality governance")
    pdf.bullet("Creating too many catalog items without lifecycle management")
    pdf.bullet("Granting admin role broadly instead of scoped roles")
    pdf.bullet("Ignoring performance: unindexed queries, heavy business rules")
    pdf.bullet("Not testing workflows with edge cases and bulk scenarios")

    pdf.section_title("19.3 Continuous Improvement")
    pdf.body_text(
        "Use Problem Management and PA trends for continuous improvement. Conduct regular "
        "CAB retrospectives, knowledge article reviews, and SLA compliance analysis. "
        "Align improvement initiatives with business outcomes."
    )

    # ========== CHAPTER 20 ==========
    pdf.chapter_title(20, "Certification Prep & Practice Questions")
    pdf.section_title("20.1 ServiceNow Certification Paths")
    pdf.table_header(["Certification", "Focus Area"], [55, 135])
    certs = [
        ("CSA (Certified System Administrator)", "Platform administration fundamentals"),
        ("CIS-ITSM (Certified Implementation Specialist)", "ITSM application implementation"),
        ("CAD (Certified Application Developer)", "Development and scripting"),
        ("CIS-APM", "Asset and Portfolio Management"),
        ("CIS-EM", "Event Management"),
        ("CIS-SAM", "Software Asset Management"),
    ]
    for i, row in enumerate(certs):
        pdf.table_row(row, [55, 135], fill=(i % 2 == 0))

    pdf.section_title("20.2 Practice Questions")
    pdf.subsection_title("Question 1")
    pdf.body_text(
        "Q: Which table do Incident, Problem, and Change Request extend?\n"
        "A) sys_task  B) task  C) sc_task  D) wm_task\n"
        "Answer: B) task"
    )
    pdf.subsection_title("Question 2")
    pdf.body_text(
        "Q: What happens to an SLA when an incident moves to 'On Hold' state?\n"
        "A) SLA stops permanently  B) SLA is cancelled  C) SLA pauses  D) SLA continues\n"
        "Answer: C) SLA pauses (if pause condition is configured)"
    )
    pdf.subsection_title("Question 3")
    pdf.body_text(
        "Q: Which component runs probes within the customer's network for Discovery?\n"
        "A) IntegrationHub  B) MID Server  C) Event Management  D) Flow Designer\n"
        "Answer: B) MID Server"
    )
    pdf.subsection_title("Question 4")
    pdf.body_text(
        "Q: What is the difference between a Standard and Normal change?\n"
        "A: Standard changes are pre-authorized with low risk; Normal changes require assessment and approval."
    )
    pdf.subsection_title("Question 5")
    pdf.body_text(
        "Q: Which record type does a Record Producer create?\n"
        "A: Always sc_req_item  B: Any configured target table  C: Only incidents  D: Only tasks\n"
        "Answer: B) Any configured target table"
    )
    pdf.subsection_title("Question 6")
    pdf.body_text(
        "Q: What does IRE stand for in CMDB context?\n"
        "A: Identification and Reconciliation Engine"
    )
    pdf.subsection_title("Question 7")
    pdf.body_text(
        "Q: Which ServiceNow tool is recommended for new automations?\n"
        "A: Workflow Editor  B) Flow Designer  C) Business Rules  D) Client Scripts\n"
        "Answer: B) Flow Designer"
    )
    pdf.subsection_title("Question 8")
    pdf.body_text(
        "Q: Priority on an incident is typically calculated from:\n"
        "A: Impact + Urgency  B) Impact x Urgency matrix  C) Category + Subcategory  D) Assignment group\n"
        "Answer: B) Impact x Urgency matrix (lookup)"
    )

    pdf.exercise_box("Final Capstone Project", [
        "Design an end-to-end scenario: User reports email outage via portal.",
        "Agent creates incident, links CI, attaches KB article, escalates priority.",
        "Major incident declared; problem record created after resolution.",
        "Change request raised for permanent fix; CAB approves and implements.",
        "Knowledge article published; SLA compliance measured on dashboard.",
        "Document each step with screenshots from your PDI.",
    ])

    # ========== APPENDIX C - DETAILED CONFIG GUIDES ==========
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 82, 147)
    pdf.cell(0, 12, "Appendix C: Step-by-Step Configuration Guides", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.section_title("C.1 Creating an SLA Definition")
    pdf.body_text(
        "Navigate to SLA > SLA Definitions > New. Set Name (e.g., P1 Response - 15 min), "
        "Table: incident, Duration: 00:15:00, Schedule: 8-5 weekdays. "
        "Set Start condition: Priority changes to 1. Stop condition: State is Resolved or Closed. "
        "Pause condition: State is On Hold. Save and test with a P1 incident."
    )
    pdf.section_title("C.2 Creating a Catalog Item")
    pdf.body_text(
        "Go to Service Catalog > Catalog Definitions > Maintain Items > New. Enter Name, Category, "
        "Catalogs (Service Catalog), and Price if applicable. In the Variables related list, add "
        "questions (e.g., Laptop Model as Select Box). Set Workflow or Flow for fulfillment. "
        "Under Availability, set roles/groups who can see the item. Publish when ready."
    )
    pdf.section_title("C.3 Setting Up Assignment Rules")
    pdf.body_text(
        "Navigate to System Policy > Rules > Assignment. Create new rule on incident table. "
        "Set Order (lower = higher priority), Condition (e.g., Category is Network), "
        "Assign to Group: Network Support. Multiple rules evaluate top-down; first match wins."
    )
    pdf.section_title("C.4 Creating a Change Model")
    pdf.body_text(
        "Open Change > Administration > Change Models > New. Define model name, table (change_request), "
        "initial state, and state transitions. Add approval activities for Authorize state. "
        "Configure conditions for auto-transition. Activate model and set as default if needed."
    )
    pdf.section_title("C.5 CMDB Identification Rules")
    pdf.body_text(
        "Go to CMDB > Identification/Reconciliation > Identification Rules. Select CI class "
        "(e.g., cmdb_ci_linux_server). Define identifier entries (serial number, name+IP). "
        "Create reconciliation rules to decide which data source wins for each field on update."
    )
    pdf.section_title("C.6 Building a Flow in Flow Designer")
    pdf.body_text(
        "Open Flow Designer > New Flow. Set name and application scope. Choose trigger: "
        "Record Created on incident. Add action: Update Record (set assignment group). "
        "Add Ask for Approval action for manager approval on high priority. "
        "Add Send Notification action. Activate flow and test."
    )
    pdf.section_title("C.7 Creating a Report and Dashboard")
    pdf.body_text(
        "Reports > Create New. Select Type: Bar chart, Source: incident, Group by: Priority, "
        "Aggregation: Count. Save as 'Incidents by Priority'. "
        "Dashboards > New Dashboard, add the report widget. Assign dashboard to itil role homepage."
    )
    pdf.section_title("C.8 REST API Testing with curl")
    pdf.set_font("Courier", "", 8)
    pdf.multi_cell(0, 4.5,
        "curl -X GET 'https://INSTANCE.service-now.com/api/now/table/incident"
        "?sysparm_limit=5' \\\n"
        "  -H 'Accept: application/json' \\\n"
        "  -u 'admin:password'"
    )
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "Replace INSTANCE with your PDI name. Use POST to create records, PATCH to update. "
        "Always use HTTPS and service accounts with minimal required roles in production."
    )

    pdf.section_title("C.9 Import Set Data Load")
    pdf.body_text(
        "System Import Sets > Load Data > Choose File (CSV). Select target import set table or "
        "create new. Map columns. Run transform map to target table (e.g., cmdb_ci). "
        "Review import set run logs for errors. Schedule recurring imports for ongoing sync."
    )
    pdf.section_title("C.10 Virtual Agent Topic Creation")
    pdf.body_text(
        "Virtual Agent > Designer > New Topic. Define trigger phrases ('reset password', 'password help'). "
        "Build conversation flow: greeting, collect email entity, call password reset subflow, "
        "confirm completion. Publish topic and test in Virtual Agent simulator."
    )

    # ========== APPENDIX D - MORE PRACTICE QUESTIONS ==========
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 82, 147)
    pdf.cell(0, 12, "Appendix D: Additional Practice Questions (20)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    extra_qs = [
        ("Q9", "Which field on incident determines who receives survey?", "Caller (caller_id)"),
        ("Q10", "What table stores approval records?", "sysapproval_approver"),
        ("Q11", "Coalesce field in transform map is used for:", "Matching/updating existing records"),
        ("Q12", "Default incident priority when Impact=2, Urgency=2:", "3 (Moderate) - verify matrix"),
        ("Q13", "Which role is minimum to create incidents?", "itil or sn_incident_write"),
        ("Q14", "sc_req_item is child of which table?", "sc_request"),
        ("Q15", "Business Rule 'before' runs:", "Before database operation commits"),
        ("Q16", "CMDB class for Windows servers:", "cmdb_ci_win_server"),
        ("Q17", "Discovery requires what on-premise component?", "MID Server"),
        ("Q18", "Knowledge article state for public visibility:", "Published"),
        ("Q19", "Change type with pre-approval:", "Standard"),
        ("Q20", "Flow Designer trigger for scheduled job:", "Scheduled - recurring or once"),
        ("Q21", "Encoded query 'active=true^priority=1' means:", "AND condition (active AND priority 1)"),
        ("Q22", "sys_id format:", "32-character hexadecimal GUID"),
        ("Q23", "UI Policy runs on:", "Client-side (browser)"),
        ("Q24", "Data Policy runs on:", "Server-side (always enforced)"),
        ("Q25", "Performance Analytics indicator collects data via:", "Scheduled collection job"),
        ("Q26", "Record producer creates records in:", "Configured target table"),
        ("Q27", "On Hold reason required when:", "Setting incident to On Hold state"),
        ("Q28", "CAB approves which change type?", "Normal (and some Emergency post-facto)"),
    ]
    for qnum, question, answer in extra_qs:
        pdf.subsection_title(qnum)
        pdf.body_text(f"{question}\nAnswer: {answer}")

    # ========== APPENDIX E - INCIDENT FIELD REFERENCE ==========
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 82, 147)
    pdf.cell(0, 12, "Appendix E: Key Field Reference by Process", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.section_title("E.1 Incident Key Fields")
    pdf.table_header(["Field", "Type", "Description"], [45, 30, 115])
    inc_fields = [
        ("number", "Auto", "INC0010001 format ticket number"),
        ("caller_id", "Reference", "User who reported (sys_user)"),
        ("category", "Choice", "High-level classification"),
        ("subcategory", "Choice", "Detailed classification"),
        ("impact", "Choice", "1=High, 2=Medium, 3=Low"),
        ("urgency", "Choice", "1=High, 2=Medium, 3=Low"),
        ("priority", "Choice", "Calculated from impact x urgency"),
        ("assignment_group", "Reference", "Team owning the ticket"),
        ("assigned_to", "Reference", "Individual agent"),
        ("cmdb_ci", "Reference", "Affected configuration item"),
        ("short_description", "String", "Brief summary (required)"),
        ("description", "Text", "Detailed description"),
        ("work_notes", "Journal", "Internal agent notes"),
        ("comments", "Journal", "Customer-visible comments"),
        ("resolution_code", "Choice", "Required on resolve"),
        ("close_code", "Choice", "Required on close"),
    ]
    for i, row in enumerate(inc_fields):
        pdf.table_row(row, [45, 30, 115], fill=(i % 2 == 0))

    pdf.section_title("E.2 Change Request Key Fields")
    pdf.table_header(["Field", "Type", "Description"], [45, 30, 115])
    chg_fields = [
        ("type", "Choice", "Standard, Normal, Emergency"),
        ("risk", "Choice", "Calculated risk level"),
        ("risk_impact", "Choice", "Impact if change fails"),
        ("implementation_plan", "Text", "Steps to implement"),
        ("backout_plan", "Text", "Rollback procedure"),
        ("test_plan", "Text", "Validation steps"),
        ("justification", "Text", "Business reason for change"),
        ("cab_date", "Date", "Scheduled CAB meeting"),
        ("start_date", "Date/Time", "Planned start"),
        ("end_date", "Date/Time", "Planned end"),
    ]
    for i, row in enumerate(chg_fields):
        pdf.table_row(row, [45, 30, 115], fill=(i % 2 == 0))

    pdf.section_title("E.3 Encoded Query Operators")
    pdf.table_header(["Operator", "Meaning", "Example"], [30, 60, 100])
    query_ops = [
        ("=", "Equals", "priority=1"),
        ("!=", "Not equal", "state!=7"),
        (">", "Greater than", "priority>2"),
        ("IN", "In list", "priorityIN1,2"),
        ("LIKE", "Contains", "short_descriptionLIKEemail"),
        ("STARTSWITH", "Begins with", "numberSTARTSWITHINC"),
        ("ISEMPTY", "Is empty", "assigned_toISEMPTY"),
        ("^", "AND", "active=true^priority=1"),
        ("^OR", "OR", "priority=1^ORpriority=2"),
        ("^ORDERBY", "Sort", "^ORDERBYDESCopened_at"),
    ]
    for i, row in enumerate(query_ops):
        pdf.table_row(row, [30, 60, 100], fill=(i % 2 == 0))

    # ========== APPENDIX A ==========
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 82, 147)
    pdf.cell(0, 12, "Appendix A: Glossary of Key Terms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    glossary = [
        ("ACL", "Access Control List - security rule for table/field access"),
        ("CAB", "Change Advisory Board - approves change requests"),
        ("CI", "Configuration Item - component in CMDB"),
        ("CMDB", "Configuration Management Database"),
        ("ECAB", "Emergency Change Advisory Board"),
        ("IRE", "Identification and Reconciliation Engine"),
        ("ITIL", "Information Technology Infrastructure Library"),
        ("ITOM", "IT Operations Management"),
        ("ITSM", "IT Service Management"),
        ("KCS", "Knowledge-Centered Service"),
        ("KEDB", "Known Error Database"),
        ("MID Server", "Management, Instrumentation, and Discovery Server"),
        ("MIT", "Major Incident Team"),
        ("NLU", "Natural Language Understanding"),
        ("OLA", "Operational Level Agreement"),
        ("PA", "Performance Analytics"),
        ("PDI", "Personal Developer Instance"),
        ("PIR", "Post-Incident Review"),
        ("RITM", "Requested Item (sc_req_item)"),
        ("SAM", "Software Asset Management"),
        ("SLA", "Service Level Agreement"),
        ("SVS", "Service Value System (ITIL 4)"),
        ("UC", "Underpinning Contract"),
    ]
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    for term, defn in glossary:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, f"{term}: ", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, defn)
        pdf.ln(1)

    # ========== APPENDIX B ==========
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 82, 147)
    pdf.cell(0, 12, "Appendix B: Useful Tables & Reference Data", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.section_title("Common ITSM Tables")
    pdf.table_header(["Table Name", "Label", "Description"], [45, 40, 105])
    ref_tables = [
        ("incident", "Incident", "Service disruption records"),
        ("problem", "Problem", "Root cause investigation"),
        ("change_request", "Change Request", "Controlled changes"),
        ("sc_request", "Request", "Catalog request header"),
        ("sc_req_item", "Requested Item", "Catalog line item"),
        ("sc_task", "Catalog Task", "Fulfillment task"),
        ("task_sla", "Task SLA", "SLA instance on task"),
        ("cmdb_ci", "Configuration Item", "Base CI table"),
        ("kb_knowledge", "Knowledge", "Knowledge articles"),
        ("sys_user", "User", "Platform users"),
        ("sys_user_group", "Group", "User groups/teams"),
        ("sysapproval_approver", "Approver", "Approval records"),
    ]
    for i, row in enumerate(ref_tables):
        pdf.table_row(row, [45, 40, 105], fill=(i % 2 == 0))

    pdf.section_title("Useful Navigation Paths (Classic UI)")
    pdf.bullet("Incidents: Self-Service > Incident > Create New (or Incident > Create New)")
    pdf.bullet("Changes: Change > Create New")
    pdf.bullet("Catalog Admin: Service Catalog > Catalog Definitions > Maintain Items")
    pdf.bullet("SLA Definitions: SLA > SLA Definitions")
    pdf.bullet("CMDB: Configuration > CI Class Manager")
    pdf.bullet("Reports: Reports > View/Run")
    pdf.bullet("Users: User Administration > Users")
    pdf.bullet("Roles: User Administration > Roles")

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5,
        "Disclaimer: This workbook is an independent educational resource and is not affiliated with "
        "or endorsed by ServiceNow, Inc. ServiceNow and the ServiceNow logo are registered trademarks "
        "of ServiceNow, Inc. For official training and certification, visit learning.servicenow.com "
        "and docs.servicenow.com."
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pdf.output(OUTPUT_PATH)
    pdf.output(WORKSPACE_COPY)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_workbook()
    size_kb = os.path.getsize(path) // 1024
    print(f"Generated: {path} ({size_kb} KB)")
