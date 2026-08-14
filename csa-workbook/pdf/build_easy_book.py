#!/usr/bin/env python3
"""Build a thorough, easy-English CSA teaching book as HTML."""
from pathlib import Path

OUT = Path(__file__).parent / "csa-easy-book.html"


def box(kind, title, body):
    return f'<div class="box box-{kind}"><div class="box-t">{title}</div><div>{body}</div></div>'


def remember(text):
    return box("rem", "Remember this", text)


def why(text):
    return box("why", "Why this matters", text)


def trap(text):
    return box("trap", "Easy trap", text)


def example(text):
    return box("ex", "Tiny example", text)


def steps(items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<ol class="steps">{lis}</ol>'


def table(head, rows):
    th = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table>{th}{body}</table>'


parts = []
parts.append(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ServiceNow CSA Easy Book</title>
<link rel="stylesheet" href="fonts.css"/>
<style>
  :root { --ink:#1c1917; --muted:#57534e; --paper:#fffaf3; --rule:#edd8b8;
          --teal:#0f766e; --violet:#6d28d9; --rose:#be123c; --amber:#b45309; }
  * { box-sizing: border-box; }
  html, body { margin:0; padding:0; background:#efe4d2; color:var(--ink);
    font-family:'Nunito','Noto Sans',sans-serif; font-size:12.2pt; line-height:1.5; }
  @page { size: A4; margin: 14mm 14mm 16mm 14mm; }
  @media print {
    html, body { background:white; }
    .chapter { break-before: auto; page-break-before: auto; }
    .cover { break-after: page; page-break-after: always; }
    h1, h2 { break-after: avoid; page-break-after: avoid; }
    .box, table, .flow { break-inside: avoid; page-break-inside: avoid; }
    a { color: inherit; text-decoration: none; }
  }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .pagewrap { max-width: 180mm; margin: 0 auto; background: var(--paper);
    padding: 8mm 10mm 12mm; }
  @media print { .pagewrap { max-width:none; padding:0; background:transparent; } }
  h1 { font-family:'Fredoka',sans-serif; font-size:26pt; line-height:1.15; margin:0 0 8px; }
  h2 { font-family:'Fredoka',sans-serif; font-size:16pt; color:#0f172a; margin:18px 0 8px;
       border-bottom: 3px solid #fde68a; display:inline-block; }
  h3 { font-size:13pt; margin:14px 0 6px; color:#6d28d9; }
  p { margin: 0 0 9px; }
  .hand { font-family:'Caveat','Patrick Hand',cursive; }
  .kicker { font-size:10pt; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:#a16207; }
  .lede { font-size:13.5pt; color:#44403c; }
  .box { border-radius:14px; padding:10px 12px; margin:10px 0 12px; border:2px solid #1c1917; }
  .box-t { font-weight:800; margin-bottom:4px; }
  .box-why { background:#ecfeff; border-color:#0f766e; }
  .box-rem { background:#fef9c3; border-color:#ca8a04; }
  .box-trap { background:#ffe4e6; border-color:#be123c; }
  .box-ex { background:#ede9fe; border-color:#6d28d9; }
  .box-story { background:#ffedd5; border-color:#c2410c; }
  ol.steps { margin:6px 0 12px; padding-left:22px; }
  ol.steps li { margin:0 0 6px; }
  table { width:100%; border-collapse:collapse; margin:8px 0 12px; font-size:11pt; }
  th { background:#1c1917; color:#fff; text-align:left; padding:6px 8px; }
  td { border:1px solid #e7e5e4; padding:6px 8px; vertical-align:top; }
  tr:nth-child(even) td { background:#fff7ed; }
  .mono { font-family:'IBM Plex Mono',monospace; font-size:10.5pt; }
  ul { margin:4px 0 12px; padding-left:20px; }
  li { margin:0 0 5px; }
  .toc a { display:block; padding:4px 0; color:#1c1917; text-decoration:none;
           border-bottom:1px dotted #e7e5e4; }
  .cover { text-align:left; padding:8mm 0 10mm; }
  .blob { display:inline-block; background:#fde68a; padding:3px 10px; border-radius:999px;
          font-family:'Caveat',cursive; font-size:16pt; }
  .stat { display:inline-block; background:#1c1917; color:#fff; border-radius:12px;
          padding:8px 12px; margin:4px 6px 4px 0; }
  .stat b { font-family:'Fredoka',sans-serif; font-size:18pt; display:block; }
  .hl { background: linear-gradient(transparent 60%, #fde047 60%); }
  .footer { margin-top:16px; font-family:'Caveat',cursive; color:#a16207; font-size:14pt; }
  .flow { font-weight:800; background:#fff; border:2px solid #1c1917; border-radius:12px;
          padding:10px; text-align:center; margin:10px 0; }
</style>
</head>
<body>
<div class="pagewrap">
""")

# COVER
parts.append("""
<section class="chapter first cover">
  <span class="blob">a friendly textbook · January 2026 exam</span>
  <p class="hand" style="font-size:22pt;color:#6d28d9;margin:18px 0 0;">ServiceNow CSA</p>
  <h1>The Easy Book</h1>
  <p class="lede">Every exam topic explained in plain English, with pictures in words, tiny examples, and “why should I care?” notes. If you are new to ServiceNow, start on page 1 and do not skip the stories.</p>
  <div style="margin:16px 0;">
    <span class="stat"><b>60</b> questions</span>
    <span class="stat"><b>90</b> minutes</span>
    <span class="stat"><b>6</b> domains</span>
    <span class="stat"><b>0</b> half marks</span>
  </div>
  <div class="box box-story">
    <div class="box-t">How to read this book</div>
    <p>Each chapter has four layers: a <b>story</b> (so it sticks), a <b>simple meaning</b>, a <b>how it looks in ServiceNow</b>, and an <b>exam trap</b>. When you finish a chapter, try the 2-minute practice on a free Personal Developer Instance (PDI) from developer.servicenow.com.</p>
    <p>Study order if you are short on time: <b>Data &amp; security → Catalog &amp; Knowledge → Lists &amp; forms → Scripts &amp; update sets → Users &amp; plugins → Navigation</b>.</p>
  </div>
  <p class="footer">Not affiliated with ServiceNow. Original teaching notes based on the public CSA blueprint.</p>
</section>
""")

# TOC
parts.append("""
<section class="chapter">
  <div class="kicker">contents</div>
  <h1>What is inside</h1>
  <div class="toc">
    <a>1. What ServiceNow actually is</a>
    <a>2. Your instance — the house you live in</a>
    <a>3. Moving around: Next Experience</a>
    <a>4. Plugins, branding, and “make it ours”</a>
    <a>5. Users, groups, and roles (the permission family)</a>
    <a>6. Lists, filters, and tags</a>
    <a>7. Forms, views, and the famous save buttons</a>
    <a>8. Tasks, assignment, SLAs, and Visual Task Boards</a>
    <a>9. Reports, dashboards, and notifications</a>
    <a>10. Knowledge Management</a>
    <a>11. Service Catalog — REQ, RITM, SCTASK</a>
    <a>12. Workflow Studio and Virtual Agent</a>
    <a>13. Tables, fields, and relationships</a>
    <a>14. Access Control Lists (the lock on every door)</a>
    <a>15. Importing data without making a mess</a>
    <a>16. CMDB and CSDM — the map of the company’s IT</a>
    <a>17. Security Center and who is responsible for what</a>
    <a>18. UI Policy, Data Policy, Business Rules, and scripts</a>
    <a>19. Update sets — moving your work to production</a>
    <a>20. Putting it all together + exam method</a>
    <a>21. Practice questions explained like a teacher</a>
  </div>
</section>
""")

# CH 1
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 1 · domain 1 · 7%</div>
  <h1>What ServiceNow actually is</h1>
  <p class="lede">If someone asks “what is ServiceNow?” and you say “a ticketing tool”, you are only 10% right. CSA wants the other 90%.</p>

  <h2>The story</h2>
  <div class="box box-story">
    <p>Imagine a big company. People email IT: “laptop broken”, “need software”, “wifi is dead”. Managers cannot see who is overloaded. Nobody knows which server supports the payroll website. HR has a different system. Customer support has another.</p>
    <p>ServiceNow is one <b>shared work platform in the cloud</b> where those processes can live together: incidents, requests, knowledge articles, approvals, and a map of the company’s technology. You do not install a giant server in your office for the core product. ServiceNow hosts it. You <b>configure</b> it.</p>
  </div>

  <h2>Simple meaning</h2>
  <p>ServiceNow is a cloud <b>application Platform-as-a-Service (aPaaS)</b>.</p>
  <ul>
    <li><b>Platform</b> = a base that many apps sit on (like Android is a platform for many apps).</li>
    <li><b>as a Service</b> = they run the computers; you log in with a browser.</li>
    <li>The same platform runs IT work, employee work (HR), customer work, and custom apps you build.</li>
  </ul>
  <p>CSA does not ask you to be an HR expert or a security-operations expert. It asks you to know that <span class="hl">lists, forms, tables, roles, flows, and CMDB are the same tools everywhere</span>.</p>

  <h2>Multi-instance (this word is loved by the exam)</h2>
  <p>Some cloud products put every customer in <b>one giant shared database</b> (multi-tenant). ServiceNow is different. Each customer gets their own <b>instance</b>: their own application and their own database.</p>
  <p>Think of an apartment building vs a hostel dorm. Multi-tenant is a dorm: many people, one big room. Multi-instance is apartments: same building company, but your furniture and your mess stay in <b>your</b> flat.</p>
  """ + remember("ServiceNow architecture = <b>multi-instance</b>, not a shared multi-tenant database.") + """
  <p>A company usually has at least two instances:</p>
  <ul>
    <li><b>Production</b> — real users, real data. Do not experiment here.</li>
    <li><b>Sub-production</b> — development / test / UAT. This is where you try things.</li>
  </ul>
  <p>A <b>clone</b> copies production onto a test instance so testers see realistic data. Cloning <span class="hl">overwrites the target instance</span>. It is like replacing all the furniture in the test apartment with a copy of production’s furniture.</p>

  <h2>Now Support (HI) vs working inside the instance</h2>
  <p>Day to day, you click around inside ServiceNow: create users, design forms, build catalog items.</p>
  <p>Some jobs need ServiceNow themselves: certain upgrades, some plugins, official support cases, sometimes clones. That portal is <b>Now Support</b>, historically called <b>HI</b> (Hosted Instance).</p>
  """ + why("If a question says “where do you request an instance upgrade?”, think Now Support / HI, not the Incident table.") + """

  <h2>Personal Developer Instance (PDI)</h2>
  <p>A PDI is a free practice instance from developer.servicenow.com. It has demo people (like Abel Tuter) and demo incidents. Break it. Reset it. It is a sandbox, not a customer production system.</p>
  <p class="footer">Next: the instance is a house. Let’s walk through the front door.</p>
</section>
""")

# CH 2
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 2 · domain 1</div>
  <h1>Your instance — the house you live in</h1>
  <p>An instance has a web address like <span class="mono">https://dev12345.service-now.com</span>. Dev, test, and prod have <b>different URLs</b>. If you build a catalog item in production “just this once”, you will eventually hurt real users.</p>

  <h2>What is stored in the instance?</h2>
  <ul>
    <li>Applications and modules (the menu)</li>
    <li>Tables and records (the data)</li>
    <li>Users, groups, roles</li>
    <li>System properties (switches)</li>
    <li>Plugins (feature packs)</li>
    <li>Your customizations</li>
  </ul>

  <h2>Logging in — proving who you are</h2>
  <p>This is <b>authentication</b> (who are you?), which is not the same as <b>authorization</b> (what may you do?). Authorization is roles and ACLs, later in the book.</p>
  """ + table(["Method", "In simple words"], [
    ["Local database", "Password stored in ServiceNow on the user record"],
    ["LDAP / Active Directory", "Company directory checks the password; often needs a MID Server"],
    ["SSO (SAML and friends)", "You log in at Okta / Azure AD / Google, then land in ServiceNow"],
    ["MFA", "Password plus a second check (app code, SMS, …)"],
    ["Smart card (CAC / PIV)", "A physical certificate card, common in government"],
  ]) + """
  <p>CSA will not ask you to paste SAML XML. It may ask which method is in use, or why a local admin account still matters if SSO breaks (break-glass).</p>

  <h2>The user menu — two magic buttons</h2>
  <p><b>Impersonate User</b> = wear someone else’s badge for testing. You do not need their password. Always unimpersonate before you keep doing admin work.</p>
  <p><b>Elevate role</b> = turn on <span class="mono">security_admin</span> for <b>this session only</b>. Needed to create or edit Access Control rules when High Security is on. It is like asking the bank vault for a time-limited key.</p>
  """ + trap("Impersonate and Elevate are not the same. Impersonate = test as them. Elevate = extra power on YOUR session.") + """

  <h2>Personalization vs customization</h2>
  <p>This pair appears forever.</p>
  """ + table(["", "Personalization", "Customization"], [
    ["Who", "Only you", "Everyone on the instance"],
    ["Examples", "Your favorite modules, your list columns, your language", "Company logo, a Business Rule, a catalog item"],
    ["Update set", "No", "Usually yes"],
  ]) + """
  <p>The <b>gear</b> on a list is personalization (often a green dot). <b>Configure → List Layout</b> changes the view for everybody who uses that view.</p>
  <p class="footer">Next: where those buttons sit on the screen.</p>
</section>
""")

# CH 3
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 3 · domain 1</div>
  <h1>Moving around: Next Experience</h1>
  <p>Next Experience is the modern native screen. Mentally divide it into three parts, like a newspaper:</p>
  <div class="flow">BANNER (top) &nbsp;+&nbsp; NAVIGATOR (left) &nbsp;+&nbsp; CONTENT FRAME (the big work area)</div>

  <h3>Banner</h3>
  <ul>
    <li><b>Logo</b> — click it to go home</li>
    <li><b>Global search</b> — search many tables at once</li>
    <li><b>Help</b>, notifications, <b>user menu</b></li>
  </ul>

  <h3>Application navigator</h3>
  <ul>
    <li><b>All</b> — full menu</li>
    <li><b>Favorites</b> — star, or drag a filter breadcrumb, or drag a record</li>
    <li><b>History</b> — recently opened things (about the last 30). This is not an audit log and not CMDB history.</li>
    <li><b>Filter box</b> — type to search the menu. Magic shortcut: <span class="mono">incident.list</span> opens the Incident list. <span class="mono">cache.do</span> (admin) flushes cache when a property change seems “stuck”.</li>
  </ul>
  <p><b>Application</b> = a menu group (Incident). <b>Module</b> = an item under it (Create New, Open, All).</p>
  <p>Training writes paths like: All → Incident → Create New.</p>

  <h2>Other doors into the same house</h2>
  <p>Not everyone uses the admin UI.</p>
  """ + table(["Door", "Who"], [
    ["Next Experience lists/forms", "Fulfillers and admins"],
    ["Service Portal / Employee Center", "Employees ordering things and reading articles"],
    ["Now Mobile (iOS and Android)", "Same, on a phone"],
    ["Workspace (UI Builder)", "Agents working cases all day"],
    ["Virtual Agent", "Chat: Teams, Slack, web, mobile"],
  ]) + """
  """ + remember("The official sample: the application available to <b>all users</b> is <b>Self-Service</b>. Incident and Change need roles.") + """
  <p><b>Additional comments</b> on a ticket are visible to the caller. <b>Work notes</b> are for fulfillers. Mixing these up sends internal gossip to the customer.</p>
  <p class="footer">Domain 1 is small. If you can sketch the three UI regions and explain impersonate vs elevate, you are fine.</p>
</section>
""")

# CH 4
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 4 · domain 2 · 10%</div>
  <h1>Plugins, branding, and “make it ours”</h1>
  <p>Domain 2 is decorating the house and turning on extra rooms.</p>

  <h2>Plugins</h2>
  <p>A plugin is a bundle ServiceNow can switch on: extra tables, menus, and logic. Path: All → System Definition → Plugins.</p>
  <ul>
    <li>Many plugins an admin can activate.</li>
    <li>Some paid / store items need entitlement or Now Support.</li>
    <li>Some cannot be cleanly turned off. Always try in sub-production first.</li>
    <li>After activation you may see <b>Guided Setup</b> (empty circle = not started).</li>
  </ul>
  <p>Demo data is often a <b>separate</b> plugin. PDIs already have demo data. Real customer prod usually should not load demo incidents of “Abel’s broken laptop”.</p>
  <p><b>Studio</b> is the workshop for building a scoped app. Custom tables in global scope often start with <span class="mono">u_</span>. Scoped apps use <span class="mono">x_company_...</span>. <b>Delegated Development</b> lets someone build in Studio without full admin.</p>

  <h2>Branding and properties</h2>
  <p>Logo, browser tab title, header colors, theme = <b>customization</b> for the whole company. Put them in an update set so they travel DEV → TEST → PROD.</p>
  <p><b>System properties</b> (<span class="mono">sys_properties.list</span>) are key-value switches. Example idea: deprecate homepages so people use <b>dashboards</b> instead. After a stubborn property change, try a new session or <span class="mono">cache.do</span>.</p>
  <p>Multiple languages: activate localization plugins. Each user still picks <b>their</b> language in settings (personalization).</p>

  <h2>Homepages vs dashboards</h2>
  <p>Homepages are the old landing page. Dashboards are the current, shareable, role-based place for charts. If the exam asks how to stop people creating homepages, the answer is a <b>system property</b>, not deleting Reports.</p>
  <p class="footer">Next: the most important admin skill — who can do what.</p>
</section>
""")

# CH 5
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 5 · domain 2 (also used in domain 5)</div>
  <h1>Users, groups, and roles</h1>
  <p class="lede">If you remember only one sentence from this chapter: <span class="hl">put roles on groups, put people in groups</span>.</p>

  <h2>The school analogy</h2>
  <div class="box box-story">
    <p>A <b>user</b> is a student (or teacher). Table <span class="mono">sys_user</span>.</p>
    <p>A <b>group</b> is a class: “Service Desk”, “Network Team”. Table <span class="mono">sys_user_group</span>. Groups can contain sub-groups. A person can be in many groups.</p>
    <p>A <b>role</b> is a hall pass: “you may enter the Incident room”. Table <span class="mono">sys_user_role</span>. Roles can contain other roles, so one hall pass can include smaller passes.</p>
  </div>
  <div class="flow">ROLE (permission) → assigned to GROUP → GROUP contains USERS</div>
  """ + why("When someone joins or leaves, you add/remove them from groups. You do not hunt through 40 roles on their user record.") + """

  <h2>Users — practical rules</h2>
  <ul>
    <li>Do <b>not delete</b> people who created tickets if you can avoid it. Set <span class="mono">active = false</span> so history still makes sense.</li>
    <li>Locked out = too many failed logins. Admins can clear it.</li>
    <li>Prefer not to attach roles on the user related list. Use groups.</li>
  </ul>

  <h2>Roles you will see on the exam</h2>
  """ + table(["Role", "Plain meaning"], [
    ["<span class='mono'>admin</span>", "Almost everything. Still elevate for some ACL work."],
    ["<span class='mono'>itil</span>", "Fulfiller: incidents, problems, changes, catalog tasks."],
    ["<span class='mono'>snc_internal</span> / <span class='mono'>snc_external</span>", "Logged-in employee vs outside customer, in the explicit-role model."],
    ["<span class='mono'>approver_user</span>", "Can approve without being a full technician."],
    ["<span class='mono'>catalog_admin</span> / <span class='mono'>knowledge_admin</span>", "Own the catalog or knowledge bases."],
    ["<span class='mono'>import_admin</span>", "Import Sets."],
    ["<span class='mono'>security_admin</span>", "ACL maintenance — elevated, session only."],
    ["<span class='mono'>flow_designer</span>", "Build flows."],
    ["<span class='mono'>report_publisher</span>", "Publish reports more widely."],
    ["<span class='mono'>personalize_form</span> / <span class='mono'>personalize_list</span>", "Change layouts for a view."],
  ]) + """

  <h2>If someone “cannot see Incident”</h2>
  <p>There are three different doors:</p>
  """ + steps([
    "Do they have a <b>role on the application menu</b>?",
    "Do they have a <b>role on the module</b> (the menu item)?",
    "Do they pass the <b>table ACL</b> (security on the data itself)?",
  ]) + """
  <p>A user can fail #1 (no menu) but still open a record if they paste a URL — or the opposite. Always impersonate to test.</p>
  <p><b>Foundation data</b> (company, department, location) should be set up early. CSDM calls this Foundation. It is usually <b>not</b> a CI in the CMDB.</p>
  <p>Record numbers like INC0000042 are configured in <b>Number Maintenance</b> (prefix, digits). Not a script.</p>
  <p class="footer">You now speak “admin”. Next: the screens fulfillers live in all day.</p>
</section>
""")

# CH 6
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 6 · domain 3 · 20%</div>
  <h1>Lists, filters, and tags</h1>
  <p>A <b>list</b> is a spreadsheet of records. Each <b>row</b> is one record. Each <b>column</b> is one field.</p>

  <h2>Parts of a list</h2>
  <ul>
    <li>Title bar — name, count, search, view</li>
    <li>Breadcrumbs — the current filter, clickable like a trail of cookies</li>
    <li>Column headers — sort, filter, charts, export, configure</li>
    <li>Rows — the data. Double-click a cell to edit in place if list editing is allowed.</li>
  </ul>
  <p>Right-click menus are a CSA favourite: list title, column header, a cell, a row, and the form header are <b>different menus</b>.</p>

  <h2>Filters in one sentence</h2>
  <p>A condition is always <span class="hl">Field + Operator + Value</span>, for example Active is true, Priority is 1. The breadcrumb shows that sentence. Click an earlier crumb to drop later conditions.</p>
  <p>You can also click a cell → Show Matching / Filter Out, or type a natural-language query if the instance allows it. Advanced people use encoded queries like <span class="mono">active=true^priority=1</span>.</p>
  <p>Save a filter as Me / Group / Everyone. Drag the breadcrumb to Favorites to pin it.</p>
  """ + trap("Gear (personalize columns) is only for you. Configure → List Layout is for everyone on that view. The gear shows a green dot when you personalized.") + """
  <p><b>Tags</b> are sticky labels you put on records for ad-hoc grouping (“release week”). They are not categories (a real field) and not user criteria (who can see a catalog item).</p>
  <p>Wildcard <span class="mono">*</span> often means “contains” in list search. If the question says “search anywhere in the field”, think wildcard.</p>
  <p class="footer">Lists find work. Forms are where you do the work.</p>
</section>
""")

# CH 7
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 7 · domain 3</div>
  <h1>Forms, views, and the famous save buttons</h1>
  <p>A form is one record opened up. Header buttons, field sections, related lists at the bottom, activity stream (comments and work notes), maybe a process-flow formatter.</p>
  <p>A <b>reference field</b> looks like a name but stores a <span class="mono">sys_id</span> (a 32-character unique ID). The lookup icon opens a popup list. <b>Dot-walking</b> follows the arrow: Incident → Caller → Department → Name.</p>

  <h2>Views</h2>
  <p>A view is a named layout of the same table: Default for fulfillers, Self Service for callers, Workspace for agents. Changing Default does <b>not</b> change Self Service. If a user “cannot see a field” but can see the record, first ask: <b>wrong view?</b></p>
  <p><b>Form Layout</b> (slushbucket) and <b>Form Designer</b> (drag-and-drop canvas) write the same layout records. In designer: handle = move, remove = off the form (field still exists in the dictionary), edit = field settings. Adding a brand-new field is a data-model change.</p>
  <p><b>Related lists</b> show records that point at this one, or many-to-many links (Affected CIs on an incident).</p>

  <h2>Memorize this table cold</h2>
  """ + table(["Button", "New record?", "Form closes?"], [
    ["<b>Submit</b>", "Yes — creating", "Yes"],
    ["<b>Update</b>", "No — saving existing", "Yes"],
    ["<b>Save</b>", "No (or first insert of a new form)", "<b>No — stays open</b>"],
    ["<b>Insert</b>", "<b>Yes</b> — clones current values to a new sys_id", "Yes"],
    ["<b>Insert and Stay</b>", "Yes — new sys_id", "<b>No — stays open</b>"],
  ]) + """
  """ + trap("Insert is not Submit. Insert copies an <b>existing</b> record into a new row. Update is not Save. Update closes; Save stays.") + """
  <p><b>Templates</b> pre-fill fields for speed. They are not security. ACLs still apply.</p>
  <p>If a field must always be mandatory for everyone, use the dictionary or a Data Policy. If it is mandatory only when Priority is 1 <b>on the form</b>, use a UI Policy. If the question is about hiding data from people who should not see it, that is an <b>ACL</b>, not a UI Policy.</p>
  <p>A <b>dictionary override</b> lets a child table (Incident) change a parent field’s behaviour (from Task) without changing Problem and Change.</p>
  <p class="footer">Next: almost every process ticket is a kind of Task.</p>
</section>
""")

# CH 8
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 8 · domain 3</div>
  <h1>Tasks, assignment, SLAs, Visual Task Boards</h1>
  <p><span class="mono">task</span> is a <b>parent table</b>. Incident, Problem, Change, catalog requests and catalog tasks all <b>extend</b> it. They share number, short description, state, assignment group, assigned to, priority, work notes, comments, SLAs.</p>
  <p>If you add a field to Task, children inherit it. That is powerful and dangerous. Prefer adding fields on the child unless you truly want it everywhere.</p>

  <h2>My Work vs My Groups Work</h2>
  """ + table(["Module", "Meaning"], [
    ["<b>My Work</b>", "Active tasks assigned <b>to me</b>"],
    ["<b>My Groups Work</b>", "Assigned to <b>one of my groups</b>, usually not yet to a person"],
  ]) + remember("Official sample wording is <b>My Groups Work</b>. ‘My Teams Work’ is a fake friend.") + """

  <h2>How a ticket gets an owner</h2>
  <ul>
    <li>Someone assigns it by hand</li>
    <li><b>Assignment rules</b> (System Policy → Rules → Assignment) match conditions → group and/or user</li>
    <li>Older Incident lookup rules</li>
    <li>Predictive Intelligence (machine learning suggestions)</li>
    <li>A flow or Business Rule</li>
  </ul>
  <p>Best practice in training: assign a <b>group</b> from category/subcategory; only assign a person when you are sure.</p>

  <h2>Clocks</h2>
  """ + table(["Clock", "Between whom"], [
    ["SLA", "You and the customer"],
    ["OLA", "Internal teams"],
    ["UC (underpinning contract)", "You and a supplier"],
    ["Inactivity monitor", "Nobody touched the ticket — not a contract"],
  ]) + """
  <p>An SLA on a task <b>starts</b>, can <b>pause</b> (waiting on customer), and <b>stops</b> (resolved). Each attached clock is a row on <span class="mono">task_sla</span>.</p>

  <h2>Visual Task Boards (VTBs)</h2>
  <p>Kanban cards. Three types:</p>
  <ul>
    <li><b>Guided</b> — lanes come from a field (State). Dragging a card <b>updates that field</b>.</li>
    <li><b>Flexible</b> — mostly a field, plus extra lanes you add.</li>
    <li><b>Freeform</b> — you invent lanes. Do not assume it changes Incident State.</li>
  </ul>
  <p>The board <b>owner</b> can change lanes and members. Members work the cards. VTBs help collaboration; they do not replace security.</p>
  <p class="footer">People also need charts and emails. That is the next chapter.</p>
</section>
""")

# CH 9
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 9 · domain 3</div>
  <h1>Reports, dashboards, and notifications</h1>

  <h2>Reports</h2>
  <p>Open Reports → View / Run, or right-click a list column → bar/pie chart (fastest demo).</p>
  <p>Report Designer tabs, in order:</p>
  <div class="flow">Data → Type → Configure → Style</div>
  <ul>
    <li><b>Data</b> — name, table or data source</li>
    <li><b>Type</b> — bar, pie, line, score, …</li>
    <li><b>Configure</b> — group by, aggregation (count, average, …), filters</li>
    <li><b>Style</b> — colours and titles</li>
  </ul>
  <p>By default only the creator sees the report. The recommended way to give it to colleagues is <span class="hl"><b>Share</b></span> with users, groups, or roles. Publish makes a wider URL (be careful). Schedule emails it. Add to dashboard places it as a widget. Homepages are legacy; dashboards are current.</p>
  <p>Platform Analytics is the newer name for visualisations and KPI signals. CSA does not need Performance Analytics admin depth.</p>

  <h2>Notifications</h2>
  <p>Path: System Notification → Email → Notifications. Usually a <b>system event</b> fires (incident inserted, a flow fires an event), then three questions:</p>
  """ + table(["Tab", "Question"], [
    ["When to send", "Which table, insert/update, conditions, which event?"],
    ["Who will receive", "User, group, Assigned to, Caller, …"],
    ["What will it contain", "Subject, body, SMS, calendar invite"],
  ]) + """
  <p>Users can subscribe/unsubscribe in System Settings (personalization). Admins can force some legal mails. Channels: email, SMS, push, meeting invite, digest.</p>
  <p>The instance can also <b>receive</b> mail. An inbound email action can create an incident from a new message or update a ticket from a reply (watermark / number in the subject).</p>
  """ + trap("If mail did not arrive: did the event fire? Did conditions match? Is the user active with a good email? Is the instance allowed to send mail (often off in sub-prod)? Did they unsubscribe?") + """
</section>
""")

# CH 10
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 10 · domain 4 · 20%</div>
  <h1>Knowledge Management</h1>
  <p>Knowledge is the company’s “how do I…?” library. The goal is to answer the question <b>without opening a ticket</b>.</p>
  """ + table(["Record", "Table", "Job"], [
    ["Knowledge base", "<span class='mono'>kb_knowledge_base</span>", "A library with an owner"],
    ["Category", "<span class='mono'>kb_category</span>", "Folders inside the library"],
    ["Article", "<span class='mono'>kb_knowledge</span>", "The actual page"],
    ["User criteria", "<span class='mono'>user_criteria</span>", "Who may read or write"],
  ]) + """
  <p><b>Can Read</b> and <b>Can Contribute</b> are different keys. A technician might write IT articles but only read HR articles.</p>
  <p>User criteria can match users, groups, roles, company, location, or a script. If an article is invisible, check published state <b>and</b> user criteria. Impersonate an employee with no itil role.</p>
  <h2>Life of an article</h2>
  <div class="flow">Draft → Review → Published (searchable) → Retired</div>
  <p>Users search <b>published</b> articles. Retired ones stay for history. Publishing may create a new version.</p>
  """ + table(["Need", "Use"], [
    ["How do I reset VPN?", "Knowledge article"],
    ["I want a laptop", "Catalog item"],
    ["VPN is down for me right now", "Incident (often a record producer)"],
  ]) + """
  <p>A record producer can still show knowledge suggestions before submit (deflection).</p>
</section>
""")

# CH 11
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 11 · domain 4</div>
  <h1>Service Catalog — shopping for work</h1>
  <p>The catalog is the ordering system. Hardware, software, access, and often “report an issue” forms live here.</p>

  <h2>The family tree (draw this in the exam)</h2>
  <p>When someone orders a normal <b>catalog item</b> (not a record producer):</p>
  <div class="flow">REQ (sc_request) — the shopping cart checkout<br>→ RITM (sc_req_item) — one per item in the cart<br>→ SCTASK (sc_task) — work for fulfiller groups</div>
  <p>One checkout = one REQ. Two laptops in the cart = two RITMs under that REQ. Each RITM can have several tasks (hardware team, software team).</p>
  """ + trap("A <b>record producer</b> skips this tree. It creates a row on a target table (often Incident). No REQ unless you build that yourself.") + """

  <h2>Item types</h2>
  """ + table(["Type", "What it does"], [
    ["Catalog Item", "Normal order → REQ / RITM / SCTASK"],
    ["Record Producer", "Creates a record on another table"],
    ["Order Guide", "One form that orders <b>several</b> items (new joiner: laptop + phone + access)"],
    ["Content Item", "A link or knowledge — not an order"],
  ]) + remember("Official sample: Multiple Choice, Single Line Text, Select Box are <b>variable types</b> — questions on the item — not order guides.") + """
  <p>A <b>variable set</b> is a reusable bundle of questions (Requested for + Location). Change once, many items update.</p>
  <p>After submit, a <b>Flow</b> should approve, create catalog tasks, and close the RITM. If you forget the flow, you get a request that sits there looking sad.</p>
  <p>Who sees the item? Active flags + <b>user criteria</b> on catalog, category, and item. Always test as a user with no itil role.</p>
</section>
""")

# CH 12
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 12 · domain 4</div>
  <h1>Workflow Studio and Virtual Agent</h1>
  <p>On the January 2026 blueprint the umbrella name is <b>Workflow Studio</b>.</p>
  """ + table(["Tool", "Job"], [
    ["Flow Designer", "When X happens, do Y. Can wait."],
    ["Process Automation Designer / playbooks", "Guided stages for agents (qualify → investigate → resolve)"],
    ["Workflow Editor", "Old drawing-board workflows. Maintain old ones; do not start new work here if Flow can do it."],
  ]) + """
  <p>A flow = <b>trigger</b> + <b>actions</b> (if, for each, wait, create record, ask approval, create catalog task, send email). A <b>spoke</b> is a pack of actions (ServiceNow Core, Slack, Teams, REST). A <b>subflow</b> is a reusable mini-flow with no trigger of its own — like a function.</p>
  <p>Triggers: record changes, a date, Service Catalog, inbound email, SLA task.</p>
  <p>Roles: <span class="mono">flow_designer</span> builds, <span class="mono">action_designer</span> builds actions, <span class="mono">flow_operator</span> watches executions. If a catalog item “does nothing”, open today’s flow executions.</p>
  """ + remember("A Flow can <b>wait</b> two days for a manager to approve. A Business Rule must not wait — it runs in the same database save.") + """

  <h2>Virtual Agent</h2>
  <p>The chat bot for employees. It searches <b>Knowledge</b>, can start <b>catalog</b> orders, checks ticket status, and can hand off to a live person. It appears on the portal, Now Mobile, Microsoft Teams, Slack, and web chat.</p>
  <p>You will not build a full language model in the CSA exam. You must know VA is a self-service channel that sits on knowledge + catalog + records.</p>
  <p class="footer">You can now describe the employee journey. Next is the biggest domain: data and locks.</p>
</section>
""")

# CH 13
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 13 · domain 5 · 30% starts here</div>
  <h1>Tables, fields, and relationships</h1>
  <p class="lede">Everything in ServiceNow is a row in a table. Incidents, users, even Business Rules. Once you see that, the platform gets quieter.</p>

  <h2>Kinds of tables</h2>
  """ + table(["Kind", "Example"], [
    ["Base (others extend it)", "<span class='mono'>task</span>, <span class='mono'>cmdb</span>"],
    ["Child / extended", "<span class='mono'>incident</span> <b>is a</b> task"],
    ["Custom global", "<span class='mono'>u_loaner</span>"],
    ["Scoped app", "<span class='mono'>x_acme_loan_loaner</span>"],
  ]) + """
  <p>Label = Incident. Name = <span class="mono">incident</span>. The exam loves database names: <span class="mono">sys_user</span>, <span class="mono">task_sla</span>, <span class="mono">cmdb_ci</span>.</p>
  <p>Every record has a <span class="hl"><b>sys_id</b></span>: 32 characters, globally unique. INC0001234 is a friendly number from Number Maintenance. The real primary key is sys_id.</p>

  <h2>Three ways tables connect</h2>
  <ol class="steps">
    <li><b>Extension</b> — “is a”. Incident is a Task. Child inherits fields. <span class="mono">sys_class_name</span> remembers the real class (important in CMDB).</li>
    <li><b>One-to-many</b> — a <b>reference field</b> on the many side. Many incidents point at one caller. On the user form, a related list can show that user’s incidents.</li>
    <li><b>Many-to-many</b> — a joining table with two references, for example incidents ↔ CIs (Affected CIs).</li>
  </ol>
  <p>The <b>Schema Map</b> (from Tables &amp; Columns) draws these links. Use it when a question says “visualize table relationships”.</p>
  <p>The <b>Dictionary</b> (<span class="mono">sys_dictionary</span>) is the ID card of every field: type, length, mandatory, default, reference table. A <b>dictionary override</b> changes a parent field only on a child table.</p>
  <p><b>Dot-walking</b> follows references in filters, notifications, and scripts: <span class="mono">caller_id.department.name</span>. In the condition builder a green plus expands the reference.</p>
  <p>Application <b>scope</b> is a fence around an app’s files so random global scripts do not wander in. User security is still ACLs, next chapter.</p>
</section>
""")

# CH 14
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 14 · domain 5 · the lock</div>
  <h1>Access Control Lists</h1>
  <p>An ACL is a rule that answers: <b>who may do this operation on this object?</b> Operations: create, read, write (update), delete, plus extras like execute.</p>
  """ + why("UI Policy can hide a field on a form. A clever person or an import can still touch the data. ACL is the real lock.") + """

  <h2>The three names</h2>
  """ + table(["ACL name", "What it locks"], [
    ["<span class='mono'>incident</span> or <span class='mono'>incident.None</span>", "The whole row — may I see this ticket at all?"],
    ["<span class='mono'>incident.*</span>", "Fields in general"],
    ["<span class='mono'>incident.work_notes</span>", "That one field"],
  ]) + """
  <p>You must pass the <b>table/row</b> ACL first. If that fails, field ACLs cannot save you. If table read passes but work_notes read fails, the incident opens and that field is blank.</p>

  <h2>AND vs OR — write this on scrap paper</h2>
  <p>Inside <b>one</b> ACL record there are up to three parts: roles, a condition, a script. All configured parts must pass. That is <span class="hl">AND</span>. Empty parts are skipped.</p>
  <p>If <b>several</b> ACL records match the same object and operation, passing <b>any one</b> of them typically grants access. That is <span class="hl">OR</span>.</p>
  <p>More specific names win: <span class="mono">table.field</span> beats <span class="mono">table.*</span>.</p>
  <p>On modern High Security instances, if nothing grants access, the answer is <b>deny</b>.</p>
  """ + steps([
    "Elevate to <span class='mono'>security_admin</span> (session only).",
    "Open System Security → Access Control, or the table’s Access Controls related list.",
    "Create the rule, then <b>impersonate</b> a normal user and prove it.",
  ]) + trap("admin override on an ACL can let admins bypass it. Clearing that checkbox is how you lock admins out of a sensitive field — rare, but know the checkbox exists.") + """
  <p>Authentication = who are you (password/SSO). Authorization = what may you do (roles + ACLs). Query Business Rules can also hide rows silently — another thing to check when “the list is empty”.</p>
</section>
""")

# CH 15
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 15 · domain 5</div>
  <h1>Importing data without making a mess</h1>
  <p>Do not type 5,000 users by hand. Use <b>Import Sets</b>.</p>
  <div class="flow">File or data source → staging table (parking lot) → Transform Map (the translator) → target table (sys_user, cmdb_ci, …)</div>
  <p>Load Data: Excel, CSV, XML, or a repeating Data Source (JDBC, LDAP, HTTP). Roles: admin or import_admin.</p>
  """ + remember("Official definition: a <b>transform map</b> maps Import Set fields to fields on an <b>existing</b> table.") + """
  <p>If column names already match, Automatic Mapping helps. If not, Mapping Assist (drag fields). Scripts can set extra fields.</p>
  <h2>Coalesce — the most important import word</h2>
  <p>Mark a unique key (email, employee number, serial number) as <b>coalesce</b>:</p>
  <ul>
    <li>Match found → <b>update</b> that person/CI</li>
    <li>No match → insert a new row</li>
    <li>Several coalesce fields → all must match (AND)</li>
    <li>No coalesce → every load <b>inserts duplicates</b></li>
  </ul>
  """ + example("Load users twice with coalesce on email. Second load only changes names. Load without coalesce and you get twins.") + trap("UI Policies do not run during import. If a field must be mandatory even from Excel, use a <b>Data Policy</b> (or dictionary mandatory).") + """
  <p>Update sets move configuration, not 10,000 user rows. Clone copies a whole instance. Import Sets move data.</p>
</section>
""")

# CH 16
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 16 · domain 5</div>
  <h1>CMDB and CSDM — the map of the company’s IT</h1>
  <p>CMDB = Configuration Management Database. It stores <b>Configuration Items (CIs)</b> — the things you manage — and <b>how they connect</b>.</p>
  <p>CIs can be physical (servers, laptops, routers) or logical (a business service, an application service, a database).</p>
  """ + why("When a server dies, you want to know which business service, which customers, which changes. Incidents and changes should point at CIs. That is impact analysis.") + """
  """ + table(["Table", "Job"], [
    ["<span class='mono'>cmdb</span> / <span class='mono'>cmdb_ci</span>", "CI classes"],
    ["<span class='mono'>cmdb_rel_ci</span>", "This CI depends on / runs on / uses that CI"],
  ]) + """
  <p><b>CI Class Manager</b> is the modern place to see the class tree. <b>Dependency Views</b> draw a map. <b>Suggested Relationships</b> say which relationship types make sense between two classes.</p>

  <h2>How data gets in</h2>
  <ul>
    <li><b>Discovery</b> — bottom-up: find devices on the network. Needs a <b>MID Server</b> (Management, Instrumentation, and Discovery): a small service <b>inside your network</b> that talks out to the instance.</li>
    <li><b>Service Mapping</b> — top-down: start from a website and walk down to the servers.</li>
    <li>Connectors / IRE (Identification and Reconciliation) — stop duplicate CIs.</li>
    <li>Manual forms and Import Sets — still exist, but identity rules matter.</li>
  </ul>

  <h2>CSDM in plain words</h2>
  <p>CSDM (Common Service Data Model) is not “another product you buy instead of ITSM”. It is ServiceNow’s <b>agreed vocabulary and table map</b> so everyone stores services the same way.</p>
  <ul>
    <li><b>Foundation</b> — users, groups, locations, companies. Referenced everywhere. Usually not linked with CI relationship records.</li>
    <li><b>Design</b> — Business Application: the idea of “Payroll App” in architecture. Not the running server.</li>
    <li><b>Build</b> — software-development objects.</li>
    <li><b>Manage technical services</b> — Application Service (a deployed running instance) plus infrastructure CIs Discovery finds. This is what often breaks at 3 a.m.</li>
    <li><b>Sell / Consume</b> — Business Service and offerings: what the business sells or uses. The <b>request catalog is not a CMDB table</b>.</li>
    <li><b>Manage Portfolio</b> — service owners looking across the above.</li>
  </ul>
  """ + trap("Business Application (design) is not the same as Application Service (running instance). The catalog is not stored as cmdb_rel_ci.") + """
</section>
""")

# CH 17
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 17 · domain 5</div>
  <h1>Security Center and shared responsibility</h1>
  <p><b>Security Center</b> is a dashboard about the <b>health of the instance’s own security settings</b>: hardening checks, risky properties, recommendations. It is not Security Incident Response (hacking cases), and it is not a list of firewall CIs in the CMDB.</p>
  <p>If the question is “where do admins review instance security posture in one place?”, Security Center.</p>

  <h2>Shared Responsibility — the cloud split</h2>
  <p>ServiceNow runs a cloud platform. They cannot log in and design your HR ACLs for you. You cannot fly to their data centre and patch a cable.</p>
  """ + table(["ServiceNow typically owns", "You (the customer) typically own"], [
    ["Physical data centres and hardware", "Who gets admin / security_admin"],
    ["Keeping the platform software running", "Users, groups, roles, ACLs"],
    ["Instance isolation (multi-instance)", "What data you put in the instance"],
    ["Offering Security Center and guidance", "Acting on those findings"],
    ["Cloud operations", "MID Servers inside <b>your</b> network, integrations, public pages you published"],
  ]) + remember("Who is responsible for designing ACLs that protect HR data? <b>The customer’s admins.</b> Who is responsible for physical security of the data centre? <b>ServiceNow.</b>") + """
</section>
""")

# CH 18
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 18 · domain 6 · 13%</div>
  <h1>UI Policy, Data Policy, Business Rules, and scripts</h1>
  <p>The exam loves “which tool?”. Pick the <b>simplest tool that actually enforces the need</b>.</p>

  <h2>Decision tree in slow motion</h2>
  """ + table(["If you need…", "Use"], [
    ["Mandatory / visible / read-only only on the <b>form</b>", "<b>UI Policy</b> (browser, no code)"],
    ["The same rule even from Import Set, API, list edit", "<b>Data Policy</b> (server)"],
    ["Wait for a manager for two days", "<b>Flow</b>"],
    ["Set a field or stop a save in the same click", "<b>before Business Rule</b>"],
    ["Slow extra work after save", "async Business Rule or a Flow"],
    ["Fancy onChange behaviour", "Client Script (onLoad / onChange / onSubmit / onCellEdit)"],
    ["Reusable server function / Ajax", "Script Include"],
    ["Stop people who should not see data", "<b>ACL</b>"],
  ]) + """

  <h2>UI Policy</h2>
  <p>No-code. When a condition is true, make fields mandatory, visible, or read-only. Usually reverse if false. Runs in the browser. Not security.</p>

  <h2>Data Policy</h2>
  <p>Looks similar (mandatory / read-only) but runs on the <b>server</b>, so imports and web services must obey too. Can also show as a UI Policy on the form.</p>

  <h2>Business Rules</h2>
  <p>Server logic when a record is inserted, updated, deleted, or queried. Table <span class="mono">sys_script</span>.</p>
  """ + table(["When", "In simple words"], [
    ["before", "While saving, before it is committed. Set fields, abort, or (on query) filter rows."],
    ["after", "Right after this record saved. Touch other records. Keep it light."],
    ["async", "Queued later. Good for slow work."],
    ["display", "When the form loads. Put notes in <span class='mono'>g_scratchpad</span> for Client Scripts."],
  ]) + """
  <p>Scripts use <span class="mono">current</span> (this record) and <span class="mono">previous</span> (old values on update). Language of the platform: <span class="hl"><b>JavaScript</b></span> — official sample. Not Java. Not Jelly.</p>

  <h2>Client vs server — two rooms</h2>
  <p><b>Browser:</b> Client Scripts, UI Policies, <span class="mono">g_form</span>, <span class="mono">g_user</span>, <span class="mono">g_scratchpad</span>.</p>
  <p><b>Server:</b> Business Rules, Script Includes, <span class="mono">gs</span>, <span class="mono">GlideRecord</span>, <span class="mono">current</span>.</p>
  """ + trap("GlideRecord inside a Client Script makes a slow round-trip to the server. Use GlideAjax + a Script Include instead.") + """
</section>
""")

# CH 19
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 19 · domain 6</div>
  <h1>Update sets — moving your work</h1>
  <p>An update set is a suitcase of <b>configuration changes</b>: new fields, Business Rules, form layouts, catalog items, ACLs, many properties.</p>
  <p>It is usually <b>not</b> a suitcase of data: not last year’s incidents, not 4,000 users, not CMDB servers. For data, use Import Sets (or clone a whole instance).</p>
  <div class="flow">DEV: create a <b>named</b> set → make it current → do the work → mark Complete<br>TEST/PROD: retrieve or import XML → <b>Preview</b> → fix errors → <b>Commit</b></div>
  """ + trap("Never develop in the Default update set and then migrate Default as your release. If you forgot, move the customer updates into a named set first. Always Preview before Commit.") + """
  <p>Customer Updates related list shows what was captured. If something is missing, you were in the wrong set, or that object is data (not captured).</p>
  <p>Scoped apps can also move as app versions / Git. CSA still tests classic update set preview and commit.</p>
  <p>A committed set can sometimes be backed out. That is an emergency tyre, not a process.</p>
</section>
""")

# CH 20
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 20</div>
  <h1>Putting it all together</h1>
  <h2>One employee story</h2>
  <div class="box box-story">
    <p>Maya’s laptop is slow. She opens Employee Center or Virtual Agent.</p>
    <p>A <b>knowledge</b> article says “restart, then order a loaner if still bad”. Still bad. She orders <b>catalog item</b> “Loaner laptop”. Variables ask size and reason.</p>
    <p>ServiceNow creates REQ + RITM. A <b>flow</b> asks her manager to approve, then creates an SCTASK for the Service Desk group (assignment rule could also help). A <b>notification</b> emails the group.</p>
    <p>The technician works My Groups Work, maybe a Guided VTB by State. They link a <b>CI</b> (the laptop in the CMDB). An <b>SLA</b> ticks. Comments go to Maya; work notes stay internal.</p>
    <p>You, the admin, built this in DEV inside a named <b>update set</b>, previewed it in TEST, committed to PROD. You tested by <b>impersonating</b> Maya (no itil role) and a technician.</p>
  </div>
  <p>If Maya should not see HR articles, that is <b>user criteria</b>. If a contractor should not see work notes, that is an <b>ACL</b>. If justification is mandatory only when “replacement” is selected, that is a <b>catalog UI policy</b>. If justification must exist even when someone imports requests, that is a <b>data policy</b>.</p>

  <h2>Exam method</h2>
  <ul>
    <li>60 questions, 90 minutes, no half marks on “select 3”.</li>
    <li>First pass fast. Flag. Second pass for ACL / coalesce / save buttons.</li>
    <li>When two answers could work, pick the most specific platform feature (UI Policy before Client Script; Data Policy if it must hold on the server; Flow if it must wait; ACL if it is access).</li>
    <li>Confirm the exam <b>release</b> on docs.servicenow.com.</li>
  </ul>
  <p>Keep the cert alive with the annual delta exam. Official courses: Welcome to ServiceNow + ServiceNow Administration Fundamentals.</p>
</section>
""")

# CH 21
parts.append("""
<section class="chapter">
  <div class="kicker">chapter 21</div>
  <h1>Practice questions — explained like a teacher</h1>
  <p>These are original drills, not leaked live items. The five official blueprint samples are included first.</p>

  <h3>Official 1. Which application is available to all users?</h3>
  <p><b>Self-Service.</b> People with no fulfiller role still get a place to read knowledge and order from the catalog. Incident/Change need roles.</p>
  <h3>Official 2. Tasks for my group, not yet for a person?</h3>
  <p><b>My Groups Work.</b> My Work = assigned to me.</p>
  <h3>Official 3. Transform map?</h3>
  <p>Maps Import Set (staging) fields to an existing target table.</p>
  <h3>Official 4. Multiple Choice, Single Line Text, Select Box?</h3>
  <p><b>Variable types</b> on a catalog item.</p>
  <h3>Official 5. Scripting language?</h3>
  <p><b>JavaScript.</b></p>

  <h3>More drills (answer then reason)</h3>
  <ol class="steps">
    <li><b>Multi-instance</b> — each customer has their own DB, not one shared dormitory database.</li>
    <li><b>Impersonate</b> to test as an employee. Elevate is for security_admin on your session.</li>
    <li>Roles on <b>groups</b>, users in groups.</li>
    <li>Gear = personalization. Configure layout = customization for the view.</li>
    <li>Insert and Stay = new sys_id, form stays open.</li>
    <li>Filter = field + operator + value.</li>
    <li>Guided VTB lanes follow a field; dragging updates it. Freeform does not reliably change Incident State.</li>
    <li>Share is the recommended way to give someone a report.</li>
    <li>Notifications: When / Who / What, usually from an event.</li>
    <li>Two catalog items in one cart = one REQ, two RITMs.</li>
    <li>Record producer writes a target table; no REQ chain by default.</li>
    <li>Flows wait; Business Rules do not.</li>
    <li>sys_id is 32 characters. INC numbers are labels.</li>
    <li>Incident extends Task = parent-child extension, not many-to-many.</li>
    <li>One ACL’s parts = AND. Several ACLs at the same spot = typically OR. Table then field.</li>
    <li>No ACL + default deny = denied.</li>
    <li>Coalesce on email = second import updates, does not twin people.</li>
    <li>cmdb_ci and cmdb_rel_ci are CMDB. sc_request is catalog.</li>
    <li>CSDM = common model/vocabulary. Discovery needs MID in your network.</li>
    <li>Customer owns ACL design. ServiceNow owns the data centre.</li>
    <li>Priority 1 makes Description mandatory on the form = UI Policy. Same rule on import = Data Policy.</li>
    <li>display Business Rule fills g_scratchpad for the client.</li>
    <li>Update sets = configuration. Preview then Commit. Named set, not Default.</li>
    <li>g_form is client. UI Policy is not a lock. Catalog is not a CMDB table. Hidden module but URL works = module role missing, not table ACL.</li>
  </ol>
  <p class="hand" style="font-size:22pt;color:#6d28d9;">You can explain the platform now — not just tick boxes. Go and click on a PDI.</p>
  <p class="footer">End of the Easy Book. Pair with the 15-page visual PDF for last-week skim, and the markdown workbook for extra labs.</p>
</section>
</div>
</body>
</html>
""")

OUT.write_text("".join(parts), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size)
