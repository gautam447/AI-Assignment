"""BI Publisher-style outputs: bursting CSV, HTML registers, XML data model dump."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def write_xml_data_model(rows: list[dict[str, Any]], root_name: str, row_name: str, path: Path) -> None:
    root = Element(root_name)
    root.set("generated", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    for row in rows:
        node = SubElement(root, row_name)
        for key, value in row.items():
            child = SubElement(node, key)
            child.text = "" if value is None else str(value)
    path.write_bytes(tostring(root, encoding="utf-8", xml_declaration=True))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bursting_file(rows: list[dict[str, Any]], key: str, email_field: str, path: Path) -> None:
    """Mimic BIP bursting query output: KEY, OUTPUT_NAME, EMAIL, DEL_CHANNEL."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["KEY", "TEMPLATE", "OUTPUT_FORMAT", "OUTPUT_NAME", "DEL_CHANNEL", "PARAMETER1"],
        )
        writer.writeheader()
        for group_key, items in grouped.items():
            email = items[0].get(email_field, "finance.controllers@aether.example")
            if group_key.endswith("BU"):
                email = f"controller.{group_key.split()[1].lower()}@aether.example"
            writer.writerow(
                {
                    "KEY": group_key,
                    "TEMPLATE": "PAYROLL_REGISTER",
                    "OUTPUT_FORMAT": "PDF",
                    "OUTPUT_NAME": f"Payroll_Register_{group_key.replace(' ', '_')}_Jul-26",
                    "DEL_CHANNEL": "EMAIL",
                    "PARAMETER1": email,
                }
            )


def html_report(title: str, subtitle: str, rows: list[dict[str, Any]], path: Path, money_cols: set[str] | None = None) -> None:
    money_cols = money_cols or set()
    headers = list(rows[0].keys()) if rows else []
    body_rows = []
    for row in rows[:80]:
        tds = []
        for key in headers:
            value = row[key]
            if key in money_cols and isinstance(value, (int, float)):
                tds.append(f"<td class='num'>{value:,.2f}</td>")
            else:
                tds.append(f"<td>{_esc(value)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{_esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{_esc(title)}</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 32px; color: #1b2430; }}
    .banner {{ display:flex; justify-content:space-between; border-bottom: 3px solid #1f4e79; padding-bottom: 12px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    .meta {{ color: #5b6770; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 12px; }}
    th {{ background: #1f4e79; color: #fff; text-align: left; padding: 8px; }}
    td {{ border-bottom: 1px solid #d9e2ec; padding: 7px 8px; }}
    tr:nth-child(even) td {{ background: #f4f7fb; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{ margin-top: 16px; font-size: 12px; color: #5b6770; }}
  </style>
</head>
<body>
  <div class="banner">
    <div>
      <h1>{_esc(title)}</h1>
      <div class="meta">{_esc(subtitle)}</div>
    </div>
    <div class="meta">Aether Global Industries<br/>Confidential – interview demo</div>
  </div>
  {table}
  <p class="note">Layout stands in for an RTF/XPT template. In Fusion this is bound to a BIP data model, then burst by KEY.</p>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_interview_pack(kpis: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(kpis, indent=2), encoding="utf-8")
