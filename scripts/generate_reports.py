#!/usr/bin/env python3
"""Generate MD, HTML, and PDF for NanoServe documentation reports."""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "documentation"
REPORTS = DOC / "reports"

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None  # type: ignore


def md_to_html_body(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    in_pre = False
    in_table = False
    table_rows: list[str] = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            return
        out.append("<table>")
        for i, row in enumerate(table_rows):
            cells = [c.strip() for c in row.strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        out.append("</table>")
        table_rows = []
        in_table = False

    for line in lines:
        if line.startswith("```"):
            if in_pre:
                out.append("</pre>")
                in_pre = False
            else:
                flush_table()
                lang = line[3:].strip()
                out.append(f'<pre class="code"><code class="{html.escape(lang)}">')
                in_pre = True
            continue
        if in_pre:
            out.append(html.escape(line))
            continue
        if line.startswith("|") and "|" in line[1:]:
            if not in_table:
                in_table = True
            if re.match(r"^\|[\s\-:|]+\|$", line):
                continue
            table_rows.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:].strip())}</h3>")
        elif line.strip() == "---":
            out.append("<hr/>")
        elif line.strip():
            text = html.escape(line)
            text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
            text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
            out.append(f"<p>{text}</p>")
        else:
            out.append("<br/>")
    flush_table()
    if in_pre:
        out.append("</pre>")
    return "\n".join(out)


def wrap_html(title: str, body: str, nav: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)} — NanoServe</title>
  <style>
    :root {{ --bg:#0f1419; --fg:#e7ecf1; --accent:#3b82f6; --muted:#94a3b8; --card:#1a2332; }}
    body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--fg);
            max-width: 920px; margin: 0 auto; padding: 2rem 1.5rem; line-height: 1.6; }}
    a {{ color: var(--accent); }}
    h1 {{ border-bottom: 2px solid var(--accent); padding-bottom: .5rem; }}
    h2 {{ margin-top: 2rem; color: #93c5fd; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #334155; padding: .5rem .75rem; text-align: left; }}
    th {{ background: var(--card); }}
    pre.code {{ background: var(--card); padding: 1rem; overflow-x: auto; border-radius: 8px;
                 font-size: .9rem; border: 1px solid #334155; }}
    nav.docs {{ background: var(--card); padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }}
    nav.docs a {{ margin-right: 1rem; }}
    .badge {{ display: inline-block; background: #166534; color: #bbf7d0; padding: .2rem .6rem;
              border-radius: 4px; font-size: .85rem; }}
  </style>
</head>
<body>
  <nav class="docs">{nav}</nav>
  {body}
  <footer><p style="color:var(--muted);font-size:.85rem;margin-top:3rem">
    Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} — NanoServe documentation
  </p></footer>
</body>
</html>"""


NAV = """
<a href="../index.html">Docs home</a>
<a href="../SETUP.html">Setup</a>
<a href="../USAGE.html">Usage</a>
<a href="FULL_TEST_REPORT.html">Tests</a>
<a href="STRESS_REPORT.html">Stress</a>
<a href="VALGRIND_REPORT.html">Valgrind</a>
"""


def md_to_pdf(md_path: Path, pdf_path: Path, title: str) -> None:
    if FPDF is None:
        print(f"[!] skip PDF {pdf_path.name} (pip install fpdf2)")
        return

    def sanitize(s: str) -> str:
        return (
            s.replace("\u2014", "-")
            .replace("\u2192", "->")
            .replace("|", " ")
            .encode("latin-1", errors="replace")
            .decode("latin-1")
        )

    text = md_path.read_text(encoding="utf-8")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    W = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(W, 10, sanitize(title))
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    for line in text.splitlines():
        if line.startswith("```") or line.strip() == "---":
            continue
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(W, 8, sanitize(line[2:].strip()))
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(W, 7, sanitize(line[3:].strip()))
            pdf.set_font("Helvetica", size=10)
        elif line.startswith("|"):
            continue
        else:
            s = sanitize(line)
            if s.strip():
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(W, 5, s[:120])
    pdf.output(str(pdf_path))
    print(f"[+] {pdf_path}")


def emit(md_path: Path) -> None:
    md = md_path.read_text(encoding="utf-8")
    stem = md_path.stem
    title_match = re.search(r"^# (.+)$", md, re.M)
    title = title_match.group(1) if title_match else stem
    html_path = md_path.with_suffix(".html")
    pdf_path = md_path.with_suffix(".pdf")
    html_path.write_text(wrap_html(title, md_to_html_body(md), NAV), encoding="utf-8")
    print(f"[+] {html_path}")
    md_to_pdf(md_path, pdf_path, title)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name in ("FULL_TEST_REPORT", "STRESS_REPORT", "VALGRIND_REPORT"):
        p = REPORTS / f"{name}.md"
        if p.exists():
            emit(p)
    for name in ("SETUP", "USAGE", "REQUIREMENTS", "SCALING"):
        p = DOC / f"{name}.md"
        if p.exists():
            emit(p)
    # index
    idx = DOC / "index.html"
    idx.write_text(
        wrap_html(
            "NanoServe Documentation",
            """<h1>NanoServe Documentation</h1>
<p class="badge">C++23 engine · Python SDK · FastAPI · 300-user native scale</p>
<h2>Getting started</h2>
<ul>
<li><a href="REQUIREMENTS.html">Prior requirements (non-Docker)</a></li>
<li><a href="SETUP.html">Setup guide</a></li>
<li><a href="USAGE.html">How to use</a></li>
<li><a href="SCALING.html">Scaling (150–300 users)</a></li>
</ul>
<h2>Reports</h2>
<ul>
<li><a href="reports/FULL_TEST_REPORT.html">Full test report</a> (MD / HTML / PDF)</li>
<li><a href="reports/STRESS_REPORT.html">User stress report</a></li>
<li><a href="reports/VALGRIND_REPORT.html">Valgrind memory report</a></li>
</ul>""",
            "",
        ),
        encoding="utf-8",
    )
    print(f"[+] {idx}")


if __name__ == "__main__":
    main()
