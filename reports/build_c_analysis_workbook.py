"""Verify the sentence analysis against its DOCX source and export its tables.

This script only reads the source DOCX, report Markdown, and attachment workbooks.
It writes C题_逐句解析.xlsx beside this script; it never fills competition templates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from zipfile import ZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / "C题.docx"
REPORT = HERE / "C题_逐句解析.md"
OUTPUT = HERE / "C题_逐句解析.xlsx"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
CATEGORIES = ["背景信息", "核心子问题", "已知条件", "参数数据", "约束规则", "行业/物理限定"]


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text)


def docx_contents() -> tuple[list[str], list[list[list[str]]]]:
    with ZipFile(SOURCE) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", NS)
    assert body is not None

    def text_of(node: ET.Element) -> str:
        return "".join(part.text or "" for part in node.iter() if part.tag.rsplit("}", 1)[-1] == "t")

    paragraphs = [text_of(p).strip() for p in body.iterfind(".//w:p", NS) if text_of(p).strip()]
    tables = [
        [[text_of(cell).strip() for cell in row.findall("w:tc", NS)] for row in table.findall("w:tr", NS)]
        for table in body.findall("w:tbl", NS)
    ]
    return paragraphs, tables


def markdown_tables(text: str) -> list[tuple[str, list[list[str]]]]:
    result: list[tuple[str, list[list[str]]]] = []
    heading = ""
    current: list[list[str]] = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip()[1:-1])]
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                current.append(cells)
            continue
        if current:
            expected = len(current[0])
            if any(len(row) != expected for row in current):
                raise ValueError(f"Inconsistent table width under {heading}")
            result.append((heading, current))
            current = []
        if line.startswith("#"):
            heading = line.lstrip("# ")
    return result


def locations(location: str) -> list[int]:
    numbers = [int(number) for number in re.findall(r"P(\d+)", location)]
    return list(range(numbers[0], numbers[-1] + 1))


def verify_source(paragraphs: list[str], statements: list[list[str]], table_rows: list[list[str]]) -> dict:
    assert [row[0] for row in statements] == [f"S{i:02d}" for i in range(1, 53)]
    assert [row[0] for row in table_rows] == [f"T{i:02d}" for i in range(1, 5)]
    clean_paragraphs = [normalized(p) for p in paragraphs]
    coverage = [set() for _ in paragraphs]
    failures = []
    for statement in statements:
        identifier, location, original, categories, *_ = statement
        unknown = set(categories.split("；")) - set(CATEGORIES)
        if unknown:
            failures.append(f"{identifier}: unknown categories {unknown}")
        refs = locations(location)
        joined = "".join(clean_paragraphs[index - 1] for index in refs)
        quote = normalized(original)
        start = joined.find(quote)
        if start < 0:
            failures.append(f"{identifier}: quote not found in {location}: {original}")
            continue
        end = start + len(quote)
        offset = 0
        for index in refs:
            length = len(clean_paragraphs[index - 1])
            lo, hi = max(start, offset), min(end, offset + length)
            coverage[index - 1].update(range(max(0, lo - offset), max(0, hi - offset)))
            offset += length
    for row in table_rows:
        for index in locations(row[1]):
            coverage[index - 1].update(range(len(clean_paragraphs[index - 1])))
    for index, paragraph in enumerate(clean_paragraphs):
        missing = "".join(char for position, char in enumerate(paragraph) if position not in coverage[index])
        if missing:
            failures.append(f"P{index + 1:02d}: uncovered text: {missing}")
    if failures:
        raise ValueError("\n".join(failures))
    days = (date(2025, 12, 31) - date(2025, 2, 1)).days + 1
    assert days == 334 and days * 144 == 48096
    return {
        "source_nonempty_paragraphs": len(paragraphs),
        "source_statements_verified": len(statements),
        "source_tables_covered": len(table_rows),
        "source_text_coverage": "100% after whitespace normalization; original tables retained separately",
        "output_days": days,
        "output_10min_intervals": days * 144,
    }


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def add_sheet(wb: Workbook, title: str, rows: list[list[str]], widths: list[float]) -> None:
    ws = wb.create_sheet(title)
    for row in rows:
        ws.append([str(value) for value in row])
    for column, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(column)].width = width
    navy, teal = "17324D", "E5F2F0"
    thin = Side(style="hair", color="DDE4EB")
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name="Microsoft YaHei", size=10, color="203347")
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=thin)
            if cell.row == 1:
                cell.fill = PatternFill("solid", fgColor=navy)
                cell.font = Font(name="Microsoft YaHei", size=10, bold=True, color="FFFFFF")
            elif cell.row % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F2F6FA")
            if cell.value == "✓":
                cell.fill = PatternFill("solid", fgColor=teal)
                cell.font = Font(name="Microsoft YaHei", size=12, bold=True, color="16715E")
                cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 32
    for index, row in enumerate(rows[1:], 2):
        lines = max(math.ceil(display_width(str(value)) / max(widths[col] - 3, 1)) for col, value in enumerate(row))
        ws.row_dimensions[index].height = min(390, max(34, lines * 16 + 12))
    ws.freeze_panes = "C2" if title == "逐句六类划分" else "B2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 80 if title == "逐句六类划分" else 90
    ws.print_title_rows = "1:1"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.oddFooter.center.text = "第 &P 页 / 共 &N 页"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    markdown = REPORT.read_text(encoding="utf-8")
    tables = markdown_tables(markdown)
    statements = [row for _, rows in tables for row in rows[1:] if re.fullmatch(r"S\d{2}", row[0])]
    original_tables = [row for _, rows in tables for row in rows[1:] if re.fullmatch(r"T\d{2}", row[0])]
    paragraphs, raw_tables = docx_contents()
    assert len(paragraphs) == 93 and len(raw_tables) == 4
    verification = verify_source(paragraphs, statements, original_tables)
    if markdown.count("\n$$\n") % 2:
        raise ValueError("Unpaired display-math delimiters")

    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = "C题逐句解析与六类划分"
    wb.properties.subject = "微网与外部电网电力调控策略：原文、参数、约束、核心原理"
    wb.properties.description = "审题报告；未计算购电结果，未改动结果模板。"
    overview = [
        ["项目", "说明"],
        ["原文", str(SOURCE)],
        ["完整解析", str(REPORT)],
        ["范围", "52条原文单元（含标题和附件条目）＋4张题面表格；非空段落P01—P93全部覆盖。"],
        ["六类划分", "逐句表中六个分类列分别标记✓；同一句可属于多个类别，可按列筛选。"],
        ["原理简称", "物＝物理定律；运＝运筹规则；统＝统计规律；生＝生态机理；社＝社会规则。"],
        ["证据口径", "【明示】题面原文；【推导】依定义推得；【附件核对】结构检查；【待约定】题面留白；【候选】后续建模选择。"],
        ["关键限制", "这是逐句审题与结构核对，不是完整数据清洗、购电模型求解或官方题面歧义解释。"],
        ["公式说明", "物理约束、信息边界和两种调减结算读法的数学表达详见Markdown第8节。"],
        ["原文校验", verification["source_text_coverage"]],
        ["DOCX SHA256", hashlib.sha256(SOURCE.read_bytes()).hexdigest()],
    ]
    add_sheet(wb, "阅读说明", overview, [23, 130])
    header = ["编号", "原文位置", "原文"] + CATEGORIES + ["逐句解析", "核心原理", "证据标签"]
    categorized = [header]
    for identifier, location, original, categories, analysis, principles in statements:
        labels = set(categories.split("；"))
        evidence = "、".join(dict.fromkeys(re.findall(r"【([^】]+)】", analysis)))
        categorized.append([identifier, location, original] + ["✓" if category in labels else "" for category in CATEGORIES] + [analysis, principles, evidence])
    add_sheet(wb, "逐句六类划分", categorized, [9, 16, 66] + [13] * 5 + [19, 88, 34, 22])

    selected = [
        ("1.", "六类信息总览", [20, 52, 100]),
        ("3.", "题面四张表", [9, 16, 85, 28, 98, 35]),
        ("4.", "四问任务", [14, 65, 65, 65, 38]),
        ("5.", "参数与数据", [31, 43, 30, 44, 68]),
        ("6.", "核心原理", [18, 37, 65, 78, 84]),
        ("7.", "约束与歧义", [9, 28, 64, 79, 95]),
        ("8.4", "调减结算口径", [55, 80, 100]),
        ("9.", "附件结构核对", [67, 77, 43, 100]),
    ]
    for prefix, sheet_name, widths in selected:
        found = [rows for heading, rows in tables if heading.startswith(prefix)]
        if len(found) != 1:
            raise ValueError(f"Expected one table for {sheet_name}, found {len(found)}")
        add_sheet(wb, sheet_name, found[0], widths)
    add_sheet(wb, "原文段落", [["位置", "DOCX原文"]] + [[f"P{i:02d}", text] for i, text in enumerate(paragraphs, 1)], [15, 145])
    cells = [["题面表格", "行号", "原表各单元格（保留顺序）"]]
    for table_index, table in enumerate(raw_tables, 1):
        for row_index, row in enumerate(table, 1):
            cells.append([f"表{table_index}", str(row_index), " ｜ ".join(cell or "[空白]" for cell in row)])
    add_sheet(wb, "原表单元格", cells, [15, 12, 150])
    wb.save(OUTPUT)
    wb.close()

    check = load_workbook(OUTPUT, read_only=False, data_only=False)
    assert check["逐句六类划分"].max_row == 53
    assert check["逐句六类划分"].max_column == 12
    assert check["原文段落"].max_row == 94
    assert all(ws.auto_filter.ref and ws.freeze_panes for ws in check)
    assert not any(cell.data_type == "f" for ws in check for row in ws for cell in row)
    verification.update({"workbook": str(OUTPUT), "sheets": check.sheetnames, "file_bytes": OUTPUT.stat().st_size})
    check.close()
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
