"""Zero-dependency Markdown -> styled HTML -> PDF converter.

Written for this handoff package so that a receiving computer can regenerate
both documents without installing anything beyond the Python standard library
(only a Chrome/Edge/Chromium binary is needed for the final print step).

Supports: ATX headings, tables, fenced code, bullet/ordered lists, blockquotes,
horizontal rules, [TOC], and inline **bold** / *italic* / `code` / [link](url).

    python tools/md2pdf.py <input.md> <output.pdf> --title "标题" --subtitle "副标题"
"""
from __future__ import annotations

import argparse
import base64
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CSS = r"""
@page { size: A4; margin: 20mm 18mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "SimSun", sans-serif;
  font-size: 10.5pt; line-height: 1.78; color: #1c2530; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.cover { text-align: center; padding: 55mm 0 30mm; page-break-after: always; }
.cover .kicker { font-size: 10pt; letter-spacing: .18em; color: #5b7691; }
.cover h1 { font-size: 25pt; line-height: 1.45; margin: 14mm 0 6mm; color: #12395b; border: 0; }
.cover .sub { font-size: 13pt; color: #4a5b6d; margin-bottom: 22mm; }
.cover .meta { font-size: 9.5pt; color: #6a7787; line-height: 2; }
h1 { font-size: 18pt; color: #12395b; margin: 0 0 8mm; padding-bottom: 3mm;
     border-bottom: 2px solid #2c6e9b; page-break-before: always; }
h1.first { page-break-before: avoid; }
h2 { font-size: 14pt; color: #1d527a; margin: 9mm 0 4mm; padding-left: 3mm;
     border-left: 4px solid #2c6e9b; page-break-after: avoid; }
h3 { font-size: 11.8pt; color: #254a68; margin: 6mm 0 3mm; page-break-after: avoid; }
h4 { font-size: 10.8pt; color: #365b78; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 3.2mm; text-align: justify; }
strong { color: #0f3d63; }
em { color: #45586b; }
a { color: #2c6e9b; text-decoration: none; }
ul, ol { margin: 0 0 4mm; padding-left: 7mm; }
li { margin-bottom: 1.4mm; }
figure { margin: 5mm 0 6mm; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 3px; }
figcaption { font-size: 9pt; color: #5b6673; margin-top: 2mm; }
p.missing { color: #b91c1c; font-size: 9.5pt; }
table { width: 100%; border-collapse: collapse; margin: 4mm 0 6mm;
        font-size: 9.3pt; page-break-inside: avoid; }
th, td { border: 1px solid #c3d0dc; padding: 2.2mm 2.6mm; vertical-align: top; }
th { background: #e8f1f8; color: #12395b; font-weight: 600; }
tr:nth-child(even) td { background: #f7fafc; }
code { font-family: "Consolas", "Courier New", monospace; font-size: 9pt;
       background: #f0f3f6; padding: .4mm 1.2mm; border-radius: 2px; color: #a03b46; }
pre { background: #f4f7fa; border: 1px solid #d6e0e8; border-left: 4px solid #2c6e9b;
      padding: 3mm 4mm; margin: 3mm 0 6mm; page-break-inside: avoid;
      white-space: pre-wrap; word-wrap: break-word; font-size: 8.8pt; line-height: 1.5; }
pre code { background: none; padding: 0; color: #243746; }
blockquote { border-left: 4px solid #b9cbd9; background: #f5f9fc; margin: 3mm 0 5mm;
             padding: 2mm 4mm; color: #40566b; page-break-inside: avoid; }
blockquote p { margin: 0; }
hr { border: 0; border-top: 1px solid #d3dde5; margin: 7mm 0; }
.toc { background: #f5f9fc; border: 1px solid #d6e4ee; padding: 5mm 7mm;
       page-break-after: always; }
.toc h2 { border: 0; padding: 0; margin: 0 0 4mm; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc li { margin: 1.2mm 0; }
.toc .l2 { padding-left: 6mm; font-size: 9.5pt; color: #4a5b6d; }
.toc a { color: #1c2530; }
.foot { margin-top: 12mm; padding-top: 3mm; border-top: 1px solid #d3dde5;
        font-size: 8.6pt; color: #6a7787; }
"""


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:\-|]+\|?", line.strip())) and "-" in line


def convert(md: str, title: str, subtitle: str, meta: list[str], base_dir: Path | None = None) -> str:
    lines = md.splitlines()
    body: list[str] = []
    toc: list[tuple[int, str, str]] = []
    base_dir = base_dir or Path('.')
    i, h1_seen = 0, False
    para: list[str] = []

    def flush_para():
        if para:
            body.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 图片：整行 ![](path)，转 base64 内嵌，避免 Chrome 打印时的相对路径问题
        mi = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if mi:
            flush_para()
            alt, src = mi.group(1), mi.group(2)
            ip = (base_dir / src)
            if ip.is_file():
                cap = f"<figcaption>{html.escape(alt)}</figcaption>" if alt else ""
                # 用相对路径而非 data URI：Chrome headless 打印时不渲染 data URI 图片。
                # HTML 会被写到 md 同目录，因此相对路径可直接解析。
                body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"/>{cap}</figure>')
            else:
                body.append(f'<p class="missing">[图片缺失: {html.escape(src)}]</p>')
            i += 1
            continue

        if stripped.startswith("```"):
            flush_para()
            lang = stripped[3:].strip()
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            body.append(f'<pre><code class="{html.escape(lang)}">'
                        + html.escape("\n".join(code)) + "</code></pre>")
            continue

        if stripped == "[TOC]":
            flush_para()
            body.append("<!--TOC-->")
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            level, text = len(m.group(1)), m.group(2).strip()
            anchor = f"sec{len(toc)}"
            toc.append((level, text, anchor))
            cls = ' class="first"' if level == 1 and not h1_seen else ""
            if level == 1:
                h1_seen = True
            body.append(f'<h{level} id="{anchor}"{cls}>{inline(text)}</h{level}>')
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            flush_para()
            header = split_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i].strip()))
                i += 1
            head = "".join(f"<th>{inline(c)}</th>" for c in header)
            trs = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                          for r in rows)
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table>")
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush_para()
            body.append("<hr>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            body.append("<blockquote><p>" + inline(" ".join(quote)) + "</p></blockquote>")
            continue

        if re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+[.)]\s+", stripped):
            flush_para()
            ordered = bool(re.match(r"^\d+[.)]\s+", stripped))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                s = lines[i].strip()
                if ordered and re.match(r"^\d+[.)]\s+", s):
                    items.append(re.sub(r"^\d+[.)]\s+", "", s))
                elif not ordered and re.match(r"^[-*+]\s+", s):
                    items.append(re.sub(r"^[-*+]\s+", "", s))
                elif s and not s.startswith(("#", "|", ">")) and items:
                    items[-1] += " " + s
                else:
                    break
                i += 1
            body.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items)
                        + f"</{tag}>")
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        para.append(stripped)
        i += 1
    flush_para()

    toc_html = ['<div class="toc"><h2>目录</h2><ul>']
    for level, text, anchor in toc:
        if level == 1:
            toc_html.append(f'<li><a href="#{anchor}">{inline(text)}</a></li>')
        elif level == 2:
            toc_html.append(f'<li class="l2"><a href="#{anchor}">{inline(text)}</a></li>')
    toc_html.append("</ul></div>")
    content = "\n".join(body).replace("<!--TOC-->", "\n".join(toc_html))

    cover = (f'<div class="cover"><div class="kicker">低空经济 · Agent 轨迹级评测</div>'
             f'<h1>{html.escape(title)}</h1>'
             f'<div class="sub">{html.escape(subtitle)}</div>'
             '<div class="meta">' + "<br>".join(html.escape(x) for x in meta) + "</div></div>")
    return ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            + cover + content + '<div class="foot">本文档由 tools/md2pdf.py 生成；'
            '全部数字来自仓库内 results/ 的实测结果文件，可复现。</div></body></html>')


def find_browser() -> str:
    candidates = [
        r"C:/Program Files/Google/Chrome/Application/chrome.exe",
        r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("未找到 Chrome/Edge/Chromium；请安装浏览器后重试，或手工打印 HTML。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--meta", nargs="*", default=[])
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    md = src.read_text(encoding="utf-8")
    page = convert(md, args.title, args.subtitle, args.meta, base_dir=src.parent)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_html:
        html_path = dst.with_suffix(".html")
    else:
        # 必须放在 md 同目录：图片用相对路径引用，临时目录会失效
        html_path = src.parent / f"._md2pdf_{os.getpid()}.html"
    html_path.write_text(page, encoding="utf-8")

    browser = find_browser()
    cmd = [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={dst.as_posix()}",
           "--virtual-time-budget=15000",   # 等 data URI 图片解码完成
           "--run-all-compositor-stages-before-draw",
           html_path.as_uri()]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not dst.exists() or dst.stat().st_size < 1000:
        raise RuntimeError(f"PDF 生成失败：{proc.stdout[-500:]} {proc.stderr[-500:]}")
    if not args.keep_html:
        html_path.unlink(missing_ok=True)
    print(f"PDF -> {dst}  ({dst.stat().st_size:,} bytes)  browser={Path(browser).name}")


if __name__ == "__main__":
    sys.exit(main())
