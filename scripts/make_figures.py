"""生成报告用的 SVG 图（零依赖，不装 matplotlib）。

图 1 `fig_trajectory_compare.svg`
    四个变体在同一道题上的节点路径对比。重点是让人一眼看出：
    `route_skip` 少了一个节点、`stale_context` 节点全在但证据链断了。
    这张图想说明的是——**最终答案可能一样，轨迹完全不同**。

图 2 `fig_noop_reroute.svg`
    模型自称「改航」的 23 次里，真正改对了多少次。

用法
----
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

FONT = "Microsoft YaHei, PingFang SC, Hiragino Sans GB, sans-serif"

BG = "#ffffff"
INK = "#1f2937"
MUTED = "#6b7280"
OK_FILL, OK_LINE = "#e8f0fe", "#1a56db"
BAD_FILL, BAD_LINE = "#fee2e2", "#dc2626"
WARN_FILL, WARN_LINE = "#fef3c7", "#d97706"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, label, sub, kind="ok", dashed=False):
    fill, line = {"ok": (OK_FILL, OK_LINE), "bad": (BAD_FILL, BAD_LINE), "warn": (WARN_FILL, WARN_LINE)}[kind]
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
        f'stroke="{line}" stroke-width="1.6"{dash}/>',
        f'<text x="{x + w/2}" y="{y + (h/2 if not sub else h/2 - 7)}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="14" font-weight="600" fill="{INK}">{esc(label)}</text>',
    ]
    if sub:
        parts.append(
            f'<text x="{x + w/2}" y="{y + h/2 + 12}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" fill="{MUTED}">{esc(sub)}</text>'
        )
    return "\n".join(parts)


def arrow(x1, y, x2, color="#9ca3af", dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 6}" y2="{y}" stroke="{color}" stroke-width="1.6" '
        f'marker-end="url(#head)"{dash}/>'
    )


def fig_trajectory() -> str:
    W, H = 1120, 470
    rows = [
        ("guarded", "正确实现", [("parse_mission", "解析任务"), ("route_decision", "条件路由"),
                                ("tool_node", "8 次工具调用"), ("generate_response", "生成决策")],
         ["safety 77.5%", "轨迹 0.978", "证据 8 条"], "ok"),
        ("route_skip", "条件边读错字段", [("parse_mission", "解析任务"), ("route_decision", "直接走 respond"),
                                         (None, "tool_node 未执行"), ("generate_response", "无证据下决策")],
         ["safety 0%", "轨迹 0.200", "证据 0 条"], "bad"),
        ("stale_context", "漏写 context", [("parse_mission", "解析任务"), ("route_decision", "条件路由"),
                                           ("tool_node", "8 次工具调用"), ("generate_response", "读到空证据")],
         ["safety 95%", "轨迹 0.978", "证据 0 条"], "warn"),
        ("repeat_tool", "工具重复调用", [("parse_mission", "解析任务"), ("route_decision", "条件路由"),
                                        ("tool_node", "10 次调用（3 次重复）"), ("generate_response", "生成决策")],
         ["safety 77.5%", "轨迹 0.677", "证据 8 条"], "warn"),
    ]

    col_x = [140, 360, 580, 800]
    bw, bh = 190, 58
    top = 74
    gap = 92

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           '<defs><marker id="head" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
           '<path d="M0,0 L0,6 L7,3 z" fill="#9ca3af"/></marker></defs>',
           f'<rect width="{W}" height="{H}" fill="{BG}"/>',
           f'<text x="40" y="34" font-family="{FONT}" font-size="19" font-weight="700" fill="{INK}">'
           f'同一道题，四条轨迹：答案可能一样，过程完全不同</text>',
           f'<text x="40" y="56" font-family="{FONT}" font-size="12" fill="{MUTED}">'
           f'LangGraph StateGraph · 每个变体跑同一批 40 道题 · 节点下方为典型工具调用情况</text>']

    for i, (name, desc, nodes, metrics, kind) in enumerate(rows):
        y = top + i * gap
        # 变体名
        out.append(f'<text x="40" y="{y + 26}" font-family="{FONT}" font-size="14" font-weight="700" fill="{INK}">{esc(name)}</text>')
        out.append(f'<text x="40" y="{y + 44}" font-family="{FONT}" font-size="11" fill="{MUTED}">{esc(desc)}</text>')

        for j, (label, sub) in enumerate(nodes):
            x = col_x[j]
            if label is None:  # 缺失节点
                out.append(box(x, y, bw, bh, sub, "跳过", kind="bad", dashed=True))
                if j > 0:
                    out.append(arrow(col_x[j - 1] + bw, y + bh / 2, x, color="#dc2626", dashed=True))
                continue
            # 判断节点颜色：只标出真正异常的那一步，不整行染红
            k = "ok"
            if name == "route_skip":
                # 解析正常；缺陷在条件边（读错字段）；tool_node 被跳过；生成节点无证据
                k = {0: "ok", 1: "warn", 2: "bad", 3: "warn"}[j]
            elif name == "stale_context":
                # 前三个节点都正常执行，问题出在证据没写回，生成节点读到空
                k = "bad" if j == 3 else "ok"
            elif name == "repeat_tool":
                k = "warn" if j == 2 else "ok"
            short = label.replace("_", "_")
            out.append(box(x, y, bw, bh, short, sub, kind=k))
            if j > 0:
                prev_label = nodes[j - 1][0]
                if prev_label is not None:
                    out.append(arrow(col_x[j - 1] + bw, y + bh / 2, x))

        # 右侧指标
        mx = 1010
        for mi, m in enumerate(metrics):
            color = BAD_LINE if ("0%" in m or "0.200" in m or "0 条" in m) else INK
            out.append(f'<text x="{mx}" y="{y + 20 + mi * 17}" font-family="{FONT}" font-size="11" fill="{color}">{esc(m)}</text>')
        out.append(f'<line x1="{mx - 12}" y1="{y}" x2="{mx - 12}" y2="{y + bh}" stroke="#e5e7eb" stroke-width="1"/>')

    # 图例
    ly = H - 34
    out.append(f'<rect x="140" y="{ly - 10}" width="14" height="12" rx="3" fill="{OK_FILL}" stroke="{OK_LINE}"/>')
    out.append(f'<text x="160" y="{ly}" font-family="{FONT}" font-size="11" fill="{MUTED}">正常执行</text>')
    out.append(f'<rect x="240" y="{ly - 10}" width="14" height="12" rx="3" fill="{WARN_FILL}" stroke="{WARN_LINE}"/>')
    out.append(f'<text x="260" y="{ly}" font-family="{FONT}" font-size="11" fill="{MUTED}">行为异常但未崩</text>')
    out.append(f'<rect x="380" y="{ly - 10}" width="14" height="12" rx="3" fill="{BAD_FILL}" stroke="{BAD_LINE}" stroke-dasharray="5 3"/>')
    out.append(f'<text x="400" y="{ly}" font-family="{FONT}" font-size="11" fill="{MUTED}">节点缺失 / 证据链断裂</text>')
    out.append(f'<text x="560" y="{ly}" font-family="{FONT}" font-size="11" fill="{MUTED}">'
               f'注：stale_context 调的 8 次工具结果未写入 context，生成节点实际读到 0 条证据</text>')

    out.append("</svg>")
    return "\n".join(out)


def fig_noop() -> str:
    W, H = 1000, 330
    data = [
        ("guarded", 15, 3, 5, "23 次声称 reroute"),
        ("repeat_tool", 16, 3, 5, "24 次"),
    ]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="{BG}"/>',
           f'<text x="60" y="38" font-family="{FONT}" font-size="19" font-weight="700" fill="{INK}">'
           f'模型说「我要改航」之后，实际改了多少</text>',
           f'<text x="60" y="60" font-family="{FONT}" font-size="12" fill="{MUTED}">'
           f'正确实现的 guarded 变体 · 40 道题 · 判据：模型输出航点与原航点逐点比对 + 圆柱几何复核</text>']

    x0, bw_total = 60, 660
    for i, (name, noop, into_nfz, ok, note) in enumerate(data):
        y = 100 + i * 78
        total = noop + into_nfz + ok
        out.append(f'<text x="{x0}" y="{y - 8}" font-family="{FONT}" font-size="13" font-weight="600" fill="{INK}">{name}</text>')
        out.append(f'<text x="{x0 + 120}" y="{y - 8}" font-family="{FONT}" font-size="11" fill="{MUTED}">{note}</text>')

        segs = [(noop, "#dc2626", f"航点原样不动 {noop}"), (into_nfz, "#d97706", f"改到禁飞区 {into_nfz}"), (ok, "#16a34a", f"真改对 {ok}")]
        cx = x0
        for val, color, label in segs:
            w = bw_total * val / total
            out.append(f'<rect x="{cx:.1f}" y="{y}" width="{w:.1f}" height="34" fill="{color}" opacity="0.88"/>')
            if w > 46:
                out.append(f'<text x="{cx + w/2:.1f}" y="{y + 22}" text-anchor="middle" font-family="{FONT}" '
                           f'font-size="13" font-weight="700" fill="#ffffff">{val}</text>')
            cx += w
        out.append(f'<rect x="{x0}" y="{y}" width="{bw_total}" height="34" fill="none" stroke="#e5e7eb" stroke-width="1"/>')

        # 右侧结论
        out.append(f'<text x="{x0 + bw_total + 24}" y="{y + 15}" font-family="{FONT}" font-size="12" fill="{INK}">'
                   f'空改航 {noop/total:.1%}</text>')
        out.append(f'<text x="{x0 + bw_total + 24}" y="{y + 32}" font-family="{FONT}" font-size="12" fill="{MUTED}">'
                   f'真改对 {ok} 次</text>')

    ly = H - 46
    for k, (color, label) in enumerate([("#dc2626", "航点原样（no-op）"), ("#d97706", "改到禁飞区"), ("#16a34a", "真正避开")]):
        x = 60 + k * 190
        out.append(f'<rect x="{x}" y="{ly - 10}" width="14" height="12" rx="3" fill="{color}" opacity="0.88"/>')
        out.append(f'<text x="{x + 22}" y="{ly}" font-family="{FONT}" font-size="11" fill="{MUTED}">{label}</text>')

    out.append(f'<text x="60" y="{H - 18}" font-family="{FONT}" font-size="11" fill="{MUTED}">'
               f'注：repeat_tool 的工具层与 guarded 不同，空改航比例却几乎一致（66.7% vs 65.2%），'
               f'说明该行为来自模型，不是变体代码。</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig_trajectory_compare.svg").write_text(fig_trajectory(), encoding="utf-8")
    (OUT / "fig_noop_reroute.svg").write_text(fig_noop(), encoding="utf-8")
    print(f"-> {OUT/'fig_trajectory_compare.svg'}")
    print(f"-> {OUT/'fig_noop_reroute.svg'}")

    # 数据快照，保证图里的数字有出处
    d = json.loads((ROOT / "results" / "noop_reroute_analysis.json").read_text(encoding="utf-8"))
    snap = {v: d["variants"][v] for v in ("guarded", "repeat_tool") if v in d["variants"]}
    (OUT / "figures_data.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT/'figures_data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
