"""环境自检：接手第一件事就跑这个。

    cd project && python tools/check_env.py

只读，不修改任何文件。输出 PASS / WARN / FAIL 三类。
FAIL 会阻止后续流程，WARN 只是提醒。
"""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT.parent

results: list[tuple[str, str, str]] = []


def add(level: str, name: str, note: str) -> None:
    results.append((level, name, note))


def check_python() -> None:
    v = sys.version_info
    note = f"{v.major}.{v.minor}.{v.micro} ({platform.system()})"
    add("PASS" if v >= (3, 10) else "FAIL", "Python >= 3.10", note)


def check_pkg(mod: str, label: str | None = None, required: bool = True) -> None:
    label = label or mod
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "")
        add("PASS", f"依赖 {label}", ver or "已安装")
    except Exception as e:  # noqa: BLE001
        add("FAIL" if required else "WARN", f"依赖 {label}", f"缺失：{type(e).__name__}")


def check_file(path: Path, name: str, required: bool = True, hint: str = "") -> None:
    if path.exists():
        if path.is_file():
            add("PASS", name, f"{path.stat().st_size / 1e6:.1f} MB")
        else:
            n = sum(1 for _ in path.rglob("*"))
            add("PASS", name, f"{n} 项")
    else:
        add("FAIL" if required else "WARN", name, hint or "缺失")


def main() -> int:
    print("=" * 68)
    print("低空 Agent 轨迹级评测 — 环境自检")
    print(f"项目根目录: {ROOT}")
    print("=" * 68)

    check_python()

    # 依赖
    for mod, label in [
        ("yaml", "pyyaml"),
        ("pydantic", "pydantic"),
        ("httpx", "httpx"),
        ("langgraph", "langgraph"),
        ("ragas", "ragas"),
        ("langchain_openai", "langchain-openai"),
        ("loguru", "loguru（上游漏声明）"),
    ]:
        check_pkg(mod, label)
    check_pkg("agent_evaluator", "agent_evaluator（-e 安装）")

    # ragas 补丁状态
    try:
        import ragas  # noqa: F401
        add("PASS", "ragas 可导入", "补丁已生效或无需补丁")
    except Exception as e:  # noqa: BLE001
        add("FAIL", "ragas 可导入", f"{type(e).__name__}：先跑 scripts/patch_ragas.py")

    # 数据
    print("-" * 68)
    check_file(ROOT / "data" / "raw" / "UAVBench_scenarios.zip", "UAVBench 场景包（必需）",
               hint="160 MB；见 models/README.md 或 tools/download_models.py --uavbench")
    check_file(ROOT / "data" / "raw" / "ab_clean.zip", "AlphaBench 包（可选）", required=False,
               hint="320 MB；仅用于结构剖析，主流程不需要")
    check_file(ROOT / "data" / "uavbench_index.json", "场景索引", required=False,
               hint="84 MB；缺失可跑 evalset/index_scenarios.py 重建（约 2.5 分钟）")
    check_file(ROOT / "data" / "evalset" / "lowalt_eval_set.json", "评测集 40 条", required=False,
               hint="跑 evalset/convert_uavbench.py --profile 40")
    check_file(ROOT / "data" / "evalset" / "lowalt_eval_set_120.json", "评测集 120 条", required=False,
               hint="跑 evalset/convert_uavbench.py --profile 120")

    # 代码结构
    print("-" * 68)
    for rel in ["agent/graph.py", "agent/geo.py", "trajectory/scorer.py",
                "runner/run_trajectory_eval.py", "tests/test_geo.py",
                "configs/lowalt_langgraph.yml"]:
        check_file(ROOT / rel, f"代码 {rel}")

    # 结果
    print("-" * 68)
    for rel, name in [
        ("results/trajectory_table_det.md", "零模型对比表"),
        ("results/trajectory_table_llm.md", "LLM 对比表"),
        ("results/significance.json", "显著性检验"),
        ("results/noop_reroute_analysis.json", "空改航分析"),
        ("docs/figures/fig_trajectory_compare.png", "图 1"),
    ]:
        check_file(ROOT / rel, f"结果 {name}", required=False, hint="可重新生成")

    # 模型（可选）
    print("-" * 68)
    model_dirs = [ROOT / "models", HANDOFF / "models"]
    found_qwen = found_bge = found_exe = None
    for d in model_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.gguf"):
            if "qwen" in p.name.lower():
                found_qwen = p
            elif "bge" in p.name.lower():
                found_bge = p
        for p in d.rglob("llama-server*"):
            found_exe = p
    add("PASS" if found_qwen else "WARN", "对话模型 Qwen2.5-3B",
        f"{found_qwen.stat().st_size/1e9:.2f} GB" if found_qwen else "未找到（只跑确定性链路则不需要）")
    add("PASS" if found_bge else "WARN", "Embedding bge-small-zh",
        f"{found_bge.stat().st_size/1e6:.0f} MB" if found_bge else "未找到（answer_relevancy 需要）")
    add("PASS" if found_exe else "WARN", "llama-server 引擎",
        str(found_exe) if found_exe else "未找到（见 models/README.md）")

    # 浏览器（PDF）
    browser = next((b for b in ["chrome", "msedge", "chromium", "google-chrome"] if shutil.which(b)), None)
    add("PASS" if browser else "WARN", "浏览器（重新生成 PDF）", browser or "未在 PATH 中找到，PDF 已随包提供")

    # 输出
    print()
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for level, name, note in sorted(results, key=lambda x: order[x[0]]):
        print(f"  [{level:4}] {name:34} {note}")

    n_fail = sum(1 for l, _, _ in results if l == "FAIL")
    n_warn = sum(1 for l, _, _ in results if l == "WARN")
    print()
    print("=" * 68)
    print(f"结果：{len(results)} 项检查，{n_fail} FAIL，{n_warn} WARN")
    if n_fail == 0:
        print("环境就绪。下一步：python tools/build_all.py")
    else:
        print("请先解决 FAIL 项。依赖安装见 README_先看这个.md 第 2 节。")
    print("=" * 68)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
