"""一键复现全部确定性实验（不需要模型、不需要 API key）。

    python scripts/run_all.py                 # 确定性链路
    python scripts/run_all.py --with-llm      # 额外跑 LLM 评测（需先起 llama-server）

依次执行：
  1. 环境自检（数据是否就位）
  2. 几何单测（24 项）
  3. 建场景索引（若不存在）
  4. 生成评测集
  5. 零模型轨迹评测（4 变体 × 40 条）
  6. 汇总对比表
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(name: str, args: list[str], timeout: int = 3600) -> bool:
    print(f"\n{'='*66}\n▶ {name}\n{'='*66}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args, cwd=ROOT, timeout=timeout)
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f"{'✓' if ok else '✗'} {name}  ({dt:.1f}s, exit={r.returncode})", flush=True)
    return ok


def check_assets() -> bool:
    print("=" * 66)
    print("▶ 环境自检")
    print("=" * 66)
    checks = []

    uav = ROOT / "data" / "raw" / "UAVBench_scenarios.zip"
    checks.append(("UAVBench 场景包", uav.exists(), f"{uav.stat().st_size/1e6:.0f} MB" if uav.exists() else "缺失"))

    ab = ROOT / "data" / "raw" / "ab_clean.zip"
    checks.append(("AlphaBench 包（可选）", ab.exists(), f"{ab.stat().st_size/1e6:.0f} MB" if ab.exists() else "缺失（不影响主流程）"))

    idx = ROOT / "data" / "uavbench_index.json"
    checks.append(("场景索引", idx.exists(), "已建" if idx.exists() else "需构建"))

    es = ROOT / "data" / "evalset" / "lowalt_eval_set.json"
    n = 0
    if es.exists():
        n = len(json.loads(es.read_text(encoding="utf-8")))
    checks.append(("评测集", es.exists(), f"{n} 条" if es.exists() else "需生成"))

    try:
        import langgraph  # noqa: F401
        checks.append(("langgraph", True, "已安装"))
    except ImportError:
        checks.append(("langgraph", False, "未安装：pip install langgraph"))

    ok = True
    for name, passed, note in checks:
        flag = "PASS" if passed else "WARN"
        print(f"  [{flag}] {name:24} {note}")
        if not passed and "langgraph" in name:
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    if not check_assets():
        print("\n关键依赖缺失，退出", file=sys.stderr)
        return 2

    if not run("几何交叉验证单测", ["-m", "unittest", "discover", "-s", "tests"]):
        return 3

    idx = ROOT / "data" / "uavbench_index.json"
    if not idx.exists() and not args.skip_index:
        if not run("建场景索引（约 2.5 分钟）", ["evalset/index_scenarios.py"]):
            return 4

    if not run("生成评测集（40 条 pilot，报告主体）", ["evalset/convert_uavbench.py", "--profile", "40"]):
        return 5
    if not run("生成评测集（120 条扩展集，稳定性检查）", ["evalset/convert_uavbench.py", "--profile", "120"]):
        return 5

    if not run("零模型轨迹评测（40 条）", ["runner/run_trajectory_eval.py", "--tag", "det"]):
        return 6
    if not run("零模型轨迹评测（120 条）", ["runner/run_trajectory_eval.py",
                                             "--evalset", "data/evalset/lowalt_eval_set_120.json",
                                             "--tag", "det120"]):
        return 6

    # 轨迹分析（不依赖 LLM）
    run("空改航分析", ["runner/analyze_noop_reroute.py"])
    run("生成图表", ["scripts/make_figures.py"])

    if args.with_llm:
        if not run("LLM 轨迹评测", ["runner/run_trajectory_eval.py", "--llm", "--tag", "llm"]):
            print("LLM 评测失败（检查 llama-server 是否在 127.0.0.1:8081）", file=sys.stderr)
        else:
            run("Ragas baseline (guarded)", ["runner/run_ragas_baseline.py", "--variant", "guarded",
                                             "--metrics", "faithfulness", "--tag", "rat"])
            run("Ragas baseline (route_skip 对照)", ["runner/run_ragas_baseline.py", "--variant", "route_skip",
                                                     "--metrics", "faithfulness", "--tag", "rat"])
            run("显著性检验", ["runner/significance.py"])

    print("\n" + "=" * 66)
    print("完成。关键产物：")
    print("  results/trajectory_table_det.md     零模型对比表（40 条）")
    print("  results/trajectory_table_det120.md  零模型对比表（120 条，稳定性）")
    print("  results/trajectory_table_llm.md     LLM 对比表（若已跑）")
    print("  results/ragas_baseline_*.json       通用指标")
    print("  data/evalset/lowalt_eval_set.json   评测集（40 条）")
    print("  data/evalset/lowalt_eval_set_120.json  扩展集（120 条）")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
