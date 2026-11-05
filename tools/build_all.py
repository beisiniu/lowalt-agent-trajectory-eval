"""一键复现全部确定性实验（不需要模型、不需要 API key、不需要联网）。

    cd project && python tools/build_all.py

依次执行：
  1. 几何交叉验证单测（24 项）
  2. 生成两档评测集（40 / 120 条）
  3. 零模型轨迹评测（4 变体 × 40 条，约 6 秒）
  4. 零模型扩展集评测（4 变体 × 120 条）
  5. 空改航分析
  6. 显著性检验
  7. 生成图表

跑完打印关键指标，和 docs/EVAL_REPORT.md 对照即可确认环境正常。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(name: str, args: list[str], timeout: int = 1800, expect: str = "") -> bool:
    print(f"\n{'='*68}\n▶ {name}\n{'='*68}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args, cwd=ROOT, timeout=timeout)
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f"{'✓' if ok else '✗'} {name}  ({dt:.1f}s, exit={r.returncode})", flush=True)
    if not ok and expect:
        print(f"   提示：{expect}", file=sys.stderr)
    return ok


def main() -> int:
    print("=" * 68)
    print("低空 Agent 轨迹级评测 — 确定性实验一键复现")
    print(f"项目目录: {ROOT}")
    print("=" * 68)

    steps = [
        ("① 几何交叉验证单测（24 项）", ["-m", "unittest", "discover", "-s", "tests"]),
        ("② 生成评测集 40 条", ["evalset/convert_uavbench.py", "--profile", "40"]),
        ("③ 生成扩展集 120 条", ["evalset/convert_uavbench.py", "--profile", "120"]),
        ("④ 零模型轨迹评测（40 条）", ["runner/run_trajectory_eval.py", "--tag", "det"]),
        ("⑤ 零模型扩展集评测（120 条）",
         ["runner/run_trajectory_eval.py", "--evalset", "data/evalset/lowalt_eval_set_120.json", "--tag", "det120"]),
        ("⑥ 空改航分析", ["runner/analyze_noop_reroute.py"]),
        ("⑦ 显著性检验", ["runner/significance.py", "--suffix", "n20"]),
        ("⑧ 生成图表", ["scripts/make_figures.py"]),
    ]

    for name, args in steps:
        if not run(name, args):
            print(f"\n在第 {name[:2]} 步失败，后续步骤中止。", file=sys.stderr)
            print("排错：先跑 python tools/check_env.py", file=sys.stderr)
            return 3

    print("\n" + "=" * 68)
    print("全部完成。关键产物：")
    print("  results/trajectory_table_det.md     零模型对比表（40 条）")
    print("  results/trajectory_table_det120.md  零模型对比表（120 条，稳定性）")
    print("  results/significance.json           四项显著性检验")
    print("  results/noop_reroute_analysis.json  空改航分解")
    print("  docs/figures/*.svg|png              两张图")
    print()
    print("对照 docs/EVAL_REPORT.md §7–§9 检查数字是否一致。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
