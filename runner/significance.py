"""对比显著性检验：通用指标 vs 轨迹维度。

为什么要做这个
--------------
12 条样本上算出的均值差（0.046）看起来像「有差异」，但 bootstrap 区间跨 0、
置换检验 p=0.755——**统计上分不开**。如果报告里直接写「区分度 0.046」，
等于把噪声说成效应，面试官一算就露馅。

所以这里把检验固化成脚本，报告里的每个 p 值都能重跑。

用法
----
    python runner/significance.py
    python runner/significance.py --baseline guarded --compare route_skip
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_metric(path: Path, metric: str) -> list[float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return [r[metric] for r in d["per_item"] if r.get(metric) is not None]


def bootstrap_diff(a: list[float], b: list[float], n_boot: int = 20000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    obs = st.fmean(a) - st.fmean(b)
    boots = []
    for _ in range(n_boot):
        sa = [rng.choice(a) for _ in a]
        sb = [rng.choice(b) for _ in b]
        boots.append(st.fmean(sa) - st.fmean(sb))
    boots.sort()
    return obs, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def permutation_p(a: list[float], b: list[float], n_perm: int = 20000, seed: int = 7) -> float:
    rng = random.Random(seed)
    obs = abs(st.fmean(a) - st.fmean(b))
    pool = list(a) + list(b)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(st.fmean(pool[: len(a)]) - st.fmean(pool[len(a) :])) >= obs:
            cnt += 1
    return cnt / n_perm


def fisher_exact_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """2x2  Fisher 精确检验（双侧），纯标准库实现。

    用于安全通过率这类二值比例：12 条里 0 通过 vs 12 条里 10 通过。
    """
    from math import comb

    def hyper(k: int, n: int, K: int, N: int) -> float:
        # 从 N 中取 n，其中 K 个成功，恰有 k 个成功的概率
        if k < max(0, n - (N - K)) or k > min(n, K):
            return 0.0
        return comb(K, k) * comb(N - K, n - k) / comb(N, n)

    # 固定边际：k1, n1, k2, n2 -> 总成功 K=k1+k2, 总样本 N=n1+n2
    K, N = k1 + k2, n1 + n2
    p_obs = hyper(k1, n1, K, N)
    total = 0.0
    for k in range(0, min(n1, K) + 1):
        p = hyper(k, n1, K, N)
        if p <= p_obs + 1e-12:
            total += p
    return min(1.0, total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="guarded")
    ap.add_argument("--compare", default="route_skip")
    ap.add_argument("--suffix", default="rat", help="读取 ragas_baseline_*_strict_<suffix>.json")
    ap.add_argument("--out", default=str(RESULTS / "significance.json"))
    args = ap.parse_args()

    out: dict = {"baseline": args.baseline, "compare": args.compare, "tests": []}

    # ---- 1. 通用指标（faithfulness）----
    ga = RESULTS / f"ragas_baseline_{args.baseline}_strict_{args.suffix}.json"
    gb = RESULTS / f"ragas_baseline_{args.compare}_strict_{args.suffix}.json"
    if ga.exists() and gb.exists():
        a = load_metric(ga, "faithfulness")
        b = load_metric(gb, "faithfulness")
        obs, lo, hi = bootstrap_diff(a, b)
        p = permutation_p(a, b)
        rec = {
            "metric": "faithfulness",
            "n_a": len(a),
            "n_b": len(b),
            "mean_a": round(st.fmean(a), 4),
            "mean_b": round(st.fmean(b), 4),
            "diff": round(obs, 4),
            "bootstrap_95ci": [round(lo, 4), round(hi, 4)],
            "ci_crosses_zero": lo < 0 < hi,
            "permutation_p": round(p, 4),
            "significant_at_0.05": p < 0.05,
            "max_b": round(max(b), 4),
            "n_perfect_b": sum(1 for x in b if x >= 0.999),
        }
        out["tests"].append(rec)
        print(f"[faithfulness] {rec['mean_a']} vs {rec['mean_b']}  diff={rec['diff']}  "
              f"CI=[{lo:.3f},{hi:.3f}]  p={p:.4f}  -> {'SIGNIFICANT' if p < 0.05 else 'NOT significant'}")
        print(f"                {args.compare} 满分条数 = {rec['n_perfect_b']}/{rec['n_b']}")

    # ---- 1b. answer_relevancy（不同文件后缀）----
    aa = RESULTS / f"ragas_baseline_{args.baseline}_strict_ar20.json"
    ab = RESULTS / f"ragas_baseline_{args.compare}_strict_ar20.json"
    if aa.exists() and ab.exists():
        a = load_metric(aa, "answer_relevancy")
        b = load_metric(ab, "answer_relevancy")
        if a and b:
            obs, lo, hi = bootstrap_diff(a, b)
            p = permutation_p(a, b)
            rec = {
                "metric": "answer_relevancy",
                "n_a": len(a), "n_b": len(b),
                "mean_a": round(st.fmean(a), 4), "mean_b": round(st.fmean(b), 4),
                "diff": round(obs, 4),
                "bootstrap_95ci": [round(lo, 4), round(hi, 4)],
                "ci_crosses_zero": lo < 0 < hi,
                "permutation_p": round(p, 4),
                "significant_at_0.05": p < 0.05,
            }
            out["tests"].append(rec)
            print(f"[answer_relevancy] {rec['mean_a']} vs {rec['mean_b']}  diff={rec['diff']}  "
                  f"CI=[{lo:.3f},{hi:.3f}]  p={p:.4f}  -> {'SIGNIFICANT' if p < 0.05 else 'NOT significant'}")

    # ---- 2. 轨迹维度（安全通过率，Fisher 精确检验）----
    ra = RESULTS / "trajectory_runs_llm.json"
    if ra.exists():
        d = json.loads(ra.read_text(encoding="utf-8"))["results"]
        ka = sum(r["safety_compliance"] for r in d if r["variant"] == args.baseline)
        na = sum(1 for r in d if r["variant"] == args.baseline)
        kb = sum(r["safety_compliance"] for r in d if r["variant"] == args.compare)
        nb = sum(1 for r in d if r["variant"] == args.compare)
        if na and nb:
            p = fisher_exact_p(ka, na, kb, nb)
            rec = {
                "metric": "safety_compliance_pass_rate",
                "n_a": na, "n_b": nb,
                "pass_a": ka, "pass_b": kb,
                "rate_a": round(ka / na, 4), "rate_b": round(kb / nb, 4),
                "fisher_exact_p": round(p, 6),
                "significant_at_0.05": p < 0.05,
            }
            out["tests"].append(rec)
            print(f"[safety]       {ka}/{na} vs {kb}/{nb}  fisher p={p:.2e}  "
                  f"-> {'SIGNIFICANT' if p < 0.05 else 'NOT significant'}")

    # ---- 3. 轨迹维度（工具一致性，bootstrap）----
    if ra.exists():
        d = json.loads(ra.read_text(encoding="utf-8"))["results"]
        ta = [r["tool_use_trajectory_consistency"] for r in d if r["variant"] == args.baseline]
        tb = [r["tool_use_trajectory_consistency"] for r in d if r["variant"] == args.compare]
        if ta and tb:
            obs, lo, hi = bootstrap_diff(ta, tb)
            p = permutation_p(ta, tb)
            rec = {
                "metric": "tool_use_trajectory_consistency",
                "n_a": len(ta), "n_b": len(tb),
                "mean_a": round(st.fmean(ta), 4), "mean_b": round(st.fmean(tb), 4),
                "diff": round(obs, 4),
                "bootstrap_95ci": [round(lo, 4), round(hi, 4)],
                "ci_crosses_zero": lo < 0 < hi,
                "permutation_p": round(p, 6),
                "significant_at_0.05": p < 0.05,
            }
            out["tests"].append(rec)
            print(f"[tool-trajectory] {rec['mean_a']} vs {rec['mean_b']}  diff={rec['diff']}  p={p:.2e}")

    out["note"] = (
        "faithfulness 的差值不显著（CI 跨 0），因此在报告中不得表述为『区分度 0.046』；"
        "安全通过率与工具一致性为确定性判定，不受 judge 随机性影响，差异显著。"
    )
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
