"""Ragas 通用指标 baseline。

设计原则：**复用同一批 Agent 输出**
------------------------------------
本脚本不重新调用 Agent，而是读 `results/trajectory_runs_<tag>.json` 里已保存的
answer / contexts。这样 baseline 与轨迹级扩展版评的是**同一份答案**，
两者的差异只来自指标，而不是采样随机性——这是对比能成立的前提。

两个模式
--------
--strict    （默认）contexts 为空就传空列表，Ragas 会因缺少上下文而给出低分/报错
--fallback  模拟 agent_evaluator `EvalSample.to_ragas_single_turn()` 的行为：
            contexts 为空时用 reference_contexts 兜底。
            这个模式专门用来量化「框架兜底把标准答案喂给 faithfulness」的影响。

输出
----
  results/ragas_baseline_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# 必须在 import openai / httpx 之前设置：本机端点不能走系统代理，
# 否则请求会被代理拦截返回 502/404，看起来像「模型不存在」。
_LOCAL = "127.0.0.1,localhost,::1"
for _k in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_k, "")
    if "127.0.0.1" not in _v:
        os.environ[_k] = f"{_v},{_LOCAL}".strip(",")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def build_rows(traces: list[dict], items: list[dict], fallback: bool, response_field: str = "raw") -> list[dict]:
    """组装 Ragas 输入。

    response_field 很关键：
      raw       —— 直接用 Agent 的完整 JSON 答案。实测 3B judge 会把 `"rationale"`
                   这类字段名当成要输出的内容，触发 OutputParserException。
      rationale —— 只取自然语言说明。Ragas 的指标本来就是为自然语言答案设计的，
                   这样才符合其前提；代价是评的不是完整答案。
    """
    by_id = {it["metadata"]["item_id"]: it for it in items}
    rows = []
    for t in traces:
        item = by_id.get(t["item_id"])
        if item is None:
            continue
        ctx = [c for c in (t.get("context") or []) if c and c.strip()]
        if not ctx and fallback:
            ctx = [c for c in (item.get("reference_contexts") or []) if c and c.strip()]
        if response_field == "rationale":
            resp = t.get("rationale") or ""
        else:
            resp = t.get("answer_raw") or ""
        rows.append(
            {
                "item_id": t["item_id"],
                "variant": t["variant"],
                "group": item["metadata"]["group"],
                "user_input": item["user_input"],
                "response": resp,
                "retrieved_contexts": ctx,
                "reference": item.get("reference") or "",
                "reference_contexts": item.get("reference_contexts") or [],
                "used_fallback": bool(not (t.get("context") or []) and fallback),
            }
        )
    return rows


def resolve_model_id(base_url: str, model: str | None) -> str:
    """llama-server 的模型 ID 常常是完整文件路径，直接用别名会 404。

    这里在未显式指定时从 /v1/models 取真实 ID。
    """
    if model:
        return model
    import json as _json
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=10) as r:
            data = _json.loads(r.read().decode("utf-8"))
        items = data.get("data") or []
        if items:
            return items[0]["id"]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"无法从 {url} 解析模型 ID: {e}") from e
    raise RuntimeError(f"{url} 未返回模型")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RESULTS / "trajectory_runs_llm.json"))
    ap.add_argument("--evalset", default=str(ROOT / "data" / "evalset" / "lowalt_eval_set.json"))
    ap.add_argument("--variant", default="guarded")
    ap.add_argument("--metrics", nargs="*", default=["faithfulness"])
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default=os.getenv("LOWALT_LLM_BASE", "http://127.0.0.1:8081/v1"))
    ap.add_argument("--model", default=os.getenv("LOWALT_LLM_MODEL", ""),
                    help="留空则从 /v1/models 自动取（llama-server 的 ID 常是完整路径）")
    ap.add_argument("--embed-base", default=os.getenv("LOWALT_EMB_BASE", ""),
                    help="embeddings 端点，answer_relevancy 需要")
    ap.add_argument("--response-field", choices=["raw", "rationale"], default="rationale",
                    help="raw=完整JSON答案；rationale=仅自然语言说明（Ragas 指标的设计前提）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    runs_path = Path(args.runs)
    if not runs_path.exists():
        print(f"缺少运行结果 {runs_path}，先跑 runner/run_trajectory_eval.py --llm", file=sys.stderr)
        return 2

    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    items = json.loads(Path(args.evalset).read_text(encoding="utf-8"))
    traces = [t for t in payload["traces"] if t["variant"] == args.variant]
    rows = build_rows(traces, items, args.fallback, args.response_field)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("没有可评分的样本", file=sys.stderr)
        return 3

    n_fb = sum(r["used_fallback"] for r in rows)
    print(f"variant={args.variant} n={len(rows)} fallback_used={n_fb} strict={not args.fallback} response={args.response_field}")

    # ---- 构造 Ragas 输入 ----
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper

    model_id = resolve_model_id(args.base_url, args.model or None)
    print(f"judge model id: {model_id}")
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=model_id, base_url=args.base_url, api_key="local-no-key", temperature=0.0)
    )

    metric_objs = []
    if "faithfulness" in args.metrics:
        from ragas.metrics import Faithfulness

        metric_objs.append(Faithfulness(llm=llm))
    if "answer_relevancy" in args.metrics:
        if not args.embed_base:
            print("[skip] answer_relevancy 需要 embeddings 端点（--embed-base），本次跳过", file=sys.stderr)
        else:
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.metrics import ResponseRelevancy

            emb_model = resolve_model_id(args.embed_base, None)
            print(f"embedding model id: {emb_model}")
            emb = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model=emb_model,
                    base_url=args.embed_base,
                    api_key="local-no-key",
                    # 本地 llama-server 不做 tiktoken 切词，显式关掉长度检查
                    check_embedding_ctx_length=False,
                )
            )
            metric_objs.append(ResponseRelevancy(llm=llm, embeddings=emb))

    if not metric_objs:
        print("没有可运行的指标", file=sys.stderr)
        return 4

    # 严格模式下丢掉空 context 的样本会让结果虚高，这里保留但记录
    ds_list = [
        {
            "user_input": r["user_input"],
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"] or ["（无工具证据）"],
            "reference": r["reference"],
        }
        for r in rows
    ]

    ds = EvaluationDataset.from_list(ds_list)

    # 本地 CPU 推理撑不住默认并发（16），会集体超时返回 NaN。
    # 这里显式限制并发并放宽单条超时——慢但能出数。
    from ragas.run_config import RunConfig

    run_cfg = RunConfig(max_workers=2, timeout=600, max_retries=2)
    result = evaluate(dataset=ds, metrics=metric_objs, run_config=run_cfg)

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy", "response_relevancy")]

    per_row = []
    for i, r in enumerate(rows):
        rec = {k: r[k] for k in ("item_id", "variant", "group", "used_fallback")}
        for c in metric_cols:
            v = df.iloc[i][c]
            rec[c] = None if v != v else float(v)  # NaN 检查
        per_row.append(rec)

    # 汇总
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_group[r["group"]].append(r)

    def agg(rs: list[dict]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rs)}
        for c in metric_cols:
            vals = [r[c] for r in rs if r.get(c) is not None]
            out[c] = round(sum(vals) / len(vals), 4) if vals else None
        return out

    summary = {
        "variant": args.variant,
        "mode": "fallback" if args.fallback else "strict",
        "n": len(per_row),
        "n_fallback": n_fb,
        "metrics": metric_cols,
        "metrics_requested": args.metrics,
        "overall": agg(per_row),
        "by_group": {g: agg(rs) for g, rs in sorted(by_group.items())},
        "config": {
            "llm_base": args.base_url,
            "model": args.model,
            "embed_base": args.embed_base,
            "runs": str(runs_path),
        },
    }

    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "fallback" if args.fallback else "strict"
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"ragas_baseline_{args.variant}_{mode}{suffix}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "per_item": per_row}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(f"\n总体: {summary['overall']}")
    for g, s in summary["by_group"].items():
        print(f"  {g:14} n={s['n']:2} " + " ".join(f"{k}={v}" for k, v in s.items() if k != "n"))
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
