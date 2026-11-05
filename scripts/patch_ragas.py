"""修复 ragas 与 langchain-community 0.4.x 的导入冲突。

问题
----
ragas 0.3.1 / 0.4.3 的 `ragas/llms/base.py` 顶部硬编码：

    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI

而 langchain-community 0.4.x 已把 VertexAI 相关实现移出主包（迁到
langchain-google-vertexai），于是 `import ragas` 直接 ModuleNotFoundError。
这不是本项目代码的问题，但会让整个评测跑不起来。

处理方式
--------
给这两个符号加一层 try/except 降级为占位类。本项目只走 OpenAI 兼容端点，
不会实例化 VertexAI，因此占位类不影响任何被测指标。

脚本幂等：重复执行不会重复改写。修改前会备份为 `base.py.bak`。

用法
----
    python scripts/patch_ragas.py          # 打补丁
    python scripts/patch_ragas.py --check  # 只检查状态
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# [lowalt-eval patch] vertexai fallback"

ORIGINAL = """from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.llms import VertexAI"""

PATCHED = f"""{MARKER}
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI
except ImportError:  # langchain-community >= 0.4 移除了 vertexai 子模块
    class ChatVertexAI:  # type: ignore[no-redef]
        \"\"\"占位：本项目不使用 VertexAI，仅需让 import 通过。\"\"\"

    class VertexAI:  # type: ignore[no-redef]
        \"\"\"占位：同上。\"\"\""""


def find_base_py() -> Path | None:
    try:
        import ragas  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    for p in sys.path:
        cand = Path(p) / "ragas" / "llms" / "base.py"
        if cand.exists():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = find_base_py()
    if path is None:
        print("未找到 ragas/llms/base.py，请先 pip install ragas", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"已打过补丁: {path}")
        return 0
    if args.check:
        print(f"未打补丁: {path}")
        return 1
    if ORIGINAL not in text:
        print("未匹配到预期 import 片段，可能 ragas 版本已变化，请手动检查：", file=sys.stderr)
        print(f"  {path}", file=sys.stderr)
        return 3

    backup = path.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"已备份 -> {backup}")

    path.write_text(text.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    print(f"已打补丁 -> {path}")

    # 立即验证
    import subprocess

    r = subprocess.run([sys.executable, "-c", "import ragas; print('ragas', ragas.__version__, 'OK')"],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-400:])
    return 0 if r.returncode == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
