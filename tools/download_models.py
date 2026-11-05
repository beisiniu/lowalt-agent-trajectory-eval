"""下载模型、推理引擎与数据集（本包未打包的大文件）。

    python tools/download_models.py --list              # 看看有哪些可下
    python tools/download_models.py --embedding         # 48 MB，带 SHA-256 校验
    python tools/download_models.py --engine            # llama.cpp b12110
    python tools/download_models.py --uavbench          # 160 MB 场景包
    python tools/download_models.py --qwen --help-only  # 只打印对话框模型的获取方式
    python tools/download_models.py --all               # 除 Qwen 外全部

Qwen2.5-3B 需要走 ModelScope，脚本只打印命令，不代下（加速器行为因环境而异）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT.parent
MODELS = ROOT / "models"
RAW = ROOT / "data" / "raw"

# 本地端点必须绕过系统代理，否则下载/自检会出现 502
_LOCAL = "127.0.0.1,localhost,::1"
for _k in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_k, "")
    if "127.0.0.1" not in _v:
        os.environ[_k] = f"{_v},{_LOCAL}".strip(",")

ITEMS = {
    "embedding": {
        "desc": "bge-small-zh-v1.5-f16.gguf（embedding，48 MB）",
        "url": "https://hf-mirror.com/CompendiumLabs/bge-small-zh-v1.5-gguf/resolve/main/bge-small-zh-v1.5-f16.gguf",
        "out": MODELS / "bge-small-zh-v1.5-f16.gguf",
        "bytes": 47886240,
        "sha256": "ab9b81d9cd329c712eee379cf0068eabe6a5e2a01d0def61535eba9384085e2c",
    },
    "engine": {
        "desc": "llama.cpp b12110 Windows CPU x64（约 25 MB 压缩）",
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b12110/llama-b12110-bin-win-cpu-x64.zip",
        "out": MODELS / "llama-b12110-bin-win-cpu-x64.zip",
        "unzip_to": MODELS,
    },
    "uavbench": {
        "desc": "UAVBench 场景包（166 MB，Git LFS）",
        "url": "https://media.githubusercontent.com/media/maferrag/UAVBench/main/data/UAVBench_scenarios.zip",
        "out": RAW / "UAVBench_scenarios.zip",
        "bytes": 166738456,
    },
    "alphabench": {
        "desc": "α³-Bench 113k episodes（320 MB，Git LFS，可选）",
        "url": "https://media.githubusercontent.com/media/maferrag/AlphaBench/main/113k_episodes%20dataset/113k_episodes%20dataset.zip",
        "out": RAW / "ab_clean.zip",
        "bytes": 336257124,
    },
}

QWEN_HELP = """
────────────────────────────────────────────────────────────────────
Qwen2.5-3B-Instruct（对话模型，约 2.0 GB）需要走 ModelScope：

    pip install modelscope
    modelscope download --model Qwen/Qwen2.5-3B-Instruct-GGUF --local_dir ./models/qwen3b

下载后取 q4_k_m 量化版本，重命名为：
    models/qwen2.5-3b-instruct-q4_k_m.gguf

在模型页确认文件名；不同镜像的文件命名可能略有差异。
参考大小：2.0 GB（2,013,698,592 B 量级）
────────────────────────────────────────────────────────────────────
"""


def human(n: int) -> str:
    return f"{n/1e9:.2f} GB" if n > 1e9 else f"{n/1e6:.1f} MB"


def download(url: str, out: Path, expect_bytes: int | None = None) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        size = out.stat().st_size
        if expect_bytes is None or size == expect_bytes:
            print(f"  已存在，跳过：{out.name}（{human(size)}）")
            return True
        print(f"  已存在但大小不符（{size} != {expect_bytes}），重新下载")
        out.unlink()

    print(f"  下载 → {out}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=1800) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            step = 8 << 20
            while True:
                chunk = r.read(step)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 / total
                    print(f"\r    {human(done)} / {human(total)}  {pct:5.1f}%", end="", flush=True)
        print()
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        print(f"  失败：{type(e).__name__}: {e}", file=sys.stderr)
        return False

    size = tmp.stat().st_size
    if expect_bytes and size != expect_bytes:
        print(f"  大小不符：{size} != {expect_bytes}，已删除", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(out)
    print(f"  完成：{out.name}（{human(size)}）")
    return True


def verify_sha256(path: Path, expect: str) -> bool:
    print("  校验 SHA-256 ...", end="", flush=True)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    ok = h.hexdigest() == expect
    print(" OK" if ok else f" 不匹配！\n    got {h.hexdigest()}\n    exp {expect}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--embedding", action="store_true")
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--uavbench", action="store_true")
    ap.add_argument("--alphabench", action="store_true")
    ap.add_argument("--qwen", action="store_true")
    ap.add_argument("--help-only", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.list or not any([args.embedding, args.engine, args.uavbench, args.alphabench, args.qwen, args.all]):
        print("可下载项：")
        for k, v in ITEMS.items():
            print(f"  --{k:12} {v['desc']}")
        print(f"  --qwen        Qwen2.5-3B-Instruct（只打印命令，需 modelscope）")
        print(f"\n当前目录：{ROOT}")
        return 0

    if args.qwen:
        print(QWEN_HELP)
        if args.help_only:
            return 0

    targets = []
    if args.all:
        targets = ["embedding", "engine", "uavbench"]   # alphabench 可选，不默认下
    else:
        if args.embedding:
            targets.append("embedding")
        if args.engine:
            targets.append("engine")
        if args.uavbench:
            targets.append("uavbench")
        if args.alphabench:
            targets.append("alphabench")

    ok_all = True
    for key in targets:
        item = ITEMS[key]
        print(f"\n▶ {item['desc']}")
        ok = download(item["url"], item["out"], item.get("bytes"))
        if ok and item.get("sha256"):
            ok = verify_sha256(item["out"], item["sha256"])
        if ok and item.get("unzip_to"):
            dest = item["unzip_to"]
            print(f"  解压到 {dest} ...", end="", flush=True)
            try:
                with zipfile.ZipFile(item["out"]) as z:
                    z.extractall(dest)
                print(" OK")
            except Exception as e:  # noqa: BLE001
                print(f" 失败：{e}")
                ok = False
        ok_all = ok_all and ok

    print("\n" + "=" * 60)
    if ok_all:
        print("完成。接着跑：python tools/check_env.py")
        print("模型启动命令见 models/README.md 第 4 节。")
    else:
        print("有项目失败，见上面的错误信息。", file=sys.stderr)
    print("=" * 60)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
