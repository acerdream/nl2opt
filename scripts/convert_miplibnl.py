"""把 MIPLIB-NL 的实例目录转成本项目 jsonl（保留 data_dir 引用）。

用途：
- 生成一份**清单**，便于按规模筛选、批量评测、统计；
- jsonl 里保留 `extra.data_dir`，因此 `nl2opt eval --data <jsonl>` 仍会走符号化链路
  （`orchestrator` 判断依据就是 `extra.data_dir` 是否存在）。

注意：**不复制 CSV**。jsonl 只记录原目录的绝对/相对路径，CSV 仍留在 MIPLIB-NL 仓库内。

用法：
    python scripts/convert_miplibnl.py                      # 全部 223 题 → data/miplibnl.jsonl
    python scripts/convert_miplibnl.py --max-rows 50        # 只导出数据 ≤50 行的题
    python scripts/convert_miplibnl.py --no-data-only       # 只导出无数据文件的题
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl2opt.dataset import load_miplibnl_instance  # noqa: E402

DEFAULT_SRC = ROOT / "data" / "MIPLIB-NL" / "dataset"
DEFAULT_OUT = ROOT / "data" / "miplibnl.jsonl"


def count_data_rows(instance_dir: Path) -> int:
    """粗算数据行数（去表头），用于规模筛选。"""
    dd = instance_dir / "data"
    if not dd.exists():
        return 0
    n = 0
    for f in dd.glob("*"):
        if not f.is_file():
            continue
        try:
            with f.open(encoding="utf-8", errors="replace") as fh:
                n += max(0, sum(1 for _ in fh) - 1)
        except OSError:
            continue
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="MIPLIB-NL → jsonl（保留 data_dir）")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="MIPLIB-NL dataset 根目录")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 jsonl 路径")
    ap.add_argument("--max-rows", type=int, default=None, help="只导出数据行数 ≤ N 的题")
    ap.add_argument("--no-data-only", action="store_true", help="只导出没有数据文件的题")
    ap.add_argument("--limit", type=int, default=None, help="最多导出 N 题（按名字排序）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"✗ 目录不存在: {src}", file=sys.stderr)
        return 1

    insts = sorted(d for d in src.iterdir() if d.is_dir() and (d / "instance.json").exists())
    if not insts:
        print(f"✗ {src} 下没有含 instance.json 的子目录", file=sys.stderr)
        return 1

    selected: list[tuple[Path, int]] = []
    for d in insts:
        rows = count_data_rows(d)
        if args.no_data_only and rows != 0:
            continue
        if args.max_rows is not None and rows > args.max_rows:
            continue
        selected.append((d, rows))
    if args.limit:
        selected = selected[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    by_bucket = {"no_data": 0, "le50": 0, "le500": 0, "le5000": 0, "gt5000": 0}
    with out.open("w", encoding="utf-8") as fh:
        for d, rows in selected:
            sample = load_miplibnl_instance(d).samples[0]
            payload = {
                "id": sample.id,
                "problem_type": sample.problem_type,
                "nl_text": sample.nl_text,
                "expected": sample.expected,
                "extra": sample.extra,
                "trusted_reference": sample.trusted_reference,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            by_bucket[
                "no_data" if rows == 0 else
                "le50" if rows <= 50 else
                "le500" if rows <= 500 else
                "le5000" if rows <= 5000 else "gt5000"
            ] += 1

    print(f"✓ 已写出 {written} 题 → {out}")
    print(f"  分层: 无数据 {by_bucket['no_data']} / ≤50 {by_bucket['le50']} / "
          f"≤500 {by_bucket['le500']} / ≤5000 {by_bucket['le5000']} / >5000 {by_bucket['gt5000']}")
    print(f"  说明: jsonl 里 extra.data_dir 指向原实例目录，CSV 未复制。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
