"""Token 成本报表：各阶段 token 汇总 × 模型单价。

计费口径对齐 DeepSeek 官方「模型 & 价格」（三段分别计价）：
    费用 = 缓存命中输入 × 命中单价 + 缓存未命中输入 × 未命中单价 + 输出 × 输出单价
（单位：元 / 百万 tokens；高峰时段为北京时间 周一至周五 9:00-12:00、14:00-18:00，
  其余含周末与法定节假日为空闲时段，空闲价为高峰价的一半）

分阶段归因的两种来源：
- **精确**：结果含 `stage_usage`（由 `_UsageTrackingLLM` 按 understand/rewrite/
  rewrite 分别记账）——新运行的评测都带此字段；
- **估算**：旧结果只有整样本合计。此时按「各阶段调用次数」比例分摊 token，
  并在报告中显著标注为估算（真实 token 分布与调用次数并不成比例）。

用法：
    python scripts/report_cost.py                          # 自动扫描 artifacts/eval_*.json
    python scripts/report_cost.py artifacts/eval_a.json artifacts/eval_b.json
    python scripts/report_cost.py --rate peak              # 强制按高峰价
    python scripts/report_cost.py --cache-hit 0.02 --cache-miss 1 --output 4   # 覆盖单价
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl2opt.config import load_config  # noqa: E402

OUT = ROOT / "artifacts" / "COST_REPORT.md"
CST = timezone(timedelta(hours=8))  # 北京时间

# 分阶段判定的路由标记（与 orchestrator 的 route 文案一致）


def _is_peak(dt: datetime) -> bool:
    """高峰时段：北京时间 周一至周五（不含法定节假日）9:00-12:00、14:00-18:00。"""
    local = dt.astimezone(CST)
    if local.weekday() >= 5:      # 周六/周日
        return False
    h = local.hour + local.minute / 60
    return (9 <= h < 12) or (14 <= h < 18)


def _resolve_pricing(cfg, model: str | None) -> tuple[str, dict]:
    """按模型名解析定价块，返回 (模型名, {idle:{...}, peak:{...}})。

    支持两种配置形态：
    - 新形态（按模型分档）：`pricing.models.<模型名>.idle/peak`
    - 旧形态（单模型平铺）：`pricing.idle/peak`（向后兼容）
    模型名优先取结果文件里记录的 `model`，其次 `--model`，最后配置默认值。
    """
    pricing = cfg.llm.get("pricing", None) or {}
    models = pricing.get("models", None)
    if not models:
        return model or cfg.llm.get("model", "unknown"), pricing

    name = model or pricing.get("default_model") or cfg.llm.get("model")
    block = models.get(name, None)
    if block is not None:
        return name, block
    # 未登记的模型：给出可用清单，避免静默按错价计费
    raise SystemExit(
        f"模型 `{name}` 未在 config.yaml 的 llm.pricing.models 中登记单价；"
        f"可用：{', '.join(sorted(str(k) for k in models))}。"
        f"请在配置中补充，或用 --cache-hit/--cache-miss/--output 显式指定单价。"
    )


def _price(block: dict, rate: str, key: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    sub = block.get(rate, None)
    if sub is None:
        return 0.0
    return float(sub.get(key, 0.0))


def _tokens(usage: dict) -> tuple[int, int, int]:
    """从 usage 提取 (缓存命中输入, 未命中输入, 输出)。

    兼容缺失字段：无 cache 明细时全部计为「未命中」，与官方按未命中价计费一致。
    """
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    if hit == 0 and miss == 0:
        miss = prompt          # 无明细 → 视为全部未命中
    elif hit + miss != prompt:
        miss = max(0, prompt - hit)
    return hit, miss, completion


def _cost(hit: int, miss: int, out: int, p_hit: float, p_miss: float, p_out: float) -> float:
    return (hit * p_hit + miss * p_miss + out * p_out) / 1_000_000.0


def _estimate_stage_calls(res: dict) -> Counter:
    """从 trajectory 推断各阶段调用次数（旧结果的估算依据）。"""
    counts: Counter = Counter()
    counts["understand"] += 1     # 走 LLM 理解的样本至少一次理解调用
    for step in res.get("trajectory") or []:
        route = step.get("route") or ""
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Token 成本报表（各阶段 × 模型单价）")
    ap.add_argument("results", nargs="*", help="评测结果 JSON（默认扫描 artifacts/eval_*.json）")
    ap.add_argument("--rate", choices=["auto", "idle", "peak"], default="auto",
                    help="单价档位：auto=按结果文件时间判断（默认）")
    ap.add_argument("--model", default=None,
                    help="按此模型名取单价（默认读结果文件记录的 model，再退回配置默认值）")
    ap.add_argument("--cache-hit", type=float, default=None, help="缓存命中输入单价（元/百万）")
    ap.add_argument("--cache-miss", type=float, default=None, help="缓存未命中输入单价（元/百万）")
    ap.add_argument("--output", type=float, default=None, help="输出单价（元/百万）")
    ap.add_argument("--out", default=str(OUT), help="报告输出路径")
    args = ap.parse_args(argv)

    cfg = load_config()

    paths = [Path(p) for p in args.results]
    if not paths:
        paths = sorted((ROOT / "artifacts").glob("eval_*.json"))

    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        samples = [r for r in data if isinstance(r, dict)]
        if not any(r.get("token_usage") for r in samples):
            continue          # 无 LLM 调用的评测跳过

        rate = args.rate
        if rate == "auto":
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            rate = "peak" if _is_peak(mtime) else "idle"
            run_local = mtime.astimezone(CST).strftime("%Y-%m-%d %H:%M")
        else:
            run_local = "—"

        # 结果文件自带 model 字段（新格式）；缺失时用 --model 或配置默认值
        file_model = next((r.get("model") for r in samples if r.get("model")), None)
        model_name, pblock = _resolve_pricing(cfg, file_model or args.model)
        p_hit = _price(pblock, rate, "input_cache_hit", args.cache_hit)
        p_miss = _price(pblock, rate, "input_cache_miss", args.cache_miss)
        p_out = _price(pblock, rate, "output", args.output)

        hit = miss = out = 0
        calls = 0
        n_with = 0
        stage_exact: dict[str, dict] = {}
        stage_est: Counter = Counter()
        exact_mode = 0
        for r in samples:
            usage = r.get("token_usage") or {}
            if not usage:
                continue
            n_with += 1
            h, m, o = _tokens(usage)
            hit += h
            miss += m
            out += o
            calls += int(usage.get("calls", 0) or 0)
            su = r.get("stage_usage") or {}
            if su:
                exact_mode += 1
                for stage, u in su.items():
                    acc = stage_exact.setdefault(stage, {"calls": 0, "hit": 0, "miss": 0, "out": 0})
                    h2, m2, o2 = _tokens(u)
                    acc["calls"] += int(u.get("calls", 0) or 0)
                    acc["hit"] += h2
                    acc["miss"] += m2
                    acc["out"] += o2
            else:
                stage_est.update(_estimate_stage_calls(r))

        rows.append({
            "file": path.name,
            "model": model_name,
            "rate": rate,
            "run_local": run_local,
            "samples": len(samples),
            "n_with": n_with,
            "calls": calls,
            "hit": hit, "miss": miss, "out": out,
            "cost": _cost(hit, miss, out, p_hit, p_miss, p_out),
            "p": (p_hit, p_miss, p_out),
            "stage_exact": stage_exact,
            "stage_est": dict(stage_est),
            "exact_mode": exact_mode,
        })

    if not rows:
        print("未找到含 token 用量的评测结果（需带 --use-llm 运行）", file=sys.stderr)
        return 2

    total_cost = sum(r["cost"] for r in rows)
    total_tokens = sum(r["hit"] + r["miss"] + r["out"] for r in rows)

    lines: list[str] = []
    lines.append("# Token 成本报表（各阶段 token × 模型单价）\n")
    _models = sorted({r["model"] for r in rows})
    _mtxt = "、".join(f"`{m}`" for m in _models)
    lines.append(f"- 模型：{_mtxt}｜计价货币："
                 f"`{(cfg.llm.get('pricing', None) or {}).get('currency', 'CNY')}`（元/百万 tokens）")
    lines.append("- 计费口径：**缓存命中输入 + 缓存未命中输入 + 输出**，三段分别计价"
                 "（对齐 DeepSeek 官方「模型 & 价格」）")
    lines.append("- 高峰/空闲：北京时间 周一至周五（不含法定节假日）9:00-12:00、14:00-18:00 为高峰，"
                 "其余（含周末）为空闲，空闲价为高峰价的一半\n")

    lines.append("## 一、总览\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 纳入评测批次 | {len(rows)} |")
    lines.append(f"| 含 token 用量的样本 | {sum(r['n_with'] for r in rows)} |")
    lines.append(f"| LLM 调用次数 | {sum(r['calls'] for r in rows)} |")
    lines.append(f"| 输入 tokens（缓存命中 / 未命中） | "
                 f"{sum(r['hit'] for r in rows):,} / {sum(r['miss'] for r in rows):,} |")
    lines.append(f"| 输出 tokens | {sum(r['out'] for r in rows):,} |")
    lines.append(f"| 总 tokens | {total_tokens:,} |")
    lines.append(f"| **总成本** | **{total_cost:.4f} 元** |")
    if sum(r["n_with"] for r in rows):
        lines.append(f"| 单样本平均成本 | {total_cost / sum(r['n_with'] for r in rows):.4f} 元 |")
    lines.append("")

    lines.append("## 二、按批次\n")
    lines.append("| 结果文件 | 模型 | 运行时间(北京) | 档位 | 样本 | 调用 | 输入命中 | 输入未命中 | 输出 | 成本(元) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['file']} | `{r['model']}` | {r['run_local']} | {'高峰' if r['rate']=='peak' else '空闲'} | "
                     f"{r['n_with']}/{r['samples']} | {r['calls']} | {r['hit']:,} | {r['miss']:,} | "
                     f"{r['out']:,} | {r['cost']:.4f} |")
    lines.append("")

    lines.append("## 三、按阶段\n")
    any_exact = any(r["stage_exact"] for r in rows)
    if any_exact:
        agg: dict[str, dict] = {}
        for r in rows:
            p_hit, p_miss, p_out = r["p"]
            for stage, u in r["stage_exact"].items():
                acc = agg.setdefault(stage, {"calls": 0, "hit": 0, "miss": 0, "out": 0, "cost": 0.0})
                acc["calls"] += u["calls"]
                acc["hit"] += u["hit"]
                acc["miss"] += u["miss"]
                acc["out"] += u["out"]
                acc["cost"] += _cost(u["hit"], u["miss"], u["out"], p_hit, p_miss, p_out)
        lines.append("| 阶段 | 调用 | 输入命中 | 输入未命中 | 输出 | tokens 合计 | 成本(元) | 占比 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        tcost = sum(a["cost"] for a in agg.values()) or 1.0
        for stage, a in sorted(agg.items(), key=lambda kv: -kv[1]["cost"]):
            tot = a["hit"] + a["miss"] + a["out"]
            lines.append(f"| `{stage}` | {a['calls']} | {a['hit']:,} | {a['miss']:,} | {a['out']:,} | "
                         f"{tot:,} | {a['cost']:.4f} | {a['cost']/tcost*100:.1f}% |")
        lines.append("")
        if not all(r["exact_mode"] == r["n_with"] for r in rows):
            lines.append("> 注：部分批次的结果不含 `stage_usage`（旧格式），其阶段数据为**估算**，"
                         "见下节。\n")
    else:
        lines.append("暂无精确的分阶段数据（结果中不含 `stage_usage`）。\n")

    est_rows = [r for r in rows if r["stage_est"]]
    if est_rows:
        lines.append("### 旧结果的阶段估算（按调用次数分摊，仅供参考）\n")
        counts: Counter = Counter()
        for r in est_rows:
            counts.update(r["stage_est"])
        est_cost = sum(r["cost"] for r in est_rows)
        est_calls = sum(counts.values()) or 1
        lines.append("| 阶段 | 调用次数（由 trajectory 推断） | 调用占比 | 分摊成本(元，估算) |")
        lines.append("|---|---|---|---|")
        for stage, c in counts.most_common():
            lines.append(f"| `{stage}` | {c} | {c/est_calls*100:.1f}% | {est_cost*c/est_calls:.4f} |")
        lines.append("")
        lines.append("> ⚠️ 该分摊只按**调用次数**比例，未考虑各阶段 prompt/completion 体量差异"
                     "（如 `rewrite` 的 prompt 含完整 IR，单次 token 较高），"
                     "因此**不能**当作精确值。要精确统计请用带 `stage_usage` 的新结果重跑。\n")

    lines.append("## 四、说明与口径\n")
    lines.append("1. **精确性**：总 tokens 与总成本来自 `usage` 实测字段（`prompt_cache_hit_tokens` /"
                 " `prompt_cache_miss_tokens` / `completion_tokens`），与官方计费口径一致。")
    lines.append("2. **分阶段**：`stage_usage` 由 `_UsageTrackingLLM.set_stage()` 在"
                 " understand / rewrite 各调用点标注，属精确归因；"
                 "旧结果无此字段时退化为按调用次数分摊的估算。")
    lines.append("3. **单价可覆盖**：`--cache-hit/--cache-miss/--output` 或 `--rate peak|idle`；"
                 "默认取 `config.yaml` 的 `llm.pricing`。")
    lines.append("4. **节假日**：脚本按「工作日 9-12/14-18」判定高峰，"
                 "**未内建中国法定节假日日历**，节假日运行可能被高估为高峰价。")
    lines.append("5. 价格随官方调整而变，请以 [DeepSeek 官方价格页]"
                 "(https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) 与实际账单为准。")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入: {out}")
    print(f"总 tokens {total_tokens:,} | 总成本 {total_cost:.4f} 元")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
