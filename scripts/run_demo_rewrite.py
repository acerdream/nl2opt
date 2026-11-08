"""离线演示完整闭环：理解（给错 IR）→ 执行 → 验证(失败) → 重写（给对 IR）→ 验证(通过)。

全程用 ScriptedProvider（不联网），按闭环的固定调用顺序返回：
  第 1 次调用 = understand（给错误 IR）
  第 2 次调用 = rewrite（给正确 IR）

> 早期还有第 2 步 `reflect`（单独一次调用产出「诊断建议」）。该环节已删除：
> 失败原因直接放进 rewrite 的提示词，让 LLM 一次完成「诊断 + 修复」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.config import load_config  # noqa: E402
from nl2opt.dataset import Sample  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ir_fixtures  # noqa: E402
from nl2opt.llm.fake import ScriptedProvider  # noqa: E402
from nl2opt.pipeline import run_sample  # noqa: E402


def main() -> int:
    cfg = load_config()
    correct_ir = _ir_fixtures.knapsack_ir().model_dump()

    # 错误 IR：把 knapsack-01 的价值系数 100 写成 1000（会得到 1080，验证失败）
    wrong_ir = json.loads(json.dumps(correct_ir))
    for t in wrong_ir["objective"]["expr"]["terms"]:
        if t["var"] == "x2":
            t["coef"] = 1000.0

    # 按调用顺序给出响应：understand → rewrite
    llm = ScriptedProvider(
        responses=[
            json.dumps(wrong_ir, ensure_ascii=False),      # understand
            json.dumps(correct_ir, ensure_ascii=False),    # rewrite
        ],
        name="fake-rewrite-demo",
    )

    # 构造一个“无内嵌 IR、无黄金 IR”的样例，强制走 LLM 理解链路
    sample = Sample(
        id="demo-rewrite",
        nl_text=(
            "有 3 件物品，价值分别为 60、100、120，重量分别为 10、20、30。"
            "背包容量为 50。每件物品只能整件取或不取。问如何选择使总价值最大？"
        ),
        problem_type="milp",
        expected={"objective_value": 220.0, "variables": {"x1": 0.0, "x2": 1.0, "x3": 1.0}},
    )

    result = run_sample(sample, llm=llm, cfg=cfg)

    print(f"样例: {result.sample_id}")
    print(f"成功: {result.success} | 总轮次: {result.rounds} | 状态: {result.status}")
    print(f"目标值: {result.objective_value}")
    print(f"解: {result.variables}")
    print(f"\n轨迹（每轮目标系数 x2 值）：")
    for step in result.trajectory:
        x2 = None
        for t in step["ir"]["objective"]["expr"]["terms"]:
            if t["var"] == "x2":
                x2 = t["coef"]
        checks = " | ".join(c["message"] for c in step["checks"])
        print(f"  round {step['round']}: x2 系数={x2}, 状态={step['status']}, 反思={refl or '-'}")
        print(f"    校验: {checks}")

    if result.success and result.rounds == 2:
        print("\n✅ 闭环验证成功：首轮理解错误 → 重写修复 → 第二轮通过。")
        return 0
    print(f"\n❌ 闭环未按预期修复。errors={result.errors}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
