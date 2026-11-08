"""理解环节：自然语言 → **SymbolicIR**（唯一输出形态）。

为什么只有一种输出：数值在数据文件里、规模中位 2,183 个变量 —— 让 LLM 直接写展平 IR
必然超输出预算（详见 README「符号化 IR」）。展平 IR 只由 `expand.expand()` 产生，
LLM 从不直接产出它。

> 早期还有一条「LLM 直接产展平 IR」的路径（`symbolic=False`），已删除：
> 它只服务于手写小样例，对真实数据集永远不可用，留着等于维护两套契约。

提示词：`SYMBOLIC_CORE_PROMPT`（领域卡片 `CARDS` 当前**未接入**符号路径）。
"""
from __future__ import annotations

from ..llm import LLMMessage, LLMProvider, complete_structured, extract_json
from ..symbolic import SymbolicIR
from .prompts import SYSTEM_PROMPT, build_system_prompt, prompt_stats  # noqa: F401

__all__ = ["understand", "SYSTEM_PROMPT", "build_system_prompt", "prompt_stats"]


def understand(
    text: str,
    llm: LLMProvider,
    meta: dict | None = None,
) -> SymbolicIR:
    """自然语言 → SymbolicIR（调用 LLM）。

    传入 `meta` 时写入 `meta["cards"]`（恒为空）/ `meta["prompt_chars"]` /
    `meta["symbolic"]`，便于归因。

    容错策略：
    - 输出被 max_tokens 截断（finish_reason=length）或 content 为空 → 重试一次
      （`reasoning_effort` 已全局固定 `low`，重试不再降档，只是一次同参数重试）；
    - 仍被截断 → 抛带指引的错误。**不**把截断断片送去解析：extract_json 的宽松兜底
      会把 `{'name': 'y', ...}` 这类碎片解析成畸形 dict，把「预算不足」误报成「IR 校验失败」。

    截断/空检测统一走 `complete_structured`（共享响应保护层），与 rewrite 一致。
    """
    stats = prompt_stats(text)
    if meta is not None:
        meta.update(stats)
    messages = [
        LLMMessage(role="system", content=build_system_prompt(text)),
        LLMMessage(role="user", content=text),
    ]
    resp = complete_structured(llm, messages, stage="understand")

    data = extract_json(resp.text)
    data = _normalize_ir_dict(data)
    return _symbolic_from_dict(data)


def _symbolic_from_dict(data: dict) -> SymbolicIR:
    """构造 SymbolicIR，并把常见写法归一化（缺省字段、字符串 var_type 等）。"""
    data = dict(data)
    # 变量族：dims 允许写成单个对象（而非列表）
    for v in data.get("variables") or []:
        if not isinstance(v, dict):
            continue
        dims = v.get("dims")
        if isinstance(dims, dict):
            v["dims"] = [dims]
        if dims is None:
            v["dims"] = []
    # 约束：允许 rhs 缺失但给出 rhs_ref
    for c in data.get("constraints") or []:
        if isinstance(c, dict) and "rhs" not in c and "rhs_ref" in c:
            c["rhs"] = c.pop("rhs_ref")
    return SymbolicIR.model_validate(data)


def _normalize_ir_dict(data: dict) -> dict:
    """对 LLM 输出的 IR dict 做健壮性归一化，避免因细微格式问题失败。

    处理：
    - 约束 sense 的 "==" → "="、"≤"/"<" → "<="、"≥"/">" → ">="
    - 目标 sense 的 "max" / "min" → "maximize" / "minimize"
    """
    for con in data.get("constraints", []):
        if not isinstance(con, dict):
            continue
        sense = con.get("sense")
        if sense == "==":
            con["sense"] = "="
        if sense in ("≤", "<"):
            con["sense"] = "<="
        if sense in ("≥", ">"):
            con["sense"] = ">="
    obj = data.get("objective")
    if isinstance(obj, dict):
        sense = obj.get("sense")
        if sense in ("max", "maximum"):
            obj["sense"] = "maximize"
        elif sense in ("min", "minimum"):
            obj["sense"] = "minimize"
    return data
