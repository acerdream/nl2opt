"""重写环节：依据验证失败原因，在**符号层**增量修正模型。

输入：原始自然语言 + 上一轮 SymbolicIR + 验证失败原因
输出：修正后的 SymbolicIR（调用方随后 `expand()` 成展平 IR 再求解）

## 为什么只在符号层重写

展平 IR 的重写在工业规模题上**根本不可行** —— 展开后动辄上千条约束，
让 LLM 重新输出整份展平 JSON 既超输出预算、又极易引用不存在的变量
（实测 `IR 解析失败: 存在未声明的变量引用 K / pS_{k}_{u}_1`）。
符号化重写只在「变量族 + 索引维 + for_each」层面修改，输出量小一到两个数量级。

> 早期还有一套展平 IR 的重写（`REWRITE_PROMPT` + `rewrite()`），已随
> 「LLM 不再产展平 IR」一并删除。`degeneracy_reason` 保留 —— 它在
> **展开之后**仍用于判断本轮是否比上一轮显著缩水。

## 为什么强调「增量」

早期实现只说「只改出错的地方」，但实测大模型会**整份重建**，
把上一轮已经建对的部分丢掉 —— 表现为目标值 253 → 51 → 0 的**波动而非收敛**。
因此提示词明确要求：输出规模不得无故缩小；失败原因未指出的部分必须原样保留；
若失败原因是「漏了 X」，应**追加** X 而不是重写整份模型。
"""
from __future__ import annotations

import json

from ..ir import IR
from ..llm import LLMMessage, LLMProvider, complete_structured, extract_json
from ..symbolic import SymbolicIR


def degeneracy_reason(old: IR, new: IR, *, ratio: float = 0.6, min_size: int = 5) -> str | None:
    """检查**展开后**的新模型是否显著退化（相对旧模型缩水），返回原因或 None。

    动机：实测工业题重写会整份重建，把已建对的约束丢掉，目标值反而更远
    （253 → 51 → 0）。缩水超过 40% 视为退化，拒绝采用。

    只对「足够大」的模型生效（`min_size`），避免小模型正常重构被误拦。
    符号层还有一道更早生效的 `symbolic_degeneracy_reason`。
    """
    oc, nc = len(old.constraints), len(new.constraints)
    ov, nv = len(old.all_var_instances), len(new.all_var_instances)
    if oc >= min_size and nc < oc * ratio:
        return f"约束数骤降 {oc} → {nc}（< {ratio:.0%}）"
    if ov >= min_size and nv < ov * ratio:
        return f"变量数骤降 {ov} → {nv}（< {ratio:.0%}）"
    # 目标退化：新目标为空但旧目标非空
    if old.objective.expr.terms and not new.objective.expr.terms:
        return "目标函数被清空"
    return None


SYMBOLIC_REWRITE_PROMPT = """你是一名数学建模专家。下面是一个工业级优化问题的自然语言描述、上一轮的**符号化模型**（SymbolicIR JSON）、以及验证失败的原因。

请**在上一轮符号模型的基础上做增量修正**。

## 硬性规则
1. 只输出修正后的完整 SymbolicIR JSON，不要输出解释文字。
2. **保留已有的正确部分**：验证失败原因没有指出的变量族、约束、系数，必须**原样保留**。
3. **规模不得无故缩小**：失败原因是「漏了 X」→ 追加 X 对应的约束/变量族，不要删别的。
4. **必须保持符号化写法**（这是关键）：
   - 变量族：`{"name": "x", "dims": [{"kind":"list","values":[...]}], "var_type": "binary"}`
   - 索引维也可来自数据列：`{"kind": "csv", "file": "a.csv", "column": "job_id"}`
   - 项：`{"var": "x_{i}_{j}", "coef": 1.0}` 或 `{"var": "x_{i}", "ref": {"file","column","key_column","key_value"}}`
   - 逐索引约束：`"for_each": "j"` 或 `["i","t"]` 或 `{"kind":"rows","file","keys","names"}`
   - rhs：数值 / DataRef / `{"var": "..."}` / `{"expr": {...}}`
   - 索引算术：`{k-1}`、`{t+1}`（越界自动跳过）
   ⚠️ **绝对不要展开成逐实例的展平 IR** —— 不要为每个实例单独写变量、也不要逐条列出约束。
5. 只引用题面【数据文件清单】里真实存在的文件，不要虚构文件名。

## 自然语言问题
<<NL>>

## 上一轮符号模型（SymbolicIR）
<<SYM>>

## 验证失败原因
<<ERRORS>>

现在请输出**增量修正后**的 SymbolicIR JSON："""


def rewrite_symbolic(
    nl_text: str,
    sym: SymbolicIR,
    errors: str,
    llm: LLMProvider,
) -> SymbolicIR:
    """在符号层增量修正模型（不经过展平 IR）。

    输入输出都是 `SymbolicIR`；调用方负责随后 `expand()` 成展平 IR 再求解。
    """
    prompt = (SYMBOLIC_REWRITE_PROMPT
        .replace('<<NL>>', nl_text)
        .replace('<<SYM>>', json.dumps(sym.model_dump(), ensure_ascii=False))
        .replace('<<ERRORS>>', errors)
        )
    messages = [
        LLMMessage(role="system", content="你是数学建模专家，负责在已有符号模型上做增量修正。"),
        LLMMessage(role="user", content=prompt),
    ]
    # 截断/空输出经共享响应保护层拦截（与 understand 一致），不再把断片送去解析
    resp = complete_structured(llm, messages, stage="rewrite")
    data = extract_json(resp.text)
    return SymbolicIR.model_validate(data)


def symbolic_degeneracy_reason(old: SymbolicIR, new: SymbolicIR,
                               *, ratio: float = 0.6) -> str | None:
    """符号层的防退化检查：变量族数、约束条数骤降即拒绝。

    比展平层更早生效 —— 展平层的 2156 条约束在符号层可能只有 20 条，
    等展开后再发现缩水就太晚了（LLM 已经把结构丢了）。
    """
    ov, nv = len(old.variables), len(new.variables)
    oc, nc = len(old.constraints), len(new.constraints)
    if ov >= 1 and nv < ov * ratio:
        return f"变量族数骤降 {ov} → {nv}"
    if oc >= 3 and nc < oc * ratio:
        return f"符号约束数骤降 {oc} → {nc}"
    if old.objective.expr.terms and not new.objective.expr.terms:
        return "目标函数被清空"
    return None
