"""符号层重写（`rewrite_symbolic`）与符号层防退化守卫的测试。

为什么需要它：展平 IR 的重写在工业规模题上不可行 —— 展开后上千条约束，
让 LLM 重写整份展平 JSON 既超输出预算、又易引用不存在的变量
（实测 chromaticindex32-8 因此失败）。符号层重写输出量小一到两个数量级。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.ir import Sense, VarType  # noqa: E402
from nl2opt.pipeline.rewrite import (  # noqa: E402
    rewrite_symbolic,
    symbolic_degeneracy_reason,
)
from nl2opt.symbolic import (  # noqa: E402
    IndexSource, SymConstraint, SymExpr, SymObjective, SymbolicIR, SymTerm, SymVar,
)


class FakeLLM:
    """返回预设文本的最小 provider。"""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        from nl2opt.llm import LLMResponse

        self.calls += 1
        return LLMResponse(text=self._text, usage={}, finish_reason="stop")


def _base_sym() -> SymbolicIR:
    return SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["a", "b"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]), rhs=1.0,
        )],
    )


def test_rewrite_symbolic_parses_symbolic_output():
    """符号重写应当解析出 SymbolicIR（而不是展平 IR）。"""
    new = _base_sym()
    new = new.model_copy(update={"constraints": list(new.constraints) + [
        SymConstraint(name="extra", sense=Sense.ge,
                      lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]), rhs=0.0),
    ]})
    llm = FakeLLM(json.dumps(new.model_dump(), ensure_ascii=False))
    out = rewrite_symbolic("nl", _base_sym(), "err", llm)
    assert isinstance(out, SymbolicIR)
    assert len(out.constraints) == 2
    assert llm.calls == 1


def test_symbolic_degeneracy_detects_family_drop():
    """变量族骤降 → 判退化。"""
    old = _base_sym()
    new = old.model_copy(update={"variables": []})
    why = symbolic_degeneracy_reason(old, new)
    assert why is not None and "变量族" in why


def test_symbolic_degeneracy_detects_constraint_drop():
    """符号约束骤降 → 判退化。"""
    old = SymbolicIR(
        variables=[SymVar(name="x", dims=[IndexSource(kind="list", values=["a"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(name=f"c{k}", sense=Sense.le,
                                   lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]),
                                   rhs=1.0) for k in range(6)],
    )
    new = old.model_copy(update={"constraints": old.constraints[:1]})
    why = symbolic_degeneracy_reason(old, new)
    assert why is not None and "符号约束数" in why


def test_symbolic_degeneracy_passes_when_growing():
    """规模持平或增长 → 不判退化。"""
    old = _base_sym()
    new = old.model_copy(update={"constraints": list(old.constraints) * 2})
    assert symbolic_degeneracy_reason(old, new) is None


def test_symbolic_degeneracy_detects_empty_objective():
    """目标被清空 → 判退化（目标为空模型无意义）。"""
    old = _base_sym()
    new = old.model_copy(update={
        "objective": SymObjective(sense=Sense.maximize, expr=SymExpr(terms=[])),
    })
    why = symbolic_degeneracy_reason(old, new)
    assert why is not None and "目标" in why
