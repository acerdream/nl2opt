"""约束名去重的回归测试。

背景：`for_each` 展开时若约束名模板漏写索引占位符，会生成多条同名约束，
而 `IR` 校验要求唯一 → 整题失败（实测 chromaticindex32-8 / wachplan 都是这个原因）。
展开器现在自动加后缀，并在 assumptions 里留痕。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.expand import DataStore, expand  # noqa: E402
from nl2opt.ir import Sense, VarType  # noqa: E402
from nl2opt.symbolic import (  # noqa: E402
    IndexSource, SymConstraint, SymExpr, SymObjective, SymbolicIR, SymTerm, SymVar,
)


def test_duplicate_constraint_names_are_suffixed():
    """模板里没放 {i} → 生成 3 条同名约束 → 自动加后缀而不是报错。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["a", "b", "c"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="pick", sense=Sense.le, for_each="i",     # ← 名字里没有 {i}
            lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]),
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    names = [c.name for c in ir.constraints]
    assert len(names) == 3
    assert len(set(names)) == 3, f"约束名必须唯一，实际 {names}"
    assert names[0] == "pick"
    assert names[1] == "pick_2" and names[2] == "pick_3"
    assert any("自动加后缀重命名" in a for a in ir.assumptions), "去重必须留痕"


def test_normal_names_untouched():
    """模板里正确放了 {i} → 名字不变、assumptions 不增加去重说明。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["a", "b"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="pick_{i}", sense=Sense.le, for_each="i",
            lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]),
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    assert [c.name for c in ir.constraints] == ["pick_a", "pick_b"]
    assert not any("自动加后缀" in a for a in ir.assumptions)
