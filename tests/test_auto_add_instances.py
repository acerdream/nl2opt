"""展开时「自动补齐缺失实例」的测试。

背景：LLM 常声明变量族的维度，却引用笛卡尔积里不存在的组合
（如 `xs_{t1}_{k}_{c}` 却写 `xs_1_2_1`，而该组合不在取值集合里）。
早期实现直接报错、整题失败（chromaticindex32-8 卡在这里）。
现在自动把该组合补进变量族，并在 assumptions 留痕。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.expand import DataStore, ExpandError, expand  # noqa: E402
from nl2opt.ir import Sense, VarType  # noqa: E402
from nl2opt.solver import solve  # noqa: E402
from nl2opt.symbolic import (  # noqa: E402
    IndexSource, SymConstraint, SymExpr, SymObjective, SymbolicIR, SymTerm, SymVar,
)


def test_missing_combination_is_auto_added():
    """族 x 声明 i∈{a,b}、j∈{1,2}，约束却引用 x_a_3 → 自动补 x_a_3。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary, lower=0.0, upper=1.0,
                          dims=[IndexSource(kind="list", values=["a", "b"]),
                                IndexSource(kind="list", values=["1", "2"])])],
        objective=SymObjective(sense=Sense.maximize, expr=SymExpr(terms=[
            SymTerm(var="x_{i}_{j}", coef=1.0),
        ])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_a_3", coef=1.0)]),   # ← 组合不存在
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    assert "x_a_3" in ir.all_var_instances, "补齐的实例必须真的进变量声明"
    assert len(ir.all_var_instances) == 5, "4 个原有 + 1 个补齐"
    assert any("自动补齐" in a for a in ir.assumptions), "补齐必须留痕"
    # 补齐后仍可求解
    sol = solve(ir)
    assert sol.status == "optimal"


def test_auto_added_instance_is_usable():
    """补齐的实例能被目标函数正常使用（值参与求解）。"""
    sym = SymbolicIR(
        variables=[SymVar(name="y", var_type=VarType.binary, lower=0.0, upper=1.0,
                          dims=[IndexSource(kind="list", values=["1"])])],
        objective=SymObjective(sense=Sense.maximize, expr=SymExpr(terms=[
            SymTerm(var="y_9", coef=1.0),      # ← 不存在，补齐
        ])),
        constraints=[],
    )
    ir = expand(sym, DataStore("."))
    assert "y_9" in ir.all_var_instances
    sol = solve(ir)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(1.0)


def test_existing_combinations_not_duplicated():
    """正常引用已有组合时，不产生补齐记录、实例数不变。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["a", "b"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_a", coef=1.0)]),
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    assert len(ir.all_var_instances) == 2
    assert not any("自动补齐" in a for a in ir.assumptions)


def test_unknown_family_still_errors():
    """变量族本身不存在 → 仍然报错（补齐只针对已声明族的组合）。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["a"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="zzz_a", coef=1.0)]),
            rhs=1.0,
        )],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, DataStore("."))
    assert "未声明" in str(ei.value)
