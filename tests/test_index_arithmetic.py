"""索引算术（`{k-1}` / `{t+1}`）的测试。

背景：MIPLIB-NL 实测中，LLM 常写「相邻颜色 `{k-1}`」「下一周期 `{t+1}`」这类索引。
早期实现只认「维名 / 字面值 / for_each 绑定」，整题在展开阶段就失败
（chromaticindex32-8、wachplan 都卡在这里）。

边界语义：**算术结果不在该维取值集合里 → 该项不产生实例（跳过）**，
与数据引用缺失时的 `on_missing: skip` 一致 —— 工业模型里「前一周期」在 t=1 时
本来就不存在，不该补 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.expand import DataStore, ExpandError, expand  # noqa: E402
from nl2opt.ir import Sense, VarType  # noqa: E402
from nl2opt.symbolic import (  # noqa: E402
    IndexSource, SymConstraint, SymExpr, SymObjective, SymbolicIR, SymTerm, SymVar,
)


def _colors(n: int = 4) -> list[str]:
    return [str(i) for i in range(1, n + 1)]


def test_index_offset_positive():
    """`x_{t+1}` 在 t 从 1..4 时 → 只生成 t=1..3 的项（t=4 的 t+1=5 越界被剔除）。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=_colors(4))])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="next", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{i+1}", coef=1.0)]),   # 无 for_each → i 求和
            rhs=10.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    got = {t.var for t in ir.constraints[0].lhs.terms}
    assert got == {"x_2", "x_3", "x_4"}, got


def test_index_offset_negative_skips_boundary():
    """`{k-1}` 在 k=1 时得 0（越界）→ 该项跳过，但整条约束仍保留其余项。"""
    sym = SymbolicIR(
        variables=[SymVar(name="a", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=_colors(4))],
                          dim_names=["k"])],
        objective=SymObjective(sense=Sense.minimize,
                               expr=SymExpr(terms=[SymTerm(var="a_{k}", coef=1.0)])),
        constraints=[SymConstraint(
            name="prev_{k}", sense=Sense.le, for_each="k",
            lhs=SymExpr(terms=[SymTerm(var="a_{k-1}", coef=1.0)]),
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    by: dict[str, set[str]] = {
        c.name: {t.var for t in c.lhs.terms} for c in ir.constraints
    }
    # k=1 → a_0 不存在 → 该项被跳过，约束变成空项（0 <= 1，恒真，保留结构）
    assert by["prev_1"] == set()
    assert by["prev_2"] == {"a_1"}
    assert by["prev_4"] == {"a_3"}
    assert any("索引越界" in a for a in ir.assumptions), "跳过必须留痕"


def test_index_offset_literal_base():
    """`{t28+1}`：`t28` 是「维名 t + 值 28」→ 28+1=29；29 在取值集合里则保留。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=[str(i) for i in range(1, 31)])],
                          dim_names=["t"])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{t}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{t28+1}", coef=1.0),
                               SymTerm(var="x_{t40+1}", coef=1.0)]),   # 41 越界
            rhs=5.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    got = {t.var for t in ir.constraints[0].lhs.terms}
    assert got == {"x_29"}, got


def test_varname_plus_number_is_value():
    """`t5`（维名+数字，无算术）→ 直接取第 5 个取值。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=[str(i) for i in range(1, 8)])],
                          dim_names=["t"])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{t}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{t5}", coef=1.0)]),
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    assert {t.var for t in ir.constraints[0].lhs.terms} == {"x_5"}


def test_index_offset_on_numeric_family():
    """连续整数维（如 1..6 的颜色）上 `{k+2}` 正常偏移。"""
    sym = SymbolicIR(
        variables=[SymVar(name="c", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=_colors(6))])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="c_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="jump", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="c_{i+2}", coef=1.0)]),
            rhs=99.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    got = {t.var for t in ir.constraints[0].lhs.terms}
    assert got == {"c_3", "c_4", "c_5", "c_6"}, got


def test_offset_on_non_numeric_family_reports():
    """对非数字取值（如 J1/J2）做算术偏移 → 明确报错，而不是静默清空。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["J1", "J2"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{i-1}", coef=1.0)]),
            rhs=1.0,
        )],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, DataStore("."))
    assert "不是数字" in str(ei.value)


def test_offset_unknown_base_reports():
    """`{zzz-1}` 的基数既不是维名/取值也没绑定 → 报错。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=_colors(3))])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{zzz-1}", coef=1.0)]),
            rhs=1.0,
        )],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, DataStore("."))
    assert "无法解析" in str(ei.value) or "zzz-1" in str(ei.value)
