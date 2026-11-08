"""IR 数值/边界/结构校验的单元测试（契约错误在入口拒绝，而非推迟成执行错误）。

对应 scripts/run_tests.py 里的 IR 校验相关断言，迁移到 pytest 以按域分层。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt.ir import (  # noqa: E402
    IR, Constraint, LinearExpr, Objective, Sense, Term, VarType, Variable,
)


def _obj(coef=1.0, var="x"):
    return Objective(sense=Sense.maximize, expr=LinearExpr(terms=[Term(var=var, coef=coef)]))


# ---------- 引用完整性（未声明变量）----------
# 动机：VRP/网络流这类按「节点对」展开的变量族规模是 n²，LLM 若把 index_set 列漏，
# 旧实现只报第一个缺失实例 → 修复环节无法定位整体缺口。


def test_undeclared_variable_rejected():
    """引用了未声明变量必须被拒绝。"""
    with pytest.raises(ValueError, match="未声明"):
        IR(variables=[Variable(name="x", var_type=VarType.continuous, lower=0)],
           objective=_obj(var="y"))


def test_undeclared_variable_error_names_family_and_index():
    """错误信息应把缺失实例归因到变量族与其缺失的索引组合。"""
    # 声明 x 族仅覆盖 (1,1)、(1,2)，但约束引用 x_8_19
    ir_args = dict(
        variables=[Variable(name="x", var_type=VarType.continuous, lower=0,
                            index_set=[[1, 1], [1, 2]])],
        objective=_obj(var="x_1_1"),
        constraints=[Constraint(name="deg_in_19", sense=Sense.ge,
                                lhs=LinearExpr(terms=[Term(var="x_8_19", coef=1.0)]),
                                rhs=1.0)],
    )
    with pytest.raises(ValueError) as exc_info:
        IR(**ir_args)
    msg = str(exc_info.value)
    assert "x_8_19" in msg
    assert "变量族 `x`" in msg          # 归因到族
    assert "缺索引 `8_19`" in msg        # 指出缺失的索引组合
    assert "约束 deg_in_19" in msg        # 指出引用位置


def test_undeclared_variable_reports_all_not_just_first():
    """多个未声明引用应一次性报全（含总数），而不是只报第一个。"""
    with pytest.raises(ValueError) as exc_info:
        IR(
            variables=[Variable(name="x", var_type=VarType.continuous, lower=0)],
            objective=_obj(var="x"),
            constraints=[
                Constraint(name=f"c{i}", sense=Sense.le,
                           lhs=LinearExpr(terms=[Term(var=f"z_{i}", coef=1.0)]), rhs=1.0)
                for i in range(9)
            ],
        )
    msg = str(exc_info.value)
    assert "共 9 处" in msg or "等共 9 处" in msg, msg
    # 且确实列出了多个具体实例
    assert msg.count("z_") >= 6


def test_ir_rejects_nan_and_inf():
    x = Variable(name="x", var_type=VarType.continuous, lower=0)

    # NaN 系数
    with pytest.raises(ValueError):
        IR(variables=[x], objective=_obj(coef=float("nan")))
    # Inf 常数
    with pytest.raises(ValueError):
        IR(variables=[x], objective=Objective(
            sense=Sense.minimize, expr=LinearExpr(terms=[], constant=float("inf"))))
    # NaN 边界
    with pytest.raises(ValueError):
        IR(variables=[Variable(name="x", lower=float("nan"))], objective=_obj())
    # Inf rhs
    with pytest.raises(ValueError):
        IR(variables=[x], objective=_obj(), constraints=[
            Constraint(name="c1", sense=Sense.le,
                       lhs=LinearExpr(terms=[Term(var="x", coef=1)]), rhs=float("inf"))])


def test_ir_rejects_bad_bounds():
    # lower > upper
    with pytest.raises(ValueError):
        IR(variables=[Variable(name="x", lower=5.0, upper=1.0)], objective=_obj())
    # binary 可行域与 [0,1] 无交集
    with pytest.raises(ValueError):
        IR(variables=[Variable(name="y", var_type=VarType.binary, lower=2.0)], objective=_obj(var="y"))
    with pytest.raises(ValueError):
        IR(variables=[Variable(name="y", var_type=VarType.binary, upper=-1.0)], objective=_obj(var="y"))


def test_ir_rejects_duplicate_constraint_names():
    x = Variable(name="x", var_type=VarType.continuous, lower=0)
    with pytest.raises(ValueError):
        IR(variables=[x], objective=_obj(), constraints=[
            Constraint(name="c1", sense=Sense.le, lhs=LinearExpr(terms=[Term(var="x", coef=1)]), rhs=1),
            Constraint(name="c1", sense=Sense.le, lhs=LinearExpr(terms=[Term(var="x", coef=1)]), rhs=2),
        ])


def test_ir_rejects_blank_or_overlong_variable_name():
    with pytest.raises(ValueError):
        Variable(name="   ")
    with pytest.raises(ValueError):
        Variable(name="x" * 201)


def test_ir_accepts_legal_bounds():
    # 合法边界仍须通过（避免过度收紧）
    for var in (Variable(name="y", var_type=VarType.binary),
                Variable(name="y", var_type=VarType.binary, lower=0.0, upper=1.0),
                Variable(name="y", var_type=VarType.binary, lower=0.5, upper=0.5)):
        IR(variables=[var], objective=_obj(var="y"))
    x = Variable(name="x", var_type=VarType.continuous, lower=0)
    IR(variables=[x], objective=_obj())


def test_ir_index_set_string_and_multidim():
    # 字符串索引与多维索引都能展开为实例名
    v1 = Variable(name="start", var_type=VarType.integer, index_set=["J1", "J2"])
    assert v1.instances() == ["start_J1", "start_J2"]

    v2 = Variable(name="assign", var_type=VarType.binary, index_set=[["J1", "M1"], ["J1", "M2"]])
    assert v2.instances() == ["assign_J1_M1", "assign_J1_M2"]

    # 空组合（多维索引里的 []）在入口拒绝
    with pytest.raises(ValueError):
        Variable(name="x", index_set=[[]])


def test_duplicate_terms_normalized():
    """重复线性项在构造时合并，表达式表示唯一（消除双后端分叉）。"""
    expr = LinearExpr(terms=[Term(var="x", coef=2.0), Term(var="x", coef=3.0),
                             Term(var="y", coef=0.0)])
    # 同变量项合并为 5*x；零系数项删除；按变量名排序
    assert [(t.var, t.coef) for t in expr.terms] == [("x", 5.0)]

    # 规范化幂等：二次验证不改变表示
    ir = IR(
        variables=[Variable(name="x", var_type=VarType.continuous, lower=0),
                   Variable(name="y", var_type=VarType.continuous, lower=0)],
        objective=Objective(sense=Sense.maximize, expr=LinearExpr(terms=[
            Term(var="x", coef=2.0), Term(var="x", coef=3.0)])),
    )
    d1 = ir.model_dump()
    d2 = IR.model_validate(d1).model_dump()
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def test_duplicate_variable_names_rejected():
    with pytest.raises(ValueError):
        IR(
            variables=[Variable(name="x"), Variable(name="x")],
            objective=_obj(),
        )
