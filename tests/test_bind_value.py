"""按分类值分派（bind）的测试。

场景（MIPLIB-NL 30n20b8 的最小复刻）：
    resources.csv: job_id → resource_type ("Mechaniker" | "Techniker")
    资源约束：每个作业占用的人员 = 它所需类型的招聘人数
    ⇒ 系数要在 `hire_Mechaniker` 与 `hire_Techniker` 之间**按数据值选择**
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.expand import DataStore, ExpandError, expand, load_data_dir  # noqa: E402
from nl2opt.ir import Sense, VarType  # noqa: E402
from nl2opt.solver import solve  # noqa: E402
from nl2opt.symbolic import (  # noqa: E402
    BindFrom, DataRef, ForEachRows, IndexSource, SymConstraint, SymExpr, SymObjective,
    SymbolicIR, SymTerm, SymVar,
)


@pytest.fixture()
def bind_data():
    root = Path(__file__).resolve().parent / "_tmp_bind_data"
    shutil.rmtree(root, ignore_errors=True)
    d = root / "data"
    d.mkdir(parents=True)
    with (d / "resources.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["job_id", "resource_type", "need"])
        w.writerow(["J1", "Mechaniker", "2"])
        w.writerow(["J2", "Techniker", "1"])
        w.writerow(["J3", "Mechaniker", "3"])
    try:
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _rtype_bind(job_key: str = "{i}") -> BindFrom:
    return BindFrom.model_validate({
        "ref": {"file": "resources.csv", "column": "resource_type",
                "key_column": "job_id", "key_value": job_key},
        "as": "rtype",
    })


def test_bind_selects_variable_by_actual_value(bind_data: Path):
    """`hire_{rtype}` 按 resources.csv 的值落到 hire_Mechaniker / hire_Techniker。"""
    store = load_data_dir(bind_data)
    jobs = IndexSource(kind="csv", file="resources.csv", column="job_id")

    sym = SymbolicIR(
        variables=[
            SymVar(name="x", var_type=VarType.binary, dims=[jobs]),
            SymVar(name="hire", lower=0.0, dim_names=["rtype"],
                   dims=[IndexSource(kind="list", values=["Mechaniker", "Techniker"])]),
        ],
        objective=SymObjective(sense=Sense.minimize, expr=SymExpr(terms=[
            SymTerm(var="hire_{rtype}", coef=1.0),
        ])),
        constraints=[
            # 对每个作业：它占用的人员 <= 该类型已招聘人数
            SymConstraint(
                name="cap_{i}", sense=Sense.le, for_each="i",
                lhs=SymExpr(terms=[
                    SymTerm(var="x_{i}", coef=1.0),
                    # 关键：系数变成 hire_Mechaniker 或 hire_Techniker，由数据值决定
                    SymTerm(var="hire_{rtype}", coef=-1.0, bind=_rtype_bind()),
                ]),
                rhs=0.0,
            ),
        ],
    )

    assert sym.validate_placeholders() == [], "rtype 由 bind 提供，应通过校验"

    ir = expand(sym, store)
    got = {c.name: {t.var: t.coef for t in c.lhs.terms} for c in ir.constraints}
    # J1/J3 是 Mechaniker，J2 是 Techniker
    assert got["cap_J1"] == {"x_J1": 1.0, "hire_Mechaniker": -1.0}, got["cap_J1"]
    assert got["cap_J2"] == {"x_J2": 1.0, "hire_Techniker": -1.0}, got["cap_J2"]
    assert got["cap_J3"] == {"x_J3": 1.0, "hire_Mechaniker": -1.0}, got["cap_J3"]


def test_bind_expanded_model_solves(bind_data: Path):
    """完整小模型：招聘人数要覆盖各类型需求 → 最优招聘 Mechaniker=3, Techniker=1。"""
    store = load_data_dir(bind_data)
    jobs = IndexSource(kind="csv", file="resources.csv", column="job_id")

    sym = SymbolicIR(
        variables=[
            SymVar(name="hire", lower=0.0, dim_names=["rtype"],
                   dims=[IndexSource(kind="list", values=["Mechaniker", "Techniker"])]),
        ],
        objective=SymObjective(sense=Sense.minimize, expr=SymExpr(terms=[
            SymTerm(var="hire_{rtype}", coef=1.0),
        ])),
        constraints=[
            # 每个作业的需求量 <= 该类型招聘数
            SymConstraint(
                name="need_{i}", sense=Sense.ge,
                for_each=ForEachRows(file="resources.csv", keys=["job_id"], names=["i"]),
                lhs=SymExpr(terms=[
                    SymTerm(var="hire_{rtype}", coef=1.0, bind=_rtype_bind()),
                ]),
                rhs=DataRef(file="resources.csv", column="need",
                            key_column="job_id", key_value="{i}"),
            ),
        ],
    )
    ir = expand(sym, store)
    sol = solve(ir)
    assert sol.status == "optimal"
    # Mechaniker 需 max(2,3)=3；Techniker 需 1
    assert sol.variables["hire_Mechaniker"] == pytest.approx(3.0)
    assert sol.variables["hire_Techniker"] == pytest.approx(1.0)
    assert sol.objective_value == pytest.approx(4.0)


def test_bind_missing_value_reports(bind_data: Path):
    """bind 的键不存在 → 明确报错。"""
    store = load_data_dir(bind_data)
    bad = BindFrom.model_validate({
        "ref": {"file": "resources.csv", "column": "resource_type",
                "key_column": "job_id", "key_value": "NOPE"},
        "as": "rtype",
    })
    sym = SymbolicIR(
        variables=[SymVar(name="hire", lower=0.0, dim_names=["rtype"],
                   dims=[IndexSource(kind="list", values=["Mechaniker", "Techniker"])])],
        objective=SymObjective(sense=Sense.minimize,
                               expr=SymExpr(terms=[SymTerm(var="hire_{rtype}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="hire_{rtype}", coef=1.0, bind=bad)]),
            rhs=1.0,
        )],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, store)
    assert "NOPE" in str(ei.value)


def test_bind_value_not_in_family_is_auto_added(bind_data: Path):
    """数据给出族里没有的分类值 → **自动补齐**该实例（而非整题失败）。

    语义变化（v30）：早期实现报错，实测会让一批题直接失败；现在补齐并留痕。
    若这些组合本不该存在，assumptions 里会提示复核索引维取值。
    """
    store = load_data_dir(bind_data)
    sym = SymbolicIR(
        variables=[SymVar(name="hire", lower=0.0, dim_names=["rtype"],
                   dims=[IndexSource(kind="list", values=["Mechaniker"])])],  # 缺 Techniker
        objective=SymObjective(sense=Sense.minimize,
                               expr=SymExpr(terms=[SymTerm(var="hire_{rtype}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c_{i}", sense=Sense.le,
            for_each=ForEachRows(file="resources.csv", keys=["job_id"], names=["i"]),
            lhs=SymExpr(terms=[SymTerm(var="hire_{rtype}", coef=1.0, bind=_rtype_bind())]),
            rhs=5.0,
        )],
    )
    ir = expand(sym, store)
    assert "hire_Techniker" in ir.all_var_instances, "缺失的分类值应被补齐"
    assert any("自动补齐" in a for a in ir.assumptions), "补齐必须留痕"


def test_bind_alias_as_parsed():
    """`as` 是 Python 关键字，Schema 用别名接收。"""
    b = BindFrom.model_validate({
        "ref": {"file": "a.csv", "column": "c", "key_column": "k", "key_value": "1"},
        "as": "rtype",
    })
    assert b.as_ == "rtype"
    # 也接受内部字段名
    b2 = BindFrom(as_="kind", ref=b.ref)
    assert b2.as_ == "kind"
