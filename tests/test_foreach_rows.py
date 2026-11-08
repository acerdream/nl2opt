"""跨表关联（ForEachRows）的测试。

场景：单机排程 + 优先级约束
    precedence.csv: predecessor_id → successor_id
    约束：对每一对 (p, s)：start_s >= start_p + dur_p
其中 p/s 不是变量族的索引维，只能从文件的行里读出来。
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
    DataRef, ForEachRows, IndexSource, SymConstraint, SymExpr, SymObjective,
    SymbolicIR, SymTerm, SymVar,
)


@pytest.fixture()
def sched_data():
    """作业集 + 优先级关系（跨表关联）。"""
    root = Path(__file__).resolve().parent / "_tmp_rows_data"
    shutil.rmtree(root, ignore_errors=True)
    d = root / "data"
    d.mkdir(parents=True)
    with (d / "jobs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["job_id", "dur"])
        w.writerow(["J1", "3"])
        w.writerow(["J2", "2"])
        w.writerow(["J3", "1"])
    with (d / "precedence.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["predecessor_id", "successor_id"])
        w.writerow(["J1", "J2"])   # J2 必须在 J1 之后
        w.writerow(["J2", "J3"])   # J3 必须在 J2 之后
    try:
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _base(extra_constraints):
    jobs = IndexSource(kind="csv", file="jobs.csv", column="job_id")
    return SymbolicIR(
        variables=[
            SymVar(name="start", dims=[jobs], lower=0.0, upper=100.0),
            SymVar(name="makespan", lower=0.0),
        ],
        objective=SymObjective(sense=Sense.minimize,
                               expr=SymExpr(terms=[SymTerm(var="makespan", coef=1.0)])),
        constraints=extra_constraints + [
            # makespan >= start_i + dur_i
            SymConstraint(name="mk_{i}", sense=Sense.ge, for_each="i",
                          lhs=SymExpr(terms=[SymTerm(var="makespan", coef=1.0),
                                             SymTerm(var="start_{i}", coef=-1.0)]),
                          rhs=DataRef(file="jobs.csv", column="dur",
                                      key_column="job_id", key_value="{i}")),
        ],
    )


def test_foreach_rows_precedence(sched_data: Path):
    """对每一对 (p,s) 建一条优先级约束 —— 这就是 30n20b8 卡住的那种写法。"""
    store = load_data_dir(sched_data)
    prec = ForEachRows(file="precedence.csv",
                       keys=["predecessor_id", "successor_id"],
                       names=["p", "s"])
    sym = _base([
        SymConstraint(
            name="prec_{p}_{s}", sense=Sense.ge, for_each=prec,
            lhs=SymExpr(terms=[SymTerm(var="start_{s}", coef=1.0),
                               SymTerm(var="start_{p}", coef=-1.0)]),
            # start_s - start_p >= dur_p（rhs 也随 {p} 取值）
            rhs=DataRef(file="jobs.csv", column="dur",
                        key_column="job_id", key_value="{p}"),
        ),
    ])

    assert sym.validate_placeholders() == [], "p/s 由 ForEachRows 提供，应通过校验"

    ir = expand(sym, store)
    names = [c.name for c in ir.constraints]
    assert "prec_J1_J2" in names and "prec_J2_J3" in names
    assert len(ir.constraints) == 2 + 3, "2 条优先级 + 3 条 makespan"

    sol = solve(ir)
    assert sol.status == "optimal"
    # 串行执行：3 + 2 + 1 = 6
    assert sol.objective_value == pytest.approx(6.0, abs=1e-6)

    # 校验优先级确实生效
    st = sol.variables
    assert st["start_J2"] >= st["start_J1"] + 3 - 1e-6
    assert st["start_J3"] >= st["start_J2"] + 2 - 1e-6


def test_foreach_rows_single_column(sched_data: Path):
    """单列遍历：直接按 jobs.csv 的行绑定 {j}（等价于索引维，但走关系对通道）。"""
    store = load_data_dir(sched_data)
    rows_spec = ForEachRows(file="jobs.csv", keys=["job_id"], names=["j"])
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="csv", file="jobs.csv", column="job_id")])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="pick_{j}", sense=Sense.le, for_each=rows_spec,
            lhs=SymExpr(terms=[SymTerm(var="x_{j}", coef=1.0)]), rhs=1.0,
        )],
    )
    ir = expand(sym, store)
    assert [c.name for c in ir.constraints] == ["pick_J1", "pick_J2", "pick_J3"]


def test_foreach_rows_missing_file(sched_data: Path):
    store = load_data_dir(sched_data)
    spec = ForEachRows(file="nope.csv", keys=["a", "b"], names=["p", "s"])
    sym = _base([SymConstraint(
        name="c_{p}_{s}", sense=Sense.ge, for_each=spec,
        lhs=SymExpr(terms=[SymTerm(var="start_{s}", coef=1.0)]), rhs=0.0,
    )])
    with pytest.raises(ExpandError) as ei:
        expand(sym, store)
    assert "不存在" in str(ei.value)


def test_foreach_rows_missing_column(sched_data: Path):
    store = load_data_dir(sched_data)
    spec = ForEachRows(file="precedence.csv", keys=["a", "b"], names=["p", "s"])
    sym = _base([SymConstraint(
        name="c_{p}_{s}", sense=Sense.ge, for_each=spec,
        lhs=SymExpr(terms=[SymTerm(var="start_{s}", coef=1.0)]), rhs=0.0,
    )])
    with pytest.raises(ExpandError) as ei:
        expand(sym, store)
    assert "可用列" in str(ei.value)


def test_foreach_rows_keys_names_mismatch():
    """keys 与 names 数量不等 → 构造期就报错。"""
    with pytest.raises(Exception) as ei:
        ForEachRows(file="a.csv", keys=["x", "y"], names=["p"])
    assert "一一对应" in str(ei.value)


def test_foreach_rows_unbound_still_reported(sched_data: Path):
    """ForEachRows 只提供 p/s，引用 {q} 仍应被占位符校验拦下。"""
    store = load_data_dir(sched_data)
    spec = ForEachRows(file="precedence.csv",
                       keys=["predecessor_id", "successor_id"], names=["p", "s"])
    sym = _base([SymConstraint(
        name="c_{p}_{s}", sense=Sense.ge, for_each=spec,
        lhs=SymExpr(terms=[SymTerm(var="start_{q}", coef=1.0)]), rhs=0.0,
    )])
    probs = sym.validate_placeholders()
    assert any("q" in probs_line for probs_line in probs), probs
    with pytest.raises(ExpandError):
        expand(sym, store)


def test_composite_key_lookup():
    """复合键 (job_id, mode)：工业数据常见（同一 job 有多个 mode 的时长）。"""
    root = Path(__file__).resolve().parent / "_tmp_ck_data"
    shutil.rmtree(root, ignore_errors=True)
    d = root / "data"
    d.mkdir(parents=True)
    try:
        with (d / "durations.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["job_id", "mode", "duration"])
            w.writerow(["J1", "1", "9"])
            w.writerow(["J1", "2", "6"])
            w.writerow(["J2", "1", "11"])
            w.writerow(["J2", "2", "7"])
        store = load_data_dir(d)

        ref = DataRef(file="durations.csv", column="duration",
                      key_column=["job_id", "mode"], key_value=["{i}", "2"])
        assert store.lookup(ref, {"i": "J1"}) == pytest.approx(6.0)
        assert store.lookup(ref, {"i": "J2"}) == pytest.approx(7.0)

        # 复合键在模型里也能用
        sym = SymbolicIR(
            variables=[SymVar(name="y", var_type=VarType.binary, dims=[
                IndexSource(kind="csv", file="durations.csv", column="job_id"),
                IndexSource(kind="list", values=["1", "2"]),
            ])],
            objective=SymObjective(sense=Sense.minimize, expr=SymExpr(terms=[
                SymTerm(var="y_{i}_{j}", ref=DataRef(
                    file="durations.csv", column="duration",
                    key_column=["job_id", "mode"], key_value=["{i}", "{j}"])),
            ])),
            constraints=[SymConstraint(
                name="one_{i}", sense=Sense.le, for_each="i",
                lhs=SymExpr(terms=[SymTerm(var="y_{i}_{j}", coef=1.0)]), rhs=1.0,
            )],
        )
        ir = expand(sym, store)
        coefs = {t.var: t.coef for t in ir.objective.expr.terms}
        assert coefs == {"y_J1_1": 9.0, "y_J1_2": 6.0, "y_J2_1": 11.0, "y_J2_2": 7.0}, coefs
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_composite_key_missing_combo_reports():
    """复合键组合不存在 → 报错并给出前几行的键示例（不能静默当 0）。"""
    root = Path(__file__).resolve().parent / "_tmp_ck2_data"
    shutil.rmtree(root, ignore_errors=True)
    d = root / "data"
    d.mkdir(parents=True)
    try:
        with (d / "a.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["k1", "k2", "v"])
            w.writerow(["a", "1", "5"])
        store = load_data_dir(d)
        ref = DataRef(file="a.csv", column="v",
                      key_column=["k1", "k2"], key_value=["a", "9"], on_missing="error")
        with pytest.raises(ExpandError) as ei:
            store.lookup(ref, {})
        msg = str(ei.value)
        assert "不存在" in msg and "示例" in msg, msg
    finally:
        shutil.rmtree(root, ignore_errors=True)
