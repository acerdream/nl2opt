"""符号化 IR + 展开器的测试。

覆盖：
- 无数据文件（纯参数）的题：b-ball 的最小复刻
- 带 CSV 数据引用的题：系数/rhs 来自文件
- 错误路径：键缺失、索引对不齐、占位符无来源、规模超限
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.expand import DataStore, ExpandError, expand, load_data_dir  # noqa: E402
from nl2opt.ir import Sense, VarType  # noqa: E402
from nl2opt.solver import solve  # noqa: E402
from nl2opt.symbolic import (  # noqa: E402
    DataRef, IndexSource, SymConstraint, SymExpr, SymObjective, SymbolicIR, SymTerm, SymVar,
)


# ---------------------------------------------------------------- b-ball

def _bball(n_players: int = 11, n_slots: int = 8, on_court: int = 5) -> SymbolicIR:
    """b-ball 的最小复刻：max-min 上场时间。"""
    return SymbolicIR(
        variables=[
            SymVar(name="x", var_type=VarType.binary, dims=[
                IndexSource(kind="list", values=[str(i) for i in range(n_players)]),
                IndexSource(kind="list", values=[str(t) for t in range(n_slots)]),
            ]),
            SymVar(name="z", var_type=VarType.continuous, lower=0.0),
        ],
        objective=SymObjective(
            sense=Sense.maximize,
            expr=SymExpr(terms=[SymTerm(var="z", coef=1.0)]),
        ),
        constraints=[
            # 每个 slot 恰好 on_court 人在场
            SymConstraint(
                name="capacity_{j}", sense=Sense.eq,
                lhs=SymExpr(terms=[SymTerm(var="x_{i}_{j}", coef=1.0)]),
                rhs=float(on_court), for_each="j",
            ),
            # 每人总时长 >= z
            SymConstraint(
                name="minplay_{i}", sense=Sense.ge,
                lhs=SymExpr(terms=[SymTerm(var="x_{i}_{j}", coef=1.0),
                                   SymTerm(var="z", coef=-1.0)]),
                rhs=0.0, for_each="i",
            ),
        ],
    )


def test_expand_bball_shapes():
    sym = _bball()
    ir = expand(sym, DataStore("."))          # b-ball 无数据文件，store 不会被用到

    assert len(ir.variables) == 2
    assert len(ir.all_var_instances) == 11 * 8 + 1, "88 个 x + 1 个 z"
    assert len(ir.constraints) == 8 + 11, "8 条容量 + 11 条最小时长"
    assert all(c.name.startswith("capacity_") for c in ir.constraints[:8])
    assert ir.is_milp is True


def test_expand_bball_solves_to_ratio():
    ir = expand(_bball(), DataStore("."))
    sol = solve(ir)
    assert sol.status == "optimal"
    # 总人次 8*5=40，11 人平均 40/11≈3.636；但 x 是 binary，每人上场次数必为整数，
    # 所以 max-min 落在 floor(40/11)=3。
    # （这正是 MIPLIB-NL 的 b-ball：optimal_value=1.5 = 3 slots × 0.5 quarter）
    assert sol.objective_value == pytest.approx(3.0, abs=1e-9)


def test_expand_bball_fewer_players_exact():
    """4 人 2 slot 每 slot 2 人 → 总 4 人次 / 4 人 = 1.0，可手算。"""
    ir = expand(_bball(n_players=4, n_slots=2, on_court=2), DataStore("."))
    sol = solve(ir)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- 带 CSV

@pytest.fixture()
def tiny_data():
    """两作业、单机：时长与交期都在 CSV 里。

    ⚠️ 不放在系统 temp（沙箱可能禁止写 workspace 之外的路径），
    而是建在 tests/ 下的临时目录，用完清理。
    """
    import shutil

    root = Path(__file__).resolve().parent / "_tmp_expand_data"
    shutil.rmtree(root, ignore_errors=True)
    d = root / "data"
    d.mkdir(parents=True)
    with (d / "jobs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["job_id", "dur", "weight"])
        w.writerow(["J1", "3", "2"])
        w.writerow(["J2", "2", "5"])
    try:
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_expand_with_data_refs(tiny_data: Path):
    """max sum_j weight_j * x_j, s.t. sum_j dur_j * x_j <= 4, x binary。

    权重与时长都来自 jobs.csv，答案可手算：x=J1 时成本 3 收益 2；x=J2 成本 2 收益 5；
    两者都选成本 5 > 4 → 最优只选 J2，收益 5。
    """
    store = load_data_dir(tiny_data)
    jobs = IndexSource(kind="csv", file="jobs.csv", column="job_id")

    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary, dims=[jobs])],
        objective=SymObjective(
            sense=Sense.maximize,
            expr=SymExpr(terms=[SymTerm(
                var="x_{i}",
                ref=DataRef(file="jobs.csv", column="weight", key_column="job_id", key_value="{i}"),
            )]),
        ),
        constraints=[SymConstraint(
            name="cap", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(
                var="x_{i}",
                ref=DataRef(file="jobs.csv", column="dur", key_column="job_id", key_value="{i}"),
            )]),
            rhs=4.0,
        )],
    )

    ir = expand(sym, store)
    assert sorted(ir.all_var_instances) == ["x_J1", "x_J2"]
    coefs = {(t.var, t.coef) for t in ir.objective.expr.terms}
    assert coefs == {("x_J1", 2.0), ("x_J2", 5.0)}, "系数应已解析成字面值"

    sol = solve(ir)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(5.0)
    assert sol.variables["x_J2"] == pytest.approx(1.0)


def test_expand_rhs_from_data(tiny_data: Path):
    """约束 rhs 也来自数据文件。"""
    store = load_data_dir(tiny_data)
    sym = SymbolicIR(
        variables=[SymVar(name="y", dims=[IndexSource(kind="csv", file="jobs.csv", column="job_id")])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="y_{i}", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="y_{i}", coef=1.0)]),
            rhs=DataRef(file="jobs.csv", column="dur", key_column="job_id", key_value="J1"),
        )],
    )
    ir = expand(sym, store)
    assert ir.constraints[0].rhs == pytest.approx(3.0)


# ---------------------------------------------------------------- 错误路径

def test_missing_key_reports_available(tiny_data: Path):
    store = load_data_dir(tiny_data)
    # 默认 on_missing="skip"（工业数据里"组合不存在"＝不可用）；
    # 需要严格时报错，要显式写 on_missing="error"
    ref = DataRef(file="jobs.csv", column="dur", key_column="job_id",
                  key_value="J9", on_missing="error")
    with pytest.raises(ExpandError) as ei:
        store.lookup(ref, {})
    msg = str(ei.value)
    assert "J9" in msg and "示例" in msg, "报错应指出缺失键并给可用键示例"


def test_missing_key_skips_by_default(tiny_data: Path):
    """默认 on_missing="skip"：键不存在 → 返回 None（调用方跳过该项）。"""
    store = load_data_dir(tiny_data)
    ref = DataRef(file="jobs.csv", column="dur", key_column="job_id", key_value="J9")
    assert store.lookup(ref, {}) is None


def test_missing_column_lists_columns(tiny_data: Path):
    store = load_data_dir(tiny_data)
    with pytest.raises(ExpandError) as ei:
        store.column("jobs.csv", "nope")
    assert "可用列" in str(ei.value)


def test_unbound_placeholder_rejected(tiny_data=None):
    """占位符既不是索引维、也没被 for_each 提供 → 校验应拦下。

    注意：`x_{i}` 在没有 for_each 时是**合法的**（`i` 视为求和维），
    所以这里用一个真正无来源的 `{zzz}` 来测。
    """
    sym = SymbolicIR(
        variables=[SymVar(name="x", dims=[IndexSource(kind="list", values=["a", "b"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{zzz}", coef=1.0)]),
            rhs=1.0,
        )],
    )
    probs = sym.validate_placeholders()
    assert any("zzz" in p for p in probs), f"应报占位符无来源，实际 {probs}"

    with pytest.raises(ExpandError):
        expand(sym, DataStore("."))


def test_no_foreach_means_sum():
    """`x_{i}` 无 for_each → 对该维求和，展开成一条含全部实例的约束。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", dims=[IndexSource(kind="list", values=["a", "b", "c"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x", coef=1.0)])),
        constraints=[SymConstraint(
            name="cap", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]),
            rhs=2.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    assert len(ir.constraints) == 1
    assert {t.var for t in ir.constraints[0].lhs.terms} == {"x_a", "x_b", "x_c"}


def test_digit_placeholder_is_literal():
    """`y_{i}_{1}` 的数字占位符是**字面索引值**，不能被当成求和维。

    回归用例：早期实现把 `{1}` 固化后丢了维对应关系，导致 `y_J1_1` 被展开 3 次、
    系数累加成 3.0（而不是三个不同实例各 1.0）。
    """
    sym = SymbolicIR(
        variables=[SymVar(name="y", var_type=VarType.binary, dims=[
            IndexSource(kind="list", values=["J1", "J2"]),
            IndexSource(kind="list", values=["1", "2", "3"]),
        ])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="y_{i}_{j}", coef=1.0)])),
        constraints=[SymConstraint(
            name="one_{i}", sense=Sense.eq, for_each="i",
            lhs=SymExpr(terms=[
                SymTerm(var="y_{i}_{1}", coef=1.0),
                SymTerm(var="y_{i}_{2}", coef=1.0),
                SymTerm(var="y_{i}_{3}", coef=1.0),
            ]),
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    assert [c.name for c in ir.constraints] == ["one_J1", "one_J2"]
    terms = [(t.var, t.coef) for t in ir.constraints[0].lhs.terms]
    assert terms == [("y_J1_1", 1.0), ("y_J1_2", 1.0), ("y_J1_3", 1.0)], terms


def test_var_on_rhs_is_moved_left():
    """`sum_i x_{i} <= cap`（cap 是变量）→ 自动移项成 `... - cap <= 0`。"""
    sym = SymbolicIR(
        variables=[
            SymVar(name="x", dims=[IndexSource(kind="list", values=["a", "b"])]),
            SymVar(name="cap", lower=0.0),
        ],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x", coef=1.0)])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]),
            rhs={"var": "cap"},
        )],
    )
    ir = expand(sym, DataStore("."))
    c = ir.constraints[0]
    assert c.rhs == 0.0
    got = {t.var: t.coef for t in c.lhs.terms}
    assert got == {"cap": -1.0, "x_a": 1.0, "x_b": 1.0}, got


def test_placeholder_count_mismatch_reports():
    """索引数与变量族维数不符 → 明确报错（而不是静默算错）。"""
    sym = SymbolicIR(
        variables=[SymVar(name="y", dims=[
            IndexSource(kind="list", values=["a"]),
            IndexSource(kind="list", values=["1", "2"]),
        ])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="y_{i}", coef=1.0)])),
        constraints=[],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, DataStore("."))
    msg = str(ei.value)
    assert "索引" in msg and "维" in msg, msg


def test_bare_index_suffix_parsed():
    """`y_{i}_1`（第二个索引没加花括号）也要能解析。"""
    sym = SymbolicIR(
        variables=[SymVar(name="y", var_type=VarType.binary, dims=[
            IndexSource(kind="list", values=["J1", "J2"]),
            IndexSource(kind="list", values=["1", "2"]),
        ])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="y_{i}_{j}", coef=1.0)])),
        constraints=[SymConstraint(
            name="pick_{i}", sense=Sense.le, for_each="i",
            lhs=SymExpr(terms=[
                SymTerm(var="y_{i}_1", coef=1.0),
                SymTerm(var="y_{i}_2", coef=1.0),
            ]),
            rhs=1.0,
        )],
    )
    ir = expand(sym, DataStore("."))
    terms = [(t.var, t.coef) for t in ir.constraints[0].lhs.terms]
    assert terms == [("y_J1_1", 1.0), ("y_J1_2", 1.0)], terms


def test_instance_cap_enforced():
    sym = SymbolicIR(
        variables=[SymVar(name="x", dims=[
            IndexSource(kind="list", values=[str(i) for i in range(500)]),
            IndexSource(kind="list", values=[str(j) for j in range(500)]),
        ])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x", coef=1.0)])),
        constraints=[],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, DataStore("."), max_instances=1000)
    assert "超过上限" in str(ei.value)


def test_empty_objective_rejected():
    sym = SymbolicIR(
        variables=[SymVar(name="x", dims=[IndexSource(kind="list", values=["a"])])],
        objective=SymObjective(sense=Sense.maximize, expr=SymExpr(terms=[])),
        constraints=[],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, DataStore("."))
    assert "目标函数展开后为空" in str(ei.value)
