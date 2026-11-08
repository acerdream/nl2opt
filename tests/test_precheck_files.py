"""虚构数据文件预检的测试。

动机：LLM 会凭空造文件名（实测 `sorrell3` 该题**没有任何数据声明**，
却被写出 `sequences.csv` / `lcs_conflicts.csv`）。早期实现要读到那一行才报错，
此时一次完整 LLM 调用已经花掉。

`check_data_files` 在展开**之前**一次性列出所有缺失文件 + 实际可用文件。
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.expand import (  # noqa: E402
    DataStore,
    ExpandError,
    check_data_files,
    expand,
    load_data_dir,
    scan_data_refs,
)
from nl2opt.ir import Sense, VarType  # noqa: E402
from nl2opt.symbolic import (  # noqa: E402
    DataRef, ForEachRows, IndexSource, SymConstraint, SymExpr, SymObjective,
    SymbolicIR, SymTerm, SymVar,
)


@pytest.fixture()
def real_data():
    root = Path(__file__).resolve().parent / "_tmp_precheck"
    shutil.rmtree(root, ignore_errors=True)
    d = root / "data"
    d.mkdir(parents=True)
    with (d / "jobs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["job_id", "dur"])
        w.writerow(["J1", "3"])
    try:
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_scan_collects_all_reference_kinds(real_data: Path):
    """扫描要覆盖：csv 索引维、项的 ref、bind、rhs、for_each rows。"""
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary, dims=[
            IndexSource(kind="csv", file="jobs.csv", column="job_id"),
        ])],
        objective=SymObjective(sense=Sense.maximize, expr=SymExpr(terms=[
            SymTerm(var="x_{i}", coef=1.0),
        ])),
        constraints=[SymConstraint(
            name="c_{i}", sense=Sense.le,
            for_each=ForEachRows(file="jobs.csv", keys=["job_id"], names=["i"]),
            lhs=SymExpr(terms=[SymTerm(
                var="x_{i}",
                ref=DataRef(file="durations.csv", column="d", key_column="job_id",
                            key_value="{i}"),
            )]),
            rhs=DataRef(file="caps.csv", column="c", key_column="job_id", key_value="J1"),
        )],
    )
    got = scan_data_refs(sym)
    assert got == {"jobs.csv", "durations.csv", "caps.csv"}, got


def test_fabricated_file_is_rejected_before_expand(real_data: Path):
    """引用了不存在的文件 → 展开前就报错，且列出实际可用文件。"""
    store = load_data_dir(real_data)
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary, dims=[
            IndexSource(kind="csv", file="sequences.csv", column="job_id"),  # ← 虚构
        ])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, store)
    msg = str(ei.value)
    assert "sequences.csv" in msg
    assert "jobs.csv" in msg, "应列出实际可用的文件"
    assert "不存在的数据文件" in msg


def test_no_data_dir_skips_check():
    """无数据引用 / data_dir 不可用时不拦截（把判断交给具体读取处）。"""
    # 用 workspace 内的目录（沙箱不允许写系统 temp）
    store = load_data_dir(Path(__file__).resolve().parent)
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["a"])])],
        objective=SymObjective(sense=Sense.maximize,
                               expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
        constraints=[],
    )
    # 无数据引用 → 不拦截
    check_data_files(sym, store)


def test_path_prefixes_are_normalized(real_data: Path):
    """`./data/jobs.csv` / `data/jobs.csv` / `jobs.csv` 视为同一文件，不误报。"""
    store = load_data_dir(real_data)
    for f in ("jobs.csv", "./jobs.csv", "data/jobs.csv", "./data/jobs.csv"):
        sym = SymbolicIR(
            variables=[SymVar(name="x", var_type=VarType.binary, dims=[
                IndexSource(kind="csv", file=f, column="job_id"),
            ])],
            objective=SymObjective(sense=Sense.maximize,
                                   expr=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)])),
            constraints=[],
        )
        check_data_files(sym, store)     # 不应抛
        ir = expand(sym, store)
        assert "x_J1" in ir.all_var_instances


def test_multiple_missing_listed_at_once(real_data: Path):
    """多个虚构文件应**一次列全**，而不是报第一个就停。"""
    store = load_data_dir(real_data)
    sym = SymbolicIR(
        variables=[SymVar(name="x", var_type=VarType.binary,
                          dims=[IndexSource(kind="list", values=["a"])])],
        objective=SymObjective(sense=Sense.maximize, expr=SymExpr(terms=[
            SymTerm(var="x_{i}", coef=1.0,
                    ref=DataRef(file="a_fake.csv", column="v",
                                key_column="k", key_value="a")),
        ])),
        constraints=[SymConstraint(
            name="c", sense=Sense.le,
            lhs=SymExpr(terms=[SymTerm(var="x_{i}", coef=1.0)]),
            rhs=DataRef(file="b_fake.csv", column="v", key_column="k", key_value="a"),
        )],
    )
    with pytest.raises(ExpandError) as ei:
        expand(sym, store)
    msg = str(ei.value)
    assert "a_fake.csv" in msg and "b_fake.csv" in msg, msg
