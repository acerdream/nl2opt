"""求解器模块：IR → scipy.optimize（linprog / milp，底层 HiGHS）。

不依赖 PuLP/OR-Tools，仅用已内置的 scipy。
- MILP：scipy.optimize.milp（integrality + LinearConstraint）
- LP：scipy.optimize.linprog（经典 A_ub/b_ub/A_eq/b_eq 形式）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from ..ir import IR, Sense, VarType


@dataclass
class SolverResult:
    """求解结果：状态 + 解向量 + 目标值。"""

    status: str  # optimal | infeasible | unbounded | error
    objective_value: float | None = None
    variables: dict[str, float] = field(default_factory=dict)
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "optimal"


def _split_constraints(ir: IR):
    """把约束按 (<=, >=, ==) 拆成经典 linprog 形式，同时保留 LinearConstraint（供 milp）。

    返回 (names, idx, c, integrality, bounds, A_ub, b_ub, A_eq, b_eq, linear_constraints)
    """
    names = ir.all_var_instances
    vmap = ir.var_map
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    # 目标系数（统一为最小化）
    # 用累加而非赋值：IR 层已规范化（合并同变量项），此处仍用 += 作**防御性正确**——
    # 即便有路径绕过规范化，这里也按**累加**处理（`2*x + 3*x = 5*x`），
    # 与 LinearExpr 的合并语义一致（历史缺陷：曾用赋值覆盖，重复项只取最后一项）。
    c = np.zeros(n)
    for t in ir.objective.expr.terms:
        c[idx[t.var]] += t.coef
    if ir.objective.sense == Sense.maximize:
        c = -c

    # 整数性
    integrality = np.zeros(n, dtype=int)
    for i, name in enumerate(names):
        vt = vmap[name].var_type
        if vt in (VarType.integer, VarType.binary):
            integrality[i] = 1

    # 边界
    lbs: list[float] = []
    ubs: list[float] = []
    for name in names:
        var = vmap[name]
        lb = var.lower if var.lower is not None else -math.inf
        ub = var.upper if var.upper is not None else math.inf
        if var.var_type == VarType.binary:
            lb = max(lb, 0.0)
            ub = min(ub, 1.0)
        lbs.append(lb)
        ubs.append(ub)
    bounds = Bounds(np.array(lbs), np.array(ubs))

    # 经典 linprog 形式
    A_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []
    A_eq_rows: list[np.ndarray] = []
    b_eq: list[float] = []

    # milp 的 LinearConstraint 形式
    rows: list[np.ndarray] = []
    cons_lb: list[float] = []
    cons_ub: list[float] = []

    for con in ir.constraints:
        row = np.zeros(n)
        for t in con.lhs.terms:
            # 累加（防御性正确）：与目标系数同理，避免重复项导致双后端分叉
            row[idx[t.var]] += t.coef
        rhs = con.rhs - con.lhs.constant

        if con.sense == Sense.le:
            A_ub_rows.append(row)
            b_ub.append(rhs)
            cons_lb.append(-math.inf)
            cons_ub.append(rhs)
        elif con.sense == Sense.ge:
            # >= 转 <= ：两边乘 -1
            A_ub_rows.append(-row)
            b_ub.append(-rhs)
            cons_lb.append(rhs)
            cons_ub.append(math.inf)
        else:  # eq
            A_eq_rows.append(row)
            b_eq.append(rhs)
            cons_lb.append(rhs)
            cons_ub.append(rhs)

        rows.append(row)

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub_arr = np.array(b_ub) if b_ub else None
    A_eq = np.vstack(A_eq_rows) if A_eq_rows else None
    b_eq_arr = np.array(b_eq) if b_eq else None

    linear_constraints = None
    if rows:
        A = np.vstack(rows)
        linear_constraints = LinearConstraint(A, np.array(cons_lb), np.array(cons_ub))

    return names, idx, c, integrality, bounds, A_ub, b_ub_arr, A_eq, b_eq_arr, linear_constraints


def solve(ir: IR, time_limit: float | None = None) -> SolverResult:
    """求解 IR，返回结构化结果。

    time_limit: 求解超时（秒）。None 表示不限时。
    HiGHS 原生支持 time_limit（通过 options 传入）；超时返回 status=1，
    映射为 "timeout" 状态。
    """
    try:
        (names, _idx, c, integrality, bounds,
         A_ub, b_ub, A_eq, b_eq, linear_constraints) = _split_constraints(ir)

        # 构造 HiGHS options（仅当设置超时时传入）
        options = None
        if time_limit is not None and time_limit > 0:
            options = {"time_limit": float(time_limit)}

        if ir.is_milp:
            res = milp(
                c=c,
                integrality=integrality,
                bounds=bounds,
                constraints=linear_constraints,
                options=options,
            )
        else:
            # linprog 的 bounds 需要 (lb, ub) 序列，而非 Bounds 对象
            bounds_seq = [(float(bounds.lb[i]), float(bounds.ub[i])) for i in range(len(c))]
            res = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds_seq,
                method="highs",
                options=options,
            )

        status = _map_status(res)
        if status != "optimal":
            return SolverResult(
                status=status,
                message=str(getattr(res, "message", "")),
                raw={"raw_status": getattr(res, "status", None)},
            )

        x = res.x if hasattr(res, "x") else None
        variables = {name: float(x[i]) for i, name in enumerate(names)} if x is not None else {}

        fun = float(res.fun) if res.fun is not None else None
        if fun is not None and ir.objective.sense == Sense.maximize:
            fun = -fun
        if fun is not None:
            # 目标常数项**不参与寻优**（不影响 argmin），但必须计入**目标值**。
            # 历史缺陷：此前漏加，导致同一 IR 在两个后端上给出不同目标值——
            # 目标常数统一由 LinearExpr.constant 承载，
            # 而本地 HiGHS 路径丢弃常数，属跨后端不一致。
            # 实测影响面很小（已评测数据中仅 1 题含非零常数），但属静默错误。
            fun += float(ir.objective.expr.constant or 0.0)

        vmap = ir.var_map
        for name, val in variables.items():
            if vmap[name].var_type == VarType.binary:
                variables[name] = 0.0 if val < 0.5 else 1.0

        return SolverResult(
            status="optimal",
            objective_value=fun,
            variables=variables,
            message=str(getattr(res, "message", "")),
            raw={"raw_status": getattr(res, "status", None)},
        )
    except Exception as exc:  # pragma: no cover - 防御性
        return SolverResult(status="error", message=f"{type(exc).__name__}: {exc}")


def _map_status(res: Any) -> str:
    """把 scipy / HiGHS 的 status 映射为统一状态。

    HiGHS 的 status 语义：
    - 0: optimal
    - 1: 达到**迭代或时间**限制
    - 2: infeasible
    - 3: unbounded

    status=1 一律视为 `timeout`，依据是**本模块只设置 time_limit**
    （见 `solve()`：`options = {"time_limit": ...}`，从不设置迭代/node 限制），
    因此 status=1 在实践中只能是时间限制触发。

    ⚠️ 不变量：若将来新增迭代上限或 node 上限，status=1 将不再等价于超时，
    此处**必须**重新区分（例如结合 `res.message` 判断），否则会把「迭代耗尽」
    误报为「超时」，进而误导失败归因。

    原始 `status` 与 `message` 已由调用方原样保留在 `SolverResult.raw` / `.message`，
    诊断信息不因这里的归并而丢失。
    """
    st = getattr(res, "status", None)
    if st is None:
        return "error"
    return {
        0: "optimal",
        1: "timeout",       # 依据见上方不变量说明
        2: "infeasible",
        3: "unbounded",
    }.get(st, "error")
