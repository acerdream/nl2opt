"""手写 IR 夹具：供开发脚本（后端对拍、沙箱校验、重写演示）使用。

这里手写几个极小的模型，只覆盖机制验证所需的形态：

    knapsack      带整数的 0-1 规划（有明确最优值）
    lp            连续 LP（单纯形路径）
    diet          多约束 LP（不等式为主）
    assignment    二维 0-1 分配（多索引变量族）
    integer      纯整数（无连续变量）
    infeasible    不可行
    unbounded     无界
    multi_index   二维变量族（检验多索引实例名）

这些模型的数值是自洽的、可手算验证的，**不来自任何数据集**。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2opt.ir import (  # noqa: E402
    Constraint,
    IR,
    LinearExpr,
    Objective,
    Sense,
    Term,
    Variable,
    VarType,
)


def _ir(variables, sense, terms, constraints, description=""):
    return IR(
        variables=variables,
        objective=Objective(sense=sense, expr=LinearExpr(terms=terms)),
        constraints=constraints,
        description=description,
    )


def knapsack_ir() -> IR:
    """max 60x1+100x2+120x3, s.t. 10x1+20x2+30x3<=50 → 最优 220（选 x2,x3）。"""
    return _ir(
        variables=[Variable(name=f"x{i}", var_type=VarType.binary) for i in (1, 2, 3)],
        sense=Sense.maximize,
        terms=[Term(var="x1", coef=60.0), Term(var="x2", coef=100.0),
               Term(var="x3", coef=120.0)],
        constraints=[Constraint(
            name="capacity", sense=Sense.le,
            lhs=LinearExpr(terms=[Term(var="x1", coef=10.0),
                                  Term(var="x2", coef=20.0),
                                  Term(var="x3", coef=30.0)]),
            rhs=50.0,
        )],
        description="0-1 knapsack, opt=220",
    )


def lp_ir() -> IR:
    """min 2x+3y, s.t. x+y>=10, x<=6, y<=6 → x=6,y=4 → 24。"""
    return _ir(
        variables=[Variable(name="x", lower=0.0), Variable(name="y", lower=0.0)],
        sense=Sense.minimize,
        terms=[Term(var="x", coef=2.0), Term(var="y", coef=3.0)],
        constraints=[
            Constraint(name="demand", sense=Sense.ge,
                       lhs=LinearExpr(terms=[Term(var="x", coef=1.0),
                                             Term(var="y", coef=1.0)]), rhs=10.0),
            Constraint(name="cap_x", sense=Sense.le,
                       lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=6.0),
            Constraint(name="cap_y", sense=Sense.le,
                       lhs=LinearExpr(terms=[Term(var="y", coef=1.0)]), rhs=6.0),
        ],
        description="small LP, opt=24",
    )


def diet_ir() -> IR:
    """min 3a+2b+1c, s.t. 2a+b>=4, b+2c>=3, a,b,c>=0 → a=2,b=0,c=1.5 → 7.5。"""
    return _ir(
        variables=[Variable(name="a", lower=0.0), Variable(name="b", lower=0.0),
                   Variable(name="c", lower=0.0)],
        sense=Sense.minimize,
        terms=[Term(var="a", coef=3.0), Term(var="b", coef=2.0),
               Term(var="c", coef=1.0)],
        constraints=[
            Constraint(name="nutrient1", sense=Sense.ge,
                       lhs=LinearExpr(terms=[Term(var="a", coef=2.0),
                                             Term(var="b", coef=1.0)]), rhs=4.0),
            Constraint(name="nutrient2", sense=Sense.ge,
                       lhs=LinearExpr(terms=[Term(var="b", coef=1.0),
                                             Term(var="c", coef=2.0)]), rhs=3.0),
        ],
        description="diet-like LP, opt=7.5",
    )


def assignment_ir() -> IR:
    """2x2 分配，代价 [[1,2],[3,1]]，每人一项、每项一人 → 最小 1+1=2。"""
    return _ir(
        variables=[Variable(name="z", var_type=VarType.binary,
                            index_set=[[i, j] for i in ("w1", "w2") for j in ("t1", "t2")])],
        sense=Sense.minimize,
        terms=[Term(var="z_w1_t1", coef=1.0), Term(var="z_w1_t2", coef=2.0),
               Term(var="z_w2_t1", coef=3.0), Term(var="z_w2_t2", coef=1.0)],
        constraints=[
            Constraint(name="w1", sense=Sense.eq,
                       lhs=LinearExpr(terms=[Term(var="z_w1_t1", coef=1.0),
                                             Term(var="z_w1_t2", coef=1.0)]), rhs=1.0),
            Constraint(name="w2", sense=Sense.eq,
                       lhs=LinearExpr(terms=[Term(var="z_w2_t1", coef=1.0),
                                             Term(var="z_w2_t2", coef=1.0)]), rhs=1.0),
            Constraint(name="t1", sense=Sense.eq,
                       lhs=LinearExpr(terms=[Term(var="z_w1_t1", coef=1.0),
                                             Term(var="z_w2_t1", coef=1.0)]), rhs=1.0),
            Constraint(name="t2", sense=Sense.eq,
                       lhs=LinearExpr(terms=[Term(var="z_w1_t2", coef=1.0),
                                             Term(var="z_w2_t2", coef=1.0)]), rhs=1.0),
        ],
        description="2x2 assignment, opt=2",
    )


def integer_ir() -> IR:
    """min 4p+5q, p,q 整数, 3p+2q>=7 → p=3,q=0 → 12（p=1,q=2 得 14，劣）。"""
    return _ir(
        variables=[Variable(name="p", var_type=VarType.integer, lower=0.0),
                   Variable(name="q", var_type=VarType.integer, lower=0.0)],
        sense=Sense.minimize,
        terms=[Term(var="p", coef=4.0), Term(var="q", coef=5.0)],
        constraints=[Constraint(name="need", sense=Sense.ge,
                                lhs=LinearExpr(terms=[Term(var="p", coef=3.0),
                                                      Term(var="q", coef=2.0)]), rhs=7.0)],
        description="small integer program, opt=12",
    )


def multi_index_ir() -> IR:
    """二维变量族：min Σ x[i][j]，x>=1，2x2 → 4。"""
    pairs = [["a", "1"], ["a", "2"], ["b", "1"], ["b", "2"]]
    return _ir(
        variables=[Variable(name="x", index_set=pairs, lower=1.0)],
        sense=Sense.minimize,
        terms=[Term(var=f"x_{i}_{j}", coef=1.0) for i, j in pairs],
        constraints=[Constraint(name="dummy", sense=Sense.ge,
                                lhs=LinearExpr(terms=[Term(var="x_a_1", coef=1.0)]),
                                rhs=0.0)],
        description="multi-index family, opt=4",
    )


def infeasible_ir() -> IR:
    """x >= 5 且 x <= 3 → 不可行。"""
    return _ir(
        variables=[Variable(name="x", lower=0.0)],
        sense=Sense.minimize,
        terms=[Term(var="x", coef=1.0)],
        constraints=[
            Constraint(name="lo", sense=Sense.ge,
                       lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=5.0),
            Constraint(name="hi", sense=Sense.le,
                       lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=3.0),
        ],
        description="x>=5 and x<=3",
    )


def unbounded_ir() -> IR:
    """max x, x >= 0 → 无界。"""
    return _ir(
        variables=[Variable(name="x", lower=0.0)],
        sense=Sense.maximize,
        terms=[Term(var="x", coef=1.0)],
        constraints=[Constraint(name="lo", sense=Sense.ge,
                                lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=0.0)],
        description="max x, x>=0",
    )


# 供脚本遍历：名字 → 构造函数（求解类）
SOLVE_CASES = {
    "knapsack": knapsack_ir,
    "lp": lp_ir,
    "diet": diet_ir,
    "assignment": assignment_ir,
    "integer": integer_ir,
    "multi_index": multi_index_ir,
}

# 供脚本遍历：名字 → 构造函数（状态类）
STATUS_CASES = {
    "infeasible": infeasible_ir,
    "unbounded": unbounded_ir,
}
