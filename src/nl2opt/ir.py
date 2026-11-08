"""IR（中间表示）Schema：自然语言理解与求解建模之间的数据契约。

IR 与具体求解库解耦。下游（建模/代码生成/验证）只依赖这里的类型，
因此可以自由切换求解后端（scipy / PuLP / OR-Tools）。
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _require_finite(v: float | None, where: str) -> float | None:
    """数值字段必须是有限数（拒绝 NaN/Inf），否则契约错误被推迟成执行错误。"""
    if v is not None and not math.isfinite(float(v)):
        raise ValueError(f"{where} 必须是有限数值，得到 {v!r}")
    return v


class VarType(str, Enum):
    """变量类型：连续 / 整数 / 0-1（二进制）。"""

    continuous = "continuous"
    integer = "integer"
    binary = "binary"


class Sense(str, Enum):
    """目标方向与约束关系。"""

    minimize = "minimize"
    maximize = "maximize"
    le = "<="  # 小于等于
    ge = ">="  # 大于等于
    eq = "="   # 等于


class Term(BaseModel):
    """单个线性项：系数 * 变量（不含偏置）。"""

    var: str = Field(..., description="变量名（含索引下标后的实例名，如 x1、y_1）")
    coef: float = Field(..., description="系数")

    @field_validator("coef")
    @classmethod
    def _coef_finite(cls, v: float) -> float:
        return _require_finite(v, "系数")


class Variable(BaseModel):
    """变量声明。支持标量、一维索引（整数/字符串）与多维索引变量族。"""

    name: str = Field(..., description="变量基名，如 x 或 y")
    var_type: VarType = Field(VarType.continuous)
    index_set: list[int | str | list[int | str]] | None = Field(
        default=None,
        description=(
            "索引集合。None 表示标量变量。实例名按 `基名_索引` 展开："
            "整数索引 [1,2,3] → y_1,y_2,y_3；"
            "字符串索引 ['J1','J2'] → start_J1,start_J2；"
            "多维索引 [['J1','M1'],['J1','M2']] → assign_J1_M1,assign_J1_M2。"
            "字符串索引用于有名字的对象（作业/机器/代理/SKU）；"
            "多维索引用于「对象×对象」的变量族（如作业×机器分配）。"
        ),
    )
    lower: float | None = Field(
        default=None,
        description="下界。**不再默认 0**——`None` 表示无下界（变量可为负）。"
        "语义上不可能为负的变量必须**显式**写 `lower=0`：非负是**建模选择**而非默认假设。"
        "（历史缺陷：默认 0 会静默把净流量/现金差额/库存变化/时间偏移这类本可为负的量锁死，"
        "导致模型不可行或漏解。）",
    )
    upper: float | None = Field(default=None, description="上界，None 表示无上界")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        """变量名不能是空白、不能超长。

        空白名会让后续任何引用（日志、序列化、求解器传参）失效，
        把契约错误推迟成执行错误；超长名（>200）几乎必是理解错误而非真实建模。
        """
        if not v or not v.strip():
            raise ValueError("变量名不能为空或纯空白")
        if len(v) > 200:
            raise ValueError(f"变量名过长（{len(v)} 字符，上限 200）: {v[:40]}…")
        return v

    @field_validator("lower", "upper")
    @classmethod
    def _bound_finite(cls, v: float | None) -> float | None:
        return _require_finite(v, "变量边界")

    @model_validator(mode="after")
    def _check_bounds(self) -> "Variable":
        """边界一致性：lower <= upper；binary 的可行域必须与 [0,1] 相交。"""
        lo, up = self.lower, self.upper
        if lo is not None and up is not None and lo > up:
            raise ValueError(f"变量 {self.name} 的下界 {lo} 大于上界 {up}")
        if self.var_type == VarType.binary:
            # binary 有效域是 [max(lo,0), min(up,1)]，须非空
            if lo is not None and lo > 1:
                raise ValueError(f"binary 变量 {self.name} 下界 {lo} 越过 1（与 [0,1] 无交集）")
            if up is not None and up < 0:
                raise ValueError(f"binary 变量 {self.name} 上界 {up} 小于 0（与 [0,1] 无交集）")
        return self

    @field_validator("index_set")
    @classmethod
    def _check_index_set(
        cls, v: list[int | str | list[int | str]] | None
    ) -> list[int | str | list[int | str]] | None:
        """索引元素必须是整数、字符串，或它们组成的非空组合。"""
        if v is None:
            return v
        for item in v:
            if isinstance(item, (list, tuple)):
                if not item:
                    raise ValueError("多维索引中不允许空组合（如 []）")
                for sub in item:
                    if isinstance(sub, bool) or not isinstance(sub, (int, str)):
                        raise ValueError(f"多维索引元素必须是整数或字符串: {sub!r}")
            elif isinstance(item, bool) or not isinstance(item, (int, str)):
                raise ValueError(f"索引必须是整数、字符串或它们的组合: {item!r}")
        return v

    def instances(self) -> list[str]:
        """返回该变量的所有实例名（多维索引按 `_` 连接）。"""
        if self.index_set is None:
            return [self.name]
        out: list[str] = []
        for item in self.index_set:
            if isinstance(item, (list, tuple)):
                out.append(f"{self.name}_" + "_".join(str(x) for x in item))
            else:
                out.append(f"{self.name}_{item}")
        return out

class LinearExpr(BaseModel):
    """线性表达式：sum(coef * var) + constant。"""

    terms: list[Term] = Field(default_factory=list)
    constant: float = Field(default=0.0)

    @field_validator("constant")
    @classmethod
    def _constant_finite(cls, v: float) -> float:
        return _require_finite(v, "表达式常数")

    @model_validator(mode="after")
    def _normalize_terms(self) -> "LinearExpr":
        """合并同变量项、删除零系数项，并按变量名排序，保证表达式表示的**唯一性**。

        否则重复项会产生歧义：赋值覆盖（只取最后一项）与累加（`2*x + 3*x = 5*x`）
        会得到**不同模型**。规范化后唯一：哈希、日志、求解全部一致。

        零系数阈值 1e-12 与 `objective_term_count` / `objective_is_degenerate`
        的阈值一致，等价于浮点噪声。
        """
        merged: dict[str, float] = {}
        for t in self.terms:
            merged[t.var] = merged.get(t.var, 0.0) + float(t.coef)
        self.terms = [Term(var=v, coef=c) for v, c in sorted(merged.items())
                      if abs(c) > 1e-12]
        return self

    def __str__(self) -> str:  # 便于调试
        parts = []
        for t in self.terms:
            coef = t.coef
            if coef == int(coef):
                coef = int(coef)
            if coef == 1:
                parts.append(t.var)
            elif coef == -1:
                parts.append(f"-{t.var}")
            else:
                parts.append(f"{coef}*{t.var}")
        s = " + ".join(parts) if parts else "0"
        if self.constant:
            sign = "+" if self.constant > 0 else "-"
            s += f" {sign} {abs(self.constant)}"
        return s


class Constraint(BaseModel):
    """线性约束：lhs (sense) rhs。"""

    name: str = Field(..., description="约束名，如 c1")
    sense: Sense = Field(...)
    lhs: LinearExpr = Field(...)
    rhs: float = Field(..., description="右端常数")

    @field_validator("rhs")
    @classmethod
    def _rhs_finite(cls, v: float) -> float:
        return _require_finite(v, "约束 rhs")


class Objective(BaseModel):
    """目标函数。"""

    sense: Sense = Field(..., description="minimize 或 maximize")
    expr: LinearExpr = Field(...)


class IR(BaseModel):
    """完整中间表示。"""

    variables: list[Variable] = Field(...)
    objective: Objective = Field(...)
    constraints: list[Constraint] = Field(default_factory=list)
    description: str = Field(default="", description="问题简述（可选）")
    assumptions: list[str] = Field(
        default_factory=list,
        description="理解过程中的假设/待澄清点，如「假设班次数为整数」「成本数字原文缺失，暂用占位」。"
        "由理解环节填充，供人工复核与追溯。",
    )

    # ---------- 派生属性 ----------

    @property
    def is_milp(self) -> bool:
        """是否含整数/0-1 变量 → 需要用 MILP 求解器。"""
        return any(v.var_type in (VarType.integer, VarType.binary) for v in self.variables)

    @property
    def all_var_instances(self) -> list[str]:
        """所有变量实例名（展开索引后）。"""
        out: list[str] = []
        for v in self.variables:
            out.extend(v.instances())
        return out

    @property
    def var_map(self) -> dict[str, Variable]:
        """实例名 → Variable 声明。"""
        m: dict[str, Variable] = {}
        for v in self.variables:
            for inst in v.instances():
                m[inst] = v
        return m

    def objective_is_degenerate(self) -> bool:
        """目标函数是否退化：**无项**或**所有系数为 0**（等价于「不优化任何东西」）。

        危害：这类模型对任何可行解都返回同一个目标值（通常 0），
        当参考答案恰好为 0 时会被误判为「求解正确」，而实际上目标根本没被建模。

        注意：纯可行性问题（只找可行解）的目标本就为空，属**合法**情形，
        因此本方法只做事实判定（是否判错由上层决定）。
        """
        terms = self.objective.expr.terms
        if not terms:
            return True
        return all(abs(float(t.coef)) < 1e-12 for t in terms)

    def objective_term_count(self) -> int:
        """目标函数中**非零系数**项的个数。"""
        return sum(1 for t in self.objective.expr.terms if abs(float(t.coef)) >= 1e-12)

    # ---------- 校验 ----------

    @field_validator("variables")
    @classmethod
    def _no_dup_variable_names(cls, v: list[Variable]) -> list[Variable]:
        seen: set[str] = set()
        for var in v:
            for inst in var.instances():
                if inst in seen:
                    raise ValueError(f"变量实例名重复: {inst}")
                seen.add(inst)
        return v

    @model_validator(mode="after")
    def _check_references(self) -> "IR":
        valid = set(self.all_var_instances)
        # 变量族基名 → Variable：用于把「未声明实例」归因到具体族（提示 index_set 不完整）
        families = [v.name for v in self.variables]

        missing: list[tuple[str, str]] = []  # (位置, 变量名)

        def collect(expr: LinearExpr, where: str) -> None:
            for t in expr.terms:
                if t.var not in valid:
                    missing.append((where, t.var))

        collect(self.objective.expr, "目标函数")
        for c in self.constraints:
            collect(c.lhs, f"约束 {c.name}")

        if missing:
            # 一次报**全部**（不是只报第一个），并尽量把缺失实例归因到变量族。
            # 动机：VRP/网络流这类按「节点对」展开的题，变量族规模是 n²；LLM 若把
            # index_set 列漏了若干组合，只报第一个会让修复环节无法定位整体缺口。
            hints: list[str] = []
            for where, name in missing[:6]:
                fam = next((b for b in families if name.startswith(b + "_")), None)
                if fam is not None:
                    idx = name[len(fam) + 1:]
                    hints.append(f"{name}（{where}；变量族 `{fam}` 缺索引 `{idx}` 的实例）")
                else:
                    hints.append(f"{name}（{where}；没有任何变量族的前缀匹配它）")
            more = f"，等共 {len(missing)} 处" if len(missing) > 6 else ""
            raise ValueError(
                "存在**未声明的变量引用**（通常表示带索引变量族的 index_set 未覆盖所有"
                f"被引用的索引组合）：{'；'.join(hints)}{more}。"
                "修复：在 variables 中把该族 index_set 补全到覆盖所有引用到的索引组合，"
                "尤其按节点对/弧展开的变量族（规模 n²）不要只列一部分。"
            )

        if self.objective.sense not in (Sense.minimize, Sense.maximize):
            raise ValueError("目标 sense 必须是 minimize 或 maximize")
        for c in self.constraints:
            if c.sense not in (Sense.le, Sense.ge, Sense.eq):
                raise ValueError(f"约束 {c.name} 的 sense 非法: {c.sense}")

        # 约束名唯一：重复名会让「按名字定位/修复约束」的重写与日志歧义
        seen: set[str] = set()
        for c in self.constraints:
            if c.name in seen:
                raise ValueError(f"约束名重复: {c.name}")
            seen.add(c.name)
        return self

    @model_validator(mode="after")
    def _check_linearity(self) -> "IR":
        """IR 天然只允许线性项，因此线性性由结构保证。此方法仅为显式约束未来扩展。"""
        return self


# 供 LLM 输出 JSON 时做启发式校验的“宽松”入口（LLM 输出未必完全符合 Schema）
def ir_from_dict(data: dict[str, Any]) -> IR:
    """把 dict 解析为 IR，失败时抛出带上下文的 ValueError。"""
    try:
        return IR.model_validate(data)
    except Exception as exc:
        raise ValueError(f"IR 解析失败: {exc}") from exc
