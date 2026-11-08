"""符号化 IR：面向「公式 + 数据分离」数据集（如 MIPLIB-NL）。

## 为什么需要独立的一层

现有的 `IR`（`ir.py`）是**完全展平**的：`Term.var` 必须是实例名（`x_1_2`）、
`Term.coef` 必须是字面 float。这套表示对中小规模、数值内嵌的题足够，
但对工业级数据集有两个硬缺口：

1. **数据在外部文件里**（CSV，最大百万行）——展平表示要求系数内嵌，做不到；
2. **规模**（中位 2183 变量 / 1388 约束）——逐项写出来超出 LLM 单次输出能力。

因此新增本模块表达「带索引变量族 + 数据引用 + 逐索引约束」的紧凑形式，
再由 `expand.py` 展开成现有 `IR`。

## 分层位置

    NL + schema ──► SymbolicIR ──expand()──► IR ──► solver / verifier
                     （本模块）    （读 CSV、展开索引）  （完全不变）

**关键**：数据必须在 `expand()` 阶段就解析成字面系数，**不能推迟到求解层** ——
否则 `verifier.check_feasibility`（把解代回约束）拿不到系数，
防「伪造 objective」的防线会失效。求解与验证共享同一份已解析的 `IR`。

## 表达力边界

支持：
- 变量族：`index_source` 为显式列表，或从 CSV 某列取值（自动去重、保序）；
- 多维族：`index_source` 给多个维度，按**笛卡尔积**展开（可用 `pairs_from` 指定只取存在的组合对）；
- 线性项：系数为字面 float，或 `DataRef`（按键从 CSV 查一个数）；
- 逐索引约束：`for_each` 指定沿哪个索引维生成多条约束；
- 求和：`SymTerm.var` 用 `{j}` 占位符表示"该维取当前值"；
- rhs 也可以是 `DataRef`。

不支持（需退回展平 IR 或人工建模）：
- 非线性、析取（需显式大 M）、目标含 min/max 算子；
- 嵌套求和的多层聚合（如 `sum_{j} sum_{k}` 需在 `SymTerm` 里直接列出组合）；
- 数据文件间的多列 join（只支持单键单值查找）。
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .ir import Sense, VarType

__all__ = [
    "DataRef",
    "IndexSource",
    "SymTerm",
    "SymExpr",
    "VarRef",
    "ForEachRows",
    "SymConstraint",
    "SymObjective",
    "SymVar",
    "SymbolicIR",
]

# 索引占位符：{j} / {i}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
# 变量族默认索引名（按维序）；可用 SymVar.dim_names 覆盖
_INDEX_NAMES = ("i", "j", "k", "l", "m")


def _opaque_index(p: str) -> bool:
    """该占位符是否属于「静态校验无法判定、留给 expand 解析」的形态。

    - 纯数字（`1`）→ 字面索引值；
    - 含数字或算术运算符（`t5`、`k-1`、`t28+1`）→ 可能是「维名+值」或算术偏移。

    `validate_placeholders` 看不到 CSV 里的实际取值，也无从知道变量族的维名，
    因此这些形态一律放行，由 `expand()` 在读到数据后按真实维度判定
    （那里能给出更精确的错误）。
    """
    return p.isdigit() or bool(re.search(r"[0-9+-]", p))


class DataRef(BaseModel):
    """从数据文件按键查找一个标量。

    语义：在 `file` 中找 `key_column == key_value` 的行，取 `column` 列的值，再 `* scale + offset`。

    **支持复合键**（工业数据常见：`(job_id, mode)` 才能定位一行）：
    `key_column` 与 `key_value` 都可写成列表，按位置配对。

    `key_value` 的每一项支持两种形式：
    - 字面值（`"1"` / `3`）：固定键；
    - 占位符（`"{j}"`）：键随当前循环索引变量 `j` 的取值变化。
    """

    file: str = Field(..., description="相对 data_dir 的文件名，如 durations.csv")
    column: str = Field(..., description="要取的列名")
    key_column: str | list[str] = Field(
        ..., description="用于匹配的键列名；复合键写成列表"
    )
    key_value: str | int | float | list[str | int | float] = Field(
        ..., description="键值；复合键写成列表，可为 {j} 占位符"
    )
    scale: float = Field(default=1.0, description="取值后乘的系数")
    offset: float = Field(default=0.0, description="取值后加的偏移")
    on_empty: Literal["skip", "zero", "error"] = Field(
        default="skip",
        description=(
            "该键的取值为空时怎么办。工业数据里「空」通常表示**该组合不可用**"
            "（如 MIPLIB-NL 的 `duration_3` 为空 = 该作业不支持 mode 3），"
            "此时对应项应视为不存在 —— `skip`（默认，跳过该项）最符合语义；"
            "`zero` 当 0 处理；`error` 直接报错。"
        ),
    )
    on_missing: Literal["skip", "zero", "error"] = Field(
        default="skip",
        description=(
            "**该键在文件里根本不存在**时怎么办。工业数据常只列出存在的组合"
            "（如 `time_windows.csv` 只给可用的 mode，job 10 没有 mode 3 那一行）——"
            "`skip`（默认）跳过该项；`zero` 当 0；`error` 报错。"
        ),
    )

    @field_validator("file", "column")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("DataRef 的文件名/列名不能为空")
        return v.strip()

    @model_validator(mode="after")
    def _align_keys(self) -> "DataRef":
        kc = self.key_column if isinstance(self.key_column, list) else [self.key_column]
        kv = self.key_value if isinstance(self.key_value, list) else [self.key_value]
        if not kc:
            raise ValueError("DataRef 至少要有一个键列")
        if len(kc) != len(kv):
            raise ValueError(
                f"DataRef 的 key_column 有 {len(kc)} 个、key_value 有 {len(kv)} 个，必须一一对应"
            )
        return self

    @property
    def key_columns(self) -> list[str]:
        return list(self.key_column) if isinstance(self.key_column, list) else [self.key_column]

    @property
    def key_values(self) -> list[str | int | float]:
        return list(self.key_value) if isinstance(self.key_value, list) else [self.key_value]

    @property
    def placeholders(self) -> set[str]:
        """键值里用到的全部索引占位符名。"""
        out: set[str] = set()
        for v in self.key_values:
            if isinstance(v, str):
                m = _PLACEHOLDER.fullmatch(v.strip())
                if m:
                    out.add(m.group(1))
        return out

    @property
    def placeholder(self) -> str | None:
        """单键且是占位符时返回该名字（向后兼容旧用法）。"""
        ph = self.placeholders
        return next(iter(ph)) if len(ph) == 1 else None


class IndexSource(BaseModel):
    """一个索引维的取值来源。

    - `{"kind": "list", "values": ["J1","J2"]}` —— 显式列举；
    - `{"kind": "csv", "file": "durations.csv", "column": "job_id"}` —— 取该列去重后的值（保序）。
    """

    kind: Literal["list", "csv"]
    values: list[str | int] | None = Field(default=None, description="kind=list 时的取值")
    file: str | None = Field(default=None, description="kind=csv 时的文件")
    column: str | None = Field(default=None, description="kind=csv 时的列名")

    @model_validator(mode="after")
    def _check(self) -> "IndexSource":
        if self.kind == "list":
            if not self.values:
                raise ValueError("kind=list 时 values 不能为空")
            for v in self.values:
                if isinstance(v, bool) or not isinstance(v, (int, str)):
                    raise ValueError(f"索引值必须是整数或字符串: {v!r}")
        else:
            if not self.file or not self.column:
                raise ValueError("kind=csv 时必须给 file 与 column")
        return self


class BindFrom(BaseModel):
    """从数据取一个**字符串值**，绑定成索引变量，用于「按分类值分派」。

    典型场景（MIPLIB-NL `30n20b8`）：`resources.csv` 给出每个作业需要
    `Mechaniker` 还是 `Techniker`。资源约束要写成"如果该作业需要 Techniker，
    就用 `hire_Techniker`，否则用 `hire_Mechaniker`" —— 这是**按列值选择变量**。

    写法（项的 `bind` 字段）：
    ```json
    {"var": "hire_{rtype}",
     "bind": {"ref": {"file": "resources.csv", "column": "resource_type",
                      "key_column": "job_id", "key_value": "{i}"},
              "as": "rtype"}}
    ```

    展开时：先查 `resource_type` 的**原始字符串**（如 "Techniker"），
    把它绑定到 `rtype`，于是 `hire_{rtype}` → `hire_Techniker`。
    """

    ref: "DataRef" = Field(..., description="取值来源（返回原始字符串，不转数值）")
    as_: str = Field(..., alias="as", description="绑定成的索引变量名")

    model_config = {"populate_by_name": True}

    @field_validator("as_")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("bind.as 不能为空")
        if not _PLACEHOLDER.fullmatch("{" + v + "}"):
            raise ValueError(f"bind.as 的 {v!r} 必须是合法标识符")
        return v


class SymTerm(BaseModel):
    """符号化线性项：系数为字面值、数据引用，或二者的乘积。

    `var` 是实例名模板，可含 `{j}` 占位符；占位符必须由所在约束的 `for_each`、
    所属变量族的索引维、或本项的 `bind` 提供。

    系数语义：
    - 只给 `coef` → 常数系数；
    - 只给 `ref` → 取数据文件里的值；
    - **两个都给** → `coef × ref值`（LLM 常用 `{"ref": dur, "coef": -1}` 表示 `-dur`）。
    """

    var: str = Field(..., description="变量实例名模板，如 x_{j} 或 hire_{rtype}")
    coef: float | None = Field(
        default=None, description="字面系数；与 ref 同时给出时作为 ref 值的倍数"
    )
    ref: DataRef | None = Field(default=None, description="数据引用（与 coef 相乘）")
    bind: BindFrom | None = Field(
        default=None, description="先按数据值绑定一个索引变量（按分类值分派）"
    )

    @model_validator(mode="after")
    def _at_least_one(self) -> "SymTerm":
        if self.coef is None and self.ref is None:
            raise ValueError(f"项 {self.var} 必须给出 coef 或 ref 至少一个")
        return self

    @property
    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.var))


class SymExpr(BaseModel):
    """符号化线性表达式：若干项 + 常数。"""

    terms: list[SymTerm] = Field(default_factory=list)
    constant: float = Field(default=0.0)


class VarRef(BaseModel):
    """右端是一个变量（而非常数/数据）。

    数学上 `lhs <= coef * var` 等价于 `lhs - coef*var <= 0`，`expand()` 会自动移项。
    存在的理由：LLM 写约束时习惯把"被比较的量"放右边（如 `sum_j ... <= hire_M`），
    硬要求它换边会显著提高失败率。
    """

    var: str = Field(..., description="变量名或其模板，如 hire_M / x_{j}")
    coef: float = Field(default=1.0, description="系数（移项时取负）")

    @field_validator("var")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("rhs 的变量名不能为空")
        return v.strip()


class ForEachRows(BaseModel):
    """按**数据文件的行**遍历，把若干列绑定成索引变量。

    用于表达「跨表关联」这类约束：例如优先级约束
    `start_s >= start_p + dur_p` 需要"对每一对 (前驱 p, 后继 s)"生成一条，
    而 p/s 只能从 `precedence.csv` 的行里读出来 —— 它们**不是**变量族的索引维。

    写法：
    ```json
    "for_each": {"kind": "rows", "file": "precedence.csv",
                 "keys": ["predecessor_id", "successor_id"],
                 "names": ["p", "s"]}
    ```

    - `keys`：文件里的列名，按顺序取；
    - `names`：绑定的索引变量名，与 `keys` 一一对应；展开时 `{p}` / `{s}` 会被替换成该行的值。
    """

    kind: Literal["rows"] = "rows"
    file: str = Field(..., description="相对 data_dir 的文件名")
    keys: list[str] = Field(..., min_length=1, description="要取的列（按顺序）")
    names: list[str] = Field(..., min_length=1, description="绑定的索引变量名（与 keys 对齐）")

    @model_validator(mode="after")
    def _align(self) -> "ForEachRows":
        if len(self.keys) != len(self.names):
            raise ValueError(
                f"for_each.keys 有 {len(self.keys)} 列，但 names 有 {len(self.names)} 个，必须一一对应"
            )
        for n in self.names:
            if not n or not n.strip():
                raise ValueError("for_each.names 不能含空名")
            if not _PLACEHOLDER.fullmatch("{" + n + "}"):
                raise ValueError(f"for_each.names 的 {n!r} 必须是合法标识符（字母/数字/下划线）")
        seen = set()
        for n in self.names:
            if n in seen:
                raise ValueError(f"for_each.names 有重复名 {n!r}")
            seen.add(n)
        return self


class SymConstraint(BaseModel):
    """符号化约束：`lhs (sense) rhs`，可对某个索引维或数据行逐值生成多条。

    `for_each` 两种形态：
    - **字符串**（`"j"`）：沿某个变量族的索引维遍历；
    - **`ForEachRows`**：沿数据文件的行遍历，把列绑定成索引变量（跨表关联）。

    `rhs` 允许四种形式：字面数值 / 数据引用 / 变量 / 整个表达式（后三者自动移项）。
    """

    name: str = Field(..., description="约束名模板，可含 {j} / {p} 等")
    sense: Sense
    lhs: SymExpr
    rhs: float | DataRef | VarRef | SymExpr = Field(
        ..., description="右端：数值 / 数据引用 / 变量 / 整个表达式（自动移项）"
    )
    for_each: str | list[str] | ForEachRows | None = Field(
        default=None,
        description=(
            "遍历方式：字符串=单个索引维；列表=多个索引维的**笛卡尔积**；"
            "`ForEachRows` 对象=按数据文件的行遍历（跨表关联）"
        ),
    )

    @field_validator("for_each", mode="before")
    @classmethod
    def _coerce_foreach(cls, v: Any) -> Any:
        """容错：`{"kind":"rows", ...}` → ForEachRows；列表 → 多索引笛卡尔积。"""
        if isinstance(v, dict):
            return ForEachRows.model_validate(v)
        if isinstance(v, (list, tuple)):
            names = [str(x) for x in v]
            if not names:
                raise ValueError("for_each 列表不能为空")
            if len(set(names)) != len(names):
                raise ValueError(f"for_each 列表有重复名: {names}")
            return names
        return v

    @field_validator("rhs", mode="before")
    @classmethod
    def _coerce_rhs(cls, v: Any) -> Any:
        """容错：接受 LLM 常见的几种写法。

        - `{"var": "hire_M"}` → `VarRef`（裸变量 dict）；
        - `"{hire_M}"` 字符串占位符 → `VarRef`（想引用变量却写成了占位符）；
        - `{"ref": {...}}` → 解开外层的 `ref`（与项里 `{"var":..., "ref":...}` 的写法同构）；
        - `{"expr": {...}}` → 解开外层的 `expr`，交给 `SymExpr`（右边的整个表达式）。
        """
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("{") and s.endswith("}") and len(s) > 2:
                return VarRef(var=s[1:-1])
            try:
                return float(s)
            except ValueError:
                raise ValueError(
                    f"rhs 是字符串 {v!r}，但既不是数字也不是 {{变量名}} 形式"
                )
        if isinstance(v, dict):
            if "var" in v and "file" not in v and "ref" not in v and "expr" not in v:
                return VarRef.model_validate(v)
            if set(v) == {"ref"} and isinstance(v["ref"], dict):
                return v["ref"]
            if set(v) == {"expr"} and isinstance(v["expr"], dict):
                return SymExpr.model_validate(v["expr"])
        return v


class SymObjective(BaseModel):
    sense: Sense
    expr: SymExpr


class SymVar(BaseModel):
    """符号化变量族：每个维度给一个 IndexSource，按笛卡尔积展开。

    `dim_names` 可显式命名各维（默认 `i, j, k, ...`）。**命名冲突时必须指定**：
    例如族 `hire` 只有一维（资源类型），若另一个族的 `for_each` 也叫 `i`，
    则 `hire_{i}` 与 `for_each: "i"` 会撞名 —— 此时应写 `dim_names: ["rtype"]`，
    并在项里用 `hire_{rtype}`。
    """

    name: str = Field(..., description="变量基名")
    var_type: VarType = Field(default=VarType.continuous)
    dims: list[IndexSource] = Field(default_factory=list, description="各维索引来源；空=标量")
    dim_names: list[str] | None = Field(
        default=None, description="各维的索引名（默认 i,j,k…）；用于避免与 for_each 撞名"
    )
    lower: float | None = None
    upper: float | None = None

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("变量名不能为空")
        return v.strip()

    @field_validator("dim_names")
    @classmethod
    def _dim_names_ok(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("dim_names 不能为空列表（不给就直接省略该字段）")
        for n in v:
            if not n or not n.strip():
                raise ValueError("dim_names 不能含空名")
            if not _PLACEHOLDER.fullmatch("{" + n + "}"):
                raise ValueError(f"dim_names 的 {n!r} 必须是合法标识符")
        if len(set(v)) != len(v):
            raise ValueError(f"dim_names 有重复名: {v}")
        return v

    @model_validator(mode="after")
    def _bounds(self) -> "SymVar":
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"变量 {self.name} 下界 > 上界")
        if self.var_type == VarType.binary:
            if self.lower is not None and self.lower > 1:
                raise ValueError(f"binary 变量 {self.name} 下界越过 1")
            if self.upper is not None and self.upper < 0:
                raise ValueError(f"binary 变量 {self.name} 上界小于 0")
        if self.dim_names is not None and len(self.dim_names) != len(self.dims):
            raise ValueError(
                f"变量 {self.name}: dim_names 有 {len(self.dim_names)} 个，但 dims 有 "
                f"{len(self.dims)} 维，必须一一对应"
            )
        return self

    @property
    def order(self) -> list[str]:
        """该族的索引名（按维序）。"""
        if self.dim_names is not None:
            return list(self.dim_names)
        return list(_INDEX_NAMES[: len(self.dims)])


class SymbolicIR(BaseModel):
    """符号化模型：变量族 + 目标 + 逐索引约束。"""

    variables: list[SymVar] = Field(...)
    objective: SymObjective = Field(...)
    constraints: list[SymConstraint] = Field(default_factory=list)
    description: str = ""
    assumptions: list[str] = Field(default_factory=list)

    @property
    def index_names(self) -> dict[str, IndexSource]:
        """索引变量名 → 来源（按各族的 `order` 汇总；同名不同源视为冲突，后者忽略）。"""
        out: dict[str, IndexSource] = {}
        for v in self.variables:
            for name, src in zip(v.order, v.dims):
                out.setdefault(name, src)
        return out

    def all_index_vars(self) -> set[str]:
        """模型里用到的全部 `for_each` 名字（索引维名 + 数据行绑定名）。"""
        out: set[str] = set()
        for c in self.constraints:
            if isinstance(c.for_each, ForEachRows):
                out.update(c.for_each.names)
            elif isinstance(c.for_each, list):
                out.update(c.for_each)
            elif c.for_each:
                out.add(c.for_each)
        return out

    def validate_placeholders(self) -> list[str]:
        """静态检查：项里的占位符是否都能被解析。

        规则：
        - **纯数字占位符视为字面索引值**（`y_{i}_{1}` 里的 `{1}` 就是第 1 种 mode）；
        - **变量族 `list` 维的实际取值也视为合法**（LLM 常直接写 `H_{Mechaniker}`）；
        - 其余占位符要么是该变量族的索引维，要么由所在约束的 `for_each` 提供
          （字符串 = 索引维名；`ForEachRows` = 它绑定的 names）。

        ⚠️ 这里只做**静态**检查，看不到 CSV 里的取值；`csv` 维的实际取值由
        `expand()` 在读到数据后按 `name in dims[d]` 判定。因此目标项不做占位符
        来源校验（否则会误杀 `H_{Mechaniker}` 这类合法写法），严格性交给展开器。

        返回问题列表（空=通过）。
        """
        problems: list[str] = []
        known = set(self.index_names) | self.all_index_vars()
        for v in self.variables:
            for src in v.dims:
                if src.kind == "list" and src.values:
                    known.update(str(x) for x in src.values)

        for con in self.constraints:
            allowed = set(known)
            if isinstance(con.for_each, ForEachRows):
                allowed.update(con.for_each.names)
            elif isinstance(con.for_each, list):
                allowed.update(con.for_each)
            elif con.for_each:
                allowed.add(con.for_each)
            for t in con.lhs.terms:
                # `bind` 自己会提供 as 名字（按分类值分派）
                if t.bind is not None:
                    allowed.add(t.bind.as_)
                unknown = {
                    p for p in t.placeholders
                    if p not in allowed and not _opaque_index(p)
                }
                if unknown:
                    problems.append(
                        f"约束 {con.name} 的项 {t.var}: 占位符 {sorted(unknown)} 无来源"
                    )
            if isinstance(con.rhs, DataRef):
                unknown = {
                    p for p in con.rhs.placeholders
                    if p not in allowed and not _opaque_index(p)
                }
                if unknown:
                    problems.append(
                        f"约束 {con.name} 的 rhs 引用占位符 {sorted(unknown)} 无来源"
                    )
        # 目标项：不校验占位符来源（交给 expand 按真实维度取值判定）
        return problems

    def stats(self) -> dict[str, Any]:
        return {
            "var_families": len(self.variables),
            "dims_per_family": [len(v.dims) for v in self.variables],
            "constraints": len(self.constraints),
            "for_each": sorted(self.all_index_vars()),
            "data_refs": sum(
                1 for c in self.constraints for t in c.lhs.terms if t.ref
            ) + sum(1 for t in self.objective.expr.terms if t.ref),
        }
