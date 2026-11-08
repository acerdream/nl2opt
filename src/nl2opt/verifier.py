"""验证器：**打分（与参考答案比对）+ 可行性断言**。

只做两件事：
- `check_optimality`：解算出的答案与参考答案是否一致（唯一判对错的检查）；
- `check_feasibility`：解是否满足 IR 的约束/界/整数性（断言「解与模型一致」）。

**能力边界**：保证「解与模型一致」，**不保证「模型与题面一致」**。
后者只能靠参考答案比对间接判断 —— 这也是本模块唯一有价值的输出。

> 历史：早期还有 objective_consistency / coefficient_grounding / objective_defined /
> semantic_coverage 四项，都是为「LLM 写求解代码 + 沙箱执行」的场景设计的
> （那时解不可信，需要独立核算）。当前架构不让 LLM 写代码、解直接来自 scipy，
> 这些检查要么恒真、要么依赖题面里的数值（工业数据集题面没有数字），已移除。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .ir import IR, Sense, VarType
from .solver import SolverResult


@dataclass
class CheckResult:
    """单条校验结论。

    `blocking` 区分校验强度：
    - `blocking=True`（默认）：`ok=False` 会阻断 `all_pass`（硬失败，进入重写）；
    - `blocking=False`：`ok=False` 仅作告警/诊断，**不阻断** `all_pass`
      （用于「弱证据」类检查，如目标系数依据的启发式扫描——证据不足以推翻
      一个本应接受的模型，只记录供人工复核）。

    这样避免「所有检查默认同等强度」：强证据（可解析的结构化权重）硬失败，
    弱证据（关键词邻近）只告警。
    """

    ok: bool
    message: str = ""
    detail: dict = field(default_factory=dict)
    kind: str = "generic"  # feasibility | optimality | status | no_solution | generic
    blocking: bool = True


def check_feasibility(ir: IR, sol: SolverResult, tol: float = 1e-6) -> CheckResult:
    """校验解是否满足全部约束、边界与整数/0-1 要求。

    解的**完整性**先于数值校验：声明变量必须全部出现、不得出现未声明变量、
    每个值必须是有限数（拒绝 NaN/Inf）。用 `x.get(t.var, 0.0)` 会把缺失变量
    静默当 0 处理，从而漏掉「下界应为 1 却缺值」这类错误；NaN 在数值比较里
    通常不触发失败，整数检查 `round(nan)` 还会抛异常——这些都必须在进入
    约束/边界/整数检验之前被明确拦截。
    """
    if not sol.success:
        return CheckResult(False, f"求解未成功，状态={sol.status}，无法做可行性校验", kind="feasibility")

    x = sol.variables
    vmap = ir.var_map
    violations: list[str] = []

    # 1) 解向量完整性：声明变量 ↔ 给出变量的对称性 + 有限值
    required = set(ir.all_var_instances)
    given = set(x)
    missing = required - given
    extra = given - required
    if missing:
        violations.append(f"解缺少已声明变量: {', '.join(sorted(missing))}")
    if extra:
        violations.append(f"解出现未声明变量: {', '.join(sorted(extra))}")
    for name, val in x.items():
        if not math.isfinite(val):
            violations.append(f"变量 {name}: 值非有限（{val!r}）")

    # 2) 约束（仅当解向量完整时才可安全求值）
    if not missing:
        for con in ir.constraints:
            lhs = con.lhs.constant
            for t in con.lhs.terms:
                lhs += t.coef * x[t.var]
            rhs = con.rhs
            if con.sense == Sense.le and lhs > rhs + tol:
                violations.append(f"约束 {con.name}: {lhs:.6f} <= {rhs} 违反")
            elif con.sense == Sense.ge and lhs < rhs - tol:
                violations.append(f"约束 {con.name}: {lhs:.6f} >= {rhs} 违反")
            elif con.sense == Sense.eq and abs(lhs - rhs) > tol:
                violations.append(f"约束 {con.name}: {lhs:.6f} == {rhs} 违反（差 {abs(lhs-rhs):.2e}）")

    # 3) 边界 + 整数性/0-1（对每个**已声明且已给出**的变量；未声明变量已在第 1 步拦截）
    for name in sorted(required & given):
        val = x[name]
        var = vmap[name]
        if var.lower is not None and val < var.lower - tol:
            violations.append(f"变量 {name}: {val} 小于下界 {var.lower}")
        if var.upper is not None and val > var.upper + tol:
            violations.append(f"变量 {name}: {val} 大于上界 {var.upper}")
        if var.var_type == VarType.integer and abs(val - round(val)) > tol:
            violations.append(f"变量 {name}: 应为整数，实际 {val}")
        if var.var_type == VarType.binary and abs(val - 0) > tol and abs(val - 1) > tol:
            violations.append(f"变量 {name}: 应为 0/1，实际 {val}")

    if violations:
        return CheckResult(False, "; ".join(violations), {"violations": violations}, kind="feasibility")
    return CheckResult(True, "可行", kind="feasibility")


def check_optimality(sol: SolverResult, expected_obj: float | None, tol: float = 1e-4,
                     ir: IR | None = None,
                     answer_var: str | None = None) -> CheckResult:
    """对比数值与参考答案。expected_obj 为 None 时跳过数值对比。

    容差为**相对**形式：`tol × max(1, |期望值|)`。
    这样小目标值（如 2.0）用绝对容差，大目标值（如 1e6）自动放宽，
    避免求解器在浮点精度内的正常抖动被误判为「目标值不符」。
    （实测案例 SCH-029：求解 1.9999989999999999 vs 期望 2.0，差 ~1e-6，
      旧实现用可行性容差 1e-6 比较目标值而误判失败。）

    **有限性前置**（P0）：`abs(nan - expected) > tol` 为 False，旧实现会把
    `objective_value=NaN` 误判为「目标值一致」；`expected=NaN/Inf` 同理。生成代码的
    `json.dumps` 默认可写出 NaN（`allow_nan=True`），故这是真实可达路径，必须显式拒绝。

    **答案变量 `answer_var`**（目标规划 / 字典序优化）：这类题的参考答案**不是目标函数值**，
    而是某个偏差变量（例「至少多少不能满足 P3」→ `d3_minus`）。expected 带
    `answer_variable` 时，本函数改为比较**该变量在解中的值**。见。

    **诊断上下文**：失败时若传入 `ir`，detail 会附「紧约束 + 卡边界变量」摘要（不进 message）。
    """
    if expected_obj is None:
        return CheckResult(True, "无参考答案，跳过目标值对比", kind="optimality")
    if not math.isfinite(float(expected_obj)):
        return CheckResult(
            False, f"参考答案目标值非有限（{expected_obj!r}），无法对比",
            {"expected": expected_obj}, kind="optimality",
        )
    if sol.objective_value is None:
        return CheckResult(False, "求解成功但目标值为空", kind="optimality")
    # 定位「被比较的实际值」：默认目标函数值；指定答案变量时改取该变量的值。
    # 目标规划/字典序题的目标函数是加权和（如 `10000*d2_minus + d3_minus`），
    # 其值没有业务含义；真正的答案在偏差变量里（如 `d3_minus`）。
    #。
    if answer_var:
        raw = sol.variables.get(answer_var)
        if raw is None:
            return CheckResult(
                False,
                f"答案变量 {answer_var!r} 不在求解结果中：目标规划/字典序题的偏差变量"
                f"须命名为 d{{k}}_minus / d{{k}}_plus 并写进模型",
                {"answer_variable": answer_var,
                 "available_vars": sorted(sol.variables)[:15]},
                kind="optimality",
            )
        actual = float(raw)
    else:
        actual = float(sol.objective_value)
    if not math.isfinite(actual):
        return CheckResult(
            False, f"被比较的数值非有限（{actual!r}）",
            {"actual": actual, "answer_variable": answer_var}, kind="optimality",
        )
    effective = tol * max(1.0, abs(float(expected_obj)))
    if abs(actual - expected_obj) > effective:
        # ⚠️ message 里**不得**出现参考值（expected_obj）。
        # 这条文本会经 `describe()` 进入重写的 prompt——LLM 看得到期望值，
        # 就能"反推出能算出该值的模型"来拟合答案，而不是重读题面修正建模。
        # 这是对**整个重写闭环证据链**的污染，不分 trusted/untrusted
        # （历史案例：converged 判定曾据此被反复拉扯）。
        # 完整数值保留在 detail 字段（供诊断/报告/人工复核），不进入 prompt。
        label = f"答案 `{answer_var}`" if answer_var else "目标值"
        msg = f"{label}与参考不符（求解 {actual:.6f}）"
        detail: dict = {"actual": actual, "expected": expected_obj,
                        "tolerance": effective}
        if answer_var:
            detail["answer_variable"] = answer_var
        return CheckResult(False, msg, detail, kind="optimality")
    if answer_var:
        return CheckResult(True, f"答案 `{answer_var}` 一致", kind="optimality")
    return CheckResult(True, "目标值一致", kind="optimality")


def check_status(expected_status: str | None, sol: SolverResult) -> CheckResult:
    """核对求解状态与参考答案声明的状态（如 infeasible/unbounded）。"""
    if expected_status is None:
        return CheckResult(True, "参考答案未声明状态", kind="status")
    if expected_status != sol.status:
        return CheckResult(
            False,
            f"状态不符：求解 {sol.status}，期望 {expected_status}",
            {"actual": sol.status, "expected": expected_status},
            kind="status",
        )
    return CheckResult(True, f"状态一致（{sol.status}）", kind="status")


def verify(
    ir: IR,
    sol: SolverResult,
    expected: dict,
    feas_tol: float = 1e-6,
    obj_tol: float = 1e-4,
    nl_text: str | None = None,
) -> list[CheckResult]:
    """验证：**最优性比对（打分）+ 可行性（断言）**。

    只保留两项，理由：
    - `check_optimality`：唯一真正判「答案对不对」的检查（与参考答案比对）；
    - `check_feasibility`：唯一能发现「求解器给回来的解与 IR 对不上」的断言
      （变量缺失/错位、界处理错、IR→矩阵转换 bug）。正常情况下恒真，成本微秒级。

    已移除的检查及其理由：
    - `check_objective_consistency`：报告的 objective 直接来自求解器返回值，
      与「由 IR 重算」同源，恒相等，无独立信息；
    - `check_coefficient_grounding`：依赖题面给出数值权重；工业数据集题面**没有数字**
      （数值在 CSV 里），恒跳过；
    - `check_objective_defined`：目标为空时 `check_optimality` 同样会失败，信息重复；
    - `check_semantic_coverage`：需要参考 IR，数据集不提供。

    **能力边界（须明示）**：本函数保证「解与模型一致」，**不保证「模型与题面一致」**。
    后者只能靠 `check_optimality` 与参考答案比对间接判断。
    """
    results: list[CheckResult] = []

    expected_status = expected.get("status")
    expected_obj = expected.get("objective_value")

    # 参考答案声明的是「状态」而非数值时，先核对状态
    if expected_status in ("infeasible", "unbounded"):
        results.append(check_status(expected_status, sol))
        return results

    if expected_status == "no_solution":
        ok = sol.status in ("infeasible", "unbounded")
        results.append(
            CheckResult(
                ok,
                f"无最优解核对：求解状态={sol.status}" + ("" if ok else "，期望无解"),
                {"actual": sol.status, "expected": "no_solution"},
                kind="no_solution",
            )
        )
        return results

    results.append(check_feasibility(ir, sol, feas_tol))
    results.append(check_optimality(sol, expected_obj, obj_tol, ir,
                                    answer_var=expected.get("answer_variable")))
    return results



def suspicious_reference(results: list[CheckResult]) -> bool:
    """判断“疑似参考答案标注错误”的情形。

    判定条件：解可行（feasibility 通过），但目标值与参考答案不符（optimality 失败）。
    此情况下求解器已给出可行解，且目标值差异只可能来自：
    (a) 建模错误（但可行说明约束基本对）——不能完全排除；
    (b) 参考答案本身标注噪声。

    ⚠️ 本函数只做**判定**，不决定后续流程。历史教训：调用方曾据此直接 `return`，
    等于在 `trusted_reference=False` 的数据集上
    **关闭了整条修复链路**，且无法区分 (a) 与 (b)。
    现由 `orchestrator.run_sample` 标记 `suspicious` 后**仍允许一次重写尝试**，
    并把结果记为 `suspicious_outcome`（converged/unresolved）作为证据，
    但**绝不升级为通过**——保留「不把可疑当通过」的安全性质。
    """
    feas = next((r for r in results if r.kind == "feasibility"), None)
    opt = next((r for r in results if r.kind == "optimality"), None)
    if feas is None or opt is None:
        return False
    return feas.ok and (not opt.ok)


def all_pass(results: list[CheckResult]) -> bool:
    """所有**阻断型**校验项是否全部通过。

    非阻断项（blocking=False，如弱证据告警）即使 ok=False 也不影响判定——
    它们只作为诊断证据记录，不进入「失败 → 重写」的闭环。
    """
    return all(r.ok for r in results if r.blocking)


def describe(results: list[CheckResult]) -> str:
    """把校验结果渲染为人类可读文本。

    PASS / FAIL / WARN 三态：非阻断项失败渲染为 WARN（告警），区别于会阻断闭环的 FAIL。
    """
    lines = []
    for r in results:
        if r.ok:
            mark = "PASS"
        elif r.blocking:
            mark = "FAIL"
        else:
            mark = "WARN"
        lines.append(f"[{mark}] {r.message}")
    return "\n".join(lines)
