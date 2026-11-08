"""闭环编排器：理解→建模→生成→执行→验证→重写，最多 N 轮。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .. import verifier
from ..config import Config
from ..dataset import Sample
from ..expand import ExpandError, empty_data_store, expand, load_data_dir
from ..ir import IR
from ..llm import LLMError, LLMProvider
from ..solver import SolverResult, solve
from ..symbolic import SymbolicIR
from .rewrite import (
    degeneracy_reason,
    rewrite_symbolic,
    symbolic_degeneracy_reason,
)
from .understand import understand


@dataclass
class PipelineResult:
    """一次端到端运行的结果。"""

    sample_id: str
    success: bool = False
    suspicious: bool = False  # 疑似参考答案标注错误（解可行但目标值不符，未重写）
    rounds: int = 0
    # 尝试次数（一次尝试 = 一次独立的「理解 → 修复若干轮」；任一尝试成功即停止）
    attempt: int = 0          # 该结果来自第几次尝试
    attempts: int = 0         # 一共跑了多少次尝试
    attempts_log: list[dict[str, Any]] = field(default_factory=list)  # 每次尝试的结局
    objective_value: float | None = None
    variables: dict[str, float] = field(default_factory=dict)
    status: str = "unknown"
    final_ir: dict[str, Any] | None = None
    generated_code: str = ""
    errors: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)  # 累计 token 用量
    stage_usage: dict[str, Any] = field(default_factory=dict)  # 按阶段（understand/rewrite）

    def to_files(self, out_dir: str | Any) -> dict[str, str]:
        """把产物落盘到 out_dir，返回各产物路径。

        产物：
        - final_code.py        生成的最终求解代码
        - solution.json        求解结果（状态/目标值/变量）
        - verification_report.json  验证报告
        - rewrite_trace.json   重写轨迹（trajectory）
        - final_error.json     最终错误（无错误时为空列表）
        - token_usage.json     token 用量
        """
        from pathlib import Path

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}

        def _write(name: str, content: str) -> str:
            p = d / name
            p.write_text(content, encoding="utf-8")
            paths[name] = str(p)
            return str(p)

        # final_code.py
        _write("final_code.py", self.generated_code or "")

        # solution.json
        _write("solution.json", json.dumps({
            "status": self.status,
            "objective_value": self.objective_value,
            "variables": self.variables,
            "success": self.success,
            "suspicious": self.suspicious,
            "rounds": self.rounds,
            "attempt": self.attempt,
            "attempts": self.attempts,
        }, ensure_ascii=False, indent=2))

        # verification_report.json
        report = {
            "sample_id": self.sample_id,
            "success": self.success,
            "suspicious": self.suspicious,
            "status": self.status,
            "rounds": self.rounds,
            "attempt": self.attempt,
            "attempts": self.attempts,
            "objective_value": self.objective_value,
            "errors": self.errors,
        }
        if self.attempts_log:
            report["attempts_log"] = self.attempts_log
        # 附加最后一轮 checks（若有）
        if self.trajectory:
            last = self.trajectory[-1]
            report["last_round_checks"] = last.get("checks", [])
        _write("verification_report.json", json.dumps(report, ensure_ascii=False, indent=2))

        # rewrite_trace.json
        _write("rewrite_trace.json", json.dumps(self.trajectory, ensure_ascii=False, indent=2))

        # final_error.json（自描述：has_error 区分"成功无错误"与"有错误"）
        _write("final_error.json", json.dumps({
            "has_error": bool(self.errors),
            "errors": self.errors,
        }, ensure_ascii=False, indent=2))

        # token_usage.json
        _write("token_usage.json", json.dumps(self.token_usage, ensure_ascii=False, indent=2))

        # stage_usage.json（分阶段 token，供成本报表按阶段归因）
        _write("stage_usage.json", json.dumps(self.stage_usage, ensure_ascii=False, indent=2))

        return paths


class _UsageTrackingLLM:
    """包装 LLMProvider，累计每次 complete() 的 usage。

    - `usage_sink`：整样本合计（原有行为，保持向后兼容）；
    - `stage_sink`：**按阶段**分别累计（understand / rewrite），
      由编排器在每次调用前用 `set_stage(...)` 指定归属。

    这样 understand/rewrite 内部无需改动，就能拿到分阶段 token 成本。
    """

    def __init__(
        self,
        inner: LLMProvider,
        usage_sink: dict[str, Any],
        stage_sink: dict[str, Any] | None = None,
    ) -> None:
        self._inner = inner
        self._sink = usage_sink
        self._stage_sink = stage_sink if stage_sink is not None else {}
        self._stage = "unknown"
        self._calls = 0

    def set_stage(self, stage: str) -> None:
        """标注后续 complete() 调用所属阶段（用于分阶段成本核算）。"""
        self._stage = stage

    def complete(self, messages, **kwargs):
        resp = self._inner.complete(messages, **kwargs)
        self._calls += 1
        self._sink["calls"] = self._calls  # calls 始终记录，不依赖 usage 非空
        self._record_usage(resp.usage)
        self._record_stage(resp.usage)
        return resp

    def __getattr__(self, name: str) -> Any:
        """委托其它属性/方法到内层 LLMProvider。

        这样将来 LLMProvider 抽象类新增任何方法，包装器无需同步改动，
        会自动透传给内层对象（透明代理）。
        """
        return getattr(self._inner, name)

    @staticmethod
    def _accumulate(sink: dict[str, Any], usage: dict[str, Any]) -> None:
        """把一次调用的 usage 累加进 sink（标量直接加，嵌套 dict 递归累加）。"""
        for key, val in usage.items():
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                sink[key] = sink.get(key, 0) + val
            elif isinstance(val, dict):
                inner = sink.setdefault(key, {})
                for k2, v2 in val.items():
                    if not isinstance(v2, bool) and isinstance(v2, (int, float)):
                        inner[k2] = inner.get(k2, 0) + v2

    def _record_usage(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        self._accumulate(self._sink, usage)

    def _record_stage(self, usage: dict[str, Any]) -> None:
        stage = self._stage_sink.setdefault(self._stage, {})
        stage["calls"] = stage.get("calls", 0) + 1
        if usage:
            self._accumulate(stage, usage)


def _error_fingerprint(errors: str) -> str:
    """归一化错误文本，得到稳定指纹。

    去掉具体数值（浮点/整数），只保留错误类型结构，使
    "目标值不符：求解 1120.000000，期望 220.0" 与
    "目标值不符：求解 1130.000000，期望 220.0" 归为同一指纹。
    """
    import re

    normalized = re.sub(r"-?\d+\.?\d*", "<N>", errors)
    # 压缩空白
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def run_sample(
    sample: Sample,
    llm: LLMProvider | None,
    cfg: Config,
) -> PipelineResult:
    """对单个样例跑「最多 N 次尝试」，任一次成功即停止。

    一次**尝试** = 一次独立的「理解 → 求解 → 修复（最多 max_rounds-1 次）」。
    每次尝试都重新问 LLM（**没有缓存**），所以 N 次尝试是真的 N 条独立轨迹。

    - `pipeline.attempts`（默认 1）：最多试几次。设 2 表示「给两次解答机会」。
    - `pipeline.max_rounds`（默认 3）：单次尝试内最多求解几轮 → **3 轮 = 2 次修复**。

    失败的尝试不会覆盖更好的结果：最终返回「成功优先、其次有解、再按与参考值距离」的
    那一次尝试；token 用量按**所有尝试累加**（成本必须如实反映）。
    """
    attempts = max(1, int(cfg.pipeline.get("attempts", 1)))
    ref = sample.expected.get("objective_value") if isinstance(sample.expected, dict) else None

    best: PipelineResult | None = None
    best_rank: tuple[int, float] | None = None
    log: list[dict[str, Any]] = []
    total_usage: dict[str, Any] = {}
    total_stage: dict[str, Any] = {}

    for attempt in range(1, attempts + 1):
        r = _run_attempt(sample, llm, cfg)
        r.attempt = attempt
        log.append({
            "attempt": attempt,
            "success": r.success,
            "status": r.status,
            "objective_value": r.objective_value,
            "rounds": r.rounds,
            "errors": list(r.errors),
        })
        _accumulate(total_usage, r.token_usage)
        _accumulate(total_stage, r.stage_usage)

        rank = _attempt_rank(r, ref)
        if best is None or best_rank is None or rank > best_rank:
            best, best_rank = r, rank
        if r.success:
            break

    assert best is not None
    best.attempts = len(log)
    best.attempts_log = log
    # token 必须跨尝试累加，否则成本报表会漏算
    best.token_usage = total_usage
    best.stage_usage = total_stage
    if len(log) > 1 and not best.success:
        best.errors.append(
            f"[尝试 {len(log)}/{attempts} 次均未成功] "
            + "；".join(f"#{x['attempt']} {x['status']}"
                        + (f" obj={x['objective_value']}" if x["objective_value"] is not None else "")
                        for x in log)
            + f"（输出取表现最好的一次：第 {best.attempt} 次）"
        )
    return best


def _attempt_rank(r: PipelineResult, ref: Any) -> tuple[int, float]:
    """尝试的优劣排序键（越大越好）：成功 > 有解 > 无解；同档比与参考值的距离（越小越好）。"""
    if r.success:
        return (2, 0.0)
    if r.objective_value is not None:
        try:
            return (1, abs(float(r.objective_value) - float(ref)))
        except (TypeError, ValueError):
            return (1, float("inf"))
    return (0, float("inf"))


def _accumulate(sink: dict[str, Any], usage: dict[str, Any]) -> None:
    """把一次尝试的 token 用量累加进 sink（支持嵌套 dict）。"""
    for k, v in (usage or {}).items():
        if isinstance(v, (int, float)):
            sink[k] = sink.get(k, 0) + v
        elif isinstance(v, dict):
            sub = sink.setdefault(k, {})
            if isinstance(sub, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        sub[k2] = sub.get(k2, 0) + v2


def _run_attempt(
    sample: Sample,
    llm: LLMProvider | None,
    cfg: Config,
) -> PipelineResult:
    """对单个样例跑完整闭环。

    模型来源只有两条：
    - 有 LLM → 符号化理解（NL → SymbolicIR）→ `expand()` 展开成展平 IR；
    - 无 LLM → 样例内嵌的展平 IR（`extra["ir"]`），供离线测试 / 复现用。
      此时没有符号模型，因此**不能重写**（一轮定结果）。

    **没有缓存**：每次调用都会真的问 LLM。这样「每题跑多次」才是多次独立采样
    （缓存会把多次尝试塌缩成同一条轨迹，见 README 的说明）。
    """
    tol = float(cfg.pipeline.get("feasibility_tolerance", 1e-6))
    # 目标值容差与可行性容差是两回事：前者应明显更松，否则会把求解器浮点抖动误判为建模错误
    obj_tol = float(cfg.pipeline.get("objective_tolerance", 1e-4))
    max_rounds = int(cfg.pipeline.get("max_rounds", 3))
    repeated_error_limit = int(cfg.pipeline.get("repeated_error_limit", 3))
    # 求解超时（秒），None 表示不限时
    solve_time_limit = cfg.executor.get("timeout_seconds", None) if hasattr(cfg, "executor") else None
    if solve_time_limit is not None:
        solve_time_limit = float(solve_time_limit)

    executor_cfg = cfg.executor if hasattr(cfg, "executor") else None

    result = PipelineResult(sample_id=sample.id)

    # 用 usage 追踪包装器包住 LLM，累计 token 用量（understand/rewrite 内部无需改动）
    if llm is not None:
        llm = _UsageTrackingLLM(llm, result.token_usage, result.stage_usage)

    # 连续相同错误跟踪
    last_fingerprint: str | None = None
    repeat_count: int = 0


    # 疑似参考噪声跟踪（untrusted reference 且「可行但目标值不符」时置位）
    # 语义：**不短路**——仍允许一次修复尝试，用于把「不知道谁错」变成证据；
    # 但无论后续修复是否成功，判定都保持 suspicious、**绝不升级为通过**。
    suspicion: str | None = None     # None / "pending" / "converged" / "unresolved"
    suspicion_rewrites = 0

    # 模型获取：**只有一条 LLM 路径** —— 符号化理解 → 展开成展平 IR。
    # （早期还有「LLM 直接产展平 IR」的分支，已删除：数值在 CSV 里、规模中位 2,183
    #   个变量，展平 IR 的 JSON 是 LLM 的输出，必然超 max_tokens。）
    ir: IR | None = None             # 最近一次**成功展开**的展平 IR（也是退化比较基准）
    sym_ir: SymbolicIR | None = None  # 当前符号模型（重写的输入）
    store = None                     # DataStore（仅符号路径有）
    # 当前 `sym_ir` 未能展开的原因。None 表示 `ir` 与 `sym_ir` 一致、可以求解。
    expand_failed: str | None = None
    data_dir = sample.extra.get("data_dir")
    if llm is not None:
        try:
            _set_stage(llm, "understand")
            sym_meta: dict[str, Any] = {}
            sym = understand(sample.nl_text, llm, sym_meta)
            result.trajectory.append({"understand_meta": sym_meta})
            # 无数据文件时给一个**保证为空**的存储（list 索引 + 字面系数照样可展开），
            # 因此「自由题面」与「工业数据集」共用同一条链路。
            store = load_data_dir(data_dir) if data_dir else empty_data_store()
            sym_ir = sym
            try:
                ir = expand(sym, store)
            except ExpandError as exc:
                # 展开失败**不再终止**：它是**建模层**失败（虚构数据文件 / 键选错 /
                # 索引对不齐 / 笛卡尔积爆炸），正是重写能修的东西。
                # 旧实现直接 return，等于「一次笔误判死刑」，max_rounds 完全用不上。
                expand_failed = f"符号模型展开失败: {exc}"
            result.trajectory[-1]["symbolic_ir"] = sym.model_dump()
            result.trajectory[-1]["symbolic_stats"] = sym.stats()
            if ir is not None:
                result.trajectory[-1]["expand_meta"] = {
                    "data_dir": data_dir,
                    "var_instances": len(ir.all_var_instances),
                    "constraints": len(ir.constraints),
                    "objective_terms": len(ir.objective.expr.terms),
                }
                result.final_ir = ir.model_dump()
            else:
                result.trajectory[-1]["expand_error"] = expand_failed
                result.errors = [expand_failed]
        except Exception as exc:
            result.errors.append(f"符号化理解失败: {exc}")
            return result
    elif "ir" in sample.extra:
        # 离线注入的展平 IR（测试 / 复现用），不经过 LLM
        ir = IR.model_validate(sample.extra["ir"])
    else:
        # 没有内嵌 IR、也没有 LLM → 无从建模。
        # （数据集每题都靠自己建模，不存在「内置答案」这种东西。）
        result.errors.append("无 LLM 且样例未内嵌 IR，无法理解")
        return result

    # 「距离参考最近」的一轮：重写不保证单调改进，实测工业题会越改越差
    # （30n20b8: 253 → 51 → 0）。最终仍失败时回退到这一轮，而不是交出最差结果。
    _exp_obj = sample.expected.get("objective_value") if isinstance(sample.expected, dict) else None
    best_dist = float("inf")
    best_round: tuple[Any, Any, int] | None = None   # (ir, sol, round_no)

    for round_no in range(1, max_rounds + 1):
        result.rounds = round_no

        exec_meta: dict[str, Any] = {}
        sol: SolverResult | None = None
        checks: list[verifier.CheckResult] = []

        if expand_failed is not None:
            # 本轮没有可求解的模型：把展开错误当成本轮失败，直接进重写。
            result.errors = [expand_failed]
            result.trajectory.append({
                "round": round_no,
                "expand_error": expand_failed,
            })
        else:
            result.final_ir = ir.model_dump()

            # ---- EXECUTE：进程内 scipy/HiGHS 直接求解 IR ----
            # 求解**不缓存**：实测 5–500 ms，而一次 LLM 调用是 10–60 s，缓存它没有意义
            checks_exec: list[verifier.CheckResult] = []
            exec_meta: dict[str, Any] = {"executor": "local"}
            sol = solve(ir, time_limit=solve_time_limit)

            checks = verifier.verify(ir, sol, sample.expected, tol, obj_tol,
                                     getattr(sample, "nl_text", None)) + checks_exec

            all_ok = verifier.all_pass(checks)

            result.trajectory.append(
                {
                    "round": round_no,
                    "ir": ir.model_dump(),
                    "status": sol.status,
                    "objective_value": sol.objective_value,
                    "variables": sol.variables,
                    "checks": [{"ok": c.ok, "message": c.message, "blocking": c.blocking,
                                "detail": c.detail} for c in checks],
                    **exec_meta,
                }
            )

            # 记录「距离参考最近」的一轮（重写不保证单调改进，供最终失败时回退）
            if sol.status == "optimal" and sol.objective_value is not None:
                try:
                    _d = (abs(float(sol.objective_value) - float(_exp_obj))
                          if _exp_obj is not None else abs(float(sol.objective_value)))
                except (TypeError, ValueError):
                    _d = float("inf")
                if _d < best_dist:
                    best_dist = _d
                    best_round = (ir.model_copy(deep=True), sol, round_no)

            if all_ok:
                if suspicion:
                    # 修复后达到了参考值 ⇒ 「参考是噪声」被证伪，原本就是模型错。
                    # 记 converged，交由循环后的收尾块统一裁决（v9 起**计为通过**，
                    # 与 trusted 数据集判定一致；status 保留 suspicious_resolved 可追溯）。
                    suspicion = "converged"
                    result.objective_value = sol.objective_value
                    result.variables = sol.variables
                    break
                result.success = True
                result.status = sol.status
                result.objective_value = sol.objective_value
                result.variables = sol.variables
                break

            # 容忍机制（仅当参考答案不可信时启用）：
            # 解可行、仅目标值与参考答案不符 → 疑似参考答案标注错误。
            #
            # 旧实现直接 `return`，后果是：**在 trusted_reference=False 的数据集上
            # 目标不符永远不会触发任何修复尝试**——
            # 闭环的修复能力在这两个最难的external数据集上等于被关闭，且无法区分
            # 「参考噪声」与「模型真错」。
            #
            # 现改为：标记疑似后**继续走修复流程**（最多一次重写），把修复结果记为证据
            # （suspicious_outcome=converged/unresolved），判定仍保持 suspicious。
            if (not sample.trusted_reference) and verifier.suspicious_reference(checks):
                if suspicion is None:
                    suspicion = "pending"
                    result.suspicious = True
                    result.objective_value = sol.objective_value
                    result.variables = sol.variables
                    result.errors = [verifier.describe(checks)]
                result.trajectory[-1]["suspicious"] = True
                if suspicion_rewrites >= 1:
                    # 已给过一次机会仍未达到参考值 → 不再消耗 token
                    suspicion = "unresolved"
                    # ⚠️ 必须用**本轮**的 sol 刷新结果字段：`if suspicion is None:` 只在第 1 轮
                    # 设置过 objective_value/variables/errors，若不在此更新，break 后报告值会
                    # 停留在第 1 轮，与 final_ir（最后一轮 IR）不一致。
                    # 实测影响 6 题，全部 rounds=2、status=suspicious：
                    # 例 报告 18.9566 而 final_ir 实为 18.9857。
                    result.objective_value = sol.objective_value
                    result.variables = sol.variables
                    result.errors = [verifier.describe(checks)]
                    result.trajectory[-1]["suspicious_outcome"] = "unresolved"
                    break

        # ---- 失败 → 连续相同错误检测 → rewrite（符号模型 + 题面 + 失败原因）→ 下一轮 ----
        # 展开失败与「求解/验证失败」在这里**合流**：两者都交给 LLM 修。
        errors = expand_failed if expand_failed is not None else verifier.describe(checks)
        result.errors = [errors]

        # 连续相同错误终止：归一化错误指纹，连续重复 N 轮即终止，避免空转烧 token
        fingerprint = _error_fingerprint(errors)
        if fingerprint == last_fingerprint:
            repeat_count += 1
        else:
            last_fingerprint = fingerprint
            repeat_count = 1
        result.trajectory[-1]["error_fingerprint"] = fingerprint
        result.trajectory[-1]["repeat_count"] = repeat_count

        if repeat_count >= repeated_error_limit:
            result.status = "repeated_error"
            if sol is not None:
                result.objective_value = sol.objective_value
                result.variables = sol.variables
            result.errors.append(
                f"[连续相同错误终止] 连续 {repeat_count} 轮错误指纹一致: {fingerprint}"
            )
            break

        # ---- 最后一轮不再重写 ----
        # 重写的产物只有「下一轮求解」会用；已经是最后一轮时，这次 LLM 调用纯属浪费
        # （实测：错误类型各不相同时，3 轮失败会白花一次重写）。
        if round_no >= max_rounds:
            result.status = "max_rounds_exceeded"
            if sol is not None:
                result.objective_value = sol.objective_value
                result.variables = sol.variables
            result.trajectory[-1]["rewrite_skipped"] = "已是最后一轮，重写无人使用"
            break

        # ---- 单一修复策略：把符号模型 + 题面 + 失败原因交给 LLM 修 ----
        # 不再做错误分层路由：LLM 自己能从失败信息判断该改什么，
        # 多一层诊断只多一次调用、多一种误判。
        if llm is None:
            if sol is not None:
                result.status = sol.status
                result.objective_value = sol.objective_value
                result.variables = sol.variables
            break

        if suspicion:
            suspicion_rewrites += 1     # 疑似态下最多允许一次重写尝试

        result.trajectory[-1]["route"] = "rewrite"
        if sym_ir is None:
            # 内嵌展平 IR（无 LLM 建模）没有符号模型可改 —— 一轮定结果。
            result.errors.append("无符号模型，无法重写（样例内嵌的是展平 IR）")
            if sol is not None:
                result.status = sol.status
                result.objective_value = sol.objective_value
                result.variables = sol.variables
            break
        try:
            _set_stage(llm, "rewrite")
            new_sym = rewrite_symbolic(sample.nl_text, sym_ir, errors, llm)
            why_sym = symbolic_degeneracy_reason(sym_ir, new_sym)
            if why_sym:
                raise ValueError(f"符号模型退化: {why_sym}")
        except (LLMError, ValueError) as exc:
            result.errors.append(f"重写失败: {exc}")
            if sol is not None:
                result.status = sol.status
                result.objective_value = sol.objective_value
                result.variables = sol.variables
            break

        sym_ir = new_sym
        result.trajectory[-1]["rewrite_layer"] = "symbolic"

        # 立刻展开一次：既校验新模型的契约，也用它判断是否缩水。
        # 展不开不算「重写失败」—— 记成下一轮的失败原因，继续重试（最多 max_rounds 轮）。
        try:
            new_ir = expand(new_sym, store)
        except ExpandError as exc:
            expand_failed = f"符号模型展开失败: {exc}"
            result.trajectory[-1]["rewrite_expand_error"] = expand_failed
            continue

        if ir is not None:
            why = degeneracy_reason(ir, new_ir)
            if why:
                # 重写把模型改缩水了：拒绝采用，保留本轮的求解结果
                result.errors.append(f"展开后退化: {why}（已拒绝本轮重写）")
                if sol is not None:
                    result.status = sol.status
                    result.objective_value = sol.objective_value
                    result.variables = sol.variables
                break

        ir = new_ir
        expand_failed = None

    else:
        # for 循环正常结束（未被 break）→ 轮次用尽
        result.status = "max_rounds_exceeded"

    # 疑似参考噪声的最终裁决（v9 修订）。
    #
    # 两种结局的**证据强度不同**，判定也不同：
    #
    #   converged  → 重写后达到参考值 ⇒ 「参考是噪声」这一假设**被证伪**，
    #                原本就是模型错。判定与 trusted 数据集**一致**：记为通过
    #                （success=True），同时保留 status=suspicious_resolved 可追溯。
    #   unresolved → 给过一次机会仍未达到 ⇒ 疑似噪声的解释才站得住，
    #                保持 success=False、status=suspicious。
    #
    # **安全不变量**：只有 unresolved 才可能"被放过"，converged 已由求解器
    # 与验证器双重确认（最终 IR 可行且目标值一致），不存在掩盖建模错误的风险。
    #
    # v8 曾规定「converged 也不计通过」，v9 撤销，原因（实测驱动）：
    #   ① **判定不一致**：同为「2 轮修复后可行且目标值一致」，trusted 数据集记
    #      通过、untrusted 记失败——同样的证据、相反的判定；
    #   ② **依据不成立**：`trusted_reference` 是**数据集级标签**（整批全 False），
    #      不携带逐题信息；把它当逐题证据降级判定属逻辑错误；
    #   ③ **理由未贯彻**：v8 反对计通过的唯一理由是「错误消息含期望值、可能拟合」
    #      （`[FAIL] 目标值不符：求解 X，期望 Y`），但该模板对 trusted 数据集
    #      同样适用，按此理由内部数据集的修复通过也不该算数——双标；
    #   ④ **实测反证**：converged 的题**全部为 2 轮**（首轮错、一次重写即达标）。
    #      "拟合答案"不会如此稳定，符合"修正建模细节即可达标"。
    if suspicion:
        result.suspicious = True
        converged = suspicion == "converged"
        result.success = converged          # 仅 converged 计通过；unresolved 仍不计
        result.status = "suspicious_resolved" if converged else "suspicious"
        result.trajectory[-1]["suspicious_outcome"] = (
            "converged" if converged else "unresolved")
        result.errors.append(
            "[疑似参考噪声] 修复结果="
            + ("converged" if converged else "unresolved")
            + ("（重写后达到参考值 ⇒ 参考可达、原本是模型错，"
               "判定与可信参考答案一致：**计为通过**，status 保留 suspicious_resolved 可追溯）"
               if converged else
               "（给过一次机会仍未达到 ⇒ 疑似参考答案噪声的解释站得住，**不计通过**）"))

    # 最终仍失败 → 回退到「距离参考最近」的那一轮，避免交出越改越差的结果
    if (not result.success) and best_round is not None and best_round[2] != result.rounds:
        _bir, _bsol, _brno = best_round
        result.final_ir = _bir.model_dump()
        result.objective_value = _bsol.objective_value
        result.variables = _bsol.variables
        result.status = _bsol.status if result.status in ("unknown", None) else result.status
        result.trajectory.append({
            "best_round_fallback": {
                "round": _brno,
                "distance": best_dist,
                "note": "最终轮不如该轮接近参考值，输出回退到该轮（原最终轮仍在 trajectory 中可查）",
            }
        })
        result.errors.append(
            f"[回退] 最终第 {result.rounds} 轮不如第 {_brno} 轮接近参考值"
            f"（距离 {best_dist:g}），输出已回退到第 {_brno} 轮"
        )

    return result


def _set_stage(llm: Any, stage: str) -> None:
    """标注后续 LLM 调用所属阶段（供分阶段 token 核算）。

    llm 可能是 `_UsageTrackingLLM` 包装器、其它 provider 或 None；
    只有包装器实现了 set_stage，其余情况静默跳过。
    """
    setter = getattr(llm, "set_stage", None)
    if callable(setter):
        setter(stage)


def _exec_error_head(sb) -> str:
    """从沙箱执行结果提取一行错误摘要（含异常类型 + 出错位置）。

    取 stderr 的**末行**（Python 回溯末行 = 异常类型 + 消息），并向上找最近的
    `File "...", line N` 帧，把「文件:行号」附上——否则只剩异常类型、丢了出错位置。
    完整 stderr（末 1500 字符）仍保留在 detail.stderr 供落盘/复核。

    """
    text = (sb.stderr or "").strip() or (sb.error or "").strip() or (sb.stdout or "").strip()
    if not text:
        return "未知错误"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "未知错误"
    exc_line = lines[-1]
    # 找异常前最近的 File "..." 帧，取其文件名:行号
    import re

    loc = None
    for ln in reversed(lines[:-1]):
        m = re.search(r'File "([^"]+)", line (\d+)', ln)
        if m:
            loc = f"{m.group(1).rsplit('/', 1)[-1]}:{m.group(2)}"
            break
    head = f"{exc_line}（@{loc}）" if loc else exc_line
    return head[:300]


def run_dataset(
    samples,
    llm: LLMProvider | None,
    cfg: Config,
) -> list[PipelineResult]:
    """批量运行，返回结果列表。"""
    return [run_sample(s, llm, cfg) for s in samples]
