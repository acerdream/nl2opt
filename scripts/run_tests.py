"""零依赖回归入口：最小 smoke + 关键历史回归 + pytest 聚合派发。

定位（P2 测试结构分层后）：
- **smoke**：确定性链路的「能跑通」级检查（黄金 IR 求解 / 特殊状态 / e2e / 重写闭环）。
- **关键历史回归**：与具体 P0/P1/P2 缺陷或协议约定绑定的**工具/脚本级**断言
  （config / pricing / 协议版本 / 状态映射 / 差异分类 / 提示词分层 / 语义不变 / 沙箱信封 /
  消融口径 / 分阶段记账）。
- **pytest 派发**：核心单元/集成断言已迁移到 `tests/`（test_expand / test_foreach_rows /
  test_bind_value / test_index_arithmetic / test_symbolic_rewrite / test_precheck_files /
  test_auto_add_instances / test_expand_dedup / test_ir_validation / test_llm_response_handling）。
  若环境中安装了 pytest，则本入口一并运行 `pytest tests/` 并聚合结果；
  否则打印清晰提示（`pip install -e .[dev]`）。

这样：无 pytest 环境仍能跑完 smoke + 关键回归；有 pytest 环境则得到完整分层覆盖。
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nl2opt.solver import solve  # noqa: E402
from nl2opt import verifier  # noqa: E402
from nl2opt.config import load_config  # noqa: E402
from nl2opt.pipeline import run_sample  # noqa: E402


# ---------------------------------------------------------------------------
# 手写的极小 IR 夹具
#
# 用手写 IR 作夹具，只验证**机制**（求解器接线、状态映射、重写闭环、
# 阶段记账），不依赖任何数据集。
# ---------------------------------------------------------------------------

def _make_ir(variables, objective, constraints):
    from nl2opt.ir import IR

    return IR(variables=variables, objective=objective, constraints=constraints)


def _knapsack_ir():
    """max 60x1+100x2+120x3, s.t. 10x1+20x2+30x3<=50 → 最优 220（选 x2,x3）。"""
    from nl2opt.ir import Constraint, LinearExpr, Objective, Sense, Term, Variable, VarType

    return _make_ir(
        variables=[Variable(name=f"x{i}", var_type=VarType.binary) for i in (1, 2, 3)],
        objective=Objective(sense=Sense.maximize, expr=LinearExpr(terms=[
            Term(var="x1", coef=60.0),
            Term(var="x2", coef=100.0),
            Term(var="x3", coef=120.0),
        ])),
        constraints=[Constraint(
            name="capacity", sense=Sense.le,
            lhs=LinearExpr(terms=[
                Term(var="x1", coef=10.0),
                Term(var="x2", coef=20.0),
                Term(var="x3", coef=30.0),
            ]),
            rhs=50.0,
        )],
    )


def _symbolic_knapsack(capacity: float, data_dir: Path) -> dict:
    """**符号版**背包（数值写在 `data_dir/items.csv`）：max Σ value_i x_i, s.t. Σ weight_i x_i ≤ capacity。

    返回符号模型 dict。用真实数据文件是刻意的 —— 它同时覆盖「csv 列做 dims」
    「DataRef 按 key 取值」「expand 展开」三件事，也就是 pipeline 现在**唯一**的建模链路。

    `data_dir` 由调用方给：**必须落在工作区内**（文件沙箱只允许写工作区，
    系统临时目录会被拒绝）。
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "items.csv").write_text(
        "item,value,weight\n1,60,10\n2,100,20\n3,120,30\n", encoding="utf-8")
    return {
        "variables": [
            {"name": "x",
             "dims": [{"kind": "csv", "file": "items.csv", "column": "item"}],
             "var_type": "binary"}
        ],
        "objective": {"sense": "maximize", "expr": {"terms": [
            {"var": "x_{i}", "ref": {"file": "items.csv", "column": "value",
                                     "key_column": "item", "key_value": "{i}"}}]}},
        "constraints": [{
            "name": "capacity", "sense": "<=", "rhs": capacity,
            "lhs": {"terms": [
                {"var": "x_{i}", "ref": {"file": "items.csv", "column": "weight",
                                         "key_column": "item", "key_value": "{i}"}}]},
        }],
    }


def _scratch_dir(name: str) -> Path:
    """在工作区 `artifacts/` 下造一个一次性数据目录。

    为什么不用 `tempfile.TemporaryDirectory`：文件沙箱不允许写系统临时目录，
    且 TemporaryDirectory 会 `chmod 0o700`（同样被拒）。调用方负责清理。
    """
    import shutil

    d = Path(__file__).resolve().parents[1] / "artifacts" / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _infeasible_ir():
    """x >= 5 且 x <= 3 → 不可行。"""
    from nl2opt.ir import Constraint, LinearExpr, Objective, Sense, Term, Variable

    return _make_ir(
        variables=[Variable(name="x", lower=0.0)],
        objective=Objective(sense=Sense.minimize,
                            expr=LinearExpr(terms=[Term(var="x", coef=1.0)])),
        constraints=[
            Constraint(name="lo", sense=Sense.ge,
                       lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=5.0),
            Constraint(name="hi", sense=Sense.le,
                       lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=3.0),
        ],
    )


def _unbounded_ir():
    """max x, x >= 0，无上界 → 无界。"""
    from nl2opt.ir import Constraint, LinearExpr, Objective, Sense, Term, Variable

    return _make_ir(
        variables=[Variable(name="x", lower=0.0)],
        objective=Objective(sense=Sense.maximize,
                            expr=LinearExpr(terms=[Term(var="x", coef=1.0)])),
        constraints=[Constraint(name="lo", sense=Sense.ge,
                                lhs=LinearExpr(terms=[Term(var="x", coef=1.0)]), rhs=0.0)],
    )


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


def _run_pytest() -> tuple[int, int]:
    """尝试运行 tests/ 下的 pytest 套件，返回 (passed, total) 或 (0, -1) 表示不可用。"""
    try:
        import pytest  # noqa: F401
    except ImportError:
        return 0, -1
    import subprocess

    root = Path(__file__).resolve().parents[1]
    # 不用 -q：需要 "N passed" 汇总行；--no-header 去掉 platform 头部
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"(\d+)\s+passed", tail)
    passed = int(m.group(1)) if m else 0
    m2 = re.search(r"(\d+)\s+failed", tail)
    failed = int(m2.group(1)) if m2 else 0
    m3 = re.search(r"(\d+)\s+error", tail)
    errors = int(m3.group(1)) if m3 else 0
    total = passed + failed + errors
    if proc.returncode != 0 or failed or errors:
        print("\n[Pytest 套件输出]")
        print(tail[-4000:])
    return passed, total


def main() -> int:
    failures: list[str] = []
    total = 0

    def check(name, fn):
        nonlocal total
        total += 1
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:
            failures.append(name)
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()

    # ===================== Smoke：确定性链路能跑通 =====================

    # 1) 求解器接线（手写 IR，不依赖任何数据集）
    def _solve_knapsack():
        sol = solve(_knapsack_ir())
        assert sol.success, sol.message
        assert approx(sol.objective_value, 220.0), sol.objective_value
        assert sol.variables["x2"] == 1.0 and sol.variables["x3"] == 1.0
    check("solver: hand-written knapsack -> 220", _solve_knapsack)

    # 2) 特殊状态映射
    def _infeasible():
        assert solve(_infeasible_ir()).status == "infeasible"
    check("infeasible status", _infeasible)

    def _unbounded():
        assert solve(_unbounded_ir()).status == "unbounded"
    check("unbounded status", _unbounded)

    # 3) 端到端 —— 用真实数据集跑：
    #    python -m nl2opt eval --data data/MIPLIB-NL/dataset/<id>

    # 4) 重写闭环 smoke：首轮错（容量写大）→ 失败 → 符号层重写 → 修复
    def _rewrite_loop():
        import shutil

        from nl2opt.dataset import Sample
        from nl2opt.llm.fake import ScriptedProvider

        data_dir = _scratch_dir("_smoke_knap")
        try:
            # 容量写在符号模型的 rhs 里（不在 CSV），所以两轮共用同一份数据目录
            wrong = _symbolic_knapsack(60.0, data_dir)   # 容量 60 → 280 ≠ 220
            right = _symbolic_knapsack(50.0, data_dir)   # 容量 50 → 220
            llm = ScriptedProvider([
                json.dumps(wrong, ensure_ascii=False),
                json.dumps(right, ensure_ascii=False),
            ])
            result = run_sample(
                Sample(id="test-rewrite", nl_text="测试重写闭环", problem_type="milp",
                       expected={"objective_value": 220.0},
                       extra={"data_dir": str(data_dir)}),
                llm=llm, cfg=load_config())
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)
        assert result.success is True, result.errors
        assert result.rounds == 2, result.rounds
        assert result.objective_value == 220.0, result.objective_value
        # 重写发生在符号层（不是展平层）；标记挂在**被重写的那一轮**上
        layers = [s.get("rewrite_layer") for s in result.trajectory if isinstance(s, dict)]
        assert "symbolic" in layers, layers
    check("rewrite loop repairs symbolic IR (smoke)", _rewrite_loop)

    # ===================== 关键历史回归（工具/脚本/协议级） =====================

    # 4b) 无数据文件的自由题面：走**同一条**符号链路（空 DataStore），不需要 CSV
    def _no_data_symbolic():
        from nl2opt.dataset import Sample
        from nl2opt.llm.fake import ScriptedProvider

        sym = {
            "variables": [{"name": "x",
                           "dims": [{"kind": "list", "values": ["1", "2", "3"]}],
                           "var_type": "binary"}],
            "objective": {"sense": "maximize",
                          "expr": {"terms": [{"var": "x_{i}", "coef": 50.0}]}},
            "constraints": [{"name": "capacity_{i}", "for_each": "i", "sense": "<=",
                             "rhs": 100.0,
                             "lhs": {"terms": [{"var": "x_{i}", "coef": 10.0}]}}],
        }
        llm = ScriptedProvider([json.dumps(sym, ensure_ascii=False)])
        # 注意：没有 extra["data_dir"] —— 关键就是这条分支也要能展开
        result = run_sample(
            Sample(id="test-nodata", nl_text="自由题面：三个物品，每个最多拿一个",
                   problem_type="milp", expected={"objective_value": 150.0}),
            llm=llm, cfg=load_config())
        assert result.success is True, result.errors
        assert result.objective_value == 150.0, result.objective_value
    check("free-form text goes through expand (no data files)", _no_data_symbolic)

    # 4c) 无 LLM 且无内嵌 IR → 明确报错，不静默产出空模型
    def _no_model_source():
        from nl2opt.dataset import Sample
        result = run_sample(Sample(id="test-nomodel", nl_text="x"), llm=None,
                            cfg=load_config())
        assert result.success is False
        assert any("无法理解" in e for e in result.errors), result.errors
    check("no LLM and no embedded IR fails loudly", _no_model_source)

    # 4d) 展开失败**可被重写修复**（旧实现直接终止，rounds=0）
    def _expand_failure_is_repairable():
        from nl2opt.dataset import Sample
        from nl2opt.llm.fake import ScriptedProvider

        ghost = {   # LLM 编了一个不存在的数据文件（sorrell3 实测就是这样）
            "variables": [{"name": "x",
                           "dims": [{"kind": "csv", "file": "ghost.csv", "column": "id"}],
                           "var_type": "binary"}],
            "objective": {"sense": "minimize",
                          "expr": {"terms": [{"var": "x_{i}", "coef": 1.0}]}},
            "constraints": [{"name": "c", "sense": ">=", "rhs": 1.0,
                             "lhs": {"terms": [{"var": "x_{i}", "coef": 1.0}]}}],
        }
        fixed = json.loads(json.dumps(ghost))
        fixed["variables"] = [{"name": "x", "dims": [{"kind": "list", "values": ["1", "2"]}],
                               "var_type": "binary"}]

        llm = ScriptedProvider([json.dumps(ghost, ensure_ascii=False),
                                json.dumps(fixed, ensure_ascii=False)])
        result = run_sample(
            Sample(id="test-expand-fix", nl_text="测试展开失败修复", problem_type="milp",
                   expected={"objective_value": 1.0},
                   extra={"data_dir": "data/MIPLIB-NL/dataset/b-ball"}),
            llm=llm, cfg=load_config())
        assert result.success is True, result.errors
        assert result.rounds == 2, result.rounds
        assert not llm.responses, "第二次 LLM 调用没被用上（展开失败仍然终止了）"

        # 重写也救不回来时：不崩溃，按轮数/重复错误收尾
        llm2 = ScriptedProvider([json.dumps(ghost, ensure_ascii=False)] * 3)
        result2 = run_sample(
            Sample(id="test-expand-stuck", nl_text="测试展开一直失败", problem_type="milp",
                   expected={"objective_value": 1.0},
                   extra={"data_dir": "data/MIPLIB-NL/dataset/b-ball"}),
            llm=llm2, cfg=load_config())
        assert result2.success is False
        assert result2.status in ("repeated_error", "max_rounds_exceeded"), result2.status
        assert any("展开" in e for e in result2.errors), result2.errors
    check("expand failure is repairable by rewrite", _expand_failure_is_repairable)

    # 4e) 重写把模型改缩水 → 拒绝采用，保留本轮结果（展开后退化守卫）
    def _degenerate_rewrite_rejected():
        from nl2opt.dataset import Sample
        from nl2opt.llm.fake import ScriptedProvider

        def model(values):
            return {
                "variables": [{"name": "x", "dims": [{"kind": "list", "values": values}],
                               "var_type": "binary"}],
                "objective": {"sense": "minimize",
                              "expr": {"terms": [{"var": "x_{i}", "coef": 1.0}]}},
                "constraints": [
                    {"name": "c1_{i}", "for_each": "i", "sense": "<=", "rhs": 1.0,
                     "lhs": {"terms": [{"var": "x_{i}", "coef": 1.0}]}},
                    {"name": "c2_{i}", "for_each": "i", "sense": ">=", "rhs": 0.0,
                     "lhs": {"terms": [{"var": "x_{i}", "coef": 1.0}]}},
                ],
            }

        full = model(["1", "2", "3", "4", "5"])   # 5 变量 / 10 约束
        shrunk = model(["1"])                     # 1 变量 / 2 约束 ← 缩水 80%
        llm = ScriptedProvider([json.dumps(full, ensure_ascii=False),
                                json.dumps(shrunk, ensure_ascii=False)])
        result = run_sample(
            Sample(id="test-degenerate", nl_text="测试退化重写", problem_type="milp",
                   expected={"objective_value": 5.0}),
            llm=llm, cfg=load_config())
        assert result.success is False
        assert result.rounds == 1, result.rounds
        assert any("退化" in e for e in result.errors), result.errors
        # 必须保留**未缩水**的那一版模型
        n_inst = sum(len(v.get("index_set") or []) for v in result.final_ir["variables"])
        assert n_inst == 5, f"采用了缩水模型（{n_inst} 个变量实例）"
    check("degenerate rewrite rejected (keeps bigger model)", _degenerate_rewrite_rejected)

    # 5) 配置包装器：`in` / 遍历（否则会 KeyError: 0 崩溃）
    def _config_contains():
        cfg = load_config()
        models = cfg.llm.pricing.models
        assert "deepseek-flash" in models, "Config.__contains__ 未生效"
        assert "no-such-model" not in models
        assert set(iter(models)) >= {"deepseek-flash", "deepseek-v4-pro"}, list(models)
        assert "models" in cfg.llm.pricing and "default_model" in cfg.llm.pricing
        assert len(models) >= 2
    check("config supports 'in' & iteration", _config_contains)

    # 6) 成本报表：按模型选价（未登记模型须显式报错）
    def _pricing_by_model():
        import report_cost as rc
        cfg = load_config()
        name, blk = rc._resolve_pricing(cfg, "deepseek-v4-pro")
        assert name == "deepseek-v4-pro"
        assert rc._price(blk, "idle", "output", None) == 13.5
        _, blk2 = rc._resolve_pricing(cfg, "deepseek-flash")
        assert rc._price(blk2, "idle", "output", None) == 4.0
        assert rc._price(blk2, "peak", "output", None) == 8.0
        try:
            rc._resolve_pricing(cfg, "gpt-9")
        except SystemExit as exc:
            assert "未在" in str(exc)
        else:
            raise AssertionError("未登记模型未报错")
        assert rc._price(blk, "idle", "output", 99.0) == 99.0
    check("cost report picks price by model", _pricing_by_model)

    # 7) 协议版本校验（跨批次对比）—— **已移除**
    # 原检查依赖 `ablation._pair_comparable`；该脚本用于旧数据集的多批次 A/B 对比，
    # 已随数据集更换删除。`PROTOCOL_VERSION` 常量本身仍在 `nl2opt/__init__.py`。

    # 8) _map_status：status=1 一律 timeout（不变量：只设 time_limit）
    def _map_status_mapping():
        import inspect
        from nl2opt.solver import engine
        from nl2opt.solver.engine import _map_status

        class R:
            def __init__(self, st, msg=""):
                self.status = st
                self.message = msg

        assert _map_status(R(0)) == "optimal"
        for msg in ("", "time limit reached", "iteration limit", "whatever"):
            assert _map_status(R(1, msg)) == "timeout", msg
        assert _map_status(R(2)) == "infeasible"
        assert _map_status(R(3)) == "unbounded"
        assert _map_status(R(99)) == "error"
        assert _map_status(R(None)) == "error"
        src = inspect.getsource(engine.solve)
        assert "time_limit" in src
        for forbidden in ("node_limit", "iteration_limit", "simplex_iteration_limit"):
            assert forbidden not in src, f"新增 {forbidden} 后须重新区分 status=1"
    check("_map_status mapping + timeout invariant", _map_status_mapping)

    # 9) 目标差异细分 —— **已移除**（原依赖的归因脚本已删除）。

    # 10) 提示词：符号核心是唯一在用的提示词；领域卡片**未接入**（接线状态断言）
    def _prompt_wiring():
        from nl2opt.pipeline.prompts import (
            CARDS, CORE_PROMPT, SYMBOLIC_CORE_PROMPT, build_system_prompt, prompt_stats,
            select_cards,
        )
        # 不管题面命中了哪些卡片，system prompt 都**等于**符号核心本身
        for text in ("at most 40% of shifts can be vans",
                     "Assign tasks to machines on a schedule",
                     "plan 3 weeks of inventory for 2 SKUs",
                     ""):
            assert build_system_prompt(text) == SYMBOLIC_CORE_PROMPT, \
                f"卡片被意外注入符号提示词（题面 {text!r}）"
        # 卡片表仍完好（将来接回时不用重新校对），但 select_cards 的结果不再影响 prompt
        assert "ratio" in select_cards("investments earn 3% and healthcare earn 1%")
        assert "收益率" in CARDS["ratio"]["body"]
        assert "fixed_cost" in CARDS and "piecewise" in CARDS
        # prompt_stats 必须如实反映「没有卡片」
        stats = prompt_stats("at most 40% of shifts can be vans")
        assert stats["cards"] == [] and stats["symbolic"] is True, stats
        assert stats["prompt_chars"] == len(SYMBOLIC_CORE_PROMPT)
        # 防过拟合：核心是跨族通用，数据集专属词不得出现在**实际使用的**核心里
        for token in ("pill", "dose", "van", "trip", "bowl", "meal", "serving", "diet"):
            assert token not in SYMBOLIC_CORE_PROMPT, f"数据集专属词 {token} 泄漏进符号核心"
        # 展平核心仍在（未接入），但不得与之混淆
        assert CORE_PROMPT != SYMBOLIC_CORE_PROMPT
    check("prompt wiring (symbolic core only, cards unwired)", _prompt_wiring)

    # 11) 消融口径：prompt_mode 记录用于 A/B 配对
    def _prompt_mode_recorded():
        import importlib
        from nl2opt import cli
        from nl2opt.pipeline import PipelineResult

        def mk(meta):
            r = PipelineResult(sample_id="x")
            r.trajectory = [{"understand_meta": meta}]
            return r

        assert cli._prompt_mode(mk({"all_cards_mode": True})) == "all_cards"
        assert cli._prompt_mode(mk({"all_cards_mode": False})) == "layered"
        assert cli._prompt_mode(mk({})) == "layered"
        assert cli._prompt_mode(PipelineResult(sample_id="x")) == "layered"  # 无轨迹

        d = cli._result_to_dict(mk({"all_cards_mode": True}), model="m")
        assert d["prompt_mode"] == "all_cards", d
        assert d["protocol_version"] == importlib.import_module("nl2opt").PROTOCOL_VERSION
    check("prompt_mode recorded for ablation pairing", _prompt_mode_recorded)

    # 12) 语义不变性检查 —— **已移除**
    # `semantic.py` 只服务「与参考 IR 对比覆盖率」，而验证器已精简为
    # 「最优性比对 + 可行性断言」两项，该模块一并删除。


    # 14) 消融判定语义 —— **已移除**
    # 原检查依赖 `ablation._degenerate_objective`，同上（ablation 已删）。
    # 「目标退化」的等价判定现在由 `nl2opt.pipeline.expand` / `ir.objective_is_degenerate` 承担。

    # 15) 分阶段 token 记账（符号链路：understand + rewrite 各一次）
    def _stage_usage():
        import shutil

        from nl2opt.dataset import Sample
        from nl2opt.llm.fake import ScriptedProvider
        data_dir = _scratch_dir("_smoke_usage")
        try:
            wrong = _symbolic_knapsack(60.0, data_dir)
            right = _symbolic_knapsack(50.0, data_dir)
            llm = ScriptedProvider([
                json.dumps(wrong, ensure_ascii=False),
                json.dumps(right, ensure_ascii=False),
            ])
            result = run_sample(
                Sample(id="test-stage-usage", nl_text="测试分阶段记账", problem_type="milp",
                       expected={"objective_value": 220.0},
                       extra={"data_dir": str(data_dir)}),
                llm=llm, cfg=load_config())
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)
        assert result.success is True, result.errors
        stages = result.stage_usage
        assert set(stages) == {"understand", "rewrite"}, stages
        for name in ("understand", "rewrite"):
            assert stages[name]["calls"] == 1, (name, stages[name])
        assert sum(s["calls"] for s in stages.values()) == result.token_usage.get("calls")
    check("stage usage attribution", _stage_usage)

    # 16) 求解超时 / 不限时接线
    def _solve_timeout():
        ir = _knapsack_ir()
        sol = solve(ir, time_limit=1e-9)
        assert sol.status in ("timeout", "optimal"), f"实际状态: {sol.status}"
    check("solve time_limit wired", _solve_timeout)

    def _solve_no_limit():
        sol = solve(_knapsack_ir())
        assert sol.status == "optimal"
        assert sol.objective_value == 220.0
    check("solve default no limit works", _solve_no_limit)

    # ===================== 汇总 + pytest 派发 =====================

    inline = total - len(failures)
    print(f"\n通过 {inline}/{total}（smoke + 关键历史回归，零依赖）")

    pytest_passed, pytest_total = _run_pytest()
    if pytest_total == -1:
        print("[提示] 未安装 pytest，tests/ 下的分层套件跳过（`pip install -e .[dev]` 后可运行）")
        return 1 if failures else 0

    print(f"\n[pytest 分层套件] {pytest_passed}/{pytest_total} passed")
    if failures or (pytest_total > 0 and pytest_passed < pytest_total):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
