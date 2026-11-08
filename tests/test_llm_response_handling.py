"""统一 LLM 响应保护层与答案防泄露的单元测试。

对应 scripts/run_tests.py 里的截断/空输出/解析失败保护、答案泄露剥离断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from nl2opt.llm import LLMError, LLMMessage, LLMProvider, LLMResponse  # noqa: E402
from nl2opt.pipeline.rewrite import rewrite_symbolic  # noqa: E402
from nl2opt.pipeline.understand import understand  # noqa: E402


class _FixedLLM(LLMProvider):
    """可控假 Provider：可指定 finish_reason（截断）与固定文本。"""

    def __init__(self, text="", finish_reason=None):
        self._text = text
        self._finish = finish_reason

    def complete(self, messages, **kwargs):
        return LLMResponse(text=self._text, finish_reason=self._finish)


def test_understand_rejects_truncated_response():
    class TruncatingProvider(LLMProvider):
        def complete(self, messages, **kwargs):
            return LLMResponse(
                text='{"variables": [{"name": "y", "var_type": "integer"',
                usage={}, finish_reason="length",
            )

    with pytest.raises(LLMError, match="截断"):
        understand("测试截断", TruncatingProvider())


def _empty_sym():
    """一个最小的符号模型（保护层只关心「响应能不能用」，不看模型内容）。"""
    from nl2opt.symbolic import SymbolicIR
    return SymbolicIR.model_validate(
        {"variables": [], "objective": {"sense": "minimize", "expr": {"terms": []}},
         "constraints": []}
    )


def test_rewrite_rejects_truncated_response():
    with pytest.raises(LLMError, match="rewrite.*截断"):
        rewrite_symbolic("测试题", _empty_sym(), "err",
                         _FixedLLM(text='{"x": ', finish_reason="length"))


def test_rewrite_rejects_empty_response():
    """空响应（只有空白）也要被保护层拦下，不能送去解析。"""
    with pytest.raises(LLMError):
        rewrite_symbolic("测试题", _empty_sym(), "err",
                         _FixedLLM(text="   ", finish_reason="stop"))


def test_no_answer_value_leak_into_prompt():
    """check_optimality 失败 message 不得含参考答案数值（污染 rewrite prompt）。"""
    from nl2opt import verifier
    from nl2opt.solver import SolverResult
    sol = SolverResult(status="optimal", objective_value=221.0, variables={})
    r = verifier.check_optimality(sol, 220.0, 1e-4)
    assert r.ok is False
    assert "220" not in r.message  # message 不含期望值
    assert r.detail.get("expected") == 220.0  # 数值保留在 detail，不进 prompt
