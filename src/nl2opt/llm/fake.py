"""离线 Fake LLM Provider：不联网，用于测试/演示「理解→重写」闭环。

两种实现都实现 LLMProvider 接口，可无缝替换 DeepSeekProvider：

- CallableProvider：complete() 调用你传入的 handler(messages) -> str，最灵活。
- ScriptedProvider：按顺序弹出预设响应（字符串或 handler），适合简单线性脚本。

用法示例（handler 依据消息内容区分环节）：

    from nl2opt.llm.fake import CallableProvider

    def handler(messages) -> str:
        joined = " ".join(m.content for m in messages)
        if "UNDERSTAND_IR" in joined:
            return json.dumps(corrupt_ir)      # 模拟“理解错”
        if "REWRITE_IR" in joined:
            return json.dumps         # 模拟“修好”
                return "建模错误：rhs 应为 100"
        return "{}"

    llm = CallableProvider(handler)
"""
from __future__ import annotations

from typing import Any, Callable, Sequence, Union

from . import LLMError, LLMMessage, LLMProvider, LLMResponse

Handler = Callable[[list[LLMMessage]], str]
ResponseItem = Union[str, Handler]


class CallableProvider(LLMProvider):
    """把 complete() 委托给一个用户提供的 handler 函数。"""

    def __init__(self, handler: Handler, name: str = "fake-callable") -> None:
        self.handler = handler
        self.name = name
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls.append([{"role": m.role, "content": m.content} for m in messages])
        text = self.handler(messages)
        return LLMResponse(text=text, raw={"provider": self.name})


class ScriptedProvider(LLMProvider):
    """按顺序弹出预设响应。响应既可以是字符串，也可以是 handler 函数。"""

    def __init__(self, responses: Sequence[ResponseItem], name: str = "fake-scripted") -> None:
        self.responses: list[ResponseItem] = list(responses)
        self.name = name
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls.append([{"role": m.role, "content": m.content} for m in messages])
        if not self.responses:
            raise LLMError("ScriptedProvider 响应队列已耗尽")
        item = self.responses.pop(0)
        text = item(messages) if callable(item) else item
        return LLMResponse(text=text, raw={"provider": self.name})


__all__ = ["CallableProvider", "ScriptedProvider", "Handler", "ResponseItem"]
