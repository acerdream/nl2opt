"""LLM 抽象层：统一调用接口，便于替换后端。

当前实现 DeepSeek（OpenAI 兼容协议），用已内置的 httpx 直连，
不依赖 openai SDK。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class LLMMessage:
    """一条对话消息。"""

    role: str  # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    """LLM 返回结果。"""

    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None  # "stop" | "length"（截断）| None（未知）

    @property
    def truncated(self) -> bool:
        """输出是否因 max_tokens 被截断。

        思考型模型常见陷阱：reasoning 吃光 max_tokens，正文 JSON 被截断，
        而 extract_json 的宽松兜底会把断片解析成畸形 dict，掩盖真实原因。
        """
        return self.finish_reason == "length"


class LLMError(RuntimeError):
    """LLM 调用失败。"""


class LLMProvider(ABC):
    """LLM 提供方抽象接口。"""

    @abstractmethod
    def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """给定消息列表，返回补全结果。"""


class DeepSeekProvider(LLMProvider):
    """DeepSeek 官方 API（OpenAI 兼容）实现。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: float = 60.0,
        reasoning_effort: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort

    def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        # 思考强度：仅在非空时透传（DeepSeek 支持 low/medium/high）
        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"网络错误: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"DeepSeek 返回 {resp.status_code}: {resp.text[:500]}")

        data = resp.json()

        # 显式校验响应骨架：服务端偶发返回 `{"choices": null}`（限流/后端异常），
        # 直接 `data["choices"][0]` 会抛 TypeError，被上层宽泛的 `except Exception`
        # 捕获后**伪装成「建模/理解失败」**，把服务故障误报成模型能力问题。
        if not isinstance(data, dict):
            raise LLMError(f"响应不是 JSON 对象: {str(data)[:300]}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            err = data.get("error") or data.get("message") or data
            raise LLMError(f"响应缺少 choices（疑似限流或服务异常）: {str(err)[:400]}")

        try:
            choice = choices[0]
            if not isinstance(choice, dict):
                raise LLMError(f"choices[0] 不是对象: {str(choice)[:200]}")
            message = choice["message"]
            text = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            # 空 content 兜底：思考型模型有时把全部输出耗尽在 reasoning_content，
            # 此时若 reasoning_content 里有内容，用它兜底（可能含最终答案）。
            if not text.strip():
                rc = message.get("reasoning_content") or ""
                if rc.strip():
                    text = rc
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应结构异常: {str(data)[:400]}") from exc

        return LLMResponse(
            text=text,
            usage=data.get("usage", {}),
            raw=data,
            finish_reason=finish_reason,
        )


def make_provider(cfg) -> LLMProvider:
    """依据配置构造 LLM 提供方。"""
    from ..config import resolve_api_key

    provider_name = cfg.llm.get("provider", "deepseek")
    if provider_name != "deepseek":
        raise LLMError(f"不支持的 LLM provider: {provider_name}")

    api_key = resolve_api_key(cfg)
    if not api_key:
        raise LLMError(
            "未找到 DEEPSEEK_API_KEY。请在 .env 中配置，或设置环境变量 DEEPSEEK_API_KEY。"
        )

    return DeepSeekProvider(
        api_key=api_key,
        base_url=cfg.llm.get("base_url", "https://api.deepseek.com"),
        model=cfg.llm.get("model", "deepseek-chat"),
        temperature=float(cfg.llm.get("temperature", 0.0)),
        max_tokens=int(cfg.llm.get("max_tokens", 2048)),
        timeout_seconds=float(cfg.llm.get("timeout_seconds", 60.0)),
        reasoning_effort=cfg.llm.get("reasoning_effort", None),
    )


def complete_structured(
    llm: LLMProvider,
    messages: list[LLMMessage],
    stage: str,
    retry_low_reasoning: bool = True,
    require_nonempty: bool = True,
) -> LLMResponse:
    """统一 LLM 响应保护层：截断 / 空输出检测 + 一次重试 + 带 stage 的明确错误。

    为什么需要它（实测缺陷）：同一个「模型截断 / 空输出」在 `understand` 阶段有明确错误，
    但在 `rewrite` 阶段却直接 `extract_json(resp.text)` 或把空文本当成建模错误——
    于是「预算不足」被误报成「建模错误」，污染失败归因。

    职责：
    - `resp.truncated`（finish_reason=length）或 `resp.text` 为空 → 重试一次
      （思考型模型常见陷阱：reasoning 吃光输出预算，正文 JSON 被截断）。
      `reasoning_effort` 已全局固定为 `low`，因此这里重试**不再降档**，
      而是一次同参数重试（对空输出/瞬时抖动有效，对真实预算不足基本无效）；
    - 重试后仍截断 → 抛带 stage 名的 `LLMError`（指引调大 `max_tokens`）；
    - 重试后仍空（且 `require_nonempty`）→ 抛带 stage 名的 `LLMError`。

    **只负责「生产失败」，不负责解析**：返回原始 `LLMResponse`，各阶段按需自行
    `extract_json` / 提取代码。这样 understand/rewrite 对「LLM 没给出可用输出」
    的判定一致，而解析失败（响应完整但 JSON 非法）仍需各阶段显式处理。
    """
    resp = llm.complete(messages)
    if retry_low_reasoning and (resp.truncated or not resp.text.strip()):
        try:
            resp = llm.complete(messages, reasoning_effort="low")
        except Exception:
            # 重试本身失败（如网络）不掩盖原始问题，继续用首次响应判定
            pass
    if resp.truncated:
        raise LLMError(
            f"[{stage}] LLM 输出被 max_tokens 截断（finish_reason=length）："
            "请调大 config.yaml 的 llm.max_tokens"
            "（reasoning_effort 已固定 low，无进一步降档空间）"
        )
    if require_nonempty and not resp.text.strip():
        raise LLMError(f"[{stage}] LLM 返回空 content（思考可能耗尽 max_tokens）")
    return resp


def extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出中尽力提取 JSON 对象（容忍 markdown 代码块包裹）。"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉首行 ```json 与末行 ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 兜底：思考型模型可能把 reasoning 文本泄漏到 content 开头，
    # 导致首个 { 之前有噪声。逐个尝试每个 { 作为起点，解析到匹配的 } 结束，
    # 返回第一个能成功解析的完整 JSON 对象。
    for i, ch in enumerate(text):
        if ch == "{":
            # 用括号匹配找到对应的 } 位置
            depth = 0
            end_idx = -1
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = j
                        break
            if end_idx != -1:
                try:
                    return json.loads(text[i : end_idx + 1])
                except json.JSONDecodeError:
                    continue

    raise LLMError(f"无法从 LLM 输出解析 JSON: {text[:300]}")
