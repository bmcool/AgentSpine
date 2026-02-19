from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass
class ProviderResponse:
    assistant_message: dict[str, Any]
    tool_calls: list[ToolCall]
    text: str
    usage: dict[str, int] | None = None


class Provider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str | None = None,
        thinking_level: str = "off",
        on_text_delta: Callable[[str], None] | None = None,
        api_key: str | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError
