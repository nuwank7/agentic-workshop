"""Claude models (Anthropic API) shared by both conversation participants."""

import os
import time
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import Field

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MAX_TOKENS = 2048
# Shared between the agent and the simulated user so one run stays well below
# the lowest Anthropic rate-limit tier (50 requests per minute).
RATE_LIMITER = InMemoryRateLimiter(requests_per_second=0.5, max_bucket_size=1)
TRANSIENT_STATUS = (429, 500, 502, 503, 504, 529)


def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise ValueError("Set ANTHROPIC_API_KEY in .env before running a live task.")
    return key


def as_claude_messages(messages):
    """Keep the role-guard reminders without depending on model-specific features.

    The middleware appends SystemMessages after tool results. Only the first
    system prompt maps to Anthropic's top-level ``system`` field, so later ones
    are sent as tagged user-turn reminders. A conversation must also open with a
    user turn, so a scripted assistant greeting gets a neutral opener before it.
    """
    result = []
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage) and index > 0:
            message = HumanMessage(content=f"<system-reminder>\n{message.text}\n</system-reminder>")
        result.append(message)
    first = next((i for i, m in enumerate(result) if not isinstance(m, SystemMessage)), None)
    if first is not None and result[first].type == "ai":
        result.insert(first, HumanMessage(content="(The support chat has opened.)"))
    return result


class WorkshopClaude(ChatAnthropic):
    """Retry only transient HTTP errors, before any tool execution occurs."""

    retry_observer: Any = Field(default=None, exclude=True)

    def bind_tools(self, tools, **kwargs):
        kwargs["parallel_tool_calls"] = False
        if any(getattr(tool, "name", None) == "ask_customer" for tool in tools):
            kwargs["tool_choice"] = "any"  # Anthropic's spelling of "required"
        return super().bind_tools(tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = as_claude_messages(messages)
        for attempt in range(6):
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status not in TRANSIENT_STATUS or attempt == 5:
                    raise
                delay = min(20 * (attempt + 1), 60)
                if self.retry_observer:
                    self.retry_observer(status, attempt + 1, delay)
                time.sleep(delay)


def preflight(name: str) -> dict:
    """Confirm the key works and the model exists. Listing models is free."""
    import anthropic

    try:
        info = anthropic.Anthropic(api_key=api_key()).models.retrieve(name)
    except anthropic.NotFoundError:
        raise ValueError(f"Unknown Claude model id: {name}") from None
    except anthropic.AuthenticationError:
        raise ValueError("ANTHROPIC_API_KEY was rejected; check the key in .env.") from None
    return {"requested_model": name, "provider": "anthropic",
            "display_name": getattr(info, "display_name", None)}


def make_model(name: str, on_retry=None) -> ChatAnthropic:
    if not name.startswith("claude-"):
        raise ValueError("Use a Claude model id, for example claude-haiku-5-5.")
    return WorkshopClaude(
        model=name, api_key=api_key(), max_tokens=DEFAULT_MAX_TOKENS,
        timeout=90, max_retries=0, rate_limiter=RATE_LIMITER, retry_observer=on_retry,
    )
