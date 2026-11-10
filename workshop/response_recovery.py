"""Recover empty or token-truncated model responses before any tool execution."""

import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .models import DEFAULT_MAX_TOKENS

MAX_ATTEMPTS = 3
MAX_RECOVERY_TOKENS = 8192
RETRY_INSTRUCTION = """Your previous response was empty, incomplete or invalid and
was not delivered. Produce a complete short reply or a real offered tool call,
following the original role, persona and policy. Do not treat a missing response
as consent or a successful action. Keep any refusal, uncertainty or departure
required by the conversation. Do not repeat completed operations. No operation
from the rejected response has executed. Do not discuss this retry in dialogue.
"""


class ResponseRecovery(AgentMiddleware):
    """Retry only the model node, inside the existing role guards."""

    def __init__(self, trace, actor):
        self.trace = trace
        self.actor = actor

    def wrap_model_call(self, request, handler):
        settings = dict(request.model_settings)
        # Preserve a caller's explicit spelling of the output-token limit.
        token_key = "max_completion_tokens" if "max_completion_tokens" in settings else "max_tokens"
        limit = settings.get(token_key) or getattr(request.model, "max_tokens", None) or DEFAULT_MAX_TOKENS
        start = max((i for i, m in enumerate(request.messages) if isinstance(m, HumanMessage)), default=-1)
        completed = {m.name for m in request.messages[start + 1:] if isinstance(m, ToolMessage)}
        available = request.tools
        messages = request.messages
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = handler(request.override(messages=messages, tools=available, model_settings=settings))
            message = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
            metadata = message.response_metadata if message is not None else {}
            # Claude reports stop_reason="max_tokens"; OpenAI-style models report "length".
            finish = metadata.get("stop_reason") or metadata.get("finish_reason")
            problem = None
            if finish in ("max_tokens", "length"):
                problem = "output token limit reached"
            elif message is None or not (message.text.strip() or message.tool_calls or message.invalid_tool_calls):
                problem = "empty model reply"
            elif attempt > 1 and any(c["name"] not in {t.name for t in available} for c in message.tool_calls):
                problem = "retry requested an unavailable or already completed tool"
            if problem is None:
                return response

            self.trace.write("model_reply_rejected", actor=self.actor, attempt=attempt,
                             reason=problem, finish_reason=finish, max_tokens=limit,
                             text=message.text if message is not None else "",
                             calls=message.tool_calls if message is not None else [],
                             usage=message.usage_metadata if message is not None else None)
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{self.actor} response recovery exhausted after {MAX_ATTEMPTS} attempts "
                    f"({problem}; finish_reason={finish}; token limit={limit}); "
                    "this is not a task failure.")
            if finish in ("max_tokens", "length"):
                limit = min(limit * 2, MAX_RECOVERY_TOKENS)
                settings = {**settings, token_key: limit}
            available = [t for t in available if t.name not in completed]
            messages = [*request.messages, SystemMessage(content=RETRY_INSTRUCTION
                        + "\nCurrently offered tools: " + json.dumps([t.name for t in available])
                        + ". If no tools are offered, reply in words without taking any action.")]
            self.trace.write("notice", actor=self.actor, behavior="response_recovery",
                             attempt=attempt, next_max_tokens=limit,
                             text=f"{self.actor}: {problem}; retry {attempt + 1}/{MAX_ATTEMPTS} "
                                  f"with token limit {limit}. No rejected tool call was executed.")
