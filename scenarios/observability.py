"""Rich console output and loguru logging for agent runs.

A LangChain callback handler sees every prompt, model reply, tool call and tool
response as it happens, so scenarios stay free of print statements.
"""

import sys
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

console = Console()

PROMPT_STYLES = {"system": "blue", "human": "green"}


def setup_logging(log_file: str = "logs/agent.log") -> None:
    """Warnings to the terminal, full detail to a file."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    logger.add(log_file, level="DEBUG", rotation="1 MB")


def note(text: str) -> None:
    """A step that happens outside the agent, such as a script."""
    logger.info("script: {}", text)
    console.print(Panel(text, title="script", border_style="white"))


class TraceHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._calls = 0
        self._started = 0.0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self._calls += 1
        self._started = time.perf_counter()
        model = kwargs.get("invocation_params", {}).get("model", "")
        console.rule(f"[dim]LLM call {self._calls} {model}")
        for msg in messages[0]:
            if isinstance(msg, (AIMessage, ToolMessage)):
                continue  # shown when they were produced
            key = (msg.type, str(msg.content))
            if key in self._seen:
                continue  # already shown on an earlier call or turn
            self._seen.add(key)
            logger.debug("prompt [{}]: {}", msg.type, msg.content)
            console.print(
                Panel(
                    str(msg.content),
                    title=msg.type,
                    border_style=PROMPT_STYLES.get(msg.type, "white"),
                )
            )

    def on_llm_end(self, response, **kwargs) -> None:
        msg = response.generations[0][0].message
        usage = getattr(msg, "usage_metadata", None) or {}
        text = msg.text.strip()
        served_by = msg.response_metadata.get("model_name", "?")  # the real model behind a router
        console.print(
            f"[dim]{served_by}, {time.perf_counter() - self._started:.1f}s, "
            f"{usage.get('input_tokens', '?')} in / "
            f"{usage.get('output_tokens', '?')} out tokens[/dim]"
        )
        logger.debug("model reply [{}]: {!r} tool_calls={}", served_by, text, msg.tool_calls)
        if text:
            console.print(Panel(text, title="assistant", border_style="cyan"))
        for call in msg.tool_calls:
            console.print(
                Panel(
                    JSON.from_data(call["args"]),
                    title=f"tool call: {call['name']}",
                    border_style="yellow",
                )
            )
        if not text and not msg.tool_calls:
            console.print(Panel("(empty reply)", title="assistant", border_style="red"))

    def on_llm_error(self, error, **kwargs) -> None:
        logger.error("model error: {}", error)
        console.print(Panel(str(error), title="model error", border_style="red"))

    def on_tool_end(self, output, **kwargs) -> None:
        name = getattr(output, "name", None) or kwargs.get("name", "")
        content = str(getattr(output, "content", output))
        logger.debug("tool response [{}]: {}", name, content)
        console.print(
            Panel(content, title=f"tool response: {name}", border_style="magenta")
        )

    def on_tool_error(self, error, **kwargs) -> None:
        logger.error("tool error: {}", error)
        console.print(Panel(str(error), title="tool error", border_style="red"))


trace = TraceHandler()
