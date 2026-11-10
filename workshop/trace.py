"""Append-only traces, readable while a simulation is running."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def display(event: dict, tools: bool = False) -> None:
    kind = event["event"]
    if kind == "start":
        console.rule(Text(f"{event['task']} · {event['model']}"))
    elif kind == "message":
        color = "cyan" if event["actor"] == "agent" else "green"
        console.print(Panel(Text(event["text"]), title=event["actor"], border_style=color))
    elif kind == "tool" and tools:
        body = json.dumps(event["arguments"], ensure_ascii=False) + "\n" + event["result"]
        console.print(Panel(Text(body), title=f"{event['actor']} tool: {event['name']}",
                            border_style="yellow"))
    elif kind == "request" and tools:
        console.print(Text("Phone tools offered to customer: " + ", ".join(event["phone_tools"]),
                           style="dim"))
    elif kind == "result":
        console.rule(Text(f"{event['outcome'].upper()} · {event['termination']}"))
        for check in event["checks"]:
            console.print(Text(f"{'PASS' if check['passed'] else 'FAIL'}  "
                               f"{check['check']} {check['arguments']}"))
        if event.get("error"):
            console.print(Text(event["error"], style="red"))
    elif kind == "notice":
        console.print(Text(event["text"], style="dim"))


class Trace:
    def __init__(self, path: Path, show_tools: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("x", encoding="utf-8")
        self.seq = 0
        self.show_tools = show_tools
        self.calls = 0
        self.cost = 0.0
        self.cost_reports = 0
        self.lock = threading.Lock()

    def write(self, event: str, **fields):
        with self.lock:
            record = {"seq": self.seq, "time": datetime.now(timezone.utc).isoformat(),
                      "event": event, **fields}
            self.seq += 1
            self.stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self.stream.flush()
            display(record, tools=self.show_tools)
            return record

    def close(self):
        self.stream.close()


class ModelTrace(BaseCallbackHandler):
    def __init__(self, trace: Trace, actor: str):
        self.trace = trace
        self.actor = actor
        self.started = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.started[run_id] = time.monotonic()

    def on_llm_end(self, response, *, run_id, **kwargs):
        message = response.generations[0][0].message
        metadata = message.response_metadata
        tokens = metadata.get("token_usage") or {}
        cost = tokens.get("cost")
        self.trace.calls += 1
        if cost is not None:
            self.trace.cost += float(cost)
            self.trace.cost_reports += 1
        self.trace.write("model", actor=self.actor,
                         served_model=metadata.get("model_name"),
                         usage=message.usage_metadata, reported_cost=cost,
                         finish_reason=metadata.get("stop_reason") or metadata.get("finish_reason"),
                         seconds=round(time.monotonic() - self.started.pop(run_id, time.monotonic()), 3))
