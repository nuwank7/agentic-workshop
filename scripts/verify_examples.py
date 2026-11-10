"""Re-execute saved tool calls and independently verify the selected outcomes.

No LLM calls, API key, or network access. Works from any working directory.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
from pydantic import BaseModel

logger.remove()
from workshop.tasks import Task, evaluate, new_environment


def verify(trace_path: Path, config_path: Path) -> dict:
    # Early recordings flushed simultaneous tool callbacks out of order. Their
    # sequence numbers retain the actual event order; preserve the raw files.
    events = sorted((json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()),
                    key=lambda event: event["seq"])
    assert [e["seq"] for e in events] == list(range(len(events))), "Missing/duplicate events"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    start = events[0]
    for actor in ("agent", "user"):
        digest = hashlib.sha256(config[f"{actor}_prompt"].encode()).hexdigest()
        assert digest == start[f"{actor}_prompt_sha256"], "Prompt snapshot differs from recording"
    task = Task.model_validate(config["task"])
    env = new_environment(task)
    allowed = set()
    for event in events:
        if event["event"] == "request":
            allowed = set(event["phone_tools"])
        elif event["event"] == "notice" and event.get("behavior") == "overwhelmed":
            allowed = set()
        elif event["event"] == "tool":
            if event["actor"] == "user":
                assert event["name"] in allowed, f"Unrequested customer action: {event['name']}"
            try:
                result = env.make_tool_call(
                    event["name"], requestor="user" if event["actor"] == "user" else "assistant",
                    **event["arguments"],
                )
                env.sync_tools()
                if isinstance(result, BaseModel):
                    result = result.model_dump(mode="json")
                result = result if isinstance(result, str) else json.dumps(result, default=str)
            except (TypeError, ValueError) as exc:
                result = f"Tool error: {exc}"
            assert result == event["result"], f"Tool output changed: {event['name']}"
    final = events[-1]
    assert final["event"] == "result" and final["outcome"] in ("pass", "fail")
    checks = evaluate(env, task)
    assert checks == final["checks"], "Recorded checks differ from replayed state"
    expected = "pass" if all(c["passed"] for c in checks) else "fail"
    assert expected == final["outcome"], "Recorded outcome differs from native assertions"
    calls = [e for e in events if e["event"] == "model"]
    assert calls and all(e.get("reported_cost") == 0 for e in calls), "Missing/nonzero cost evidence"
    assert all(e.get("served_model", "").endswith(":free") for e in calls)
    return {"task": task.id, "outcome": expected, "checks": len(checks), "model_calls": len(calls)}


def main():
    directory = ROOT / "examples"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = {"easy": "pass", "hard-user": "fail", "hard-env": "fail", "hard-both": "fail"}
    assert {e["task"]: e["outcome"] for e in manifest["selected"]} == expected
    assert len({e["agent_prompt_sha256"] for e in manifest["selected"]}) == 1
    for entry in manifest["selected"]:
        trace = directory / entry["trace"]
        config = directory / entry["config"]
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == entry["trace_sha256"]
        assert hashlib.sha256(config.read_bytes()).hexdigest() == entry["config_sha256"]
        result = verify(trace, config)
        assert result["outcome"] == entry["outcome"]
        print(f"{result['task']}: {result['outcome'].upper()} reproduced from native tool replay; "
              f"{result['checks']} checks; {result['model_calls']} zero-cost model calls")


if __name__ == "__main__":
    main()
