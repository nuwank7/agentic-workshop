"""Task authoring and native Tau assertions; never passed to the agent."""

import json
import os
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Pin data to this checkout, even if another Tau installation set this variable.
os.environ["TAU2_DATA_DIR"] = str(ROOT / "data")

from tau2.data_model.tasks import EnvAssertion, Task
from tau2.domains.telecom.environment import get_environment
from tau2.domains.telecom.tasks.const import PERSONAS

TASKS = {
    "easy": (False, False),
    "hard-user": (True, False),
    "hard-env": (False, True),
    "hard-both": (True, True),
}


def load_task(name: str) -> Task:
    """Same goal; vary only the persona and the initial faults/checks."""
    hard_user, hard_env = TASKS[name]
    sources = json.loads((ROOT / "data/source_tasks.json").read_text(encoding="utf-8"))
    record = deepcopy(sources[1 if hard_env else 0])
    record["id"] = name  # This ID is for the runner, not a model input.
    record["user_scenario"]["persona"] = PERSONAS["Hard" if hard_user else "Easy"]
    if hard_user:
        record["user_scenario"]["persona"] += "\n" + (
            ROOT / "prompts/hard_user_addition.txt"
        ).read_text(encoding="utf-8")
    # Keep the original goal, known information and consent limits identical.
    # Native checks expose the extra prerequisites explicitly for teaching.
    extras = [
        ("assert_airplane_mode_status", {"expected_status": False}),
        ("assert_service_status", {"expected_status": "connected"}),
    ]
    if hard_env:
        extras += [
            ("assert_mobile_data_saver_mode_status", {"expected_status": False}),
            ("assert_mobile_data_usage_exceeded", {"expected_status": False}),
        ]
    for func, args in extras:
        record["evaluation_criteria"]["env_assertions"].append(
            EnvAssertion(env_type="user", func_name=func, arguments=args,
                         assert_value=True).model_dump()
        )
    return Task.model_validate(record)


def new_environment(task: Task):
    env = get_environment()
    initial = task.initial_state
    env.set_state(initialization_data=initial.initialization_data,
                  initialization_actions=initial.initialization_actions,
                  message_history=initial.message_history or [])
    return env


def evaluate(env, task: Task) -> list[dict]:
    """Grade real state, never the model's claim that it succeeded."""
    env.sync_tools()
    return [
        {"check": check.func_name, "arguments": check.arguments,
         "passed": bool(env.run_env_assertion(check, raise_assertion_error=False))}
        for check in task.evaluation_criteria.env_assertions
    ]


def validate_tasks() -> list[dict]:
    """Offline solvability check using upstream reference actions, not an agent."""
    results = []
    for name in TASKS:
        task = load_task(name)
        env = new_environment(task)
        assert not all(c["passed"] for c in evaluate(env, task)), name
        for action in task.evaluation_criteria.actions:
            env.make_tool_call(tool_name=action.name, requestor=action.requestor,
                               **action.arguments)
            env.sync_tools()
        checks = evaluate(env, task)
        assert all(c["passed"] for c in checks), (name, checks)
        results.append({"task": name, "solvable": True, "checks": len(checks)})
    return results
