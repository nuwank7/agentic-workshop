"""Run one of four telecom workshop tasks using Claude models (Anthropic API)."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from workshop.models import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["easy", "hard-user", "hard-env", "hard-both", "all", "validate"])
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Claude model id for BOTH participants (default %(default)s)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Optional conversation-turn cap; omitted means unlimited")
    parser.add_argument("--tools", action=argparse.BooleanOptionalAction, default=True,
                        help="Display tool calls/results (enabled by default; --no-tools hides them)")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--agent-prompt", type=Path, help="Optional prompt for later workshop improvements")
    args = parser.parse_args()
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns must be positive")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    logger.remove()
    from workshop.tasks import TASKS, validate_tasks
    if args.task == "validate":
        for result in validate_tasks():
            print(f"{result['task']}: solvable; {result['checks']} native environment checks")
        return 0
    from workshop.runner import run
    results = []
    try:
        for name in TASKS if args.task == "all" else [args.task]:
            results.append(run(name, args.model, args.output, args.max_turns,
                               args.tools, args.agent_prompt))
            if results[-1]["outcome"] == "error":
                break  # Do not repeat a provider outage for every condition.
    except Exception as exc:
        print(f"Setup error: {type(exc).__name__}: {exc}")
        return 2
    return 2 if any(r["outcome"] == "error" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
