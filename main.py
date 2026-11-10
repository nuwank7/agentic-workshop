"""CLI: route a task to one or all of the agent scenarios."""

import os
from enum import Enum
from typing import Annotated

import anthropic
import typer
from dotenv import load_dotenv

from loguru import logger
from rich.panel import Panel

from scenarios import DEFAULT_MODEL, TASKS
from scenarios.observability import console, setup_logging, trace

load_dotenv()
setup_logging()

Task = Enum("Task", {name: name for name in TASKS}, type=str)


def main(
    task: Annotated[Task, typer.Argument(help="Task to run")],
    scenario: Annotated[
        str | None, typer.Option(help="Run only one scenario of the task")
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            envvar="ANTHROPIC_MODEL",
            help="Claude model id, for example claude-haiku-5-5 or claude-sonnet-5-5",
        ),
    ] = DEFAULT_MODEL,
) -> None:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key or key == "your-key-here":
        console.print("[red]Set ANTHROPIC_API_KEY in .env before running a task.[/red]")
        raise typer.Exit(2)
    module = TASKS[task.value]
    scenarios = module.SCENARIOS
    if scenario and scenario not in scenarios:
        raise typer.BadParameter(
            f"must be one of: {', '.join(scenarios)}", param_hint="--scenario"
        )
    for name in [scenario] if scenario else scenarios:
        trace.reset()
        logger.info("start {} / {} (model {})", task.value, name, model)
        console.print()
        console.rule(f"[bold]{task.value} / {name}[/bold]")
        console.print(Panel(module.QUESTION, title="task", border_style="green"))
        try:
            outcome = scenarios[name](module.QUESTION, model)
        except anthropic.AuthenticationError:
            console.print("[red]ANTHROPIC_API_KEY was rejected; check the key in .env.[/red]")
            raise typer.Exit(2)
        except anthropic.NotFoundError:
            console.print(f"[red]Unknown Claude model id: {model}[/red]")
            raise typer.Exit(2)
        if outcome:
            console.print(Panel(outcome, title="outcome", border_style="bold white"))
        logger.info("end {} / {}: {}", task.value, name, outcome)


if __name__ == "__main__":
    typer.run(main)
