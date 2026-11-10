# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A two-session hands-on workshop on agentic AI, cloned from
[kaushithamsilva/icter-workshop](https://github.com/kaushithamsilva/icter-workshop)
and adapted to run on Claude through the Anthropic API.

- **Session 01** (`main.py`, `scenarios/`): LangGraph agents on Claude showing how
  context, tool descriptions and error messages change agent behaviour.
- **Session 02** (`telecom.py`, `showprogress.py`, `workshop/`): a support agent and a
  simulated customer run Sierra τ²-bench telecom tasks; the environment state is graded.

## Commands

Always use `uv` from the repository root. Do not activate a venv manually.

```bash
uv sync --frozen                                   # install locked deps
uv run main.py <task> [--scenario NAME] [--model ID]
uv run telecom.py validate                         # offline: task environments load
uv run telecom.py easy|hard-user|hard-env|hard-both|all [--model ID] [--max-turns N]
uv run showprogress.py [TRACE.jsonl] [--follow] [--tools] [--delay S]
uv run python scripts/verify_examples.py           # offline: replay recorded examples
uv run python -m unittest discover -s tests -v     # offline test suite
```

Session 01 tasks: `check-my-balance`, `wrong-tool-description`, `wrong-error-message`,
`blocked-account`.

Before finishing any code change, run the two offline checks (`verify_examples.py` and
the unittest suite). They need no API key and make no model calls. Live runs
(`main.py`, `telecom.py <task>`) are billed to the user's Anthropic account, so only run
them when asked.

## Configuration

- `ANTHROPIC_API_KEY` and optional `ANTHROPIC_MODEL` live in `.env` (copy `.env.example`).
- Model precedence for `main.py`: `--model`, then `ANTHROPIC_MODEL`, then `claude-haiku-5-5`.
  `telecom.py` ignores `ANTHROPIC_MODEL`; use `--model`.

## Layout

- `main.py` — Session 01 CLI; `scenarios/__init__.py` registers tasks in `TASKS`.
- `scenarios/common.py` — model setup, mock user data (Alex Perera, `ACC-1001`), `ask` helper.
- `scenarios/observability.py` — rich trace callback + loguru (`logs/agent.log`).
- `telecom.py` — Session 02 CLI; `workshop/runner.py` drives the two-role conversation.
- `workshop/support_role.py`, `workshop/customer_role.py` — role prompts and reply validation.
- `workshop/response_recovery.py` — retries for empty/truncated replies (2048 → 4096 → 8192 tokens).
- `workshop/prompts/` — baseline agent prompt and hard-user addition.
- `vendor/tau2-core/` — vendored subset of τ²-bench (MIT). Treat as read-only;
  `PROVENANCE.json` records upstream SHA-256 hashes.
- `examples/` — recorded baseline traces, verified by hash.
- `docs/telecom-details.md` — instructor notes.

## Rules

- Never commit `.env` or paste API keys anywhere; `.env`, `logs/` and `runs/` are git-ignored.
- Do not edit files under `examples/` or `vendor/tau2-core/`: `verify_examples.py` checks
  them by hash, and line endings must stay LF (see `.gitattributes`).
- Keep the scenarios free of print statements; tracing goes through the callback handler.
- To add a Session 01 task, create a module in `scenarios/` with `QUESTION` and a
  `SCENARIOS` dict, then add it to `TASKS`.
- Models are non-deterministic; a live run differing from a recording is not a bug by itself.
