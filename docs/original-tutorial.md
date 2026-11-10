# LangGraph agent tutorial

A small CLI that sends one task to several LangGraph agents (running on Claude) so you
can see how context and tool design change the result.

| Task | Lesson |
|---|---|
| `check-my-balance` | Context and tools change what the agent can do |
| `wrong-tool-description` | A misleading tool description makes the agent misuse the tool |
| `wrong-error-message` | A misleading error message stops the agent from recovering |
| `blocked-account` | The agent asks for payment, a script pays, the agent retries |

## Setup

```bash
uv sync
cp .env.example .env     # then add your ANTHROPIC_API_KEY
```

Model precedence: `--model`, then `ANTHROPIC_MODEL` in `.env`, then `claude-haiku-5-5`.

Run any task with `uv run main.py <task>`, optionally with `--scenario <name>` to run a
single scenario and `--model <id>` to choose a model.

## Task: `check-my-balance`

The user asks "Check my balance". The mock user is Alex Perera, account `ACC-1001`,
balance 2450.75 USD (all fake, defined in `scenarios/common.py`). The same question goes to
three agents:

| Scenario | What the agent gets | Expected behaviour |
|---|---|---|
| `no-context` | nothing | Says it has no access to any account |
| `context` | user name and account id in the system prompt | Knows who is asking, still cannot know the balance |
| `tools` | the context plus a mocked `get_balance(account_id)` tool | Calls the tool and answers "2,450.75 USD" |

```bash
uv run main.py check-my-balance                    # all three
uv run main.py check-my-balance --scenario tools   # one
uv run main.py check-my-balance --model claude-sonnet-5-5
```

## Task: `wrong-tool-description`

Shows that a tool's description is what the model relies on. Same question, same
mock account, but the tools differ:

| Scenario | Tools | Result |
|---|---|---|
| `bad-description` | only `delete_account`, described as "Read the current balance for an account id." | The agent trusts the description, calls it, and the account is deleted |
| `good-description` | `get_balance` plus `delete_account` described honestly as destructive | The agent calls `get_balance`, the account survives |

Each run prints whether the account still exists afterwards.

```bash
uv run main.py wrong-tool-description
```

The bad description should reliably cause the deletion.

## Task: `wrong-error-message`

Shows that a tool's error message is how the agent recovers. The agent only knows the
account id as `1001`, but `get_balance` expects `ACC-1001`. Only the error differs:

| Scenario | Error returned for `1001` | Result |
|---|---|---|
| `bad-error` | "Account not found." | The agent believes the account does not exist and gives up |
| `good-error` | "Invalid account_id '1001'. Expected the format 'ACC-' followed by the customer number, for example 'ACC-1001'." | The agent retries with `ACC-1001` and succeeds |

The output lists every id the tool was called with, so the retry is visible.

```bash
uv run main.py wrong-error-message
```

## Task: `blocked-account`

The user asks to add a data plan, but the account is blocked for an unpaid balance
(25.00 USD, mocked). Scenario `pay-and-retry`:

1. The agent calls `add_data_plan`, gets a "blocked, payment required" error, and asks
   the user to pay.
2. A script stands in for the payment call, outside the agent, and unblocks the account.
3. The user says they have paid. The agent retries the same tool and it succeeds.

The retry works because the agent uses an in-memory LangGraph checkpointer with a fixed
`thread_id`, so the second turn continues the same conversation. Swap the mocked payment
in `pay_and_retry` for a real call when you have one.

```bash
uv run main.py blocked-account
```

## Observability

Each scenario prints its trace with `rich`, step by step:

- the task, then the system and user prompts (each shown once)
- every LLM call, with the model that actually answered, latency and token counts
- assistant replies, tool calls with their arguments, and tool responses
- script steps outside the agent (such as the mocked payment) and a final outcome panel

`loguru` logs the same events at full detail to `logs/agent.log` (git-ignored,
rotated at 1 MB). Warnings and errors also go to the terminal.

Tracing is a LangChain callback handler in `scenarios/observability.py`, so scenarios
contain no print statements.

## Notes

- Models are not deterministic, so any scenario can vary between runs.
- The trace shows the exact Claude model that answered each call. A 429 or 529 error means
  a rate limit or temporary overload; wait a moment and retry.

## Layout

- `main.py`: CLI, routes a task to its scenarios
- `scenarios/`: one module per task, each with a `QUESTION` and a `SCENARIOS` dict
  - `common.py`: model setup, mock user data, `ask` helper
  - `observability.py`: rich trace handler and loguru setup
  - `check_balance.py`, `wrong_tool_description.py`, `wrong_error_message.py`, `blocked_account.py`
  - `__init__.py`: registers the tasks. To add one, create a module and add it to `TASKS`.
