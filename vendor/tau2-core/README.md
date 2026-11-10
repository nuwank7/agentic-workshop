# Bundled τ² telecom core

Source: [Sierra τ²-bench](https://github.com/sierra-research/tau2-bench), commit
`363133ada1936491fb5bcec33cd62c3518a99f65`, MIT license (included).

This local Python dependency contains the native telecom environment, tools,
database models, assertions, and their import dependencies. Copied modules are
unchanged. Minimal package initializers replace upstream initializers that eagerly
import the full benchmark/voice stack. It does not provide the upstream Tau CLI.

`PROVENANCE.json` records the upstream SHA-256 hashes. Matching telecom data lives
in `workshop/data/`; `source_tasks.json` contains the two unmodified source tasks
used to author the four workshop variants. `workshop/tasks.py` documents the
persona swaps and additional native assertions. Model prompts receive neither
initialization actions nor reference actions nor grader checks.

The workshop uses a custom LangGraph conversation runner, an explicit
`ask_customer` request boundary, a relevant subset of tools, a fixed opening, and
a conditional user-departure rule. Its results are teaching examples, not scores
from the official Tau orchestrator or leaderboard.
