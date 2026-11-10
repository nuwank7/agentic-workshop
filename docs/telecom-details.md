# Instructor notes: telecom adaptation

## Source and boundary

The native telecom environment, tools, policies, assertions, Easy/Hard personas
and source tasks come from [Sierra τ²-bench](https://github.com/sierra-research/tau2-bench),
commit `363133ada1936491fb5bcec33cd62c3518a99f65`.

The two source task IDs are:

- `[mobile_data_issue]airplane_mode_on|bad_network_preference[PERSONA:Hard]`
- `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|data_usage_exceeded[PERSONA:Hard]`

Their unmodified records are in `workshop/data/source_tasks.json`. The workshop
swaps the persona independently of the environment, giving a 2×2 comparison with
the same goal. All four initially fail at least one check; upstream reference
actions satisfy every check. `telecom.py validate` verifies this offline.

This is a **custom LangGraph harness**, not the official Tau orchestrator or a
leaderboard evaluation. Package initializers were reduced to avoid installing
the unrelated benchmark/voice stack; native telecom modules are unchanged.
See `vendor/tau2-core/PROVENANCE.json` for hashes and the included MIT license.

The support model receives the same baseline prompt, native policy and tool
catalog in every condition. The customer receives its scenario and persona.
Neither sees initialization actions, reference solutions or grading assertions.
The support model has account tools; only the customer has phone tools.
`ask_customer` lists the phone tools requested in that support turn. The runner
offers those tools to the customer without executing them automatically.
This prevents a weak simulator from independently solving the task.

The initial greeting and customer request are scripted and identical across
conditions. Subsequent dialogue and tool choices come from the two model roles,
except the conditional departure described below. Histories are separate, while
the native telecom state is shared.

## Keeping both speakers in character

The incorrect replies in earlier traces came from the customer model itself,
not swapped display labels. After phone-tool results it sometimes adopted a
support voice, calling the representative John or discussing “your internet”.
Those drafts were previously accepted into its conversation history.
Support could also copy John's self-introduction into `ask_customer.message`.
The initial correction guarded only customer output; a support prompt reminder
alone did not prevent this second failure.

Both prompts now explicitly identify John Smith as the customer, with named
speaker messages. `workshop/customer_role.py` adds a customer identity reminder
to every model call, including calls after tool observations. It rejects clear
role mistakes and malformed or unoffered tool calls before they reach the
conversation or execute. It gives the same model at most three attempts.
Rejected drafts are retained as `customer_reply_rejected` trace events.

On a customer retry, the model receives a dialogue-and-observations view of its
history. Completed tool results remain intact as phone observations, but earlier
function-call messages are omitted from that input view. Malformed tool markup
is not repeated in the repair prompt. This prevents the repair context from
reinforcing old calls when no tools are currently allowed. For example, after a
speed test reports “No Connection”, the customer should report that result to
support; it cannot independently run an unrequested network check. The graph's
canonical history and raw trace retain the original tool calls and results.

Wording-only retries have no tools and must preserve stop/transfer markers.
Protocol retries cannot repeat a tool already completed in that customer turn.
The repair prompt preserves observations, refusal and uncertainty; it does not
provide a solution. The original persona, patience rule, state and grading
assertions remain unchanged. Exhausted retries are `ERROR`, never scored `FAIL`.
The guard catches concrete patterns, not every possible semantic role error;
native assertions also do not grade dialogue quality.

`workshop/support_role.py` provides the corresponding support identity reminder
and checks both model text and every `ask_customer.message` in a tool batch
before any of that batch executes. Support cannot introduce itself as John.
Wording retries must preserve the original tool names, arguments and phone-tool
lists; they cannot add a repeat refuel or change which checks the customer is
asked to perform. A final check inside `ask_customer` also blocks invalid identity
claims. Rejected drafts are stored as `support_reply_rejected`; three failed
attempts end as `ERROR`. Neither guard changes saved historical recordings.

## Empty or truncated model responses

In the reported hard-user run, the provider returned `finish_reason="length"`
after using all 2,048 output tokens on reasoning, with no visible reply. This
was a model-output error, not the customer's intentional silence or departure.

`workshop/response_recovery.py` runs inside both role guards. It retries an empty
or truncated response before tool execution, with at most three attempts per
model invocation. Token truncation raises the output limit from 2,048 to 4,096,
then 8,192. It uses the same model, persona, history and tool permissions;
completed tools are excluded on retry. An overwhelmed customer still has no
phone tools. Partial replies and partial tool calls are not delivered or run.
The recovered response still passes through the corresponding role guard.

Every real model call, including rejected attempts, remains in usage/cost counts.
Rejections are recorded as `model_reply_rejected` with the finish reason and
token limit. Exhausted recovery ends as `ERROR`; no customer reply is fabricated,
and no provider error is counted as an intentional task failure.

## Hard user

Tau's native Hard persona describes an anxious, inexperienced older customer.
It does **not** itself implement the text-conversation hang-up behavior used here.
We combine it with native minimal verbosity and an explicit workshop addition:

1. A request offering more than one distinct phone tool overwhelms the user.
   The user takes no action and is prompted to ask for one clear step.
2. A second such request, without a single-tool request between them, triggers
   the saved scripted departure. Ordinary conversation does not reset patience.
3. A single-tool request resets patience and allows the user to act normally.

The rule is based on the structured request, not an unreliable count of words
in the dialogue. Leaving is conditional on agent behavior. The hard task remains
solvable through clear instructions; success is not disabled in code.

## What the checks mean

Easy checks require working mobile data, speed at least 200 Mbps rated Excellent,
airplane mode off and service connected. Hard checks add Data Saver off, allowance
not exceeded and exactly 2 GB refueled on account C1001, line L1002. The speed
requirement also catches a bad VPN or slow network preference.

These are native state assertions. They do not independently grade every policy
rule, consent wording or conversation quality. A model claiming success never
overwrites the state. Conversation turns are unlimited by default (`max_turns`
is `null` in the saved configuration); `--max-turns N` enables an optional cap.
The separate limit of 30 graph steps within a speaker's turn still stops a tool
loop. Provider/protocol errors and interrupted runs are labeled ERROR, not task
failures. Removing the conversation cap does not change the persona's departure
rule or turn a stopped historical recording into a continued run.

## Recorded evidence

`examples/manifest.json` lists four selected recordings and all 17 development
attempts. Every raw trace and available prompt/task/result snapshot is retained
under `examples/attempts/`. Selected traces are exact copies, not edited scripts.

The four selected runs used the same model and identical agent prompt. They have
67 successful model replies, all reporting zero cost. Across all retained
attempts, 237 successful replies report zero cost. These are model-response cost
reports, not an independent billing audit.

Earlier development included free-provider capacity errors, interrupted runs,
malformed output, and a pilot where the customer self-solved before request gating.
These are excluded as classroom baseline evidence. One earlier combined trace
recorded FAIL despite malformed customer markup; its manifest explicitly excludes
it. New runs attempt bounded protocol repair and report unrecoverable output as
ERROR.

Run code hashes can differ because protocol-error detection, documentation and
trace-write ordering were corrected during development. Final maintenance also
ensures a tool-step limit still passes if every state check is already satisfied;
none of the selected grades changes. Actual prompt hashes and
task definitions are saved. Some simultaneous callbacks in the original hard-env
recording flushed out of order; recorded `seq` numbers preserve their order.
The verifier sorts by that sequence and reproduces **every tool result and final
check**. New writes are serialized. Raw historical files remain unchanged.

Free models may repeat instructions, confuse customer/support roles or emit
invalid tool output. The historical recordings preserve those imperfections;
the role guards apply to new runs only. They are
selected classroom cases, not controlled success-rate estimates, and a fresh
live attempt can behave differently. Use replays for a predictable demonstration.

### Runtime validation on 2 November 2026

`examples/runtime-validation/manifest.json` indexes four additional, unchanged
traces with prompt/task snapshots, results and SHA-256 hashes. They were recorded
while successive runtime fixes were being developed, not as a controlled batch:

| Task | Outcome | Checks true | Model replies | Termination |
|---|---|---:|---:|---|
| Easy | PASS | 4/4 | 19 | Explicit four-turn smoke-test cap; predates response recovery. |
| Hard user | PASS | 4/4 | 27 | Customer stop after nine turns; unlimited conversation. |
| Hard environment | FAIL | 6/7 | 39 | Tool-step limit; repeated refuels violate the exact 2 GB check. |
| Both hard | ERROR | 5/7 diagnostic | 56 | Recovery exhausted after thirteen completed turns; unlimited conversation. |

All 141 replies reported zero cost from the free NVIDIA model. Native tool replay
reproduced each recorded tool output and final state. In the ERROR run, the false
checks were excellent speed and exactly 2 GB refueled; this is not a scored FAIL.
Its support recovery rejected an unavailable/already completed tool request.
Bounded recovery therefore remains a best-effort guard, not a guarantee that a
free-model run completes. The hard-environment recording predates the final
small wording refinement to customer repair instructions. The original
`examples/manifest.json` and its selected classroom recordings are unchanged.

Normal `telecom.py` commands now display the conversation and tool details by
default; `--no-tools` hides those details. `showprogress.py` retains its original
behavior: add `--tools` when replaying or following a trace to see tool details.

## Later exercise

The failed variants intentionally have no improved agent. Keep the scenario,
checks and user fixed while changing `workshop/prompts/baseline.txt` via a copied
`--agent-prompt` file. Compare the conversation and tool trace, then inspect the
same final assertions. Useful discussion points are clear instructions, verifying
the actual speed, and preventing repeated account writes. Implement and test
those improvements with the instructor in the next session.

## File map

| File | Purpose |
|---|---|
| `telecom.py` | Run one condition, all four, or offline solvability validation. |
| `workshop/tasks.py` | Load native source tasks, swap personas and add native assertions. |
| `workshop/runner.py` | Alternate the two agents, constrain tool ownership and grade state. |
| `workshop/customer_role.py` | Reinforce customer identity and retry invalid drafts without replaying completed phone actions. |
| `workshop/support_role.py` | Check support identity in dialogue and tool arguments before delivery; preserve the tool plan during repair. |
| `workshop/response_recovery.py` | Retry empty/truncated model responses with bounded token growth before any tool execution. |
| `workshop/models.py` | Build the Claude (Anthropic API) models, adapt role reminders to Claude's message format, and retry transient HTTP errors. |
| `workshop/prompts/baseline.txt` | Unimproved starting support-agent prompt. |
| `workshop/prompts/hard_user_addition.txt` | Workshop patience rule layered onto the native persona. |
| `workshop/trace.py`, `showprogress.py` | Record, display, replay and follow terminal conversations. |
| `tests/test_workshop.py` | Offline scenario, protocol, tool-ownership and grading checks. |
| `tests/test_customer_role.py` | Role-drift regressions, safe reply repair and preserved early-exit behavior. |
| `tests/test_support_role.py` | Support identity regressions, unchanged phone requests and prevention of repeated account writes during repair. |
| `tests/test_response_recovery.py` | Reproduce the hard-user truncation and check retry bounds, tool safety and unchanged patience. |
| `scripts/verify_examples.py` | Independently replay saved native calls and check recorded outcomes. |
| `examples/` | Selected recordings, their snapshots and the complete attempt archive. |
| `vendor/tau2-core/` | Bundled native modules, license and source hashes. |
