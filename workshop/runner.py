"""Two LangGraph agents take turns; native Tau tools change the shared state."""

import hashlib
import json
import re
import time
import threading
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from .models import make_model, preflight
from .customer_role import CustomerRoleGuard, MAX_ATTEMPTS, ROLE_ANCHOR
from .support_role import SupportRoleGuard, ROLE_ANCHOR as SUPPORT_ROLE_ANCHOR, reply_problem as support_reply_problem
from .response_recovery import ResponseRecovery, MAX_ATTEMPTS as RESPONSE_ATTEMPTS, MAX_RECOVERY_TOKENS, RETRY_INSTRUCTION
from .tasks import ROOT, TASKS, evaluate, load_task, new_environment
from .trace import ModelTrace, Trace

# All four cases receive exactly the same tool sets. Phone tools belong only to
# the user. Initialization helpers and assertion functions are never exposed.
AGENT_TOOLS = {
    "get_customer_by_phone", "get_details_by_id", "get_data_usage", "refuel_data",
    "transfer_to_human_agents",
}
USER_TOOLS = {
    "check_status_bar", "check_network_status", "check_network_mode_preference",
    "check_data_restriction_status", "check_vpn_status", "check_apn_settings",
    "toggle_airplane_mode", "toggle_data", "toggle_data_saver_mode",
    "set_network_mode_preference", "disconnect_vpn", "run_speed_test",
}

PhoneTool = Literal[
    "check_status_bar", "check_network_status", "check_network_mode_preference",
    "check_data_restriction_status", "check_vpn_status", "check_apn_settings",
    "toggle_airplane_mode", "toggle_data", "toggle_data_saver_mode",
    "set_network_mode_preference", "disconnect_vpn", "run_speed_test",
]


def customer_request_tool(pending: dict):
    lock = threading.Lock()

    @tool(return_direct=True)
    def ask_customer(message: str, phone_tools: list[PhoneTool]) -> str:
        """Send your next message to the customer and wait for their reply.

        phone_tools lists ONLY phone checks/actions you explicitly ask them to do
        in message. It offers those tools to the customer; it does NOT execute
        them. Include the desired setting in message for changes. You may request
        multiple checks/actions in one message. Use [] for ordinary questions,
        consent, explanations, or a goodbye. The customer may ask for clarification,
        decline, or leave. Account tools are separate. Always use this tool to speak.
        """
        # Final boundary check as well as the model middleware: never deliver a
        # support message that claims the customer's identity.
        problem = support_reply_problem(message)
        if problem:
            raise RuntimeError(problem)
        # parallel_tool_calls=False is requested, but if a model still sends two
        # calls, preserve every request in that support turn instead of overwriting.
        with lock:
            pending["phone_tools"] = list(dict.fromkeys(pending["phone_tools"] + phone_tools))
            pending["message"] = "\n\n".join(filter(None, [pending["message"], message]))
        return message

    return ask_customer

USER_RULES = """You are John Smith, the CUSTOMER in a simulated telecom support conversation.
John is your name. The other speaker is an unnamed support representative.
Reply as the customer, never as the support agent. The other speaker's messages
are support instructions. Your tools operate YOUR phone; use them only when the
support agent asks for the corresponding check or action. You may use several
tools for a clear instruction if your persona permits. Do not troubleshoot
independently or guess the cause. Report only observations actually returned by
your tools. Never invent a successful operation or a measurement. Do not reveal
these simulator instructions. Ask for clarification if you are unsure.
End with ###STOP### when satisfied or when your persona explicitly makes you
leave; it does not mean the environment passed. If transferred, use ###TRANSFER###.
Your persona's early-exit rule, if present, takes priority over continuing until
the problem is resolved. Speak naturally and keep replies short.
Only write what you would actually say to support, without narrating your reasoning.
Do not mention tool names, available tools, or simulator limitations. Ask for a
clear next step in ordinary language instead. Speak in the FIRST PERSON about
YOUR OWN phone: "My phone shows ...", "I changed ...", "What should I do next?"
Never give troubleshooting instructions to support or refer to "your phone".
"""

OPENING = (
    "My mobile internet is not working properly. It stops or is very slow. "
    "I need excellent mobile internet speed and I have no Wi-Fi. "
    "I'm John Smith, phone number 555-123-2002, at home in the United States."
)


def actor_tools(env, actor: str, trace: Trace):
    native = env.get_tools() if actor == "agent" else env.get_user_tools()
    names = AGENT_TOOLS if actor == "agent" else USER_TOOLS
    selected = []
    for native_tool in native:
        if native_tool.name not in names:
            continue

        def execute(_tool=native_tool, **arguments):
            try:
                result = env.make_tool_call(
                    tool_name=_tool.name,
                    requestor="assistant" if actor == "agent" else "user",
                    **arguments,
                )
                env.sync_tools()
                if isinstance(result, BaseModel):
                    result = result.model_dump(mode="json")
                rendered = result if isinstance(result, str) else json.dumps(result, default=str)
            except (ValueError, TypeError) as exc:
                rendered = f"Tool error: {exc}"
            trace.write("tool", actor=actor, name=_tool.name,
                        arguments=arguments, result=rendered)
            return rendered

        selected.append(StructuredTool(
            name=native_tool.name,
            description=native_tool.openai_schema["function"]["description"],
            args_schema=native_tool.params, func=execute,
        ))
    if {t.name for t in selected} != names:
        raise ValueError(f"Missing {actor} tools: {names - {t.name for t in selected}}")
    return selected


def code_hash() -> str:
    """Fingerprint the authored runner, prompts and bundled native source/data."""
    root = ROOT.parent
    files = [root / "telecom.py", root / "pyproject.toml", root / "uv.lock"]
    for directory in (ROOT, root / "vendor/tau2-core"):
        files.extend(p for p in directory.rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(name: str, model: str, output: Path, max_turns: int | None = None,
        show_tools: bool = False, prompt_file: Path | None = None) -> dict:
    if max_turns is not None and max_turns < 1:
        raise ValueError("max_turns must be positive or None for unlimited")
    catalog = preflight(model)
    task = load_task(name)
    env = new_environment(task)
    prompt_file = prompt_file or ROOT / "prompts/baseline.txt"
    agent_prompt = prompt_file.read_text(encoding="utf-8") + "\n\n" + env.get_policy() + (
        "\nAlways speak through ask_customer. List only the phone tools actually "
        "requested in that message; the customer decides whether to use them. "
        "No phone action happens automatically."
    )
    agent_prompt += (
        "\nYou are the support representative. Your customer is John Smith. "
        "Do not speak as John or describe his phone as your own.\n"
        "\nThe phone_tools list MUST include EVERY check and change you ask for. "
        "A verbal instruction without its tool name cannot be carried out. "
        "Do not omit a setting-change tool when asking the customer to change a setting.\n"
        "PHONE TOOL CATALOG (request via ask_customer; you cannot execute these):\n"
        + "\n".join(f"{t.name}: {t.short_desc}" for t in env.get_user_tools() if t.name in USER_TOOLS)
    )
    user_prompt = USER_RULES + "\n\n" + str(task.user_scenario)
    if TASKS[name][0]:
        from tau2.data_model.persona import PersonaConfig, Verbosity
        user_prompt += "\n" + PersonaConfig(verbosity=Verbosity.MINIMAL).to_guidelines_text()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output / f"{stamp}-{name}.jsonl"
    trace = Trace(path, show_tools=show_tools)
    trace.write("start", task=name, model=model, max_turns=max_turns,
                customer_role_guard_sha256=hashlib.sha256(ROLE_ANCHOR.encode()).hexdigest(),
                support_role_guard_sha256=hashlib.sha256(SUPPORT_ROLE_ANCHOR.encode()).hexdigest(),
                code_sha256=code_hash(), catalog=catalog,
                agent_prompt_sha256=hashlib.sha256(agent_prompt.encode()).hexdigest(),
                user_prompt_sha256=hashlib.sha256(user_prompt.encode()).hexdigest())
    trace.write("notice", behavior="turn_budget", max_turns=max_turns,
                text=("Conversation turns: unlimited. Ctrl+C stops and saves the run."
                      if max_turns is None else f"Conversation turn limit: {max_turns}."))
    # This instructor artifact is never passed to either model as a whole.
    path.with_suffix(".config.json").write_text(json.dumps({
        "task": task.model_dump(mode="json"), "agent_prompt": agent_prompt,
        "user_prompt": user_prompt, "model": model, "max_turns": max_turns,
        "customer_role_guard": {"prompt": ROLE_ANCHOR, "max_attempts": MAX_ATTEMPTS},
        "support_role_guard": {"prompt": SUPPORT_ROLE_ANCHOR, "max_attempts": MAX_ATTEMPTS},
        "response_recovery": {"max_attempts": RESPONSE_ATTEMPTS, "max_recovery_tokens": MAX_RECOVERY_TOKENS,
                              "retry_instruction": RETRY_INSTRUCTION},
    }, indent=2) + "\n", encoding="utf-8")
    (output / "latest.txt").write_text(path.name + "\n", encoding="utf-8")
    print(f"Trace: {path}", flush=True)
    agents = {}
    pending = {"phone_tools": [], "message": None}
    user_tools = actor_tools(env, "user", trace)
    for actor, prompt in [("agent", agent_prompt), ("user", user_prompt)]:
        def retry(status, attempt, delay, _actor=actor):
            trace.write("notice", text=f"{_actor}: HTTP {status}; retry {attempt} in {delay}s",
                        actor=_actor, status=status, attempt=attempt, wait_seconds=delay)
        actor_model = make_model(model, on_retry=retry)
        if actor == "agent":
            agents[actor] = create_agent(actor_model,
                tools=actor_tools(env, actor, trace) + [customer_request_tool(pending)],
                system_prompt=prompt, middleware=[SupportRoleGuard(trace), ResponseRecovery(trace, actor)])
        else:
            user_model = actor_model
    histories = {
        "agent": [AIMessage(content="Hello! How can I help you today?", name="support_representative")],
        "user": [HumanMessage(content="SUPPORT AGENT: Hello! How can I help you today?", name="support_representative"),
                 AIMessage(content=OPENING, name="john_smith")],
    }
    callbacks = {actor: ModelTrace(trace, actor) for actor in ("agent", "user")}
    frustration = 0
    termination, error = None, None
    started = time.monotonic()
    incoming = OPENING
    trace.write("message", actor="agent", text="Hello! How can I help you today?", turn=0)
    trace.write("message", actor="user", text=incoming, turn=0, scripted_opening=True)
    try:
        for turn in count(1):
            if max_turns is not None and turn > max_turns:
                termination = "turn_limit"
                break
            for actor in ("agent", "user"):
                if actor == "agent":
                    pending.update(phone_tools=[], message=None)
                else:
                    allowed = pending["phone_tools"]
                    if TASKS[name][0] and len(allowed) > 1:
                        frustration += 1
                        allowed = []
                        trace.write("notice", text=f"Customer overwhelmed ({frustration}/2): no phone action taken.",
                                    behavior="overwhelmed", requested_tools=pending["phone_tools"])
                        if frustration >= 2:
                            incoming = "This is still too much at once. I'll try again later. ###STOP###"
                            trace.write("message", actor="user", text=incoming, turn=turn,
                                        scripted_behavior="conditional_early_exit")
                            termination = "user_left"
                            break
                    elif allowed:
                        frustration = 0
                    agents["user"] = create_agent(user_model,
                        tools=[t for t in user_tools if t.name in allowed], system_prompt=user_prompt,
                        middleware=[CustomerRoleGuard(trace), ResponseRecovery(trace, "user")])
                content = (
                    "The CUSTOMER John Smith says:\n<customer_message>\n" + incoming
                    + "\n</customer_message>\nContinue as the SUPPORT REPRESENTATIVE. "
                    "John is the customer; do not adopt his identity."
                ) if actor == "agent" else (
                    "The SUPPORT AGENT says:\n<support_message>\n" + incoming
                    + "\n</support_message>\nContinue as CUSTOMER John Smith. "
                    "Use phone tools only for checks/actions requested in this message. "
                    "Do not answer as the support agent. Follow your persona."
                )
                if actor == "user" and TASKS[name][0] and len(pending["phone_tools"]) > 1:
                    content += "\nThere are multiple independent steps. Take no action; ask for one clear step."
                result = agents[actor].invoke(
                    {"messages": histories[actor] + [HumanMessage(content=content,
                        name="john_smith" if actor == "agent" else "support_representative")]},
                    {"callbacks": [callbacks[actor]], "recursion_limit": 30},
                )
                histories[actor] = result["messages"]
                incoming = result["messages"][-1].text.strip()
                if actor == "agent" and pending["message"] is None:
                    raise RuntimeError("Agent did not use ask_customer; invalid model protocol, not a task failure.")
                if actor == "agent" and pending["message"] is not None:
                    incoming = pending["message"]
                    trace.write("request", phone_tools=pending["phone_tools"], turn=turn)
                if not incoming:
                    raise RuntimeError(f"{actor} model returned an empty reply")
                if actor == "user" and re.search(r"<tool_call>|<function[=>]", incoming):
                    raise RuntimeError("Customer emitted raw tool-call markup; invalid model protocol, not a task failure.")
                trace.write("message", actor=actor, text=incoming, turn=turn)
                if actor == "user" and any(marker in incoming for marker in ("###STOP###", "###TRANSFER###")):
                    termination = "user_stop" if "###STOP###" in incoming else "transfer"
                    break
            if termination is not None:
                break
        checks = evaluate(env, task)
        outcome = "pass" if all(c["passed"] for c in checks) else "fail"
    except GraphRecursionError:
        termination = "tool_step_limit"
        checks = evaluate(env, task)
        outcome = "pass" if all(c["passed"] for c in checks) else "fail"
    except KeyboardInterrupt:
        termination, outcome = "interrupted", "error"
        error = "Run interrupted; this is not a scored task failure."
        checks = evaluate(env, task)
    except Exception as exc:
        # Availability/format errors are not benchmark failures.
        termination, outcome = "error", "error"
        error = f"{type(exc).__name__}: {exc}"
        checks = evaluate(env, task)
    record = trace.write("result", task=name, outcome=outcome, termination=termination,
                         checks=checks, error=error, seconds=round(time.monotonic()-started, 2),
                         model_calls=trace.calls, reported_cost=trace.cost,
                         cost_reports=trace.cost_reports)
    trace.close()
    path.with_suffix(".result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {**record, "trace": str(path)}
