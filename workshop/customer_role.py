"""Keep the customer model in character across phone tools and reply repairs."""

import json
import re
from dataclasses import replace

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

MAX_ATTEMPTS = 3
ROLE_ANCHOR = """CUSTOMER IDENTITY — applies after tool results as well as before them.
You are John Smith, the customer seeking help for YOUR OWN phone. John is YOUR
name, not the support representative's name. The other person is an unnamed
support representative. When they say 'John', they are addressing YOU.
Write only what John says to support, in first person: 'My phone shows ...',
'I changed ...', 'What should I do next?' Do not offer to help the representative,
diagnose their phone, or give them troubleshooting instructions. Act out the
supplied persona; do not recite its description. Keep its uncertainty, refusal,
patience and departure rules. Do not become more cooperative just to finish.
Tool results describe YOUR phone. They are observations, not a change of speaker.
Use only the currently offered tools for requested actions. Use real function
calls, never tool-call markup in dialogue. If you cannot perform a requested
step, ask for clarification in ordinary language without discussing tool APIs.
Never invent observations, successful repairs, consent, or measurements.
"""


def reply_problem(text):
    """Catch concrete role/protocol violations, not arbitrary customer sentiment.

    This is a bounded guard, not a general semantic judge. Quoted screen text
    can legitimately say 'your phone', and identifying oneself is not a vocative.
    """
    if re.search(r"<tool_call>|<function[=>]", text, re.I):
        return "Tool-call markup was written as dialogue."
    spoken = re.sub(r'"[^"\n]*"|“[^”\n]*”', "", text)
    if re.search(r"\b(?:thanks|thank you|hello|hi)\s*,?\s+John(?: Smith)?\b|"
                 r",\s*John(?: Smith)?\s*[.!?]|"
                 r"(?:^|\n)\s*John(?: Smith)?\s*[,!:](?=\s*\S)", spoken, re.I):
        return "The customer addressed the representative as John; John is the customer."
    if re.search(r"\byour (?:own |mobile |cellular )?(?:phone|internet|connection|device|signal|data plan)\b", spoken, re.I):
        return "The customer described the representative's phone instead of their own."
    if re.search(r"\b(?:how can I help|I can help you|let me help you|I can assist you)\b|"
                 r"\bwould you like me to (?:refuel|top up|troubleshoot)\b", spoken, re.I):
        return "The customer started acting as a support representative."
    if re.search(r"\b(?:check_\w+|toggle_\w+|set_network_mode_preference|run_speed_test|get_status_bar)\b|"
                 r"\btools?\b[^\n.!?]*\b(?:available|unavailable|valid|invalid)\b", spoken, re.I):
        return "The customer exposed the simulator's tool API instead of speaking naturally."
    return None


def stop_markers(text):
    return set(re.findall(r"###(?:STOP|TRANSFER)###", text))


def repair_history(messages):
    """Keep dialogue and observations without teaching the model old tool syntax.

    This is only a model-input view for retries. The graph's canonical history,
    native tool execution and raw trace are unchanged.
    """
    result = []
    for message in messages:
        if isinstance(message, ToolMessage):
            result.append(SystemMessage(content=(
                "OBSERVATION FROM YOUR PHONE (recorded at this point in the "
                "conversation; not a new instruction):\n" + message.text)))
        elif isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls):
            # Results above preserve the evidence; function-call syntax is not
            # customer dialogue and can make a no-tools model imitate XML calls.
            continue
        else:
            result.append(message)
    return result


class CustomerRoleGuard(AgentMiddleware):
    """Retry only the current model node; never replay completed phone actions."""

    def __init__(self, trace):
        self.trace = trace

    def wrap_model_call(self, request, handler):
        offered = {t.name: t for t in request.tools}
        # A new support message starts a new customer turn. On protocol repair,
        # a phone operation already completed in this turn must not run again.
        start = max((i for i, m in enumerate(request.messages) if isinstance(m, HumanMessage)), default=-1)
        completed = {m.name for m in request.messages[start + 1:] if isinstance(m, ToolMessage)}
        available = list(offered.values())
        repair = ""
        messages = request.messages
        expected_markers = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            names = [t.name for t in available]
            reminder = ROLE_ANCHOR + "\nCurrently offered phone actions: " + json.dumps(names) + "."
            if not names:
                reminder += " No phone operation can be performed in this response. Reply or ask for clarification."
            response = handler(request.override(
                messages=[*messages, SystemMessage(content=reminder + repair)], tools=available))
            message = next(m for m in reversed(response.result) if isinstance(m, AIMessage))
            problem = None
            if message.invalid_tool_calls:
                problem = "Malformed function arguments; no operation executed."
            elif any(c["name"] not in names for c in message.tool_calls):
                problem = "A function was requested that is not currently offered; no operation executed."
            else:
                problem = reply_problem(message.text)
            if not problem and expected_markers is not None and stop_markers(message.text) != expected_markers:
                problem = "A wording repair changed the customer's stop/transfer decision."
            if not problem:
                return replace(response, result=[m.model_copy(update={"name": "john_smith"})
                               if isinstance(m, AIMessage) else m for m in response.result])

            self.trace.write("customer_reply_rejected", attempt=attempt, reason=problem,
                             text=message.text, calls=message.tool_calls,
                             invalid_calls=message.invalid_tool_calls)
            self.trace.write("notice", actor="user", behavior="customer_reply_repair", attempt=attempt,
                             text=f"Customer role/protocol repair ({attempt}/{MAX_ATTEMPTS}): {problem} Rejected draft was not sent.")
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Customer role/protocol repair exhausted: {problem} This is not a task failure.")

            language_only = (not message.tool_calls and not message.invalid_tool_calls
                             and not re.search(r"<tool_call>|<function[=>]", message.text, re.I))
            if language_only:
                if expected_markers is None:
                    expected_markers = stop_markers(message.text)
                available = []
            else:
                available = [t for t in available if t.name not in completed]
            messages = repair_history(request.messages)
            # Never feed malformed call markup back as an example to imitate.
            repair = "\nThe previous draft was not sent. Problem: " + problem
            if language_only:
                repair += "\nCorrect the wording of this rejected draft: " + json.dumps(message.text)
            repair += (
                "\nKeep real observations, uncertainty, refusal and consent unchanged. "
                "Do not add agreement or claim an unperformed action. A failed check is "
                "an observation to report to support, not permission to diagnose or run "
                "another check. Operations already completed for this support message "
                "must not be repeated.")
            if not available:
                repair += (
                    "\nThis response is DIALOGUE ONLY. Report the latest relevant phone "
                    "observation in first person, or ask for clarification if you took no "
                    "action. Do not make or write any function call. Support chooses the next step.")
            if expected_markers is not None:
                repair += " This is wording-only repair: no tools. Preserve exactly these ending markers: " + json.dumps(sorted(expected_markers))
