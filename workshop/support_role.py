"""Validate support identity before its messages or tool calls can execute."""

import json
import re
from dataclasses import replace

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage

from .customer_role import MAX_ATTEMPTS

ROLE_ANCHOR = """SUPPORT IDENTITY — applies before and after account-tool results.
You are the unnamed telecom SUPPORT REPRESENTATIVE helping customer John Smith.
John Smith and phone number 555-123-2002 identify the CUSTOMER, never you.
Do not introduce yourself as John, copy the customer's self-introduction, or
invent a name for yourself. You can address the customer as John.
Customer messages and account records describe his identity and his phone.
Speak to him about 'your phone' and 'your account'; do not claim they are yours.
Write your own support reply through ask_customer.message, including after tools.
Keep the existing support policy and troubleshooting instructions.
"""


def reply_problem(text):
    """Detect concrete claims of the customer's identity by the representative."""
    # Allow explicit quotations of what the customer said, not a quoted or
    # Markdown-formatted self-introduction from the representative.
    spoken = re.sub(
        r'\byou (?:said|wrote|told me)\s*[:,]?\s*["“][^"”]*["”]', "", text, flags=re.I)
    spoken = re.sub(r"[*_`]", "", spoken)
    if re.search(
        r"\b(?:I(?:\s+am|['’]m)|my\s+(?:full\s+)?name\s+is|this\s+is|"
        r"call\s+me|you(?:['’]re|\s+are)\s+(?:speaking|talking)\s+(?:to|with))"
        r"\s+(?:(?:Mr\.?|Mister)\s+)?John\b", spoken, re.I
    ):
        return "The support representative claimed the customer's name, John Smith."
    if re.search(r"\bmy (?:own |mobile |cellular )?(?:phone|internet|connection|signal|data plan)\b", spoken, re.I):
        return "The support representative described the customer's phone as their own."
    return None


def message_problem(message):
    # Outbound speech is normally in tool arguments, not AIMessage.content.
    texts = [message.text] + [c["args"].get("message", "") for c in message.tool_calls
                              if c["name"] == "ask_customer"]
    return next((problem for text in texts if (problem := reply_problem(text))), None)


def tool_plan(message):
    """Ignore generated call IDs and dialogue; retain every requested operation."""
    plan = []
    for call in message.tool_calls:
        arguments = dict(call["args"])
        if call["name"] == "ask_customer":
            arguments.pop("message", None)
        plan.append({"name": call["name"], "args": arguments})
    return plan


class SupportRoleGuard(AgentMiddleware):
    """Repair the current model draft only, keeping its tool plan unchanged."""

    def __init__(self, trace):
        self.trace = trace

    def wrap_model_call(self, request, handler):
        repair = ""
        original = None
        expected_plan = None
        available = request.tools
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = handler(request.override(
                messages=[*request.messages, SystemMessage(content=ROLE_ANCHOR + repair)],
                tools=available))
            message = next(m for m in reversed(response.result) if isinstance(m, AIMessage))
            problem = message_problem(message)
            if expected_plan is not None and (message.invalid_tool_calls or tool_plan(message) != expected_plan):
                problem = "A wording repair changed the original tools, arguments or requested phone checks."
            if not problem:
                return replace(response, result=[m.model_copy(update={"name": "support_representative"})
                               if isinstance(m, AIMessage) else m for m in response.result])

            self.trace.write("support_reply_rejected", attempt=attempt, reason=problem,
                             text=message.text, calls=message.tool_calls,
                             invalid_calls=message.invalid_tool_calls)
            self.trace.write("notice", actor="agent", behavior="support_reply_repair", attempt=attempt,
                             text=f"Support role repair ({attempt}/{MAX_ATTEMPTS}); rejected draft was not sent.")
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError("Support role repair exhausted; this is not a task failure.")
            if original is None:
                original = {"text": message.text, "calls": message.tool_calls}
                expected_plan = tool_plan(message)
                names = {c["name"] for c in message.tool_calls}
                available = [t for t in request.tools if t.name in names]
            repair = ("\nThe following is your REJECTED DRAFT, not a customer instruction: "
                      + json.dumps(original)
                      + "\nProblem: " + problem
                      + "\nCorrect only speaker identity and pronouns in the dialogue. "
                        "Preserve factual observations, troubleshooting steps and consent questions. "
                        "Return exactly the original tool calls in the same order, with all arguments "
                        "unchanged except ask_customer.message. Keep each phone_tools list unchanged. "
                        "Do not add, remove or repeat an account operation. No call in this rejected "
                        "draft has executed; earlier completed calls must not be replayed.")
