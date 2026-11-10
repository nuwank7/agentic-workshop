"""Support messages must not claim customer identity or replay account writes."""

import unittest
from unittest.mock import Mock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import test_workshop as base_tests
from test_workshop import ScriptedModel, call
from workshop.runner import customer_request_tool
from workshop.support_role import SupportRoleGuard, reply_problem


SCREENSHOT_REPLY = (
    "I'm John Smith, and I see you have line L1002 (555-123-2002) on the Premium Plan "
    "with 15GB monthly limit. You've used 8.7GB so far, with 6.3GB remaining until "
    "your cycle ends on 2025-02-28. To help restore excellent mobile internet speed, "
    "please run the following checks on your phone and report the results:"
)


def ask(text, identity="request", phones=None):
    return call("ask_customer", {"message": text, "phone_tools": phones or []}, identity)


class SupportRoleTests(unittest.TestCase):
    simulate = base_tests.RunnerTests.simulate

    def test_customer_identity_claims_are_detected(self):
        for text in (SCREENSHOT_REPLY, "I’m **John Smith**, your support agent.",
                     '"My name is John Smith."', "This is John from support.",
                     "You're speaking with John Smith.", "My mobile internet isn't working."):
            with self.subTest(text=text):
                self.assertIsNotNone(reply_problem(text))

    def test_addressing_customer_and_explicit_quotes_are_allowed(self):
        for text in ("Hello John! I can help with your phone.", "Your account name is John Smith.",
                     "John, please check your phone.", 'You said, "I am John Smith". Is that correct?',
                     "My records show your phone is active."):
            with self.subTest(text=text):
                self.assertIsNone(reply_problem(text))

    def test_identity_is_reasserted_after_account_results(self):
        messages = [HumanMessage(content="I'm John Smith."),
                    call("get_customer_by_phone", {"phone_number": "555-123-2002"}, "lookup"),
                    ToolMessage(content='{"full_name":"John Smith"}',
                                tool_call_id="lookup", name="get_customer_by_phone")]
        model = ScriptedModel(responses=[ask("Hello John, how can I help?")])
        handler = Mock(return_value=ModelResponse(result=[model.responses[0]]))
        request = ModelRequest(model=model, messages=messages, tools=[])
        response = SupportRoleGuard(Mock()).wrap_model_call(request, handler)
        actual = handler.call_args.args[0]
        self.assertIsInstance(actual.messages[-1], SystemMessage)
        self.assertIn("identify the CUSTOMER, never you", actual.messages[-1].text)
        self.assertEqual(request.messages, messages)
        self.assertEqual(response.result[-1].name, "support_representative")

    def test_screenshot_message_in_tool_arguments_is_repaired_before_delivery(self):
        phones = ["check_network_status", "check_data_restriction_status", "check_apn_settings"]
        corrected = SCREENSHOT_REPLY.replace("I'm John Smith, and I see", "John, I see")
        agent = ScriptedModel(responses=[ask(SCREENSHOT_REPLY, "bad", phones), ask(corrected, "fixed", phones)])
        result, events = self.simulate(agent, ScriptedModel(responses=[AIMessage(content="I'll leave. ###STOP###")]))
        self.assertEqual(result["termination"], "user_stop")
        self.assertEqual(sum(e["event"] == "support_reply_rejected" for e in events), 1)
        self.assertFalse(any(e["event"] == "message" and "I'm John Smith, and I see" in e["text"] for e in events))
        self.assertEqual([e["phone_tools"] for e in events if e["event"] == "request"], [phones])
        self.assertEqual([e["text"] for e in events if e["event"] == "message" and e["actor"] == "agent"][-1], corrected)

    def test_wording_repair_cannot_repeat_completed_refuel(self):
        refuel = call("refuel_data", {"customer_id": "C1001", "line_id": "L1002", "gb_amount": 2}, "refuel")
        bad = ask("I'm John Smith. Please check your network.", "bad", ["check_network_status"])
        fixed = ask("John, please check your network.", "fixed", ["check_network_status"])
        repeated = AIMessage(content="", tool_calls=[*refuel.tool_calls, *fixed.tool_calls])
        agent = ScriptedModel(responses=[refuel, bad, repeated, fixed])
        result, events = self.simulate(agent, ScriptedModel(responses=[AIMessage(content="I'll leave. ###STOP###")]), "hard-env")
        self.assertEqual(result["termination"], "user_stop")
        self.assertEqual(sum(e["event"] == "tool" and e["name"] == "refuel_data" for e in events), 1)
        self.assertEqual(sum(e["event"] == "support_reply_rejected" for e in events), 2)
        self.assertTrue(next(c["passed"] for c in result["checks"] if c["check"] == "assert_data_refueling_amount"))

    def test_every_message_in_a_batch_is_checked_and_phone_requests_are_preserved(self):
        first = ask("Please check network status.", "a", ["check_network_status"])
        bad_second = ask("I'm John Smith. Please check your status bar.", "b", ["check_status_bar"])
        fixed_second = ask("John, please check your status bar.", "fixed-b", ["check_status_bar"])
        changed_second = ask("John, please change Airplane Mode.", "changed-b", ["toggle_airplane_mode"])
        agent = ScriptedModel(responses=[AIMessage(content="", tool_calls=[*first.tool_calls, *second.tool_calls])
                                        for second in (bad_second, changed_second, fixed_second)])
        result, events = self.simulate(agent, ScriptedModel(responses=[AIMessage(content="I'll leave. ###STOP###")]))
        self.assertEqual(result["termination"], "user_stop")
        self.assertEqual(sum(e["event"] == "support_reply_rejected" for e in events), 2)
        self.assertEqual([e["phone_tools"] for e in events if e["event"] == "request"],
                         [["check_network_status", "check_status_bar"]])

    def test_persistent_identity_error_blocks_entire_batch_before_account_mutation(self):
        refuel = call("refuel_data", {"customer_id": "C1001", "line_id": "L1002", "gb_amount": 2}, "refuel")
        bad = AIMessage(content="", tool_calls=[*refuel.tool_calls, *ask("I'm John Smith.").tool_calls])
        result, events = self.simulate(ScriptedModel(responses=[bad]),
                                      ScriptedModel(responses=[AIMessage(content="Hello")]), "hard-env")
        self.assertEqual(result["outcome"], "error")
        self.assertEqual(sum(e["event"] == "support_reply_rejected" for e in events), 3)
        self.assertFalse(any(e["event"] in ("tool", "request") for e in events))
        self.assertEqual(sum(e["event"] == "message" and e["actor"] == "agent" for e in events), 1)

    def test_customer_request_boundary_also_rejects_identity_confusion(self):
        pending = {"phone_tools": [], "message": None}
        with self.assertRaisesRegex(RuntimeError, "claimed the customer's name"):
            customer_request_tool(pending).invoke({"message": SCREENSHOT_REPLY, "phone_tools": []})
        self.assertEqual(pending, {"phone_tools": [], "message": None})


if __name__ == "__main__":
    unittest.main()
