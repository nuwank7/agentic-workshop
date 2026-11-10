"""Reproduce role drift without rewriting persona decisions or replaying tools."""

import unittest
from unittest.mock import Mock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import test_workshop as base_tests
from test_workshop import ScriptedModel, call
from workshop.customer_role import CustomerRoleGuard, reply_problem


class CustomerRoleTests(unittest.TestCase):
    simulate = base_tests.RunnerTests.simulate

    def test_screenshot_role_errors_are_detected(self):
        for text in (
            "Thank you for asking me to run those checks, John. Here's what I found.",
            "That explains why your internet isn't working. What should I do?",
            "John, I can help you restore your phone's internet.",
            "Would you like me to refuel 2GB?",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(reply_problem(text))

    def test_identity_consent_uncertainty_and_quoted_screen_text_remain_valid(self):
        for text in (
            "John Smith.", "My name is John Smith, and my phone is offline.",
            "Please check my account. Can you help me turn it off?",
            'My screen says "Your phone is offline". What should I do?',
            "Yes, please add 2GB for $4 to my line.",
            "No, I don't approve the charge. One step please. ###STOP###",
        ):
            with self.subTest(text=text):
                self.assertIsNone(reply_problem(text))

    def test_identity_is_reasserted_after_tool_results_without_changing_history(self):
        history = [HumanMessage(content="John, check your phone."),
                   call("check_network_status", {}, "check"),
                   ToolMessage(content="Airplane Mode: ON", name="check_network_status", tool_call_id="check")]
        model = ScriptedModel(responses=[AIMessage(content="My phone shows Airplane Mode ON.")])
        request = ModelRequest(model=model, messages=history, tools=[])
        handler = Mock(return_value=ModelResponse(result=[model.responses[0]]))
        response = CustomerRoleGuard(Mock()).wrap_model_call(request, handler)
        actual = handler.call_args.args[0]
        self.assertIsInstance(actual.messages[-1], SystemMessage)
        self.assertIn("John is YOUR", actual.messages[-1].text)
        self.assertIn("Currently offered phone actions: []", actual.messages[-1].text)
        self.assertEqual(request.messages, history)
        self.assertEqual(response.result[-1].name, "john_smith")

    def test_bad_final_reply_is_repaired_without_repeating_a_completed_toggle(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Turn Airplane Mode off and report the result.",
            "phone_tools": ["toggle_airplane_mode"]}, "req")])
        user = ScriptedModel(responses=[
            call("toggle_airplane_mode", {}, "toggle"),
            AIMessage(content="Your phone has Airplane Mode off. ###STOP###"),
            call("toggle_airplane_mode", {}, "must-not-repeat"),
            AIMessage(content="My phone has Airplane Mode off. ###STOP###"),
        ])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["termination"], "user_stop")
        self.assertEqual(sum(e["event"] == "tool" and e["name"] == "toggle_airplane_mode" for e in events), 1)
        self.assertTrue(next(c["passed"] for c in result["checks"] if c["check"] == "assert_airplane_mode_status"))
        self.assertEqual(sum(e["event"] == "customer_reply_rejected" for e in events), 2)
        self.assertFalse(any(e["event"] == "message" and "Your phone has" in e["text"] for e in events))

    def test_raw_markup_can_be_reissued_as_a_real_call_before_any_action(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Check network status.", "phone_tools": ["check_network_status"]}, "req")])
        user = ScriptedModel(responses=[
            AIMessage(content="<tool_call><function=check_network_status>{}"),
            call("check_network_status", {}, "real-call"),
            AIMessage(content="My phone has no service. I'll leave now. ###STOP###"),
        ])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["termination"], "user_stop")
        self.assertEqual(sum(e["event"] == "tool" for e in events), 1)

    def test_overwhelmed_customer_repair_cannot_execute_phone_actions(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Check status and turn off Airplane Mode.",
            "phone_tools": ["check_network_status", "toggle_airplane_mode"]}, identity)
            for identity in ("first", "second")])
        user = ScriptedModel(responses=[
            AIMessage(content="<tool_call><function=toggle_airplane_mode>{}"),
            AIMessage(content="Too much at once. Please give me one step."),
        ])
        result, events = self.simulate(agent, user, "hard-user")
        self.assertEqual(result["termination"], "user_left")
        self.assertFalse(any(e["event"] == "tool" for e in events))

    def test_wording_repair_preserves_refusal_and_stop_marker(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Would you approve 2GB for $4?", "phone_tools": []}, "req")])
        user = ScriptedModel(responses=[
            AIMessage(content="No, John. I won't approve the charge. ###STOP###"),
            AIMessage(content="No, I won't approve the charge."),
            AIMessage(content="No, I won't approve the charge. ###STOP###"),
        ])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["termination"], "user_stop")
        accepted = [e for e in events if e["event"] == "message" and e["actor"] == "user"][-1]
        self.assertEqual(accepted["text"], "No, I won't approve the charge. ###STOP###")
        self.assertEqual(sum(e["event"] == "customer_reply_rejected" for e in events), 2)

    def test_persistent_role_errors_stop_as_error_instead_of_being_published(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "What does your phone show?", "phone_tools": []}, "req")])
        result, events = self.simulate(agent, ScriptedModel(responses=[AIMessage(content="I can help you fix your phone.")]))
        self.assertEqual(result["outcome"], "error")
        self.assertEqual(sum(e["event"] == "customer_reply_rejected" for e in events), 3)
        self.assertEqual(sum(e["event"] == "message" and e["actor"] == "user" for e in events), 1)  # opening only

    def test_failed_speed_test_is_reported_without_unrequested_network_check(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Please run a speed test and report the result.",
            "phone_tools": ["run_speed_test"]}, "req")])
        user = ScriptedModel(responses=[
            call("run_speed_test", {}, "speed"),
            call("check_network_status", {}, "unrequested"),
            AIMessage(content="<tool_call>\n<function=check_network_status>\n</function>\n</tool_call>"),
            AIMessage(content="My speed test failed: No Connection. I'll leave now. ###STOP###"),
        ])
        result, events = self.simulate(agent, user, "hard-env")
        self.assertEqual((result["outcome"], result["termination"]), ("fail", "user_stop"))
        actions = [e for e in events if e["event"] == "tool"]
        self.assertEqual([e["name"] for e in actions], ["run_speed_test"])
        self.assertEqual(actions[0]["result"], "Speed test failed: No Connection.")
        self.assertEqual(sum(e["event"] == "customer_reply_rejected" for e in events), 2)

    def test_protocol_repairs_keep_evidence_but_remove_tool_call_syntax(self):
        history = [HumanMessage(content="Check network status."),
                   call("check_network_status", {}, "network"),
                   ToolMessage(content="Airplane Mode: OFF", name="check_network_status", tool_call_id="network"),
                   AIMessage(content="My phone has Airplane Mode off."),
                   HumanMessage(content="Run a speed test."), call("run_speed_test", {}, "speed"),
                   ToolMessage(content="Speed test failed: No Connection.", name="run_speed_test", tool_call_id="speed")]
        phone_tool = Mock()
        phone_tool.name = "run_speed_test"
        request = ModelRequest(model=ScriptedModel(responses=[]), messages=history, tools=[phone_tool])
        handler = Mock(side_effect=[
            ModelResponse(result=[call("check_network_status", {}, "unrequested")]),
            ModelResponse(result=[AIMessage(content="<tool_call><function=check_network_status></function></tool_call>")]),
            ModelResponse(result=[AIMessage(content="My speed test failed with No Connection.")]),
        ])
        CustomerRoleGuard(Mock()).wrap_model_call(request, handler)
        for invocation in handler.call_args_list[1:]:
            retry = invocation.args[0]
            self.assertEqual(retry.tools, [])
            self.assertFalse(any(isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
                                 for m in retry.messages))
            context = "\n".join(m.text for m in retry.messages)
            self.assertIn("Airplane Mode: OFF", context)
            self.assertIn("Speed test failed: No Connection.", context)
            self.assertIn("DIALOGUE ONLY", context)
            self.assertNotIn("<tool_call>", context)
            self.assertNotIn("<function=check_network_status>", context)
        self.assertIs(request.messages, history)
        self.assertIsInstance(history[-1], ToolMessage)


if __name__ == "__main__":
    unittest.main()
