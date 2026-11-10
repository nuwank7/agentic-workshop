"""Provider output failures must not become simulated customer behavior."""

import unittest
from unittest.mock import Mock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

import test_workshop as base_tests
from test_workshop import ScriptedModel, call
from workshop.response_recovery import ResponseRecovery


def truncated(**kwargs):
    return AIMessage(content=kwargs.pop("content", ""),
                     response_metadata={"stop_reason": "max_tokens"},
                     usage_metadata={"input_tokens": 2089, "output_tokens": 2048, "total_tokens": 4137,
                                     "output_token_details": {"reasoning": 2048}}, **kwargs)


def ask(message, phones, identity):
    return call("ask_customer", {"message": message, "phone_tools": phones}, identity)


class ResponseRecoveryTests(unittest.TestCase):
    simulate = base_tests.RunnerTests.simulate

    def test_screenshot_sequence_recovers_without_changing_patience_or_phone(self):
        agent = ScriptedModel(responses=[
            ask("Check status and network.", ["check_status_bar", "check_network_status"], "first"),
            ask("Check your status bar.", ["check_status_bar"], "single"),
            ask("Turn Airplane Mode off, then check status.", ["toggle_airplane_mode", "check_status_bar"], "third"),
            ask("Turn Airplane Mode off, then check status.", ["toggle_airplane_mode", "check_status_bar"], "fourth"),
        ])
        user = ScriptedModel(responses=[
            AIMessage(content="One step please."), call("check_status_bar", {}, "status"),
            AIMessage(content="My phone shows Airplane Mode on and battery at 80%."),
            truncated(), AIMessage(content="That's too much again. Please give me one step."),
        ])
        result, events = self.simulate(agent, user, "hard-user")
        self.assertEqual((result["outcome"], result["termination"]), ("fail", "user_left"))
        self.assertEqual([e["name"] for e in events if e["event"] == "tool"], ["check_status_bar"])
        self.assertEqual(sum(e["event"] == "model_reply_rejected" for e in events), 1)
        self.assertEqual(result["model_calls"], 9)  # Includes the rejected API response.
        self.assertFalse(any(e["event"] == "message" and not e["text"].strip() for e in events))

    def test_budget_grows_on_length_without_mutating_original_request(self):
        request = ModelRequest(model=ScriptedModel(responses=[]), tools=[],
                               messages=[HumanMessage(content="Please give me one step.")])
        good = ModelResponse(result=[AIMessage(content="One step please.")])
        handler = Mock(side_effect=[ModelResponse(result=[truncated()]), ModelResponse(result=[truncated()]), good])
        self.assertIs(ResponseRecovery(Mock(), "user").wrap_model_call(request, handler), good)
        self.assertEqual([c.args[0].model_settings.get("max_tokens", 2048) for c in handler.call_args_list],
                         [2048, 4096, 8192])
        self.assertEqual(request.model_settings, {})
        self.assertEqual(len(request.messages), 1)
        self.assertTrue(all(c.args[0].tools == [] for c in handler.call_args_list))

    def test_truncated_tool_call_is_not_executed_until_complete_response(self):
        agent = ScriptedModel(responses=[ask("Turn off Airplane Mode.", ["toggle_airplane_mode"], "req")])
        toggle = call("toggle_airplane_mode", {}, "real-toggle")
        user = ScriptedModel(responses=[truncated(tool_calls=toggle.tool_calls), toggle,
                                       AIMessage(content="My phone has Airplane Mode off. ###STOP###")])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["termination"], "user_stop")
        mutations = [e for e in events if e["event"] == "tool"]
        self.assertEqual(len(mutations), 1)
        rejection = next(e for e in events if e["event"] == "model_reply_rejected")
        self.assertLess(rejection["seq"], mutations[0]["seq"])

    def test_empty_reply_after_toggle_cannot_repeat_completed_action(self):
        agent = ScriptedModel(responses=[ask("Turn off Airplane Mode.", ["toggle_airplane_mode"], "req")])
        user = ScriptedModel(responses=[call("toggle_airplane_mode", {}, "first"), AIMessage(content="  "),
                                       call("toggle_airplane_mode", {}, "must-not-repeat"),
                                       AIMessage(content="My phone has Airplane Mode off. ###STOP###")])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["termination"], "user_stop")
        self.assertEqual(sum(e["event"] == "tool" for e in events), 1)
        self.assertEqual(sum(e["event"] == "model_reply_rejected" for e in events), 2)
        self.assertTrue(next(c["passed"] for c in result["checks"] if c["check"] == "assert_airplane_mode_status"))

    def test_persistent_empty_response_is_bounded_error_not_invented_reply(self):
        agent = ScriptedModel(responses=[ask("What does your phone show?", [], "req")])
        result, events = self.simulate(agent, ScriptedModel(responses=[AIMessage(content="")]))
        self.assertEqual(result["outcome"], "error")
        self.assertIn("recovery exhausted after 3 attempts", result["error"])
        self.assertEqual(sum(e["event"] == "model_reply_rejected" for e in events), 3)
        self.assertEqual(sum(e["event"] == "message" and e["actor"] == "user" for e in events), 1)

    def test_valid_tool_call_with_empty_text_is_not_a_missing_reply(self):
        response = ModelResponse(result=[call("check_status_bar", {}, "status")])
        handler = Mock(return_value=response)
        request = ModelRequest(model=ScriptedModel(responses=[]), messages=[], tools=[])
        self.assertIs(ResponseRecovery(Mock(), "user").wrap_model_call(request, handler), response)
        handler.assert_called_once()

    def test_partial_visible_text_is_not_published_or_executed(self):
        agent = ScriptedModel(responses=[truncated(content="I am John", tool_calls=[
            {"name": "ask_customer", "args": {"message": "Incomplete", "phone_tools": []}, "id": "partial"}]),
            ask("John, what does your phone show?", [], "complete")])
        result, events = self.simulate(agent, ScriptedModel(responses=[AIMessage(content="I'll leave. ###STOP###")]))
        self.assertEqual(result["termination"], "user_stop")
        self.assertEqual([e["text"] for e in events if e["event"] == "message" and e["actor"] == "agent"],
                         ["Hello! How can I help you today?", "John, what does your phone show?"])


if __name__ == "__main__":
    unittest.main()
