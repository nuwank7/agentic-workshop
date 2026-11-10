"""Offline checks for scenario integrity, tool ownership, grading and traces."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from workshop.models import as_claude_messages, make_model
from workshop.runner import actor_tools, customer_request_tool, run
from workshop.tasks import TASKS, evaluate, load_task, new_environment, validate_tasks
from workshop.trace import Trace


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, identity):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identity}])


class TaskTests(unittest.TestCase):
    def test_all_four_start_unsolved_and_are_solvable(self):
        self.assertEqual(len(validate_tasks()), 4)

    def test_persona_changes_do_not_change_environment_or_criteria(self):
        for left, right in [("easy", "hard-user"), ("hard-env", "hard-both")]:
            a, b = load_task(left), load_task(right)
            self.assertEqual(a.initial_state, b.initial_state)
            self.assertEqual(a.evaluation_criteria, b.evaluation_criteria)
            self.assertEqual(a.user_scenario.instructions, b.user_scenario.instructions)
            self.assertNotEqual(a.user_scenario.persona, b.user_scenario.persona)

    def test_hard_environment_extends_easy_and_keeps_user_goal(self):
        easy, hard = load_task("easy"), load_task("hard-env")
        self.assertEqual(easy.user_scenario, hard.user_scenario)
        for init in easy.initial_state.initialization_actions:
            self.assertIn(init, hard.initial_state.initialization_actions)
        for check in easy.evaluation_criteria.env_assertions:
            self.assertIn(check, hard.evaluation_criteria.env_assertions)

    def test_new_attempt_has_fresh_state(self):
        task = load_task("easy")
        a = new_environment(task)
        a.make_tool_call("toggle_airplane_mode", requestor="user")
        b = new_environment(task)
        self.assertFalse(a.user_tools.device.airplane_mode)
        self.assertTrue(b.user_tools.device.airplane_mode)

    def test_first_fix_alone_does_not_pass(self):
        task = load_task("easy")
        env = new_environment(task)
        env.make_tool_call("toggle_airplane_mode", requestor="user")
        checks = {c["check"]: c["passed"] for c in evaluate(env, task)}
        self.assertTrue(checks["assert_airplane_mode_status"])
        self.assertFalse(checks["assert_internet_speed"])

    def test_easy_repairs_do_not_solve_hard_environment(self):
        hard = load_task("hard-env")
        env = new_environment(hard)
        for action in load_task("easy").evaluation_criteria.actions:
            env.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
            env.sync_tools()
        self.assertFalse(all(c["passed"] for c in evaluate(env, hard)))

    def test_tools_have_correct_owners_and_exclude_hidden_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Trace(Path(tmp) / "trace.jsonl")
            for name in TASKS:
                env = new_environment(load_task(name))
                agent = {t.name for t in actor_tools(env, "agent", trace)}
                user = {t.name for t in actor_tools(env, "user", trace)}
                self.assertFalse(agent & user)
                self.assertNotIn("toggle_airplane_mode", agent)
                self.assertIn("refuel_data", agent)
                self.assertIn("toggle_airplane_mode", user)
                self.assertFalse(any(n.startswith("assert_") for n in agent | user))
            trace.close()


class ClaudeModelTests(unittest.TestCase):
    def test_non_claude_ids_rejected(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy"}):
            for name in ["openai/gpt-4o", "openrouter/free", "nvidia/nemotron:free"]:
                with self.assertRaises(ValueError):
                    make_model(name)
            make_model("claude-haiku-5-5")

    def test_missing_key_does_not_fall_back_to_another_provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy", "OPENROUTER_API_KEY": "dummy"}, clear=True):
            with self.assertRaises(ValueError):
                make_model("claude-haiku-5-5")

    def test_ask_customer_forces_tool_use_without_parallel_calls(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy"}):
            model = make_model("claude-haiku-5-5")
        bound = model.bind_tools([customer_request_tool({"phone_tools": [], "message": None})])
        self.assertEqual(bound.kwargs["tool_choice"], {"type": "any", "disable_parallel_tool_use": True})

    def test_role_reminders_become_user_turns(self):
        messages = as_claude_messages([
            SystemMessage(content="policy"), AIMessage(content="Hello!"),
            HumanMessage(content="My internet is slow."),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ToolMessage(content="ok", tool_call_id="1"), SystemMessage(content="stay in role"),
        ])
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)  # conversation opens with a user turn
        self.assertEqual(sum(isinstance(m, SystemMessage) for m in messages), 1)
        self.assertIn("<system-reminder>", messages[-1].text)


class RunnerTests(unittest.TestCase):
    def test_multiple_requests_in_one_turn_are_not_lost(self):
        pending = {"phone_tools": [], "message": None}
        request = customer_request_tool(pending)
        request.invoke({"message": "Check the status bar.", "phone_tools": ["check_status_bar"]})
        request.invoke({"message": "Then test the speed.", "phone_tools": ["run_speed_test"]})
        self.assertEqual(pending["phone_tools"], ["check_status_bar", "run_speed_test"])
        self.assertIn("Check the status bar.", pending["message"])
        self.assertIn("Then test the speed.", pending["message"])

    def simulate(self, agent, user, name="easy", max_turns=4):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with patch("workshop.runner.preflight", return_value={"test": True}), patch(
            "workshop.runner.make_model", side_effect=[agent, user]
        ), contextlib.redirect_stdout(io.StringIO()):
            result = run(name, "claude-haiku-5-5", Path(temp.name), max_turns=max_turns)
        events = [json.loads(line) for line in Path(result["trace"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[-1]["event"], "result")
        return result, events

    def test_real_tool_mutations_produce_a_pass(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Turn airplane mode off, select 4G/5G, and test the speed.",
            "phone_tools": ["toggle_airplane_mode", "set_network_mode_preference", "run_speed_test"],
        }, "request")])
        user = ScriptedModel(responses=[
            call("toggle_airplane_mode", {}, "a"),
            call("set_network_mode_preference", {"mode": "4g_5g_preferred"}, "b"),
            call("run_speed_test", {}, "c"),
            AIMessage(content="Excellent speed. ###STOP###"),
        ])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["outcome"], "pass")
        self.assertEqual(sum(e["event"] == "tool" for e in events), 3)

    def test_default_cli_continues_beyond_twelve_turns_until_customer_stops(self):
        import telecom
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Do you need clarification?", "phone_tools": []}, f"question-{i}")
            for i in range(14)])
        user = ScriptedModel(responses=[AIMessage(content=f"I'm still unsure, question {i}.")
                                       for i in range(13)] + [AIMessage(content="I'll leave. ###STOP###")])
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["telecom.py", "hard-user", "--output", tmp]), \
             patch("workshop.runner.preflight", return_value={"test": True}), \
             patch("workshop.runner.make_model", side_effect=[agent, user]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(telecom.main(), 0)
            path = next(Path(tmp).glob("*.jsonl"))
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            config = json.loads(path.with_suffix(".config.json").read_text(encoding="utf-8"))
        self.assertIsNone(config["max_turns"])
        self.assertEqual(events[-1]["termination"], "user_stop")
        self.assertEqual(max(e.get("turn", 0) for e in events), 14)

    def test_explicit_turn_cap_remains_available(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Do you need clarification?", "phone_tools": []}, f"question-{i}")
            for i in range(3)])
        user = ScriptedModel(responses=[AIMessage(content=f"Please explain again, question {i}.")
                                       for i in range(3)])
        result, events = self.simulate(agent, user, max_turns=2)
        self.assertEqual(result["termination"], "turn_limit")
        self.assertEqual(max(e.get("turn", 0) for e in events), 2)
        self.assertEqual(result["model_calls"], 4)

    def test_user_claiming_success_cannot_fake_pass(self):
        result, _ = self.simulate(ScriptedModel(responses=[call("ask_customer", {"message": "Is it fixed?", "phone_tools": []}, "req")]),
                                 ScriptedModel(responses=[AIMessage(content="Everything works. ###STOP###")]))
        self.assertEqual(result["outcome"], "fail")
        self.assertEqual(result["termination"], "user_stop")

    def test_empty_model_reply_is_error_not_task_failure(self):
        result, _ = self.simulate(ScriptedModel(responses=[call("ask_customer", {"message": "Is it fixed?", "phone_tools": []}, "req")]),
                                 ScriptedModel(responses=[AIMessage(content="")]))
        self.assertEqual(result["outcome"], "error")

    def test_two_overwhelming_requests_cause_conditional_departure(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Check the status bar and network status.",
            "phone_tools": ["check_status_bar", "check_network_status"],
        }, identity) for identity in ("first", "second")])
        user = ScriptedModel(responses=[AIMessage(content="One step please.")])
        result, events = self.simulate(agent, user, "hard-user")
        self.assertEqual(result["termination"], "user_left")
        self.assertEqual(result["outcome"], "fail")
        self.assertFalse(any(e["event"] == "tool" for e in events))

    def test_customer_cannot_operate_unrequested_phone_tools(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Check the status bar.", "phone_tools": ["check_status_bar"],
        }, "req")])
        user = ScriptedModel(responses=[call("toggle_airplane_mode", {}, "forbidden"),
                                       AIMessage(content="I'll leave. ###STOP###")])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["outcome"], "fail")
        self.assertFalse(any(e["event"] == "tool" and e["name"] == "toggle_airplane_mode"
                             for e in events))

    def test_malformed_agent_protocol_is_not_a_task_failure(self):
        result, _ = self.simulate(ScriptedModel(responses=[AIMessage(content="Raw tool-shaped text")]),
                                 ScriptedModel(responses=[AIMessage(content="Hello")]))
        self.assertEqual(result["outcome"], "error")

    def test_malformed_customer_protocol_is_not_a_task_failure(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Check the status bar.", "phone_tools": ["check_status_bar"],
        }, "req")])
        user = ScriptedModel(responses=[AIMessage(content="<tool_call><function=check_status_bar>{}")])
        result, events = self.simulate(agent, user)
        self.assertEqual(result["outcome"], "error")
        self.assertFalse(any(e["event"] == "tool" for e in events))

    def test_tool_step_limit_still_grades_the_actual_state(self):
        agent = ScriptedModel(responses=[call("ask_customer", {
            "message": "Turn airplane mode off, select 4G/5G, and test speed.",
            "phone_tools": ["toggle_airplane_mode", "set_network_mode_preference", "run_speed_test"],
        }, "req")])
        user = ScriptedModel(responses=[
            call("toggle_airplane_mode", {}, "airplane"),
            call("set_network_mode_preference", {"mode": "4g_5g_preferred"}, "network"),
        ] + [call("run_speed_test", {}, f"speed-{i}") for i in range(40)])
        result, _ = self.simulate(agent, user)
        self.assertEqual(result["termination"], "tool_step_limit")
        self.assertEqual(result["outcome"], "pass")


if __name__ == "__main__":
    unittest.main()
