"""Shared model setup, mock user data and the ask helper."""

import os

from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from .observability import trace

DEFAULT_MODEL = "claude-haiku-5-5"

USER = {"name": "Alex Perera", "account_id": "ACC-1001", "account_type": "savings"}
USER_CONTEXT = (
    "You are a banking assistant. The current user is "
    f"{USER['name']} (account {USER['account_id']}, {USER['account_type']})."
)

MOCK_BALANCES = {"ACC-1001": 2450.75}


def make_model(name: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )


def ask(agent: CompiledStateGraph, question: str, config: RunnableConfig | None = None) -> str:
    """Run one turn. The trace handler prints and logs every step."""
    config = {**(config or {}), "callbacks": [trace]}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    answer = result["messages"][-1].text
    logger.info("final answer: {}", answer)
    return answer
