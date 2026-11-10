"""Same question, three agents: no context, user context, user context plus a tool."""

from langchain.agents import create_agent
from langchain_core.tools import tool

from .common import MOCK_BALANCES, USER_CONTEXT, ask, make_model

QUESTION = "Check my balance"


@tool
def get_balance(account_id: str) -> str:
    """Return the current balance for an account id."""
    balance = MOCK_BALANCES.get(account_id)
    if balance is None:
        return f"No account found for {account_id}"
    return f"{balance:.2f} USD"


def no_context(question: str, model: str) -> None:
    ask(create_agent(make_model(model)), question)


def with_context(question: str, model: str) -> None:
    agent = create_agent(make_model(model), system_prompt=USER_CONTEXT)
    ask(agent, question)


def with_tools(question: str, model: str) -> None:
    agent = create_agent(
        make_model(model), tools=[get_balance], system_prompt=USER_CONTEXT
    )
    ask(agent, question)


SCENARIOS = {
    "no-context": no_context,
    "context": with_context,
    "tools": with_tools,
}
