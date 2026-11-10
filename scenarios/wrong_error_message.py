"""The tool needs "ACC-1001" but the agent only knows "1001": the error decides recovery."""

from langchain.agents import create_agent
from langchain_core.tools import tool

from .common import MOCK_BALANCES, USER, ask, make_model

QUESTION = "Check my balance"

NUMBER_CONTEXT = (
    "You are a banking assistant. The current user is "
    f"{USER['name']}, account id 1001, already authenticated. Use your tools to answer."
)


def _error_message_scenario(question: str, model: str, not_found: str) -> str:
    attempts: list[str] = []

    @tool
    def get_balance(account_id: str) -> str:
        """Return the current balance for an account id."""
        attempts.append(account_id)
        balance = MOCK_BALANCES.get(account_id)
        return f"{balance:.2f} USD" if balance is not None else not_found.format(account_id)

    agent = create_agent(
        make_model(model), tools=[get_balance], system_prompt=NUMBER_CONTEXT
    )
    ask(agent, question)
    return f"get_balance called with: {attempts}"


def bad_error(question: str, model: str) -> str:
    """The failure says the account does not exist, which is misleading."""
    return _error_message_scenario(question, model, "Account not found.")


def good_error(question: str, model: str) -> str:
    """The failure explains what was wrong and how to fix it."""
    return _error_message_scenario(
        question,
        model,
        "Invalid account_id '{}'. Expected the format 'ACC-' followed by the "
        "customer number, for example 'ACC-1001'.",
    )


SCENARIOS = {
    "bad-error": bad_error,
    "good-error": good_error,
}
