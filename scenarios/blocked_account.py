"""The agent asks for payment, a script pays, the agent retries."""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from .common import USER_CONTEXT, ask, make_model
from .observability import note

QUESTION = "Add the basic data plan to my account"

AMOUNT_OWED = 25.00


def pay_and_retry(question: str, model: str) -> str:
    account = {"blocked": True, "plans": []}

    @tool
    def add_data_plan(account_id: str, plan: str = "basic-5gb") -> str:
        """Add a data plan to an account. The default plan is 'basic-5gb'."""
        if account["blocked"]:
            return (
                f"Account {account_id} is blocked because of an unpaid balance of "
                f"{AMOUNT_OWED:.2f} USD. The customer must pay it before a plan can be added."
            )
        account["plans"].append(plan)
        return f"Plan {plan} added to {account_id}"

    agent = create_agent(
        make_model(model),
        tools=[add_data_plan],
        system_prompt=USER_CONTEXT,
        checkpointer=InMemorySaver(),  # keeps the conversation so the retry has context
    )
    config = {"configurable": {"thread_id": "blocked-account"}}

    ask(agent, question, config)

    # Stand-in for the real payment call the user would trigger outside the agent.
    account["blocked"] = False
    note(f"payment of {AMOUNT_OWED:.2f} USD received, account unblocked")

    ask(agent, "I have paid. Please try again.", config)
    return f"plans on account: {account['plans']}"


SCENARIOS = {"pay-and-retry": pay_and_retry}
