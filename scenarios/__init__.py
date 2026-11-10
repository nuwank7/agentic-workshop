"""Task name -> (question, scenarios that handle it)."""

from . import blocked_account, check_balance, wrong_error_message, wrong_tool_description
from .common import DEFAULT_MODEL

TASKS = {
    "check-my-balance": check_balance,
    "wrong-tool-description": wrong_tool_description,
    "wrong-error-message": wrong_error_message,
    "blocked-account": blocked_account,
}

__all__ = ["DEFAULT_MODEL", "TASKS"]
