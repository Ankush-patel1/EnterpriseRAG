import re
import time
from collections import defaultdict

# Patterns that indicate personally identifiable information
PII_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',  # email address
    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',                    # phone number
    r'\b\d{3}-\d{2}-\d{4}\b',                                 # SSN
    r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b',      # credit card number
]

# Common phrases used in prompt injection attacks
INJECTION_TRIGGERS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard the above",
    "you are now",
    "pretend you are",
    "act as if",
    "forget everything",
    "new instructions:",
    "system prompt",
    "jailbreak",
    "override",
]

DAILY_TOKEN_BUDGET = 50_000  # approximate tokens per user per day

# In-memory usage tracker: { ip_address: { "tokens": int, "reset_at": timestamp } }
# This resets on server restart; fine for a capstone demo
_usage = defaultdict(lambda: {"tokens": 0, "reset_at": 0})


def contains_pii(text):
    """Returns True if the text matches any PII pattern."""
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def is_injection_attempt(text):
    """Returns True if the text contains known prompt injection phrases."""
    lowered = text.lower()
    return any(trigger in lowered for trigger in INJECTION_TRIGGERS)


def _estimate_tokens(text):
    """Rough token count estimate: words * 1.3 (GPT-style tokenisation approximation)."""
    return int(len(text.split()) * 1.3)


def check_token_budget(user_ip, question):
    """
    Checks if the user has exceeded their daily token budget.
    Returns True if they are within budget (request allowed), False if over.
    """
    now = time.time()
    usage = _usage[user_ip]

    # Reset counter at the start of a new day
    if now > usage["reset_at"]:
        usage["tokens"] = 0
        usage["reset_at"] = now + 86400  # 24 hours from now

    tokens_needed = _estimate_tokens(question)

    if usage["tokens"] + tokens_needed > DAILY_TOKEN_BUDGET:
        return False

    usage["tokens"] += tokens_needed
    return True


def validate_question(question, user_ip):
    """
    Runs all safety checks on an incoming question.
    Returns (allowed: bool, error_message: str).
    """
    if contains_pii(question):
        return False, "Your question appears to contain personal information. Please remove it and try again."

    if is_injection_attempt(question):
        return False, "Your question was flagged as a potential prompt injection attempt."

    if not check_token_budget(user_ip, question):
        return False, "Daily usage limit reached. Please try again tomorrow."

    return True, ""
