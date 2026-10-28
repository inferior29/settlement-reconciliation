"""Root-cause classification of exceptions: a rule baseline and a Claude backend.

Both return {"root_cause": one of ROOT_CAUSES, "next_action": str}.
Claude receives only the evidence columns below: synthetic merchant ids and amounts, no personal data.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from .config import AMOUNT_TOLERANCE, NEXT_ACTION, ROOT_CAUSES

EVIDENCE_COLUMNS = ["exception_type", "amount", "expected", "settled_amount", "shortfall", "delay_business_days",
                    "holiday_in_delay_window", "refund_amount", "bank_entries", "age_days"]


def evidence(row: pd.Series) -> dict[str, Any]:
    """The facts about one exception that are shown to a classifier (plain Python types, JSON-safe)."""
    out: dict[str, Any] = {}
    for c in EVIDENCE_COLUMNS:
        v = row[c]
        if not isinstance(v, str) and pd.isna(v):
            out[c] = None
        elif c == "holiday_in_delay_window":
            out[c] = bool(v)
        else:
            out[c] = v.item() if hasattr(v, "item") else v
    return out


class RuleClassifier:
    """Transparent baseline: the exception type plus two arithmetic checks decide the cause."""

    name = "rules"

    def classify(self, ev: dict[str, Any]) -> dict[str, str]:
        t = ev["exception_type"]
        if t == "duplicate_settlement":
            cause = "duplicate_posting"
        elif t == "reference_mismatch":
            cause = "reference_id_mismatch"
        elif t == "missing_settlement":
            cause = "settlement_file_missing"
        elif t == "short_settled":
            refund, short = ev.get("refund_amount") or 0, ev.get("shortfall") or 0
            cause = "refund_deduction" if refund > 0 and abs(short - refund) <= AMOUNT_TOLERANCE else "fee_calculation_error"
        elif t == "delayed_settlement":
            cause = "bank_holiday_delay"
        else:  # unmatched_bank_entry or anything unexpected
            cause = "needs_review"
        return {"root_cause": cause, "next_action": NEXT_ACTION.get(cause, "Route to a reconciliation analyst for manual review.")}


SYSTEM_PROMPT = (
    "You are a payments reconciliation analyst. For one exception between a payment gateway and a bank "
    "settlement file, choose the single most likely root cause from the allowed list and write a one-sentence "
    "next action for the operations team. Base the answer only on the evidence provided: "
    "shortfall equal to a refund_amount means the refund was netted off; a shortfall that is a small percentage of "
    "the amount with no matching refund suggests a wrong fee rate; holiday_in_delay_window true explains a delay; "
    "bank_entries of 2 means a duplicate posting. Always answer by calling the record_root_cause tool."
)

ROOT_CAUSE_TOOL = {
    "name": "record_root_cause",
    "description": "Record the most likely root cause and next action for one reconciliation exception.",
    "input_schema": {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string", "enum": ROOT_CAUSES},
            "next_action": {"type": "string", "maxLength": 240},
        },
        "required": ["root_cause", "next_action"],
    },
}

FALLBACK = {"root_cause": "needs_review", "next_action": "Model output invalid or unavailable - route to an analyst."}


def validate(result: Any) -> dict[str, str] | None:
    if not isinstance(result, dict) or result.get("root_cause") not in ROOT_CAUSES:
        return None
    if not isinstance(result.get("next_action"), str):
        return None
    return {"root_cause": result["root_cause"], "next_action": result["next_action"]}


class ClaudeClassifier:
    """Structured JSON via a forced tool call. Reads ANTHROPIC_API_KEY from the environment."""

    name = "claude"

    def __init__(self, client: Any = None, model: str | None = None):
        if client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set ANTHROPIC_API_KEY to use the Claude backend.")
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("RECON_MODEL", "claude-haiku-5-5")

    def classify(self, ev: dict[str, Any]) -> dict[str, str]:
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=300, system=SYSTEM_PROMPT, tools=[ROOT_CAUSE_TOOL],
                tool_choice={"type": "tool", "name": "record_root_cause"},
                messages=[{"role": "user", "content": json.dumps(ev, default=str)}],
            )
        except Exception:
            return dict(FALLBACK)
        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "record_root_cause":
                return validate(block.input) or dict(FALLBACK)
        return dict(FALLBACK)


def classify_all(backend: Any, exceptions: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    evs = [evidence(r) for _, r in exceptions.iterrows()]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(backend.classify, evs))
    else:
        results = [backend.classify(e) for e in evs]
    out = exceptions.copy()
    out["predicted_root_cause"] = [r["root_cause"] for r in results]
    out["next_action"] = [r["next_action"] for r in results]
    return out
