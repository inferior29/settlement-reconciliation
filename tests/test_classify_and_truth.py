from types import SimpleNamespace

import pandas as pd

from recon.classify import FALLBACK, ClaudeClassifier, RuleClassifier, classify_all, evidence, validate
from recon.config import ROOT_CAUSES
from recon.generate_data import generate
from recon.matching import reconcile
from recon.report import evaluate


def test_engine_finds_exactly_the_seeded_and_derived_exceptions():
    """On 4,000 generated transactions the engine matches ground truth with no misses and no false alarms."""
    g, b, r, t = generate(n=4000, seed=21)
    _, exc = reconcile(g, b, r)
    ev = evaluate(classify_all(RuleClassifier(), exc), t)
    assert (ev["missed"], ev["false_positives"]) == (0, 0)
    assert ev["detected"] == len(t) > 100


def test_rule_baseline_decisions():
    c = RuleClassifier().classify
    assert c({"exception_type": "duplicate_settlement"})["root_cause"] == "duplicate_posting"
    assert c({"exception_type": "short_settled", "shortfall": 300.0, "refund_amount": 300.0})["root_cause"] == "refund_deduction"
    assert c({"exception_type": "short_settled", "shortfall": 25.0, "refund_amount": 0.0})["root_cause"] == "fee_calculation_error"
    assert c({"exception_type": "unmatched_bank_entry"})["root_cause"] == "needs_review"


def test_evidence_contains_no_identifiers():
    g, b, r, _ = generate(n=1500, seed=2)
    _, exc = reconcile(g, b, r)
    ev = evidence(exc.iloc[0])
    assert set(ev) == {"exception_type", "amount", "expected", "settled_amount", "shortfall", "delay_business_days",
                       "holiday_in_delay_window", "refund_amount", "bank_entries", "age_days"}


class FakeClient:
    def __init__(self, content=None, error=None):
        self.calls, self._content, self._error = [], content, error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self._error:
            raise self._error
        return SimpleNamespace(content=self._content)


def block(data):
    return SimpleNamespace(type="tool_use", name="record_root_cause", input=data)


EV = {"exception_type": "short_settled", "shortfall": 300.0, "refund_amount": 300.0}


def test_claude_classifier_forces_tool_and_returns_structured_result():
    client = FakeClient([block({"root_cause": "refund_deduction", "next_action": "Confirm refund."})])
    out = ClaudeClassifier(client=client, model="m").classify(EV)
    assert out == {"root_cause": "refund_deduction", "next_action": "Confirm refund."}
    call = client.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "record_root_cause"}
    assert call["tools"][0]["input_schema"]["properties"]["root_cause"]["enum"] == ROOT_CAUSES


def test_claude_classifier_never_guesses_on_bad_output_or_errors():
    bad = FakeClient([block({"root_cause": "cosmic_rays", "next_action": "x"})])
    assert ClaudeClassifier(client=bad, model="m").classify(EV) == FALLBACK
    assert ClaudeClassifier(client=FakeClient(error=TimeoutError()), model="m").classify(EV) == FALLBACK
    assert validate({"root_cause": "duplicate_posting", "next_action": 5}) is None


def test_evaluate_counts_wrong_root_causes():
    g, b, r, t = generate(n=2000, seed=4)
    _, exc = reconcile(g, b, r)
    wrong = classify_all(RuleClassifier(), exc)
    wrong["predicted_root_cause"] = "duplicate_posting"            # deliberately bad classifier
    ev = evaluate(wrong, t)
    assert ev["root_cause_accuracy"] < 0.5
