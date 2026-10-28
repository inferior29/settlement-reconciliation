import pandas as pd

from recon.config import AS_OF
from recon.matching import reconcile

# AS_OF is Tue 2026-10-27. Standard holiday: 2026-10-02. Regional (bank-only) holiday: 2026-10-08.
GATEWAY = pd.DataFrame([
    ("GW1", "M1", "2026-10-19", 1000.00),   # settles normally
    ("GW2", "M2", "2026-10-19", 1000.00),   # never settles, due Wed 10-21 -> missing
    ("GW3", "M3", "2026-10-26", 1000.00),   # due 10-28, not yet due -> not an exception
    ("GW4", "M4", "2026-10-19", 2000.00),   # short by a 300 refund
    ("GW5", "M5", "2026-10-19", 500.00),    # posted twice by the bank
    ("GW6", "M6", "2026-10-19", 1000.00),   # bank carries a corrupted reference
    ("GW7", "M7", "2026-10-06", 1000.00),   # settles after the regional holiday -> delayed
], columns=["txn_id", "merchant_id", "txn_date", "amount"])

BANK = pd.DataFrame([
    ("B1", "GW1", "M1", 980.00, "2026-10-20"),
    ("B4", "GW4", "M4", 1660.00, "2026-10-20"),
    ("B5a", "GW5", "M5", 490.00, "2026-10-20"),
    ("B5b", "GW5", "M5", 490.00, "2026-10-22"),
    ("B6", "gw6", "M6", 980.00, "2026-10-20"),
    ("B7", "GW7", "M7", 980.00, "2026-10-09"),
    ("B8", "UNKNOWN", "M9", 777.00, "2026-10-20"),
], columns=["bank_ref", "gateway_ref", "merchant_id", "settled_amount", "settle_date"])

REFUNDS = pd.DataFrame([("GW4", "2026-10-21", 300.00)], columns=["txn_id", "refund_date", "refund_amount"])


def run():
    matches, exc = reconcile(GATEWAY, BANK, REFUNDS, as_of=AS_OF)
    return matches, exc.set_index(["txn_id", "exception_type"]) if len(exc) else exc


def test_exactly_the_expected_exceptions_are_flagged():
    _, exc = run()
    flagged = set(exc.index)
    assert flagged == {
        ("GW2", "missing_settlement"), ("GW4", "short_settled"), ("GW5", "duplicate_settlement"),
        ("GW6", "reference_mismatch"), ("GW7", "delayed_settlement"), ("UNKNOWN", "unmatched_bank_entry"),
    }


def test_clean_and_not_yet_due_transactions_raise_nothing():
    _, exc = run()
    ids = {i for i, _ in exc.index}
    assert "GW1" not in ids and "GW3" not in ids


def test_match_types():
    matches, _ = run()
    mt = matches.set_index("txn_id")["match_type"]
    assert mt["GW1"] == "exact" and mt["GW6"] == "fuzzy" and mt["GW2"] == "" and mt["GW3"] == ""


def test_short_settlement_carries_the_evidence_for_classification():
    _, exc = run()
    r = exc.loc[("GW4", "short_settled")]
    assert (r.expected, r.settled_amount, r.shortfall, r.refund_amount) == (1960.00, 1660.00, 300.00, 300.00)
    assert r.value_at_risk == 300.00


def test_missing_settlement_value_and_age():
    _, exc = run()
    r = exc.loc[("GW2", "missing_settlement")]
    assert r.value_at_risk == 980.00          # expected settlement is at risk
    assert r.age_days == 6 and r.age_bucket == "3-7 days"  # due 10-21, as-of 10-27


def test_duplicate_is_flagged_with_the_amount_to_recover():
    _, exc = run()
    r = exc.loc[("GW5", "duplicate_settlement")]
    assert r.value_at_risk == 490.00 and r.bank_entries == 2


def test_delay_is_explained_by_the_regional_holiday():
    _, exc = run()
    r = exc.loc[("GW7", "delayed_settlement")]
    assert r.delay_business_days == 1 and bool(r.holiday_in_delay_window) is True
    assert r.value_at_risk == 0.0             # timing only


def test_orphan_bank_credit_is_flagged_not_force_matched():
    _, exc = run()
    assert exc.loc[("UNKNOWN", "unmatched_bank_entry")].value_at_risk == 777.00


def test_fuzzy_match_respects_amount_tolerance_and_date_window():
    bank = BANK.copy()
    bank.loc[bank.bank_ref == "B6", "settled_amount"] = 900.00        # too far from expected 980
    _, exc = reconcile(GATEWAY, bank, REFUNDS, as_of=AS_OF)
    kinds = set(zip(exc.txn_id, exc.exception_type))
    assert ("GW6", "reference_mismatch") not in kinds and ("GW6", "missing_settlement") in kinds
