"""Business rules in one place. Change a rule here and every module follows."""
from __future__ import annotations

from datetime import date

AS_OF = date(2026, 10, 27)          # fixed "today" so runs are reproducible

FEE_RATE = 0.02                      # gateway/bank fee deducted from each settlement
AMOUNT_TOLERANCE = 0.50              # INR; differences at or below this are treated as equal
DATE_WINDOW_DAYS = 10                # stage-2 match: bank date within this many days after the txn
SETTLEMENT_SLA_BUSINESS_DAYS = 2     # T+2

# Illustrative calendars (not a real bank calendar).
# Our own calendar knows STANDARD holidays; the bank also observes REGIONAL ones we do not model.
STANDARD_HOLIDAYS = ["2026-09-17", "2026-10-02"]
REGIONAL_HOLIDAYS = ["2026-08-26", "2026-09-04", "2026-10-08"]
BANK_CALENDAR = STANDARD_HOLIDAYS + REGIONAL_HOLIDAYS

EXCEPTION_TYPES = ["missing_settlement", "short_settled", "duplicate_settlement",
                   "delayed_settlement", "reference_mismatch", "unmatched_bank_entry"]

ROOT_CAUSES = [
    "settlement_file_missing",     # bank never received/processed the settlement
    "refund_deduction",            # a refund was netted off the settlement
    "fee_calculation_error",       # wrong fee rate applied
    "duplicate_posting",           # bank posted the same settlement twice
    "bank_holiday_delay",          # a bank holiday pushed settlement past T+2
    "reference_id_mismatch",       # bank carried a corrupted transaction reference
]

NEXT_ACTION = {
    "settlement_file_missing": "Raise a ticket with the bank for the missing settlement and attach the gateway batch reference.",
    "refund_deduction": "Confirm the refund in the refunds ledger and mark the shortfall as explained.",
    "fee_calculation_error": "Recalculate the fee at the contracted rate and claim the difference from the bank.",
    "duplicate_posting": "Ask the bank to reverse the duplicate credit before month end.",
    "bank_holiday_delay": "No action beyond monitoring; confirm the delay matches the bank holiday calendar.",
    "reference_id_mismatch": "Correct the reference on the bank record so the entry matches the gateway transaction.",
}
