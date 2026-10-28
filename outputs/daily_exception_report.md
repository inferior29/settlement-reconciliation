# Daily reconciliation exception report - 27 Oct 2026

Synthetic data. Classifier backend: **rules**.

## Match summary
- Gateway transactions: **20,000**
- Matched on exact reference: **19,372**
- Matched by amount/date tolerance (reference mismatch): **160**
- No bank entry: **468** (of which past due: 240; the rest are not yet due)
- Exceptions flagged: **1,688** | Total value at risk: **INR 1,076,509.01**

## Exceptions and value at risk, by type
| exception_type | exceptions | value_at_risk_inr |
|---|---|---|
| delayed_settlement | 868 | 0.00 |
| short_settled | 300 | 147,355.14 |
| missing_settlement | 240 | 589,701.63 |
| reference_mismatch | 160 | 0.00 |
| duplicate_settlement | 120 | 339,452.24 |

## Ageing (days since the exception became visible)
| exception_type | 0-2 days | 3-7 days | 8-14 days | 15+ days |
|---|---|---|---|---|
| delayed_settlement | 0 | 0 | 62 | 806 |
| duplicate_settlement | 4 | 20 | 25 | 71 |
| missing_settlement | 0 | 36 | 51 | 153 |
| reference_mismatch | 0 | 26 | 25 | 109 |
| short_settled | 0 | 44 | 63 | 193 |
| VALUE AT RISK (INR) | 17,452.58 | 235,387.49 | 187,044.06 | 636,624.88 |

## Evaluation against seeded truth
- Detection: 1,688 found, 0 false positives, 0 missed (precision 100.0%, recall 100.0%)
- Root-cause classification (rules): **100.0%** on 1,688 classified exceptions

| true_root_cause | accuracy | n |
|---|---|---|
| bank_holiday_delay | 100.0% | 868 |
| duplicate_posting | 100.0% | 120 |
| fee_calculation_error | 100.0% | 150 |
| reference_id_mismatch | 100.0% | 160 |
| refund_deduction | 100.0% | 150 |
| settlement_file_missing | 100.0% | 240 |

Value at risk: missing = expected settlement; short-settled = shortfall; duplicate and unmatched bank credits = amount to
recover; delayed and reference-mismatch items are timing/process issues and carry no value at risk.
