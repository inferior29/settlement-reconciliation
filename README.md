# AI-Assisted Settlement Reconciliation

Matches payment-gateway transactions to bank settlement records, flags what does not reconcile, uses Claude to
classify each exception into a root cause with a next action, and produces a daily exception report with ageing and
value at risk. **All data is synthetic.**

```
gateway.csv ─┐
bank.csv ────┼─► stage 1: exact reference ─► stage 2: merchant + amount tolerance + date window ─► exceptions
refunds.csv ─┘                                                                                          │
                                       root-cause classifier (Claude | rule baseline) ◄─────────────────┘
                                                               │
                                          daily report: ageing, value at risk, evaluation vs seeded truth
```

## What gets flagged

| Exception | Rule |
|---|---|
| `missing_settlement` | gateway transaction past its T+2 business-day due date with no bank entry |
| `short_settled` | bank paid less than gross minus the 2% fee, beyond a INR 0.50 tolerance |
| `duplicate_settlement` | bank posted the same reference more than once |
| `delayed_settlement` | settled after the T+2 due date |
| `reference_mismatch` | no exact reference match; matched only on merchant + amount + date |
| `unmatched_bank_entry` | bank credit that matches no gateway transaction |

Root-cause categories (6): `settlement_file_missing`, `refund_deduction`, `fee_calculation_error`, `duplicate_posting`,
`bank_holiday_delay`, `reference_id_mismatch`.

## Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src                                  # or: pip install -e .
python -m recon.generate_data --out data               # 20,000 transactions, bank file, refunds, truth.csv
python -m recon.run --backend rules                    # baseline, no API key needed
pytest -q                                              # 15 tests

# Claude backend (own API key, read from the environment; --limit controls cost)
export ANTHROPIC_API_KEY=...
python -m recon.run --backend claude --limit 200 --workers 8
```

Outputs: `outputs/exceptions.csv` and `outputs/daily_exception_report.md`.

## Results (rule baseline, synthetic data, seed 42)

| Measure | Result |
|---|---|
| Transactions / exact matches / fuzzy matches | 20,000 / 19,372 / 160 |
| Exceptions flagged | 1,688 (868 delayed, 300 short-settled, 240 missing, 160 reference mismatch, 120 duplicate) |
| Detection vs ground truth | 0 missed, 0 false positives |
| Value at risk | INR 1,076,509.01 (missing 589,701.63, duplicates 339,452.24, short-settled 147,355.14) |
| Root-cause accuracy, rule baseline | 100% |

**Read the 100% correctly.** In synthetic data the exception type plus two arithmetic checks almost determines the
cause, so a rule baseline is perfect by construction, and exact detection is a check that the engine's logic matches how
the generator built the data. Neither says anything about real bank files, which have messier references, partial
settlements and several overlapping causes. **The Claude backend has not been run against the live API yet.** It is
unit-tested against a fake client. Run it with your own key and record its accuracy in this table; the interesting question
is how it behaves on cases the rules cannot decide.

## Design notes

- **Two-stage matching.** Exact reference first. Only the leftovers go to fuzzy matching, which is one-to-one (a bank row can be used once) and limited by merchant, amount tolerance and a date window.
- **Two calendars.** Due dates use a standard holiday list. The bank also observes regional holidays that list does not model, which is what produces genuine `bank_holiday_delay` exceptions.
- **Value at risk has a stated definition.** Missing = expected settlement; short-settled = shortfall; duplicate and unmatched credits = amount to recover. Delays and reference mismatches are timing/process issues and carry none.
- **Evidence-only prompts.** Claude receives ten numeric/boolean fields about one exception: no names, no account data. It must answer through a forced `record_root_cause` tool whose schema enumerates the six causes. Invalid output or an API error becomes `needs_review` for a human.
- **Decoys in the data.** Some transactions have a refund that was *not* netted off, so "a refund exists" alone does not explain a shortfall.

## Limitations

- Synthetic data; amounts, fees and calendars are illustrative.
- One fee rate, one currency, one bank file format.
- A bank row can match only one gateway transaction; many-to-one settlements (batching) are not modelled.
- The `claude-haiku-5-5` default is configurable with `RECON_MODEL`.

## Layout

```
src/recon/  config.py  generate_data.py  matching.py  classify.py  report.py  run.py
tests/      matching rules, ground-truth agreement, rule + Claude (fake client) classifiers
data/ outputs/  generated inputs and the latest report
```
