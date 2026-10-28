"""Deterministic synthetic gateway transactions, bank settlements, refunds and ground truth.

Written files:
  gateway.csv     txn_id, merchant_id, txn_date, amount
  bank.csv        bank_ref, gateway_ref, merchant_id, settled_amount, settle_date
  refunds.csv     txn_id, refund_date, refund_amount
  truth.csv       txn_id, exception_type, true_root_cause     (evaluation only)
"""
from __future__ import annotations

import argparse
import math
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (AMOUNT_TOLERANCE, AS_OF, BANK_CALENDAR, FEE_RATE, SETTLEMENT_SLA_BUSINESS_DAYS,
                     STANDARD_HOLIDAYS)


def bday(d, lag: int, calendar) -> date:
    return np.busday_offset(np.datetime64(d, "D"), lag, roll="forward", holidays=calendar).astype(object)


def generate(n: int = 20_000, seed: int = 42):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    first_day = date(2026, 8, 1)
    n_days = (AS_OF - first_day).days  # txns from Aug 1 up to the day before AS_OF

    merchants = [f"M{i:03d}" for i in range(1, 151)]
    ids = rng.sample(range(10**9, 10**10), n)
    gateway = pd.DataFrame({
        "txn_id": [f"GW{i}" for i in ids],
        "merchant_id": [rng.choice(merchants) for _ in range(n)],
        "txn_date": [first_day + timedelta(days=rng.randrange(n_days)) for _ in range(n)],
        "amount": np.clip(nprng.lognormal(np.log(1800), 0.9, n), 200, 250_000).round(2),
    })
    gateway["expected"] = (gateway["amount"] * (1 - FEE_RATE)).round(2)

    # ---- pick which transactions get which seeded exception --------------------------------
    # Eligible: old enough that the T+2 due date has passed. Recent transactions are more likely to be picked
    # (weighted shuffle), because in real life old exceptions get resolved and recent ones dominate the queue.
    eligible = gateway.index[gateway["txn_date"] <= AS_OF - timedelta(days=6)].tolist()
    key = {i: rng.random() ** (1.0 / math.exp(-(AS_OF - gateway.at[i, "txn_date"]).days / 30)) for i in eligible}
    old_enough = sorted(eligible, key=key.get)  # pop() takes the highest key first
    take = lambda k: [old_enough.pop() for _ in range(k)]
    plan = {"missing": take(round(n * 0.012)), "refund": take(round(n * 0.0075)),
            "fee": take(round(n * 0.0075)), "duplicate": take(round(n * 0.006)),
            "id_mismatch": take(round(n * 0.008))}
    role = {i: k for k, idxs in plan.items() for i in idxs}

    bank_rows, refund_rows, truth_rows = [], [], []
    bank_counter = 0

    def new_ref() -> str:
        nonlocal bank_counter
        bank_counter += 1
        return f"UTR{bank_counter:012d}"

    all_ids = set(gateway["txn_id"])
    for idx, t in gateway.iterrows():
        kind = role.get(idx)
        if kind == "missing":
            truth_rows.append((t.txn_id, "missing_settlement", "settlement_file_missing"))
            continue
        lag = 1 if rng.random() < 0.65 else 2
        settle = bday(t.txn_date, lag, BANK_CALENDAR)
        if settle > AS_OF:  # not settled yet; nothing to reconcile
            continue
        settled, ref = t.expected, t.txn_id
        if kind == "refund":
            refund = round(t.amount * rng.uniform(0.10, 0.60), 2)
            refund_rows.append((t.txn_id, t.txn_date + timedelta(days=rng.randint(1, 5)), refund))
            settled = round(t.expected - refund, 2)
            truth_rows.append((t.txn_id, "short_settled", "refund_deduction"))
        elif kind == "fee":
            settled = round(t.amount * (1 - rng.uniform(0.03, 0.05)), 2)  # wrong fee rate
            truth_rows.append((t.txn_id, "short_settled", "fee_calculation_error"))
        elif kind == "id_mismatch":
            variants = [t.txn_id[:-1], t.txn_id.lower(), "TX-" + t.txn_id, t.txn_id[:5] + t.txn_id[6] + t.txn_id[5] + t.txn_id[7:]]
            ref = rng.choice([v for v in variants if v not in all_ids])  # never collide with a real txn id
            truth_rows.append((t.txn_id, "reference_mismatch", "reference_id_mismatch"))
        elif rng.random() < 0.03 and kind is None:  # decoy: refund exists but was NOT netted off this settlement
            refund_rows.append((t.txn_id, t.txn_date + timedelta(days=rng.randint(1, 5)), round(t.amount * 0.3, 2)))
        bank_rows.append((new_ref(), ref, t.merchant_id, settled, settle))
        if kind == "duplicate":
            second = min(bday(settle, rng.randint(1, 3), BANK_CALENDAR), AS_OF)  # never future-dated
            bank_rows.append((new_ref(), ref, t.merchant_id, settled, second))
            truth_rows.append((t.txn_id, "duplicate_settlement", "duplicate_posting"))
        # Natural delays: the bank observes regional holidays our T+2 calendar does not model.
        due = bday(t.txn_date, SETTLEMENT_SLA_BUSINESS_DAYS, STANDARD_HOLIDAYS)
        if settle > due:  # a seeded txn can also be late; that is a real second exception
            truth_rows.append((t.txn_id, "delayed_settlement", "bank_holiday_delay"))

    bank = pd.DataFrame(bank_rows, columns=["bank_ref", "gateway_ref", "merchant_id", "settled_amount", "settle_date"])
    refunds = pd.DataFrame(refund_rows, columns=["txn_id", "refund_date", "refund_amount"])
    truth = pd.DataFrame(truth_rows, columns=["txn_id", "exception_type", "true_root_cause"])
    bank = bank.sample(frac=1, random_state=seed).reset_index(drop=True)  # bank file order is unrelated to gateway order
    return gateway.drop(columns="expected"), bank, refunds, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g, b, r, t = generate(a.n, a.seed)
    for name, df in (("gateway", g), ("bank", b), ("refunds", r), ("truth", t)):
        df.to_csv(out / f"{name}.csv", index=False)
    print(f"gateway {len(g)} | bank {len(b)} | refunds {len(r)} | seeded/derived exceptions {len(t)} -> {out}/")


if __name__ == "__main__":
    main()
