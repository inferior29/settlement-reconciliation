"""Reconciliation engine: match gateway transactions to bank settlements and flag exceptions.

Stage 1  exact match on the transaction reference.
Stage 2  for what is left, match on merchant + settled amount within tolerance + a date window
         (these are flagged as `reference_mismatch` because the reference did not agree).

Exceptions flagged:
  missing_settlement    gateway txn past its T+2 due date with no bank entry
  short_settled         bank paid less than gross - fee (beyond tolerance)
  duplicate_settlement  bank posted the same reference more than once
  delayed_settlement    settled after the T+2 due date
  reference_mismatch    matched only in stage 2
  unmatched_bank_entry  bank credit that matches no gateway transaction
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (AMOUNT_TOLERANCE, AS_OF, DATE_WINDOW_DAYS, FEE_RATE, REGIONAL_HOLIDAYS,
                     SETTLEMENT_SLA_BUSINESS_DAYS, STANDARD_HOLIDAYS)


def _d(series) -> np.ndarray:
    return pd.to_datetime(series).values.astype("datetime64[D]")


def due_dates(txn_dates) -> np.ndarray:
    return np.busday_offset(_d(txn_dates), SETTLEMENT_SLA_BUSINESS_DAYS, roll="forward", holidays=STANDARD_HOLIDAYS)


def reconcile(gateway: pd.DataFrame, bank: pd.DataFrame, refunds: pd.DataFrame | None = None,
              as_of=AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (matches, exceptions)."""
    as_of_d = np.datetime64(as_of, "D")
    g = gateway.copy()
    g["txn_date"] = pd.to_datetime(g["txn_date"])
    g["expected"] = (g["amount"] * (1 - FEE_RATE)).round(2)
    g["due_date"] = pd.to_datetime(due_dates(g["txn_date"]))
    b = bank.copy()
    b["settle_date"] = pd.to_datetime(b["settle_date"])
    refund_by_txn = (refunds.groupby("txn_id")["refund_amount"].sum() if refunds is not None and len(refunds)
                     else pd.Series(dtype=float))

    # ---- stage 1: exact reference ------------------------------------------------------------
    b = b.sort_values(["gateway_ref", "settle_date", "bank_ref"]).reset_index(drop=True)
    b["entry_no"] = b.groupby("gateway_ref").cumcount() + 1
    b["entries"] = b.groupby("gateway_ref")["bank_ref"].transform("size")
    primary = b[b["entry_no"] == 1][["gateway_ref", "bank_ref", "settled_amount", "settle_date", "entries"]]
    m = g.merge(primary, left_on="txn_id", right_on="gateway_ref", how="left").drop(columns="gateway_ref")
    m["match_type"] = np.where(m["bank_ref"].notna(), "exact", "")

    extras = b[(b["entry_no"] > 1) & b["gateway_ref"].isin(g["txn_id"])]
    orphans = b[~b["gateway_ref"].isin(g["txn_id"])]

    # ---- stage 2: merchant + amount tolerance + date window ----------------------------------
    unmatched = m[m["bank_ref"].isna()][["txn_id", "merchant_id", "txn_date", "expected"]]
    cand = unmatched.merge(orphans[["bank_ref", "merchant_id", "settled_amount", "settle_date"]], on="merchant_id")
    gap = (cand["settle_date"] - cand["txn_date"]).dt.days
    cand = cand[((cand["settled_amount"] - cand["expected"]).abs() <= AMOUNT_TOLERANCE) & gap.between(0, DATE_WINDOW_DAYS)]
    cand = cand.assign(gap=(cand["settle_date"] - cand["txn_date"]).dt.days,
                       diff=(cand["settled_amount"] - cand["expected"]).abs())
    cand = cand.sort_values(["gap", "diff"]).drop_duplicates("txn_id").drop_duplicates("bank_ref")
    fuzzy = cand.set_index("txn_id")
    hit = m["txn_id"].isin(fuzzy.index)
    if hit.any():  # on a clean day stage 2 finds nothing; skip so empty frames never reach the assignment
        for col in ("bank_ref", "settled_amount", "settle_date"):
            m.loc[hit, col] = m.loc[hit, "txn_id"].map(fuzzy[col]).values
        m.loc[hit, "entries"] = 1
        m.loc[hit, "match_type"] = "fuzzy"
    leftover_orphans = orphans[~orphans["bank_ref"].isin(fuzzy["bank_ref"])]

    # ---- exceptions --------------------------------------------------------------------------
    rows = []
    base = ["txn_id", "merchant_id", "txn_date", "due_date", "amount", "expected", "bank_ref", "settled_amount", "settle_date"]

    def add(frame: pd.DataFrame, kind: str, **extra) -> None:
        if len(frame):
            out = frame[base].copy()
            out["exception_type"] = kind
            for k, v in extra.items():
                out[k] = v
            rows.append(out)

    settled = m[m["bank_ref"].notna()]
    due_passed = m["due_date"] < pd.Timestamp(as_of)
    add(m[m["bank_ref"].isna() & due_passed], "missing_settlement")
    add(settled[settled["match_type"] == "fuzzy"], "reference_mismatch")
    short = settled[(settled["expected"] - settled["settled_amount"]) > AMOUNT_TOLERANCE]
    add(short, "short_settled")
    add(settled[settled["settle_date"] > settled["due_date"]], "delayed_settlement")
    dup = extras.merge(g, left_on="gateway_ref", right_on="txn_id", suffixes=("_bank", ""))
    if len(dup):
        dup = dup.assign(bank_ref=dup["bank_ref"], settled_amount=dup["settled_amount"], settle_date=dup["settle_date"])
        add(dup, "duplicate_settlement")
    if len(leftover_orphans):
        o = leftover_orphans.rename(columns={"gateway_ref": "txn_id"}).copy()
        for col in ("txn_date", "due_date", "amount", "expected"):
            o[col] = pd.NaT if "date" in col else np.nan
        add(o, "unmatched_bank_entry")

    exc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=base + ["exception_type"])
    exc = _features(exc, refund_by_txn, m, as_of_d)
    return m, exc


def _features(exc: pd.DataFrame, refund_by_txn: pd.Series, matches: pd.DataFrame, as_of_d) -> pd.DataFrame:
    """Add the evidence columns a person (or an AI model) needs to explain each exception."""
    if exc.empty:
        return exc.assign(shortfall=[], delay_business_days=[], holiday_in_delay_window=[], refund_amount=[],
                          bank_entries=[], age_days=[], age_bucket=[], value_at_risk=[])
    e = exc.copy()
    e["shortfall"] = (e["expected"] - e["settled_amount"]).where(e["exception_type"] == "short_settled").round(2)
    settle, due = e["settle_date"], e["due_date"]
    both = settle.notna() & due.notna()
    delay = pd.Series(0, index=e.index)
    delay[both] = np.busday_count(_d(due[both]), _d(settle[both]), holidays=STANDARD_HOLIDAYS)
    e["delay_business_days"] = delay.clip(lower=0)
    reg = pd.to_datetime(REGIONAL_HOLIDAYS)
    e["holiday_in_delay_window"] = [
        bool(both.iloc[i] and ((reg >= due.iloc[i]) & (reg <= settle.iloc[i])).any()) for i in range(len(e))
    ]
    e["refund_amount"] = e["txn_id"].map(refund_by_txn).fillna(0.0)
    entries = matches.set_index("txn_id")["entries"]
    e["bank_entries"] = e["txn_id"].map(entries).fillna(1).astype(int)
    e.loc[e["exception_type"] == "duplicate_settlement", "bank_entries"] = 2

    visible_from = e["settle_date"].where(e["exception_type"] != "missing_settlement", e["due_date"])
    e["age_days"] = (pd.Timestamp(as_of_d) - pd.to_datetime(visible_from)).dt.days.clip(lower=0)
    e["age_bucket"] = pd.cut(e["age_days"], [-1, 2, 7, 14, 10_000], labels=["0-2 days", "3-7 days", "8-14 days", "15+ days"]).astype(str)
    # Value at risk = money not correctly in hand (timing-only exceptions carry no value at risk).
    var = pd.Series(0.0, index=e.index)
    var[e["exception_type"] == "missing_settlement"] = e["expected"]
    var[e["exception_type"] == "short_settled"] = e["shortfall"]
    var[e["exception_type"] == "duplicate_settlement"] = e["settled_amount"]       # over-settled, recoverable by reversal
    var[e["exception_type"] == "unmatched_bank_entry"] = e["settled_amount"]
    e["value_at_risk"] = var.round(2)
    e.insert(0, "exception_id", [f"X{i:05d}" for i in range(1, len(e) + 1)])
    return e
