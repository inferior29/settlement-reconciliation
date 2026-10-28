"""Daily exception report: ageing, value at risk, and evaluation against seeded truth."""
from __future__ import annotations

import pandas as pd

from .config import AS_OF

BUCKETS = ["0-2 days", "3-7 days", "8-14 days", "15+ days"]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def by_type(exc: pd.DataFrame) -> pd.DataFrame:
    g = exc.groupby("exception_type").agg(exceptions=("exception_id", "count"),
                                          value_at_risk_inr=("value_at_risk", "sum")).reset_index()
    g["value_at_risk_inr"] = g["value_at_risk_inr"].round(2).map("{:,.2f}".format)
    return g.sort_values("exceptions", ascending=False)


def ageing(exc: pd.DataFrame) -> pd.DataFrame:
    counts = exc.pivot_table(index="exception_type", columns="age_bucket", values="exception_id", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=BUCKETS, fill_value=0)
    var = exc.groupby("age_bucket")["value_at_risk"].sum().reindex(BUCKETS, fill_value=0)
    counts.loc["VALUE AT RISK (INR)"] = var.round(2).map("{:,.2f}".format)
    return counts.reset_index()


def evaluate(exc: pd.DataFrame, truth: pd.DataFrame) -> dict:
    key = ["txn_id", "exception_type"]
    found = exc.merge(truth, on=key, how="outer", indicator=True)
    tp = int((found["_merge"] == "both").sum())
    fp = int((found["_merge"] == "left_only").sum())
    fn = int((found["_merge"] == "right_only").sum())
    both = found[(found["_merge"] == "both") & found["predicted_root_cause"].notna() & (found["predicted_root_cause"] != "")]
    correct = both["predicted_root_cause"] == both["true_root_cause"]
    per_cause = (both.assign(correct=correct).groupby("true_root_cause")["correct"].agg(["mean", "count"])
                 .rename(columns={"mean": "accuracy", "count": "n"}).reset_index())
    return {"detected": tp, "false_positives": fp, "missed": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "classified": len(both), "root_cause_accuracy": float(correct.mean()) if len(both) else float("nan"),
            "per_cause": per_cause}


def write_report(path, backend: str, matches: pd.DataFrame, exc: pd.DataFrame, ev: dict) -> None:
    pct = lambda x: f"{100 * x:.1f}%"
    per = ev["per_cause"].assign(accuracy=lambda d: d["accuracy"].map(pct))
    text = f"""# Daily reconciliation exception report - {AS_OF:%d %b %Y}

Synthetic data. Classifier backend: **{backend}**.

## Match summary
- Gateway transactions: **{len(matches):,}**
- Matched on exact reference: **{(matches['match_type'] == 'exact').sum():,}**
- Matched by amount/date tolerance (reference mismatch): **{(matches['match_type'] == 'fuzzy').sum():,}**
- No bank entry: **{(matches['match_type'] == '').sum():,}** (of which past due: {(exc['exception_type'] == 'missing_settlement').sum():,}; the rest are not yet due)
- Exceptions flagged: **{len(exc):,}** | Total value at risk: **INR {exc['value_at_risk'].sum():,.2f}**

## Exceptions and value at risk, by type
{_md(by_type(exc))}

## Ageing (days since the exception became visible)
{_md(ageing(exc))}

## Evaluation against seeded truth
- Detection: {ev['detected']:,} found, {ev['false_positives']:,} false positives, {ev['missed']:,} missed (precision {pct(ev['precision'])}, recall {pct(ev['recall'])})
- Root-cause classification ({backend}): **{pct(ev['root_cause_accuracy'])}** on {ev['classified']:,} classified exceptions

{_md(per)}

Value at risk: missing = expected settlement; short-settled = shortfall; duplicate and unmatched bank credits = amount to
recover; delayed and reference-mismatch items are timing/process issues and carry no value at risk.
"""
    path.write_text(text)
