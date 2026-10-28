"""Pipeline: reconcile -> classify exceptions -> report -> evaluate.

    python -m recon.run --backend rules
    ANTHROPIC_API_KEY=... python -m recon.run --backend claude --limit 200 --workers 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .classify import ClaudeClassifier, RuleClassifier, classify_all
from .matching import reconcile
from .report import evaluate, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backend", choices=["rules", "claude"], default="rules")
    ap.add_argument("--limit", type=int, default=0, help="classify only a random sample of N exceptions (cost control)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gateway, bank = pd.read_csv(data / "gateway.csv"), pd.read_csv(data / "bank.csv")
    refunds, truth = pd.read_csv(data / "refunds.csv"), pd.read_csv(data / "truth.csv")

    matches, exc = reconcile(gateway, bank, refunds)
    backend = ClaudeClassifier() if a.backend == "claude" else RuleClassifier()
    sample = exc.sample(a.limit, random_state=7) if a.limit else exc
    classified = classify_all(backend, sample, workers=a.workers)
    exc = exc.merge(classified[["exception_id", "predicted_root_cause", "next_action"]], on="exception_id", how="left")

    ev = evaluate(exc, truth)
    exc.to_csv(out / "exceptions.csv", index=False)
    write_report(out / "daily_exception_report.md", backend.name, matches, exc, ev)
    print((out / "daily_exception_report.md").read_text())


if __name__ == "__main__":
    main()
