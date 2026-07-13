"""
demo_before_after.py — the project's central demo artifact.

Runs the full agent (deterministic tools + conditional LLM tool) on a
"before" notebook with real injected issues and an "after" notebook where
they've all been fixed, using the same synthetic dataset for both so the
only thing that changes is the code. This is the script the README's demo
GIF is built from.

Run: python -m eval.demo_before_after
(requires GROQ_API_KEY set in .env — the LLM tool fires here since the
dataset has a genuine timestamp column)
"""

import os
import pandas as pd
from pipelineguardian import agent

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "synthetic_notebooks")


def make_demo_df() -> pd.DataFrame:
    n = 200
    return pd.DataFrame({
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D"),
        "tenure_months": [float(i % 24) if i % 7 != 0 else None for i in range(n)],  # some NaNs
        "monthly_spend": [50 + (i % 30) for i in range(n)],
        "churn": [0, 1] * (n // 2),
    })


def print_report(label: str, report, elapsed_note: str = ""):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"Tools run: {report.tools_run}")
    print(f"Tools skipped: {report.tools_skipped}")
    if not report.issues:
        print("\nNo issues found.")
    for issue in report.issues:
        print(f"\n[{issue.severity.upper()}] {issue.check_name}  (confidence: {issue.confidence})")
        print(f"  {issue.message}")
        print(f"  evidence: {issue.evidence}")
        print(f"  fix: {issue.suggested_fix}")


def main():
    df = make_demo_df()

    before_path = os.path.join(FIXTURES_DIR, "demo_before.ipynb")
    after_path = os.path.join(FIXTURES_DIR, "demo_after.ipynb")

    before_report = agent.audit(before_path, df=df, target_col="churn")
    after_report = agent.audit(after_path, df=df, target_col="churn")

    print_report("BEFORE (demo_before.ipynb) — injected issues", before_report)
    print_report("AFTER (demo_after.ipynb) — fixed", after_report)

    print(f"\n{'='*70}")
    print(f"SUMMARY: before={len(before_report.issues)} issues, after={len(after_report.issues)} issues")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
