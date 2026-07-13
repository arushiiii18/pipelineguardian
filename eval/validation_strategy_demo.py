"""
validation_strategy_demo.py — qualitative demonstration of validation_strategy_reviewer
against real fixtures, using the real Groq-backed LLM (not mocked).

This is intentionally NOT part of eval.py's precision/recall scoring — see
eval.py's docstring for why a P/R number on a handful of hand-labeled LLM
judgments would be an unfalsifiable-sounding metric. This script instead
prints each case's schema signal, the LLM's actual verdict and reasoning,
and whether that matches what the fixture was built to test. Read the
output, don't just check it ran — this is your actual interview evidence
that the tool discriminates rather than reflexively flagging.

Run: python -m eval.validation_strategy_demo
(requires GROQ_API_KEY set in .env)
"""

import os
import pandas as pd
from pipelineguardian import agent

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "synthetic_notebooks")


def timestamp_df():
    return pd.DataFrame({
        "event_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "feature": range(100),
        "target": [0] * 50 + [1] * 50,
    })


def group_df():
    # 20 customers, 5 rows each -> group column with real repeats
    customer_ids = [f"cust_{i}" for i in range(20) for _ in range(5)]
    return pd.DataFrame({
        "customer_id": customer_ids,
        "feature": range(100),
        "target": ([0, 1] * 50)[:100],
    })


CASES = [
    {
        "name": "Timestamp present, random split (expect: FLAGGED)",
        "file": "10_all_clean_baseline.py",
        "df": timestamp_df(),
        "target_col": "target",
        "expect_flag": True,
    },
    {
        "name": "Timestamp present, already chronologically split (expect: CLEAN)",
        "file": "11_timestamp_correct_split.py",
        "df": timestamp_df(),
        "target_col": "target",
        "expect_flag": False,
    },
    {
        "name": "Group/customer_id present, random split (expect: FLAGGED)",
        "file": "12_group_leak_random_split.py",
        "df": group_df(),
        "target_col": "target",
        "expect_flag": True,
    },
]


def main():
    print(f"{'='*70}")
    for case in CASES:
        path = os.path.join(FIXTURES_DIR, case["file"])
        report = agent.audit(path, df=case["df"], target_col=case["target_col"])
        flagged = any(i.check_name == "validation_strategy_mismatch" for i in report.issues)

        status = "MATCH" if flagged == case["expect_flag"] else "MISMATCH — read this one closely"
        print(f"\n{case['name']}")
        print(f"  file: {case['file']}")
        print(f"  expected flag: {case['expect_flag']}, got flag: {flagged}  [{status}]")
        for issue in report.issues:
            if issue.check_name == "validation_strategy_mismatch":
                print(f"  reasoning: {issue.message}")
                print(f"  suggested fix: {issue.suggested_fix}")
        print(f"{'-'*70}")


if __name__ == "__main__":
    main()
