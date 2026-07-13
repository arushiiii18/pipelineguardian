"""
cli.py — standalone CLI, usable independent of any hosting uptime.

Usage:
    python -m pipelineguardian.cli audit path/to/notebook.ipynb
    python -m pipelineguardian.cli audit path/to/script.py --data data.csv --target churn
    python -m pipelineguardian.cli audit path/to/notebook.ipynb --project-dir .
    python -m pipelineguardian.cli audit path/to/notebook.ipynb --json
"""

from __future__ import annotations
import argparse
import json
import sys
import pandas as pd

from pipelineguardian import agent


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _print_human(report) -> None:
    print(f"\nAudit: {report.source_path}")
    print(f"Tools run: {', '.join(report.tools_run)}")
    if report.tools_skipped:
        print(f"Tools skipped: {', '.join(report.tools_skipped)}")

    if not report.issues:
        print("\nNo issues found.\n")
        return

    issues = sorted(report.issues, key=lambda i: SEVERITY_ORDER.get(i.severity, 9))
    print(f"\n{len(issues)} issue(s) found:\n")
    for issue in issues:
        loc = ""
        if issue.line is not None:
            loc = f" (line {issue.line}" + (f", cell {issue.cell}" if issue.cell is not None else "") + ")"
        print(f"[{issue.severity.upper()}] {issue.check_name}{loc}")
        print(f"  confidence: {issue.confidence}")
        print(f"  {issue.message}")
        print(f"  evidence:   {issue.evidence}")
        print(f"  fix:        {issue.suggested_fix}\n")


def cmd_audit(args: argparse.Namespace) -> int:
    df = None
    if args.data:
        try:
            df = pd.read_csv(args.data)
        except Exception as e:
            print(f"Warning: could not read --data {args.data}: {e}. Continuing without schema_inspector.", file=sys.stderr)

    try:
        report = agent.audit(
            args.source_path,
            project_dir=args.project_dir,
            df=df,
            target_col=args.target,
        )
    except Exception as e:
        print(f"Error auditing {args.source_path}: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)

    # Non-zero exit if any HIGH severity issue was found — useful for CI gating.
    has_high = any(i.severity == "high" for i in report.issues)
    return 1 if has_high else 0


def main():
    parser = argparse.ArgumentParser(prog="pipelineguardian", description="Audit ML notebooks/scripts for engineering correctness bugs.")
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit", help="Audit a notebook or script")
    audit_p.add_argument("source_path", help="Path to a .ipynb or .py file")
    audit_p.add_argument("--data", help="Path to a CSV to enable schema_inspector (timestamp/group/imbalance checks)")
    audit_p.add_argument("--target", help="Target column name in --data, required for imbalance detection")
    audit_p.add_argument("--project-dir", help="Project directory to check for requirements.txt")
    audit_p.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable text")
    audit_p.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
