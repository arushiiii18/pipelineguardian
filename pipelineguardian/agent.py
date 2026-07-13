"""
agent.py — top-level orchestrator.

leakage_detector and reproducibility_checker always run: cheap, deterministic,
always relevant regardless of dataset shape.

schema_inspector always runs if a dataframe is available, but it produces
SIGNALS, not Issues. Whether validation_strategy_reviewer (the one LLM call
in the whole system) runs is a plain Python decision based on those signals.
No timestamp/group/imbalance signal -> it's skipped, and the report says
exactly why. This if-statement IS the "agent decides" story the project
rests on — see validation_strategy_reviewer.py's docstring for why that's
implemented as a readable branch rather than dressed up as LLM autonomy.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd

from pipelineguardian.models import AuditReport
from pipelineguardian.tools.notebook_parser import load_source
from pipelineguardian.tools import leakage_detector, reproducibility_checker, schema_inspector
from pipelineguardian.tools import validation_strategy_reviewer


def audit(
    source_path: str,
    project_dir: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    llm=None,
) -> AuditReport:
    """
    llm: optional pre-built structured LLM, injected for testing. When None,
    validation_strategy_reviewer builds the real LangChain client lazily —
    only if the conditional branch actually fires.
    """
    pset = load_source(source_path)
    tools_run: list[str] = []
    tools_skipped: list[str] = []
    issues = []

    issues += leakage_detector.run(pset)
    tools_run.append("leakage_detector")

    issues += reproducibility_checker.run(pset, project_dir)
    tools_run.append("reproducibility_checker")

    if df is not None:
        signals = schema_inspector.inspect(df, target_col)
        tools_run.append("schema_inspector")

        if signals.has_trigger:
            issues += validation_strategy_reviewer.review(pset, signals, llm=llm)
            tools_run.append("validation_strategy_reviewer")
        else:
            tools_skipped.append(
                "validation_strategy_reviewer (no timestamp/group/imbalance signal found)"
            )
    else:
        tools_skipped.append("schema_inspector (no dataframe provided)")
        tools_skipped.append("validation_strategy_reviewer (schema_inspector did not run)")

    return AuditReport(
        source_path=source_path,
        issues=issues,
        tools_run=tools_run,
        tools_skipped=tools_skipped,
    )
