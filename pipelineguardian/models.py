"""
Shared data model for PipelineGuardian.

Every tool be it deterministic or LLM-backed, emits Issue objects in this shape.
There is deliberately no composite "quality score" field anywhere in this file.
See README.md, "Non-goals", for why.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    HIGH = "high"       # will silently corrupt results (e.g. leakage)
    MEDIUM = "medium"   # will hurt reproducibility / trust in results
    LOW = "low"          # low blast radius


class Confidence(str, Enum):
    DETERMINISTIC = "deterministic"  # exact AST/regex pattern match, no ambiguity
    INFERRED = "inferred"            # LLM judgment call (validation_strategy_reviewer only)


class Issue(BaseModel):
    check_name: str = Field(..., description="Machine-readable id, e.g. 'scaler_fit_before_split'")
    severity: Severity
    confidence: Confidence
    message: str = Field(..., description="Human-readable one-line explanation")
    evidence: str = Field(..., description="The matched line/pattern, or the LLM's stated reasoning")
    line: Optional[int] = Field(None, description="1-indexed line number in the flattened source")
    cell: Optional[int] = Field(None, description="Notebook cell index, if source was a .ipynb")
    suggested_fix: str = Field(..., description="Concrete, actionable fix — not generic advice")

    model_config = ConfigDict(use_enum_values=True)


class AuditReport(BaseModel):
    """Top-level output of a full audit run. No aggregate score, by design."""
    source_path: str
    issues: list[Issue] = Field(default_factory=list)
    tools_run: list[str] = Field(default_factory=list)
    tools_skipped: list[str] = Field(default_factory=list)  # e.g. validation_strategy_reviewer, with why
