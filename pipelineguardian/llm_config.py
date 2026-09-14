"""
llm_config.py — single source of truth for all LLM configuration.

Every LLM-backed component (validation_strategy_reviewer, llm_only_auditor,
llm_context_auditor, and the LLM sub-path of hybrid_router) imports its
model/temperature from this file. No component hardcodes its own model string
or temperature value.

IMPORTANT: if you change LLM_MODEL or LLM_TEMPERATURE, you MUST bump
PROMPT_VERSION and/or CONTEXT_SCHEMA_VERSION (as applicable) and re-tag
`experiment-frozen` before running any more Yang-corpus evaluations. The
run_manifest.jsonl and experiment_config.yaml record these values at each run.
"""

LLM_MODEL = "qwen/qwen3.8-27b"
LLM_PROVIDER = "groq"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1500
LLM_RUNS_PER_NOTEBOOK = 3   # for llm_only, llm_with_context, hybrid during final Yang eval
PROMPT_VERSION = "v1"       # bump this and re-freeze if any prompt text changes
CONTEXT_SCHEMA_VERSION = "v1"  # bump this and re-freeze if NotebookContext fields change
