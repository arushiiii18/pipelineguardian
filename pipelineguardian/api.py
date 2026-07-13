"""
api.py — FastAPI backend, thin wrapper around agent.audit().

POST /audit accepts a notebook/script file upload, and optionally a CSV
(for schema_inspector) plus a target column name. Returns the same
AuditReport JSON shape as the CLI's --json mode.

Run locally: uvicorn pipelineguardian.api:app --reload
"""

import os
import tempfile
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipelineguardian import agent
from pipelineguardian.models import AuditReport

app = FastAPI(title="PipelineGuardian API")

# CORS wide open deliberately: this is a public audit tool with no user data
# persistence and no auth, so there's nothing sensitive to protect by
# restricting origins. Tighten this if that ever changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/audit", response_model=AuditReport)
async def audit_endpoint(
    source_file: UploadFile = File(..., description=".ipynb or .py file"),
    data_file: UploadFile | None = File(None, description="Optional CSV for schema_inspector"),
    target_col: str | None = Form(None),
):
    if not (source_file.filename.endswith(".ipynb") or source_file.filename.endswith(".py")):
        raise HTTPException(400, "source_file must be a .ipynb or .py file")

    suffix = ".ipynb" if source_file.filename.endswith(".ipynb") else ".py"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp:
        tmp.write(await source_file.read())
        tmp_path = tmp.name

    df = None
    if data_file is not None:
        try:
            df = pd.read_csv(data_file.file)
        except Exception as e:
            raise HTTPException(400, f"Could not parse data_file as CSV: {e}")

    try:
        report = agent.audit(tmp_path, df=df, target_col=target_col)
        report.source_path = source_file.filename  # don't leak the server tempfile path to the client
        return report
    finally:
        os.unlink(tmp_path)
