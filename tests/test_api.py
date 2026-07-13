import os
from fastapi.testclient import TestClient
from pipelineguardian.api import app

client = TestClient(app)

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "..", "eval", "synthetic_notebooks")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_audit_endpoint_detects_issue():
    path = os.path.join(FIXTURES_DIR, "05_target_in_features.py")
    with open(path, "rb") as f:
        r = client.post("/audit", files={"source_file": ("05_target_in_features.py", f, "text/x-python")})
    assert r.status_code == 200
    body = r.json()
    assert any(i["check_name"] == "target_column_in_feature_list" for i in body["issues"])
    # server tempfile path must never leak to the client
    assert body["source_path"] == "05_target_in_features.py"
    assert "tmp" not in body["source_path"].lower()


def test_audit_endpoint_clean_file_no_issues():
    path = os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")
    with open(path, "rb") as f:
        r = client.post("/audit", files={"source_file": ("10_all_clean_baseline.py", f, "text/x-python")})
    assert r.status_code == 200
    assert r.json()["issues"] == []


def test_audit_endpoint_rejects_wrong_file_type():
    r = client.post("/audit", files={"source_file": ("evil.exe", b"nope", "application/octet-stream")})
    assert r.status_code == 400
