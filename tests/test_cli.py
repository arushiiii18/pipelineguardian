import json
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..")
FIXTURES_DIR = os.path.join(REPO_ROOT, "eval", "synthetic_notebooks")


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "pipelineguardian.cli", "audit", *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_cli_exits_nonzero_on_high_severity_issue():
    path = os.path.join(FIXTURES_DIR, "01_scaler_leak.py")
    result = _run_cli(path)
    assert result.returncode == 1
    assert "scaler_fit_before_split" in result.stdout


def test_cli_exits_zero_on_clean_file():
    path = os.path.join(FIXTURES_DIR, "10_all_clean_baseline.py")
    result = _run_cli(path)
    assert result.returncode == 0
    assert "No issues found" in result.stdout


def test_cli_json_mode_produces_valid_json():
    path = os.path.join(FIXTURES_DIR, "05_target_in_features.py")
    result = _run_cli(path, "--json")
    parsed = json.loads(result.stdout)
    assert parsed["source_path"] == path
    assert any(i["check_name"] == "target_column_in_feature_list" for i in parsed["issues"])
