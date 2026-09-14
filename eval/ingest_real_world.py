"""
ingest_real_world.py — Ingestion and ground-truth template generator for real-world notebooks.

Parses input .ipynb / .py files, copies them to eval/real_world_reproducibility/notebooks/,
and populates or updates eval/real_world_reproducibility/ground_truth.json with starter
annotation templates.

Usage:
    python -m eval.ingest_real_world --input_dir path/to/raw_notebooks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from pipelineguardian.tools.notebook_parser import load_source

HERE = os.path.dirname(__file__)
BENCHMARK_DIR = os.path.join(HERE, "real_world_reproducibility")
TARGET_NB_DIR = os.path.join(BENCHMARK_DIR, "notebooks")
GT_PATH = os.path.join(BENCHMARK_DIR, "ground_truth.json")


def load_existing_gt() -> dict:
    if os.path.exists(GT_PATH):
        try:
            with open(GT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "description": "Real-world reproducibility benchmark ground truth",
        "version": "1.0",
        "status": "UNFROZEN",
        "notebooks": {}
    }


def ingest(input_dir: str):
    os.makedirs(TARGET_NB_DIR, exist_ok=True)
    gt = load_existing_gt()

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: input directory {input_dir} does not exist.")
        return

    files = [p for p in input_path.glob("*") if p.suffix.lower() in (".py", ".ipynb")]
    print(f"Found {len(files)} candidate notebook/script files in {input_dir}.")

    for idx, fpath in enumerate(sorted(files), start=len(gt["notebooks"]) + 1):
        nb_id = f"rw_nb_{idx:03d}"
        dest_filename = f"{nb_id}_{fpath.name}"
        dest_path = os.path.join(TARGET_NB_DIR, dest_filename)

        shutil.copy2(fpath, dest_path)

        # Parse source to confirm loadability
        pset = load_source(dest_path)

        if nb_id not in gt["notebooks"]:
            gt["notebooks"][nb_id] = {
                "file_name": dest_filename,
                "original_path": str(fpath),
                "is_notebook": pset.is_notebook,
                "labels": {
                    "missing_python_seed": None,
                    "missing_numpy_seed": None,
                    "missing_pytorch_seed": None,
                    "missing_random_state": None,
                    "unpinned_dependencies": None
                },
                "annotator_notes": "",
                "labeled": False
            }

    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

    print(f"Ingestion complete. Updated ground truth written to {GT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest real-world notebooks for reproducibility benchmark")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw .ipynb or .py files")
    args = parser.parse_args()
    ingest(args.input_dir)
