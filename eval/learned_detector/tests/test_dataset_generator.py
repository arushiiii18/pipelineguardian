"""
Unit and regression tests for DatasetGenerator, base pipelines catalog, and manifest integrity.
"""

import os
import sys
import json
import pytest

# Ensure src is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from base_pipelines import BASE_PIPELINES_REGISTRY, get_base_pipelines
from generator import DatasetGenerator, HELD_OUT_BASE_IDS


def test_base_pipelines_diversity_and_count():
    pipelines = get_base_pipelines()
    assert len(pipelines) == 30, f"Expected exactly 30 base pipelines, found {len(pipelines)}"
    
    # Check diversity of styles and categories
    styles = {spec.style for spec in pipelines.values()}
    categories = {spec.category for spec in pipelines.values()}
    
    assert len(styles) >= 4, f"Insufficient style diversity: {styles}"
    assert len(categories) >= 6, f"Insufficient category diversity: {categories}"
    
    # Check that each base spec contains valid code template with seed placeholder
    for base_id, spec in pipelines.items():
        assert "{seed}" in spec.code_template, f"{base_id} missing {{seed}} placeholder"
        assert len(spec.description) > 10, f"{base_id} missing descriptive explanation"


def test_manifest_file_exists_and_schema_valid():
    manifest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "manifest.json")
    assert os.path.exists(manifest_path), "manifest.json does not exist"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    records = data["records"]
    assert len(records) >= 600, f"Expected at least 600 records, found {len(records)}"
    
    required_keys = {
        "sample_id", "base_id", "mutation_type", "label", "reason",
        "generator_version", "seed", "source_type", "source_reference",
        "input_hash", "split", "verification_status"
    }
    
    for r in records[:50]:  # check sample records
        missing = required_keys - set(r.keys())
        assert not missing, f"Record {r.get('sample_id')} missing keys: {missing}"
        assert r["label"] in [0, 1]
        assert r["verification_status"] in ["verified_clean", "verified_leakage", "rejected", "unverified", "inconclusive"]


def test_split_isolation_by_base_id():
    manifest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    records = data["records"]
    train_val_bases = {r["base_id"] for r in records if r["split"] == "train_val"}
    test_bases = {r["base_id"] for r in records if r["split"] == "test"}
    
    # Strictly zero overlap between train_val and held-out test base IDs
    overlap = train_val_bases.intersection(test_bases)
    assert len(overlap) == 0, f"Data leakage between base pipelines across splits: {overlap}"
    assert test_bases == HELD_OUT_BASE_IDS, f"Held out bases do not match expected set: {test_bases}"
    assert len(train_val_bases) == 24, f"Expected 24 train_val base pipelines, found {len(train_val_bases)}"


def test_generator_determinism():
    gen1 = DatasetGenerator(seed=42)
    spec = BASE_PIPELINES_REGISTRY["base_01"]
    mutations1 = gen1._create_mutations_for_base(spec)
    
    gen2 = DatasetGenerator(seed=42)
    mutations2 = gen2._create_mutations_for_base(spec)
    
    assert len(mutations1) == len(mutations2)
    for m1, m2 in zip(mutations1, mutations2):
        assert m1["code"] == m2["code"]
        assert m1["label"] == m2["label"]
        assert m1["mutation_type"] == m2["mutation_type"]
