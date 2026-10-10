"""
features.py - Feature extraction for Python ML pipelines.
Extracts structural AST metrics, API call frequencies, and order-of-operation indicators.
"""

import ast
import re
from typing import List, Dict, Any
import numpy as np


FEATURE_NAMES = [
    "ast_num_calls",
    "ast_num_assigns",
    "ast_num_imports",
    "ast_num_attributes",
    "count_train_test_split",
    "count_fit",
    "count_fit_transform",
    "count_transform",
    "count_pipeline",
    "count_columntransformer",
    "count_scaler",
    "count_imputer",
    "count_concat",
    "fit_before_split_flag",
    "fit_arg_is_full_data_flag",
    "concat_before_fit_flag",
    "split_to_fit_line_delta"
]


def extract_pipeline_features(code_str: str) -> np.ndarray:
    """
    Extracts fixed 17-dimensional structural feature vector from Python code string.
    """
    feat = np.zeros(len(FEATURE_NAMES), dtype=float)

    try:
        tree = ast.parse(code_str)
    except Exception:
        # Fallback to regex if parsing fails
        tree = None

    if tree is not None:
        calls = 0
        assigns = 0
        imports = 0
        attrs = 0
        split_line = None
        earliest_fit_line = None
        fit_arg_is_full = 0.0
        concat_line = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                calls += 1
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name == "train_test_split":
                    split_line = getattr(node, "lineno", None)

                if func_name in ["concat", "vstack", "hstack"]:
                    concat_line = getattr(node, "lineno", None)

                if func_name in ["fit", "fit_transform"]:
                    lineno = getattr(node, "lineno", None)
                    if lineno is not None:
                        if earliest_fit_line is None or lineno < earliest_fit_line:
                            earliest_fit_line = lineno

                    # Check argument passed to fit
                    if node.args:
                        first_arg = node.args[0]
                        arg_id = getattr(first_arg, "id", "")
                        if arg_id in ["X", "X_mat", "X_arr", "df", "data", "X_combined"]:
                            fit_arg_is_full = 1.0

            elif isinstance(node, ast.Assign):
                assigns += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
            elif isinstance(node, ast.Attribute):
                attrs += 1

        feat[0] = calls
        feat[1] = assigns
        feat[2] = imports
        feat[3] = attrs

        # Order indicators
        if split_line is not None and earliest_fit_line is not None:
            feat[13] = 1.0 if earliest_fit_line < split_line else 0.0
            feat[16] = float(split_line - earliest_fit_line)
        feat[14] = fit_arg_is_full
        if concat_line is not None and earliest_fit_line is not None:
            feat[15] = 1.0 if concat_line < earliest_fit_line else 0.0

    # Lexical / API occurrences via fast regex
    feat[4] = len(re.findall(r"\btrain_test_split\b", code_str))
    feat[5] = len(re.findall(r"\.fit\(", code_str))
    feat[6] = len(re.findall(r"\.fit_transform\(", code_str))
    feat[7] = len(re.findall(r"\.transform\(", code_str))
    feat[8] = len(re.findall(r"\bPipeline\b", code_str))
    feat[9] = len(re.findall(r"\bColumnTransformer\b", code_str))
    feat[10] = len(re.findall(r"\b(StandardScaler|MinMaxScaler|RobustScaler|Normalizer|MaxAbsScaler)\b", code_str))
    feat[11] = len(re.findall(r"\b(SimpleImputer|KNNImputer)\b", code_str))
    feat[12] = len(re.findall(r"\b(concat|vstack|hstack)\b", code_str))

    return feat


def extract_features_matrix(code_list: List[str]) -> np.ndarray:
    return np.array([extract_pipeline_features(c) for c in code_list], dtype=float)
