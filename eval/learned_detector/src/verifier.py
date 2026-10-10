"""
verifier.py - Execution-based oracle and AST-assisted verification for preprocessing leakage.
Independently verifies whether a generated ML script fitted its preprocessor on the full dataset (leakage)
or strictly on the training partition (clean).
Rejects or flags ambiguous, failing, or unverifiable examples.
"""

import ast
import traceback
from typing import Dict, Any, Optional, Tuple
import numpy as np


class LabelVerificationOracle:
    """
    Independent verification oracle.
    Executes the candidate pipeline in a restricted namespace, extracts fitted transformers,
    and quantifies whether fitted parameters (e.g. mean_, data_min_, statistics_) match the
    full dataset (label=1) or the training partition only (label=0).
    """

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def verify_code(self, code_str: str, expected_label: int, mutation_type: str) -> Dict[str, Any]:
        """
        Executes code, inspects runtime objects, and returns verification status and quantitative details.
        """
        outcome = {
            "verified": False,
            "verification_status": "unverified",
            "oracle_method": "none",
            "fitted_object_found": False,
            "stat_matches_full": False,
            "stat_matches_train": False,
            "reason": "",
            "details": {}
        }

        # 1. Syntax check
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            outcome["verification_status"] = "rejected"
            outcome["reason"] = f"Syntax error during AST parse: {e}"
            return outcome

        # 2. Execution check in isolated namespace
        exec_globals = {"__name__": "__main__"}
        try:
            exec(code_str, exec_globals)
        except Exception as e:
            outcome["verification_status"] = "rejected"
            outcome["reason"] = f"Runtime execution failed: {type(e).__name__}: {e}"
            outcome["details"]["traceback"] = traceback.format_exc(limit=2)
            return outcome

        # 3. Locate fitted transformer and feature arrays
        fitted_scaler = self._find_fitted_scaler(exec_globals)
        X_full, X_train, X_test = self._find_data_partitions(exec_globals)

        if fitted_scaler is not None and X_full is not None and X_train is not None:
            outcome["fitted_object_found"] = True
            oracle_res = self._check_scaler_statistics(fitted_scaler, X_full, X_train, X_test)
            outcome.update(oracle_res)
            if not outcome.get("stat_matches_full", False) and not outcome.get("stat_matches_train", False):
                ast_res = self._check_ast_order(tree, code_str)
                outcome.update(ast_res)
        else:
            # Fallback to AST order analysis if fitted continuous statistics cannot be extracted
            ast_res = self._check_ast_order(tree, code_str)
            outcome.update(ast_res)


        # 4. Final verification alignment with expected label
        if outcome["verification_status"] not in ["rejected", "inconclusive"]:
            if expected_label == 1:
                if outcome["stat_matches_full"] or outcome.get("ast_leakage_detected", False):
                    outcome["verified"] = True
                    outcome["verification_status"] = "verified_leakage"
                else:
                    outcome["verified"] = False
                    outcome["verification_status"] = "rejected"
                    outcome["reason"] = "Expected leakage (label 1), but runtime stats match train partition or are clean."
            elif expected_label == 0:
                if outcome["stat_matches_train"] or outcome.get("ast_clean_detected", False):
                    outcome["verified"] = True
                    outcome["verification_status"] = "verified_clean"
                else:
                    outcome["verified"] = False
                    outcome["verification_status"] = "rejected"
                    outcome["reason"] = "Expected clean (label 0), but runtime stats match full dataset or show leakage."

        return outcome

    def _find_fitted_scaler(self, scope: Dict[str, Any]) -> Optional[Any]:
        # Search direct variables
        for k, v in scope.items():
            if k.startswith("__") or isinstance(v, type):
                continue
            if hasattr(v, "mean_") or hasattr(v, "scale_") or hasattr(v, "data_min_") or hasattr(v, "statistics_") or hasattr(v, "center_"):
                return v
            # Look inside Pipeline instance
            if hasattr(v, "named_steps") and isinstance(getattr(v, "named_steps", None), dict):
                for step_name, step in v.named_steps.items():
                    if hasattr(step, "mean_") or hasattr(step, "scale_") or hasattr(step, "data_min_") or hasattr(step, "statistics_") or hasattr(step, "center_"):
                        return step
            # Look inside ColumnTransformer instance
            if hasattr(v, "transformers_") and isinstance(getattr(v, "transformers_", None), (list, tuple)):
                for trans_tuple in v.transformers_:
                    if len(trans_tuple) >= 2:
                        step = trans_tuple[1]
                        if hasattr(step, "mean_") or hasattr(step, "scale_") or hasattr(step, "data_min_") or hasattr(step, "statistics_"):
                            return step
            # Look inside CustomPreprocessor instance
            if hasattr(v, "scaler") and (hasattr(v.scaler, "mean_") or hasattr(v.scaler, "scale_")):
                return v.scaler
        return None


    def _find_data_partitions(self, scope: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        X_full, X_train, X_test = None, None, None

        # Detect full dataset
        for name in ["X", "X_mat", "X_arr", "df", "data"]:
            if name in scope:
                val = scope[name]
                arr = self._to_numpy_numeric(val)
                if arr is not None:
                    X_full = arr
                    break

        # Detect train partition
        for name in ["X_train", "X_tr", "X_train_raw", "X_train_s", "X_train_scaled"]:
            if name in scope:
                val = scope[name]
                arr = self._to_numpy_numeric(val)
                if arr is not None:
                    X_train = arr
                    break

        # Detect test partition
        for name in ["X_test", "X_te", "X_val"]:
            if name in scope:
                val = scope[name]
                arr = self._to_numpy_numeric(val)
                if arr is not None:
                    X_test = arr
                    break

        return X_full, X_train, X_test

    def _to_numpy_numeric(self, val: Any) -> Optional[np.ndarray]:
        try:
            if hasattr(val, "to_numpy"):
                arr = val.to_numpy()
            elif isinstance(val, np.ndarray):
                arr = val
            elif isinstance(val, (list, tuple)):
                arr = np.array(val)
            else:
                return None

            if np.issubdtype(arr.dtype, np.number):
                return arr.astype(float)
        except Exception:
            pass
        return None

    def _check_scaler_statistics(self, scaler: Any, X_full: np.ndarray, X_train: np.ndarray, X_test: Optional[np.ndarray]) -> Dict[str, Any]:
        res = {
            "oracle_method": "fitted_statistics",
            "stat_matches_full": False,
            "stat_matches_train": False,
            "stat_name": None,
            "details": {}
        }

        fitted_stat = None
        stat_name = None

        if hasattr(scaler, "mean_") and scaler.mean_ is not None:
            fitted_stat = np.array(scaler.mean_, dtype=float)
            stat_name = "mean_"
            full_stat = np.nanmean(X_full[:, :len(fitted_stat)], axis=0)
            train_stat = np.nanmean(X_train[:, :len(fitted_stat)], axis=0)
        elif hasattr(scaler, "data_min_") and scaler.data_min_ is not None:
            fitted_stat = np.array(scaler.data_min_, dtype=float)
            stat_name = "data_min_"
            full_stat = np.nanmin(X_full[:, :len(fitted_stat)], axis=0)
            train_stat = np.nanmin(X_train[:, :len(fitted_stat)], axis=0)
        elif hasattr(scaler, "center_") and scaler.center_ is not None:
            fitted_stat = np.array(scaler.center_, dtype=float)
            stat_name = "center_"
            full_stat = np.nanmedian(X_full[:, :len(fitted_stat)], axis=0)
            train_stat = np.nanmedian(X_train[:, :len(fitted_stat)], axis=0)
        elif hasattr(scaler, "statistics_") and scaler.statistics_ is not None:
            fitted_stat = np.array(scaler.statistics_, dtype=float)
            stat_name = "statistics_"
            full_stat = np.nanmean(X_full[:, :len(fitted_stat)], axis=0)
            train_stat = np.nanmean(X_train[:, :len(fitted_stat)], axis=0)

        if fitted_stat is None:
            res["oracle_method"] = "inconclusive_stats"
            return res

        if len(fitted_stat) != len(full_stat) or len(fitted_stat) != len(train_stat):
            res["oracle_method"] = "feature_dimension_shift"
            return res

        res["stat_name"] = stat_name
        diff_full = float(np.max(np.abs(fitted_stat - full_stat)))
        diff_train = float(np.max(np.abs(fitted_stat - train_stat)))

        res["details"]["diff_with_full"] = diff_full
        res["details"]["diff_with_train"] = diff_train

        # Check if train and full stats are indistinguishable
        stat_separation = float(np.max(np.abs(full_stat - train_stat)))
        if stat_separation < self.tolerance:
            res["verification_status"] = "inconclusive"
            res["reason"] = f"Train and full dataset statistics are too close ({stat_separation:.6f}) to distinguish."
            return res

        if diff_full < self.tolerance:
            res["stat_matches_full"] = True
        if diff_train < self.tolerance:
            res["stat_matches_train"] = True


        return res

    def _check_ast_order(self, tree: ast.AST, code_str: str) -> Dict[str, Any]:
        """
        AST structural inspection of order of operations between train_test_split and transformer.fit.
        """
        res = {
            "oracle_method": "ast_order_trace",
            "ast_leakage_detected": False,
            "ast_clean_detected": False,
            "split_line": None,
            "fit_lines": []
        }

        split_line = None
        fit_lines = []
        concat_before_fit = False

        for node in ast.walk(tree):
            # Detect train_test_split call
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name == "train_test_split":
                    split_line = getattr(node, "lineno", None)

                if func_name in ["concat", "vstack", "hstack"]:
                    # Check if concat call appears around fit
                    concat_before_fit = True

                # Detect fit or fit_transform calls
                if func_name in ["fit", "fit_transform"]:
                    # Ignore classifier fit (e.g. clf.fit(X_train, y_train))
                    # Check if object being fitted is a transformer/scaler/imputer
                    if isinstance(node.func, ast.Attribute):
                        caller = node.func.value
                        caller_name = getattr(caller, "id", "")
                        if any(term in caller_name.lower() for term in ["scaler", "imputer", "norm", "prep", "encoder", "col_trans", "poly", "trans"]):
                            fit_lines.append(getattr(node, "lineno", 0))

        res["split_line"] = split_line
        res["fit_lines"] = fit_lines

        if split_line is not None and fit_lines:
            earliest_fit = min(fit_lines)
            if earliest_fit < split_line or concat_before_fit:
                res["ast_leakage_detected"] = True
            else:
                res["ast_clean_detected"] = True
        elif "cross_val_score" in code_str and "Pipeline" in code_str:
            res["ast_clean_detected"] = True

        return res
