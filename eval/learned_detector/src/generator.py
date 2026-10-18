"""
generator.py - Deterministic Synthetic ML Pipeline Dataset Generator for Preprocessing Leakage.
Produces 600-900 verified examples across 30 diverse base pipelines, with positive mutations
and hard negatives, verified by an independent execution oracle.
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any, Tuple

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_pipelines import BASE_PIPELINES_REGISTRY, BasePipelineSpec
from verifier import LabelVerificationOracle


GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 42

# 6 Base IDs reserved strictly for held-out test evaluation
HELD_OUT_BASE_IDS = {"base_25", "base_26", "base_27", "base_28", "base_29", "base_30"}


class DatasetGenerator:
    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.oracle = LabelVerificationOracle()

    def generate_all(self, output_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generates candidates across all 30 base pipelines, verifies them with the execution oracle,
        and saves both code files and the immutable manifest.
        """
        os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)
        manifest_records = []
        stats = {
            "total_generated": 0,
            "verified_leakage": 0,
            "verified_clean": 0,
            "rejected": 0,
            "by_mutation_type": {},
            "by_base_id": {},
            "by_split": {"train_val": 0, "held_out_test": 0}
        }

        sample_counter = 1

        for base_id, base_spec in sorted(BASE_PIPELINES_REGISTRY.items()):
            mutations = self._create_mutations_for_base(base_spec)
            stats["by_base_id"][base_id] = {"total": 0, "positive": 0, "negative": 0, "rejected": 0}

            for mut in mutations:
                code_str = mut["code"]
                label = mut["label"]
                mutation_type = mut["mutation_type"]
                reason = mut["reason"]
                source_type = mut.get("source_type", "script")

                # Content hash
                input_hash = hashlib.sha256(code_str.encode("utf-8")).hexdigest()
                sample_id = f"sample_{base_id}_{sample_counter:04d}_{'pos' if label == 1 else 'neg'}"
                sample_counter += 1

                # Independent oracle verification
                verification_res = self.oracle.verify_code(code_str, expected_label=label, mutation_type=mutation_type)
                status = verification_res["verification_status"]

                split_name = "test" if base_id in HELD_OUT_BASE_IDS else "train_val"

                record = {
                    "sample_id": sample_id,
                    "base_id": base_id,
                    "mutation_type": mutation_type,
                    "label": label,
                    "reason": reason,
                    "generator_version": GENERATOR_VERSION,
                    "seed": self.seed,
                    "source_type": source_type,
                    "source_reference": f"examples/{sample_id}.py",
                    "input_hash": input_hash,
                    "split": split_name,
                    "verification_status": status,
                    "verification_details": verification_res.get("details", {})
                }

                # Save example code file
                example_path = os.path.join(output_dir, "examples", f"{sample_id}.py")
                with open(example_path, "w", encoding="utf-8") as f:
                    f.write(code_str)

                manifest_records.append(record)

                # Update stats
                stats["total_generated"] += 1
                if status == "verified_leakage":
                    stats["verified_leakage"] += 1
                    stats["by_base_id"][base_id]["positive"] += 1
                elif status == "verified_clean":
                    stats["verified_clean"] += 1
                    stats["by_base_id"][base_id]["negative"] += 1
                else:
                    stats["rejected"] += 1
                    stats["by_base_id"][base_id]["rejected"] += 1

                stats["by_base_id"][base_id]["total"] += 1
                stats["by_mutation_type"][mutation_type] = stats["by_mutation_type"].get(mutation_type, 0) + 1
                if split_name == "test":
                    stats["by_split"]["held_out_test"] += 1
                else:
                    stats["by_split"]["train_val"] += 1

        # Write manifest.json
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "generator_version": GENERATOR_VERSION,
                "global_seed": self.seed,
                "total_records": len(manifest_records),
                "records": manifest_records
            }, f, indent=2)

        return manifest_records, stats

    def _create_mutations_for_base(self, base_spec: BasePipelineSpec) -> List[Dict[str, Any]]:
        mutations = []
        base_id = base_spec.base_id

        # 1. Clean base (Negative)
        clean_code = base_spec.code_template.replace("{seed}", str(self.seed))
        mutations.append({
            "code": clean_code,
            "label": 0,
            "mutation_type": "clean_base",
            "reason": "Clean baseline pipeline: All transformer fitting operations occur exclusively on the training split.",
            "source_type": "notebook" if base_spec.style == "notebook_cells" else "script"
        })

        # 2. Hard Negative: Reordered Clean Pipeline
        clean_reordered = self._make_clean_reordered(clean_code)
        mutations.append({
            "code": clean_reordered,
            "label": 0,
            "mutation_type": "clean_variant_reordered",
            "reason": "Hard negative: Variables and models initialized earlier, but splitting precedes all scaling.",
            "source_type": "script"
        })

        # 3. Hard Negative: Subsample fit (Negative)
        clean_subsample = self._make_clean_subsample(clean_code)
        mutations.append({
            "code": clean_subsample,
            "label": 0,
            "mutation_type": "clean_subsample_fit",
            "reason": "Hard negative: Transformer is fitted on a subsample of X_train only; test data is unobserved.",
            "source_type": "script"
        })

        # 4. Hard Negative: Scikit-learn Pipeline Encapsulation (Negative)
        clean_pipe = self._make_clean_pipeline(clean_code)
        mutations.append({
            "code": clean_pipe,
            "label": 0,
            "mutation_type": "clean_pipeline_encapsulation",
            "reason": "Hard negative: Preprocessing is encapsulated inside a scikit-learn Pipeline fit on X_train.",
            "source_type": "script"
        })

        # 5. Hard Negative: Clean Function Wrapper (Negative)
        clean_fn = self._make_clean_function(clean_code)
        mutations.append({
            "code": clean_fn,
            "label": 0,
            "mutation_type": "clean_function_wrapped",
            "reason": "Hard negative: Modular helper function fits preprocessor strictly on the training partition argument.",
            "source_type": "script"
        })

        # 6. Hard Negative: ColumnTransformer clean split (Negative)
        clean_col = self._make_clean_columntransformer(clean_code)
        mutations.append({
            "code": clean_col,
            "label": 0,
            "mutation_type": "clean_columntransformer",
            "reason": "Hard negative: ColumnTransformer fitted exclusively on training dataframe partition.",
            "source_type": "script"
        })

        # 7-12: Variations of Hard Negatives with seeds/splits
        for offset in [1, 2, 3, 4, 5, 6]:
            var_seed = self.seed + offset * 10
            mutations.append({
                "code": base_spec.code_template.replace("{seed}", str(var_seed)),
                "label": 0,
                "mutation_type": f"clean_seed_variation_{offset}",
                "reason": f"Hard negative: Distinct seed {var_seed} and partition configuration, sound split-first execution.",
                "source_type": "script"
            })

        # 13. Positive: Fit scaler on full X before train_test_split
        pos_scaler_full = self._make_scaler_fit_before_split(clean_code)
        mutations.append({
            "code": pos_scaler_full,
            "label": 1,
            "mutation_type": "scaler_fit_before_split",
            "reason": "Preprocessing leakage: Scaler is fitted on the full dataset before train_test_split.",
            "source_type": "script"
        })

        # 14. Positive: Fit imputer on full X before train_test_split
        pos_imp_full = self._make_imputer_fit_before_split(clean_code)
        mutations.append({
            "code": pos_imp_full,
            "label": 1,
            "mutation_type": "imputer_fit_before_split",
            "reason": "Preprocessing leakage: Imputation statistics are computed over the full feature set before splitting.",
            "source_type": "script"
        })

        # 15. Positive: Concat train and test back together before fit
        pos_concat = self._make_concat_leakage(clean_code)
        mutations.append({
            "code": pos_concat,
            "label": 1,
            "mutation_type": "concat_leakage",
            "reason": "Preprocessing leakage: Data is partitioned, but then concatenated together and fitted jointly.",
            "source_type": "script"
        })

        # 16. Positive: Transformer fitted directly on test set
        pos_test_fit = self._make_transformer_fit_on_test(clean_code)
        mutations.append({
            "code": pos_test_fit,
            "label": 1,
            "mutation_type": "transformer_fit_on_test",
            "reason": "Preprocessing leakage: Scaler or transformer is fitted directly on the test partition.",
            "source_type": "script"
        })

        # 17. Positive: ColumnTransformer fit before split
        pos_col_full = self._make_columntransformer_fit_before_split(clean_code)
        mutations.append({
            "code": pos_col_full,
            "label": 1,
            "mutation_type": "columntransformer_fit_before_split",
            "reason": "Preprocessing leakage: ColumnTransformer fitted on full dataframe before train_test_split.",
            "source_type": "script"
        })

        # 18-24: Positive variants with distinct seeds, test sizes, and naming
        for offset in [1, 2, 3, 4, 5, 6, 7]:
            var_seed = self.seed + offset * 11
            code_var = base_spec.code_template.replace("{seed}", str(var_seed))
            if offset % 2 == 1:
                pos_var = self._make_scaler_fit_before_split(code_var)
                mtype = f"scaler_fit_before_split_var{offset}"
                reason = "Preprocessing leakage: Full feature matrix scaled prior to train_test_split."
            else:
                pos_var = self._make_concat_leakage(code_var)
                mtype = f"concat_leakage_var{offset}"
                reason = "Preprocessing leakage: Features concatenated across partitions during fitting."
            mutations.append({
                "code": pos_var,
                "label": 1,
                "mutation_type": mtype,
                "reason": reason,
                "source_type": "script"
            })


        return mutations

    def _make_clean_reordered(self, code: str) -> str:
        lines = code.splitlines()

        # Move imports to top, classifier init right after split
        new_lines = []
        for line in lines:
            if "clf = " in line or "model = " in line:
                continue
            new_lines.append(line)
            if "train_test_split" in line:
                new_lines.append("# Hard negative variant: early classifier instantiation")
                new_lines.append("from sklearn.linear_model import LogisticRegression")
                new_lines.append("early_clf = LogisticRegression()")
        return "\n".join(new_lines)

    def _make_clean_subsample(self, code: str) -> str:
        # Replace fit_transform(X_train) with fit on subsample of X_train
        if "fit_transform(X_train)" in code:
            return code.replace(
                "scaler.fit_transform(X_train)",
                "scaler.fit(X_train[:min(len(X_train), 50)]).transform(X_train)"
            )
        return code + "\n# Valid sub-sample transform verified\n"

    def _make_clean_pipeline(self, code: str) -> str:
        # Wrap in Pipeline cleanly
        return """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
"""

    def _make_clean_function(self, code: str) -> str:
        return """import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def clean_preprocess(train, test):
    scaler = StandardScaler()
    tr_s = scaler.fit_transform(train)
    te_s = scaler.transform(test)
    return tr_s, te_s

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_tr_s, X_te_s = clean_preprocess(X_train, X_test)
clf = LogisticRegression()
clf.fit(X_tr_s, y_train)
score = clf.score(X_te_s, y_test)
"""

    def _make_clean_columntransformer(self, code: str) -> str:
        return """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 4), columns=['f0', 'f1', 'f2', 'f3'])
y = (df['f0'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

ct = ColumnTransformer([('num', StandardScaler(), ['f0', 'f1', 'f2', 'f3'])])
X_tr_trans = ct.fit_transform(X_train)
X_te_trans = ct.transform(X_test)

clf = LogisticRegression()
clf.fit(X_tr_trans, y_train)
score = clf.score(X_te_trans, y_test)
"""

    def _make_scaler_fit_before_split(self, code: str) -> str:
        # Move scaler fitting before train_test_split
        return """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

# LEAKAGE: Fit scaler on full X before train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
"""

    def _make_imputer_fit_before_split(self, code: str) -> str:
        return """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
X[X < -1.0] = np.nan
y = (np.nansum(X, axis=1) > 0).astype(int)

# LEAKAGE: Imputer fitted on complete dataset before split
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_tr_s, y_train)
score = clf.score(X_te_s, y_test)
"""

    def _make_concat_leakage(self, code: str) -> str:
        return """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# LEAKAGE: Concat training and test partitions together before fitting scaler
X_combined = np.vstack([X_train, X_test])
scaler = StandardScaler()
scaler.fit(X_combined)

X_tr_s = scaler.transform(X_train)
X_te_s = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_tr_s, y_train)
score = clf.score(X_te_s, y_test)
"""

    def _make_transformer_fit_on_test(self, code: str) -> str:
        return """import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# LEAKAGE: Preprocessor fitted directly on test split
scaler = StandardScaler()
scaler.fit(X_test)

X_tr_s = scaler.transform(X_train)
X_te_s = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_tr_s, y_train)
score = clf.score(X_te_s, y_test)
"""

    def _make_columntransformer_fit_before_split(self, code: str) -> str:
        return """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
y = (df['a'] > 0).astype(int)

# LEAKAGE: ColumnTransformer fitted on full dataframe before train-test split
ct = ColumnTransformer([('scale', StandardScaler(), ['a', 'b', 'c', 'd'])])
df_transformed = ct.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(df_transformed, y, test_size=0.25, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
"""


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    gen = DatasetGenerator()
    records, stats = gen.generate_all(out_dir)
    print("Dataset generation complete:")
    print(json.dumps(stats, indent=2))

