"""
base_pipelines.py - Catalog of 30 distinct, clean base ML pipelines.
Provides diverse realistic preprocessing and modeling patterns for synthetic leakage generation.
Each pipeline is self-contained, deterministic, and executes cleanly without internet.
"""

from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import hashlib


@dataclass
class BasePipelineSpec:
    base_id: str
    name: str
    category: str
    style: str
    description: str
    code_template: str


# 30 Base Pipeline Templates
BASE_PIPELINES_REGISTRY: Dict[str, BasePipelineSpec] = {}


def _register(spec: BasePipelineSpec):
    BASE_PIPELINES_REGISTRY[spec.base_id] = spec


# 1. Base 01: Standard numeric classification with StandardScaler & LogisticRegression
_register(BasePipelineSpec(
    base_id="base_01",
    name="StandardScaler LogisticRegression Script",
    category="standard_numeric",
    style="script",
    description="Numeric tabular features, train_test_split, StandardScaler fit on train only, LogisticRegression.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Synthetic deterministic data
np.random.seed({seed})
X_arr = np.random.randn(120, 4)
y_arr = (X_arr[:, 0] + X_arr[:, 1] > 0).astype(int)
df = pd.DataFrame(X_arr, columns=['f0', 'f1', 'f2', 'f3'])
df['target'] = y_arr

X = df[['f0', 'f1', 'f2', 'f3']]
y = df['target']

# Clean split-first workflow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(random_state={seed})
clf.fit(X_train_scaled, y_train)
score = clf.score(X_test_scaled, y_test)
"""
))

# 2. Base 02: MinMaxScaler with RandomForestClassifier
_register(BasePipelineSpec(
    base_id="base_02",
    name="MinMaxScaler RandomForest Script",
    category="standard_numeric",
    style="script",
    description="Continuous features scaled into [0, 1] using MinMaxScaler on train split, RandomForest.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

np.random.seed({seed})
X = np.random.uniform(10.0, 50.0, size=(100, 5))
y = (X[:, 0] * 0.4 + X[:, 2] * 0.6 > 25.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state={seed})

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=10, random_state={seed})
model.fit(X_train_s, y_train)
acc = model.score(X_test_s, y_test)
"""
))

# 3. Base 03: Missing value SimpleImputer (median) + RobustScaler + GradientBoosting
_register(BasePipelineSpec(
    base_id="base_03",
    name="Imputer RobustScaler GradientBoosting",
    category="missing_data_robust",
    style="script",
    description="Data with synthetic NaNs, SimpleImputer with median and RobustScaler fit on train.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed({seed})
X_mat = np.random.randn(110, 4) * 2.5
mask = np.random.rand(*X_mat.shape) < 0.1
X_mat[mask] = np.nan
y = (np.nansum(X_mat, axis=1) > 0).astype(int)

df = pd.DataFrame(X_mat, columns=['a', 'b', 'c', 'd'])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state={seed})

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

gb = GradientBoostingClassifier(n_estimators=10, random_state={seed})
gb.fit(X_train_scaled, y_train)
score = gb.score(X_test_scaled, y_test)
"""
))

# 4. Base 04: scikit-learn Pipeline with StandardScaler and RidgeClassifier
_register(BasePipelineSpec(
    base_id="base_04",
    name="Sklearn Pipeline RidgeClassifier",
    category="pipeline_encapsulation",
    style="pipeline",
    description="Encapsulated scikit-learn Pipeline with StandardScaler and RidgeClassifier fit on X_train.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

np.random.seed({seed})
X = np.random.randn(100, 6)
y = (X[:, 0] - X[:, 1] + X[:, 2] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RidgeClassifier(alpha=1.0))
])

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)
"""
))

# 5. Base 05: ColumnTransformer with numeric scaling and OneHotEncoder
_register(BasePipelineSpec(
    base_id="base_05",
    name="ColumnTransformer Mixed Types",
    category="column_transformer",
    style="column_transformer",
    description="Heterogeneous dataset with numeric and categorical columns handled via ColumnTransformer.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

np.random.seed({seed})
n = 120
num_f = np.random.randn(n, 2)
cat_f = np.random.choice(['low', 'med', 'high'], size=(n, 1))
df = pd.DataFrame({'num1': num_f[:, 0], 'num2': num_f[:, 1], 'cat': cat_f.ravel()})
y = (df['num1'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state={seed})

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['num1', 'num2']),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['cat'])
])

X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

clf = LogisticRegression(random_state={seed})
clf.fit(X_train_trans, y_train)
res = clf.score(X_test_trans, y_test)
"""
))

# 6. Base 06: SimpleImputer (most_frequent) + OrdinalEncoder + DecisionTreeClassifier
_register(BasePipelineSpec(
    base_id="base_06",
    name="Imputer OrdinalEncoder DecisionTree",
    category="categorical_imputation",
    style="script",
    description="Categorical data with missing entries, most_frequent imputation and ordinal encoding.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

np.random.seed({seed})
n = 100
cities = np.random.choice(['Paris', 'Tokyo', 'London', np.nan], size=(n, 1), p=[0.3, 0.3, 0.3, 0.1])
sizes = np.random.choice(['S', 'M', 'L', np.nan], size=(n, 1), p=[0.3, 0.3, 0.3, 0.1])
y = np.random.choice([0, 1], size=n)

df = pd.DataFrame(np.hstack([cities, sizes]), columns=['city', 'size'])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state={seed})

imputer = SimpleImputer(strategy='most_frequent')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_enc = encoder.fit_transform(X_train_imp)
X_test_enc = encoder.transform(X_test_imp)

dt = DecisionTreeClassifier(max_depth=3, random_state={seed})
dt.fit(X_train_enc, y_train)
acc = dt.score(X_test_enc, y_test)
"""
))

# 7. Base 07: Function-wrapped preprocessing with Normalizer and SVC
_register(BasePipelineSpec(
    base_id="base_07",
    name="Function-wrapped Preprocessing SVC",
    category="modular_functions",
    style="function_wrapped",
    description="Preprocessing logic wrapped in dedicated clean helper functions.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

def load_data(seed):
    np.random.seed(seed)
    X = np.random.exponential(scale=2.0, size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 3.0).astype(int)
    return X, y

def fit_and_apply_scaling(train_data, test_data):
    scaler = Normalizer(norm='l2')
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    return scaled_train, scaled_test, scaler

X, y = load_data({seed})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

X_train_norm, X_test_norm, normalizer = fit_and_apply_scaling(X_train, X_test)

model = SVC(kernel='linear', random_state={seed})
model.fit(X_train_norm, y_train)
val_acc = model.score(X_test_norm, y_test)
"""
))

# 8. Base 08: Aliased imports with MaxAbsScaler and ExtraTreesClassifier
_register(BasePipelineSpec(
    base_id="base_08",
    name="Aliased Imports ExtraTrees",
    category="aliased_imports",
    style="aliased",
    description="Demonstrates alias conventions: skprep for preprocessing and skmodel for ensemble models.",
    code_template="""import numpy as np
import sklearn.preprocessing as skprep
import sklearn.model_selection as skms
import sklearn.ensemble as skens

np.random.seed({seed})
X_mat = np.random.randn(110, 5)
y_vec = (X_mat[:, 0] > 0.2).astype(int)

X_tr, X_te, y_tr, y_te = skms.train_test_split(X_mat, y_vec, test_size=0.25, random_state={seed})

scaler_inst = skprep.MaxAbsScaler()
X_tr_s = scaler_inst.fit_transform(X_tr)
X_te_s = scaler_inst.transform(X_te)

clf = skens.ExtraTreesClassifier(n_estimators=10, random_state={seed})
clf.fit(X_tr_s, y_tr)
score = clf.score(X_te_s, y_te)
"""
))

# 9. Base 09: Pure NumPy array workflows with KNeighborsClassifier
_register(BasePipelineSpec(
    base_id="base_09",
    name="Pure NumPy Array KNeighbors",
    category="numpy_operations",
    style="numpy_direct",
    description="Raw numpy operations and slicing with StandardScaler and KNeighborsClassifier.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

np.random.seed({seed})
data = np.random.normal(loc=5.0, scale=2.0, size=(100, 3))
labels = (data[:, 0] + data[:, 1] > 10.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state={seed})

norm = StandardScaler()
norm.fit(X_train)
X_train_p = norm.transform(X_train)
X_test_p = norm.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_p, y_train)
accuracy = knn.score(X_test_p, y_test)
"""
))

# 10. Base 10: Notebook-style cell markers with QuantileTransformer and GaussianNB
_register(BasePipelineSpec(
    base_id="base_10",
    name="Notebook-Style QuantileTransformer GaussianNB",
    category="notebook_representation",
    style="notebook_cells",
    description="Notebook cell representation with explicit %% cell delimiters and QuantileTransformer.",
    code_template="""# %% [markdown]
# # Pipeline 10: Non-linear quantile transformation and Naive Bayes

# %% [code]
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.naive_bayes import GaussianNB

# %% [code]
np.random.seed({seed})
X = np.random.gamma(shape=2.0, scale=2.0, size=(120, 3))
y = (X[:, 0] > 4.0).astype(int)

# %% [code]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

# %% [code]
qt = QuantileTransformer(n_quantiles=50, random_state={seed})
X_train_qt = qt.fit_transform(X_train)
X_test_qt = qt.transform(X_test)

# %% [code]
nb = GaussianNB()
nb.fit(X_train_qt, y_train)
acc = nb.score(X_test_qt, y_test)
"""
))

# 11. Base 11: Outlier clipping + RobustScaler + SGDClassifier
_register(BasePipelineSpec(
    base_id="base_11",
    name="Outlier Clipping RobustScaler SGD",
    category="custom_preprocessing",
    style="script",
    description="Custom pandas percentile clipping followed by RobustScaler and SGDClassifier.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier

np.random.seed({seed})
X_raw = np.random.randn(100, 4) * 3.0
X_raw[0, 0] = 50.0  # outlier
y = (X_raw[:, 1] > 0).astype(int)

df = pd.DataFrame(X_raw, columns=['c1', 'c2', 'c3', 'c4'])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state={seed})

# Compute limits only from training set
lower_lim = X_train.quantile(0.05)
upper_lim = X_train.quantile(0.95)

X_train_clipped = X_train.clip(lower=lower_lim, upper=upper_lim, axis=1)
X_test_clipped = X_test.clip(lower=lower_lim, upper=upper_lim, axis=1)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train_clipped)
X_test_s = scaler.transform(X_test_clipped)

sgd = SGDClassifier(loss='log_loss', random_state={seed})
sgd.fit(X_train_s, y_train)
score = sgd.score(X_test_s, y_test)
"""
))

# 12. Base 12: PowerTransformer (Yeo-Johnson) with LinearSVC
_register(BasePipelineSpec(
    base_id="base_12",
    name="PowerTransformer LinearSVC",
    category="transformations",
    style="script",
    description="Box-Cox / Yeo-Johnson power transform applied to skewed positive data.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import LinearSVC

np.random.seed({seed})
X = np.random.exponential(scale=1.5, size=(100, 4))
y = (X[:, 0] + X[:, 2] > 3.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

pt = PowerTransformer(method='yeo-johnson')
X_train_pt = pt.fit_transform(X_train)
X_test_pt = pt.transform(X_test)

svc = LinearSVC(dual='auto', random_state={seed})
svc.fit(X_train_pt, y_train)
acc = svc.score(X_test_pt, y_test)
"""
))

# 13. Base 13: SimpleImputer (mean) + MaxAbsScaler + AdaBoostClassifier
_register(BasePipelineSpec(
    base_id="base_13",
    name="SimpleImputer MaxAbsScaler AdaBoost",
    category="missing_data_robust",
    style="script",
    description="Missing continuous features imputed with mean, scaled with MaxAbsScaler, AdaBoost.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import AdaBoostClassifier

np.random.seed({seed})
X = np.random.randn(110, 4)
X[X < -1.0] = np.nan
y = (np.nansum(X, axis=1) > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state={seed})

imp = SimpleImputer(strategy='mean')
X_tr_imp = imp.fit_transform(X_train)
X_te_imp = imp.transform(X_test)

scaler = MaxAbsScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_te_scaled = scaler.transform(X_te_imp)

model = AdaBoostClassifier(n_estimators=10, random_state={seed})
model.fit(X_tr_scaled, y_train)
res = model.score(X_te_scaled, y_test)
"""
))

# 14. Base 14: PolynomialFeatures + StandardScaler + LogisticRegression
_register(BasePipelineSpec(
    base_id="base_14",
    name="PolynomialFeatures StandardScaler LogReg",
    category="feature_generation",
    style="script",
    description="Interaction and polynomial feature expansion followed by StandardScaler.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed({seed})
X = np.random.randn(100, 3)
y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_poly)
X_test_s = scaler.transform(X_test_poly)

clf = LogisticRegression(random_state={seed})
clf.fit(X_train_s, y_train)
score = clf.score(X_test_s, y_test)
"""
))

# 15. Base 15: FunctionTransformer (log1p) + StandardScaler + MLPClassifier
_register(BasePipelineSpec(
    base_id="base_15",
    name="FunctionTransformer StandardScaler MLP",
    category="transformations",
    style="script",
    description="Log transform on right-skewed data with FunctionTransformer, then StandardScaler.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.neural_network import MLPClassifier

np.random.seed({seed})
X = np.abs(np.random.randn(100, 4) * 5.0)
y = (X[:, 0] > 3.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

log_trans = FunctionTransformer(np.log1p, validate=True)
X_train_log = log_trans.fit_transform(X_train)
X_test_log = log_trans.transform(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_log)
X_test_s = scaler.transform(X_test_log)

mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=50, random_state={seed})
mlp.fit(X_train_s, y_train)
score = mlp.score(X_test_s, y_test)
"""
))

# 16. Base 16: KNNImputer + MinMaxScaler + RandomForestClassifier
_register(BasePipelineSpec(
    base_id="base_16",
    name="KNNImputer MinMaxScaler RandomForest",
    category="missing_data_robust",
    style="script",
    description="Multi-variate nearest-neighbor imputation followed by MinMaxScaler.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

np.random.seed({seed})
X = np.random.randn(100, 4)
mask = np.random.rand(*X.shape) < 0.08
X[mask] = np.nan
y = (np.nansum(X, axis=1) > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

knn_imp = KNNImputer(n_neighbors=3)
X_tr_imp = knn_imp.fit_transform(X_train)
X_te_imp = knn_imp.transform(X_test)

scaler = MinMaxScaler()
X_tr_s = scaler.fit_transform(X_tr_imp)
X_te_s = scaler.transform(X_te_imp)

rf = RandomForestClassifier(n_estimators=10, random_state={seed})
rf.fit(X_tr_s, y_train)
acc = rf.score(X_te_s, y_test)
"""
))

# 17. Base 17: SelectKBest (f_classif) + StandardScaler + SVC
_register(BasePipelineSpec(
    base_id="base_17",
    name="SelectKBest StandardScaler SVC",
    category="feature_selection",
    style="script",
    description="Supervised feature selection via SelectKBest fitted strictly on training fold.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

np.random.seed({seed})
X = np.random.randn(100, 8)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state={seed})

selector = SelectKBest(score_func=f_classif, k=4)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_sel)
X_test_s = scaler.transform(X_test_sel)

svc = SVC(kernel='rbf', random_state={seed})
svc.fit(X_train_s, y_train)
acc = svc.score(X_test_s, y_test)
"""
))

# 18. Base 18: Disparate column subsets scaled separately with StandardScaler and MinMaxScaler
_register(BasePipelineSpec(
    base_id="base_18",
    name="Sub-dataframe Split Scalers",
    category="column_transformer",
    style="script",
    description="Different feature blocks scaled with separate scalers independently on train data.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

np.random.seed({seed})
df = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
y = (df['a'] + df['c'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state={seed})

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

tr_ab = scaler_std.fit_transform(X_train[['a', 'b']])
te_ab = scaler_std.transform(X_test[['a', 'b']])

tr_cd = scaler_minmax.fit_transform(X_train[['c', 'd']])
te_cd = scaler_minmax.transform(X_test[['c', 'd']])

X_tr_comb = np.hstack([tr_ab, tr_cd])
X_te_comb = np.hstack([te_ab, te_cd])

clf = LogisticRegression(random_state={seed})
clf.fit(X_tr_comb, y_train)
score = clf.score(X_te_comb, y_test)
"""
))

# 19. Base 19: DictVectorizer sparse data + LogisticRegression
_register(BasePipelineSpec(
    base_id="base_19",
    name="DictVectorizer Tabular Records",
    category="encoding",
    style="script",
    description="Dictionary record conversion into sparse matrices via DictVectorizer on train split.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

np.random.seed({seed})
records = [
    {{'age': np.random.randint(20, 60), 'tier': np.random.choice(['silver', 'gold', 'platinum'])}}
    for _ in range(100)
]
y = np.array([1 if r['age'] > 35 else 0 for r in records])

train_idx, test_idx = train_test_split(np.arange(len(records)), test_size=0.25, random_state={seed})
rec_train = [records[i] for i in train_idx]
rec_test = [records[i] for i in test_idx]
y_train, y_test = y[train_idx], y[test_idx]

vec = DictVectorizer(sparse=False)
X_train_vec = vec.fit_transform(rec_train)
X_test_vec = vec.transform(rec_test)

clf = LogisticRegression(random_state={seed})
clf.fit(X_train_vec, y_train)
score = clf.score(X_test_vec, y_test)
"""
))

# 20. Base 20: K-Fold cross validation with StandardScaler inside fold
_register(BasePipelineSpec(
    base_id="base_20",
    name="KFold Loop with In-Fold Scaling",
    category="cross_validation",
    style="script",
    description="Manual K-Fold loop demonstrating proper in-fold fitting of StandardScaler.",
    code_template="""import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

np.random.seed({seed})
X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)

kf = KFold(n_splits=3, shuffle=True, random_state={seed})
scores = []

for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    model = RidgeClassifier()
    model.fit(X_tr_s, y_tr)
    scores.append(model.score(X_val_s, y_val))

mean_acc = float(np.mean(scores))
"""
))

# 21. Base 21: Custom class OOP Preprocessor wrapping StandardScaler
_register(BasePipelineSpec(
    base_id="base_21",
    name="OOP Custom Preprocessor Class",
    category="object_oriented",
    style="oop",
    description="Clean object-oriented abstraction encapsulating fit and transform on training data.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class CustomPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

np.random.seed({seed})
X = np.random.randn(100, 4)
y = (X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

prep = CustomPreprocessor()
X_tr_s = prep.fit_transform(X_train)
X_te_s = prep.transform(X_test)

clf = LogisticRegression(random_state={seed})
clf.fit(X_tr_s, y_train)
res = clf.score(X_te_s, y_test)
"""
))

# 22. Base 22: HistGradientBoostingClassifier with QuantileTransformer
_register(BasePipelineSpec(
    base_id="base_22",
    name="HistGradientBoosting QuantileTransformer",
    category="gradient_boosting",
    style="script",
    description="Tree-based boosting model evaluated after quantile transformation fitted on train fold.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

np.random.seed({seed})
X = np.random.exponential(scale=3.0, size=(120, 5))
y = (X[:, 0] + X[:, 1] > 4.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state={seed})

qt = QuantileTransformer(n_quantiles=40, random_state={seed})
X_tr_qt = qt.fit_transform(X_train)
X_te_qt = qt.transform(X_test)

hgb = HistGradientBoostingClassifier(max_iter=15, random_state={seed})
hgb.fit(X_tr_qt, y_train)
score = hgb.score(X_te_qt, y_test)
"""
))

# 23. Base 23: StratifiedKFold cross_val_score with Pipeline & LinearDiscriminantAnalysis
_register(BasePipelineSpec(
    base_id="base_23",
    name="StratifiedKFold Pipeline LDA",
    category="pipeline_encapsulation",
    style="pipeline",
    description="cross_val_score combined with Pipeline to guarantee split integrity across folds.",
    code_template="""import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed({seed})
X = np.random.randn(100, 4)
y = (X[:, 0] * 2 + X[:, 2] > 0).astype(int)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state={seed})
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis())
])

scores = cross_val_score(pipe, X, y, cv=cv)
mean_score = float(np.mean(scores))
"""
))

# 24. Base 24: Cyclic Feature Engineering + MinMaxScaler + LogisticRegression
_register(BasePipelineSpec(
    base_id="base_24",
    name="Cyclic Feature Engineering MinMaxScaler",
    category="feature_generation",
    style="script",
    description="Cyclic sin/cos time feature engineering followed by train-only MinMaxScaler.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

np.random.seed({seed})
hours = np.random.randint(0, 24, size=100)
df = pd.DataFrame({{'hour': hours, 'sin_hour': np.sin(2 * np.pi * hours / 24), 'cos_hour': np.cos(2 * np.pi * hours / 24)}})
y = (hours > 12).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state={seed})

scaler = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_te_scaled = scaler.transform(X_test)

clf = LogisticRegression(random_state={seed})
clf.fit(X_tr_scaled, y_train)
res = clf.score(X_te_scaled, y_test)
"""
))

# 25. Base 25: Imbalance handling with RobustScaler and class_weight Balanced LogReg
_register(BasePipelineSpec(
    base_id="base_25",
    name="Class-weighted Balanced LogReg RobustScaler",
    category="class_imbalance",
    style="script",
    description="Imbalanced synthetic classes with RobustScaler fitted strictly on training data.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

np.random.seed({seed})
X = np.random.randn(120, 4)
# 85% negative, 15% positive
y = (np.random.rand(120) < 0.15).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed}, stratify=y)

scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

clf = LogisticRegression(class_weight='balanced', random_state={seed})
clf.fit(X_tr_s, y_train)
score = clf.score(X_te_s, y_test)
"""
))

# 26. Base 26: Multi-class classification with StandardScaler and LogisticRegression multinomial
_register(BasePipelineSpec(
    base_id="base_26",
    name="Multiclass Multinomial LogReg StandardScaler",
    category="multiclass",
    style="script",
    description="3-class target with StandardScaler fitted exclusively on training instances.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed({seed})
X = np.random.randn(120, 4)
logits = X[:, :3]
y = np.argmax(logits, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(multi_class='multinomial', random_state={seed})
clf.fit(X_train_scaled, y_train)
acc = clf.score(X_test_scaled, y_test)
"""
))

# 27. Base 27: Chained Transformers SimpleImputer -> PowerTransformer -> StandardScaler
_register(BasePipelineSpec(
    base_id="base_27",
    name="Chained 3-Step Transformer SVC",
    category="transformations",
    style="script",
    description="Triple sequential preprocessing steps fitted sequentially on training split.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC

np.random.seed({seed})
X = np.abs(np.random.randn(100, 3) * 2.0)
X[X < 0.5] = np.nan
y = (np.nansum(X, axis=1) > 2.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={seed})

imp = SimpleImputer(strategy='mean')
X_tr_1 = imp.fit_transform(X_train)
X_te_1 = imp.transform(X_test)

pt = PowerTransformer(method='yeo-johnson')
X_tr_2 = pt.fit_transform(X_tr_1)
X_te_2 = pt.transform(X_te_1)

scaler = StandardScaler()
X_tr_3 = scaler.fit_transform(X_tr_2)
X_te_3 = scaler.transform(X_te_2)

model = SVC(kernel='linear', random_state={seed})
model.fit(X_tr_3, y_train)
acc = model.score(X_te_3, y_test)
"""
))

# 28. Base 28: NumPy Structured Array conversion + StandardScaler + RandomForest
_register(BasePipelineSpec(
    base_id="base_28",
    name="Structured Array Records RandomForest",
    category="numpy_operations",
    style="script",
    description="Custom structured numpy dtype converted to 2D feature matrix with train-only scaling.",
    code_template="""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

np.random.seed({seed})
n = 100
struct_data = np.zeros(n, dtype=[('sensor_a', 'f8'), ('sensor_b', 'f8'), ('label', 'i4')])
struct_data['sensor_a'] = np.random.randn(n) * 10
struct_data['sensor_b'] = np.random.randn(n) * 2
struct_data['label'] = (struct_data['sensor_a'] > 0).astype(int)

X = np.column_stack([struct_data['sensor_a'], struct_data['sensor_b']])
y = struct_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state={seed})

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_te_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=10, random_state={seed})
rf.fit(X_tr_scaled, y_train)
res = rf.score(X_te_scaled, y_test)
"""
))

# 29. Base 29: Categorical frequency target encoding simulation + MinMaxScaler + DecisionTree
_register(BasePipelineSpec(
    base_id="base_29",
    name="Frequency Map Encoding MinMaxScaler DecisionTree",
    category="encoding",
    style="script",
    description="Category frequency map computed strictly from X_train, then scaled with MinMaxScaler.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

np.random.seed({seed})
cats = np.random.choice(['catA', 'catB', 'catC', 'catD'], size=120)
nums = np.random.randn(120)
df = pd.DataFrame({{'category': cats, 'num_val': nums}})
y = (nums > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state={seed})

# Compute frequencies strictly on train set
freq_map = X_train['category'].value_counts(normalize=True).to_dict()
X_tr_encoded = X_train.copy()
X_te_encoded = X_test.copy()
X_tr_encoded['cat_freq'] = X_tr_encoded['category'].map(freq_map).fillna(0.0)
X_te_encoded['cat_freq'] = X_te_encoded['category'].map(freq_map).fillna(0.0)

feat_cols = ['num_val', 'cat_freq']
scaler = MinMaxScaler()
X_tr_s = scaler.fit_transform(X_tr_encoded[feat_cols])
X_te_s = scaler.transform(X_te_encoded[feat_cols])

dt = DecisionTreeClassifier(max_depth=3, random_state={seed})
dt.fit(X_tr_s, y_train)
acc = dt.score(X_te_s, y_test)
"""
))

# 30. Base 30: ColumnTransformer with remainder='passthrough' & HistGradientBoosting
_register(BasePipelineSpec(
    base_id="base_30",
    name="Full ColumnTransformer Passthrough HistGradientBoosting",
    category="column_transformer",
    style="column_transformer",
    description="Production-grade ColumnTransformer with passthrough remainder, Pipeline, HistGB.",
    code_template="""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

np.random.seed({seed})
n = 110
df = pd.DataFrame({{
    'num_scale': np.random.randn(n) * 10,
    'num_raw': np.random.uniform(0, 1, n),
    'flag': np.random.choice([0, 1], n)
}})
y = (df['num_scale'] + df['num_raw'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state={seed})

col_trans = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), ['num_scale'])
    ],
    remainder='passthrough'
)

pipe = Pipeline([
    ('prep', col_trans),
    ('model', HistGradientBoostingClassifier(max_iter=15, random_state={seed}))
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
"""
))


def get_base_pipelines() -> Dict[str, BasePipelineSpec]:
    return dict(BASE_PIPELINES_REGISTRY)
