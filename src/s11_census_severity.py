"""Stage 11 - Severity measurement. Quantifies how much each defect found in
Stage 10 actually distorts measured model performance: the effect of
de-duplicating the canonical training data, the effect of Mirror 1's silent
feature discretization, and whether a standard pipeline silently accepts the
label-encoding divergence found in Stage 10.

Three models (logistic regression, Random Forest, Gradient Boosting) bound a
wider part of the model-flexibility spectrum than a linear/tree-ensemble pair
alone. Where a train/test partition is not fixed by the canonical archive's
own official split, results are replicated across five seeds and reported as
mean +/- standard deviation, rather than a single point estimate."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from auditlib import save_json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'adult_census_income')
MIRR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mirrors', 'adult_census_income')

COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'class']
NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CAT = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
       'race', 'sex', 'native-country']
SEEDS = [42, 43, 44, 45, 46]
MODELS = ['logreg', 'rf', 'gb']

print("=" * 72); print(" STAGE 11: CENSUS-INCOME SEVERITY MEASUREMENT"); print("=" * 72)

def load_canonical():
    train = pd.read_csv(os.path.join(RAW, 'adult.data'), header=None, names=COLS,
                         sep=r',\s*', engine='python', na_values='?').dropna(how='all')
    test = pd.read_csv(os.path.join(RAW, 'adult.test'), header=None, names=COLS,
                        sep=r',\s*', engine='python', na_values='?', skiprows=1).dropna(how='all')
    test = test.copy(); test['class'] = test['class'].str.rstrip('.')
    return train, test

def _models(seed):
    return [('logreg', LogisticRegression(max_iter=1000, random_state=seed)),
            ('rf', RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(random_state=seed))]

def pipeline(model, num_cols, cat_cols):
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                           ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])
    return Pipeline([('pre', pre), ('clf', model)])

def fit_eval(Xtr, ytr, Xte, yte, seed, num_cols=NUM, cat_cols=CAT):
    out = {}
    for name, model in _models(seed):
        pipe = pipeline(model, num_cols, cat_cols)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        out[name] = float(roc_auc_score(yte, p))
    return out

def replicate_fixed_split(Xtr, ytr, Xte, yte, num_cols=NUM, cat_cols=CAT):
    """Model-seed replication only; the train/test partition itself is fixed
    (used where the canonical archive ships its own official split)."""
    runs = [fit_eval(Xtr, ytr, Xte, yte, s, num_cols, cat_cols) for s in SEEDS]
    return {m: (round(float(np.mean([r[m] for r in runs])), 4),
                round(float(np.std([r[m] for r in runs])), 4)) for m in MODELS}

def replicate_resplit(X, y, num_cols, cat_cols):
    """Both the split and model seeds vary per replicate (used where no
    official split exists, so the split itself is a source of variance)."""
    runs = []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
        runs.append(fit_eval(Xtr, ytr, Xte, yte, s, num_cols, cat_cols))
    return {m: (round(float(np.mean([r[m] for r in runs])), 4),
                round(float(np.std([r[m] for r in runs])), 4)) for m in MODELS}

def fmt(md):
    return {m: f"{v[0]:.4f} +/- {v[1]:.4f}" for m, v in md.items()}

# 1. Baseline: canonical official train/test split, as correctly used ------
train, test = load_canonical()
ytr = (train['class'] == '>50K').astype(int)
yte = (test['class'] == '>50K').astype(int)
Xtr, Xte = train.drop(columns=['class']), test.drop(columns=['class'])
baseline_auc = replicate_fixed_split(Xtr, ytr, Xte, yte)
print(f"\n--- 1. BASELINE: canonical adult.data -> adult.test (official split, as shipped) ---")
print(f"  train n={len(train):,}  test n={len(test):,}  (5-seed replication, model randomness only)")
print(f"  AUC: {fmt(baseline_auc)}")

# 2. De-duplication effect: drop exact-duplicate rows from the TRAINING file only
full_train = pd.concat([Xtr, ytr.rename('target')], axis=1)
n_before = len(full_train)
dedup_train = full_train.drop_duplicates()
n_after = len(dedup_train)
Xtr_dedup, ytr_dedup = dedup_train.drop(columns=['target']), dedup_train['target']
dedup_auc = replicate_fixed_split(Xtr_dedup, ytr_dedup, Xte, yte)
print(f"\n--- 2. DE-DUPLICATION EFFECT: exact duplicates removed from training file only ---")
print(f"  training duplicates removed: {n_before - n_after} ({100*(n_before-n_after)/n_before:.3f}% of adult.data)")
print(f"  AUC after de-duplication: {fmt(dedup_auc)}")
delta_dedup = {m: round(dedup_auc[m][0] - baseline_auc[m][0], 4) for m in MODELS}
print(f"  delta (mean) vs baseline: {delta_dedup}")

# 3. Discretized mirror (OpenML did=179): stratified 80/20, matched protocol -
def read_arff_data(path):
    names, rows, in_data = [], [], False
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            low = s.lower()
            if low.startswith('@attribute'):
                names.append(s.split()[1].strip("'\""))
            elif low.startswith('@data'):
                in_data = True
            elif in_data:
                rows.append(s.split(','))
    return pd.DataFrame(rows, columns=names)

m1 = read_arff_data(os.path.join(MIRR, 'adult_openml_v1.arff'))
# Same 6 numeric + 8 categorical features as the canonical model (NUM/CAT above),
# so the only difference between the two fitted pipelines is that four of these
# six numeric columns are discretized in the mirror rather than continuous --
# isolating the discretization effect rather than confounding it with a
# different feature set.
m1_num = ['age', 'fnlwgt', 'education-num', 'capitalgain', 'capitalloss', 'hoursperweek']
m1_cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
          'race', 'sex', 'native-country']
m1_cat = [c for c in m1_cat if c in m1.columns]
m1_num = [c for c in m1_num if c in m1.columns]
assert len(m1_num) == 6 and len(m1_cat) == 8, f"mirror feature set mismatch: {m1_num} {m1_cat}"
m1_y = (m1['class'] == '>50K').astype(int)
m1_X = m1[m1_num + m1_cat].copy()
for c in m1_num:
    m1_X[c] = pd.to_numeric(m1_X[c], errors='coerce')
m1_X['fnlwgt'] = pd.to_numeric(m1_X['fnlwgt'], errors='coerce')

mirror_auc = replicate_resplit(m1_X, m1_y, m1_num, m1_cat)

# Canonical, matched protocol (stratified 80/20 on the full corrected 48,842-row
# corpus rather than the official split) for a fair like-for-like comparison
full = pd.concat([train, test], ignore_index=True)
y_full = (full['class'] == '>50K').astype(int)
X_full = full.drop(columns=['class'])
canonical_matched_auc = replicate_resplit(X_full, y_full, NUM, CAT)

print(f"\n--- 3. DISCRETIZATION EFFECT: continuous features silently binned to 5 levels (5-seed, resplit each seed) ---")
print(f"  canonical (continuous features): {fmt(canonical_matched_auc)}")
print(f"  OpenML mirror (discretized features): {fmt(mirror_auc)}")
delta_discretization = {m: round(mirror_auc[m][0] - canonical_matched_auc[m][0], 4) for m in MODELS}
print(f"  delta (mean): {delta_discretization}")

# 4. Label-bug severity: does a standard stratified split silently "succeed"
#    on the corrupted 4-class labels, without raising any error? -------------
naive_combined = pd.concat([train.assign(**{'class': train['class']}),
                             pd.read_csv(os.path.join(RAW, 'adult.test'), header=None, names=COLS,
                                         sep=r',\s*', engine='python', na_values='?', skiprows=1).dropna(how='all')],
                            ignore_index=True)
naive_labels = naive_combined['class']
naive_n_classes = naive_labels.nunique()
split_succeeded = False
try:
    _, _, _, y_te_naive = train_test_split(naive_combined.drop(columns=['class']), naive_labels,
                                            test_size=0.2, stratify=naive_labels, random_state=42)
    split_succeeded = True
    naive_test_class_counts = y_te_naive.value_counts().to_dict()
except Exception as e:
    naive_test_class_counts = {'error': str(e)}

print(f"\n--- 4. LABEL-BUG SEVERITY: does a standard pipeline silently accept the corrupted labels? ---")
print(f"  naive combined n_classes = {naive_n_classes} (should be 2)")
print(f"  stratified train_test_split on corrupted labels succeeds without error: {split_succeeded}")
print(f"  resulting test-partition class counts: {naive_test_class_counts}")
print(f"  interpretation: a standard scikit-learn call raises no error and returns a technically\n"
      f"  valid-looking split; the corruption is silent and would only surface as a spurious\n"
      f"  4-way class imbalance downstream, not as a crash.")

# 5. Persist -------------------------------------------------------------
def serialize(md):
    return {m: {'mean': v[0], 'std': v[1]} for m, v in md.items()}

out = {
    'seeds': SEEDS,
    'models': MODELS,
    'baseline_official_split': {'train_n': len(train), 'test_n': len(test), 'auc': serialize(baseline_auc)},
    'deduplication_effect': {
        'training_duplicates_removed': int(n_before - n_after),
        'pct_of_training_file': round(100*(n_before-n_after)/n_before, 3),
        'auc_after_dedup': serialize(dedup_auc),
        'delta_mean_vs_baseline': delta_dedup,
    },
    'discretization_effect': {
        'canonical_matched_split_auc': serialize(canonical_matched_auc),
        'mirror_discretized_auc': serialize(mirror_auc),
        'delta_mean': delta_discretization,
    },
    'label_bug_severity': {
        'naive_n_classes': int(naive_n_classes),
        'stratified_split_succeeds_silently': split_succeeded,
        'test_partition_class_counts': {str(k): int(v) for k, v in naive_test_class_counts.items()} if split_succeeded else naive_test_class_counts,
    },
}
save_json(out, 's11_census_severity.json')
print("\n[Stage 11 complete]")
