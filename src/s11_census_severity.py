"""Stage 11 - Severity measurement. Quantifies how much each defect found in
Stage 10 actually distorts measured model performance: the effect of
de-duplicating the canonical training data, the effect of Mirror 1's silent
feature discretization, and whether a standard pipeline silently accepts the
label-encoding divergence found in Stage 10."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from auditlib import save_json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

print("=" * 72); print(" STAGE 11: CENSUS-INCOME SEVERITY MEASUREMENT"); print("=" * 72)

def load_canonical():
    train = pd.read_csv(os.path.join(RAW, 'adult.data'), header=None, names=COLS,
                         sep=r',\s*', engine='python', na_values='?').dropna(how='all')
    test = pd.read_csv(os.path.join(RAW, 'adult.test'), header=None, names=COLS,
                        sep=r',\s*', engine='python', na_values='?', skiprows=1).dropna(how='all')
    test = test.copy(); test['class'] = test['class'].str.rstrip('.')
    return train, test

def pipeline(model):
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUM),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                           ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    return Pipeline([('pre', pre), ('clf', model)])

def fit_eval(Xtr, ytr, Xte, yte, seed=42):
    out = {}
    for name, model in [('logreg', LogisticRegression(max_iter=1000, random_state=seed)),
                         ('rf', RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))]:
        pipe = pipeline(model)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        out[name] = round(float(roc_auc_score(yte, p)), 4)
    return out

# 1. Baseline: canonical official train/test split, as correctly used ------
train, test = load_canonical()
ytr = (train['class'] == '>50K').astype(int)
yte = (test['class'] == '>50K').astype(int)
Xtr, Xte = train.drop(columns=['class']), test.drop(columns=['class'])
baseline_auc = fit_eval(Xtr, ytr, Xte, yte)
print(f"\n--- 1. BASELINE: canonical adult.data -> adult.test (official split, as shipped) ---")
print(f"  train n={len(train):,}  test n={len(test):,}")
print(f"  AUC: {baseline_auc}")

# 2. De-duplication effect: drop exact-duplicate rows from the TRAINING file only
full_train = pd.concat([Xtr, ytr.rename('target')], axis=1)
n_before = len(full_train)
dedup_train = full_train.drop_duplicates()
n_after = len(dedup_train)
Xtr_dedup, ytr_dedup = dedup_train.drop(columns=['target']), dedup_train['target']
dedup_auc = fit_eval(Xtr_dedup, ytr_dedup, Xte, yte)
print(f"\n--- 2. DE-DUPLICATION EFFECT: exact duplicates removed from training file only ---")
print(f"  training duplicates removed: {n_before - n_after} ({100*(n_before-n_after)/n_before:.3f}% of adult.data)")
print(f"  AUC after de-duplication: {dedup_auc}")
print(f"  delta vs baseline: {{'logreg': {round(dedup_auc['logreg']-baseline_auc['logreg'],4)}, "
      f"'rf': {round(dedup_auc['rf']-baseline_auc['rf'],4)}}}")

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

Xtr_m1, Xte_m1, ytr_m1, yte_m1 = train_test_split(m1_X, m1_y, test_size=0.2, stratify=m1_y, random_state=42)

def pipeline_generic(model, num_cols, cat_cols):
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                           ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])
    return Pipeline([('pre', pre), ('clf', model)])

def fit_eval_generic(Xtr, ytr, Xte, yte, num_cols, cat_cols, seed=42):
    out = {}
    for name, model in [('logreg', LogisticRegression(max_iter=1000, random_state=seed)),
                         ('rf', RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))]:
        pipe = pipeline_generic(model, num_cols, cat_cols)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        out[name] = round(float(roc_auc_score(yte, p)), 4)
    return out

mirror_auc = fit_eval_generic(Xtr_m1, ytr_m1, Xte_m1, yte_m1, m1_num, m1_cat)

# Canonical, matched protocol (stratified 80/20 on the full corrected 48,842-row
# corpus rather than the official split) for a fair like-for-like comparison
full = pd.concat([train, test], ignore_index=True)
y_full = (full['class'] == '>50K').astype(int)
X_full = full.drop(columns=['class'])
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
canonical_matched_auc = fit_eval(Xtr_c, ytr_c, Xte_c, yte_c)

print(f"\n--- 3. DISCRETIZATION EFFECT: continuous features silently binned to 5 levels ---")
print(f"  canonical (continuous features, matched 80/20 split): {canonical_matched_auc}")
print(f"  OpenML mirror (discretized features, same split protocol): {mirror_auc}")
print(f"  delta: {{'logreg': {round(mirror_auc['logreg']-canonical_matched_auc['logreg'],4)}, "
      f"'rf': {round(mirror_auc['rf']-canonical_matched_auc['rf'],4)}}}")

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
out = {
    'baseline_official_split': {'train_n': len(train), 'test_n': len(test), 'auc': baseline_auc},
    'deduplication_effect': {
        'training_duplicates_removed': int(n_before - n_after),
        'pct_of_training_file': round(100*(n_before-n_after)/n_before, 3),
        'auc_after_dedup': dedup_auc,
        'delta_vs_baseline': {k: round(dedup_auc[k] - baseline_auc[k], 4) for k in baseline_auc},
    },
    'discretization_effect': {
        'canonical_matched_split_auc': canonical_matched_auc,
        'mirror_discretized_auc': mirror_auc,
        'delta': {k: round(mirror_auc[k] - canonical_matched_auc[k], 4) for k in baseline_auc},
    },
    'label_bug_severity': {
        'naive_n_classes': int(naive_n_classes),
        'stratified_split_succeeds_silently': split_succeeded,
        'test_partition_class_counts': {str(k): int(v) for k, v in naive_test_class_counts.items()} if split_succeeded else naive_test_class_counts,
    },
}
save_json(out, 's11_census_severity.json')
print("\n[Stage 11 complete]")
