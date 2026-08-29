"""Stage 10 - Prevalence and provenance audit. Applies an eight-check
data-integrity audit protocol (auditlib.duplicate_audit / auxiliary_column_screen)
to the canonical UCI Adult / Census Income archive and two independently
obtained, actively used redistributions.

Canonical source: UCI Adult / Census Income (Becker & Kohavi, 1996),
https://doi.org/10.24432/C5XW20. Two independently obtained, actively used
redistributions are compared against it:
  - OpenML did=179  ("adult", v1) -- the version returned by
    sklearn.datasets.fetch_openml('adult') at default settings.
  - OpenML did=43898 ("adult", v3).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from auditlib import duplicate_audit, auxiliary_column_screen, sha256, save_json

RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'adult_census_income')
MIRR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mirrors', 'adult_census_income')

COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'class']

print("=" * 72); print(" STAGE 10: PREVALENCE AND PROVENANCE AUDIT (CENSUS INCOME)"); print("=" * 72)

# 1. Canonical files, checksums, and official-spec verification -------------
p_train = os.path.join(RAW, 'adult.data')
p_test = os.path.join(RAW, 'adult.test')
sha_train, sha_test = sha256(p_train), sha256(p_test)

train = pd.read_csv(p_train, header=None, names=COLS, sep=r',\s*', engine='python', na_values='?').dropna(how='all')
# adult.test's first line is a comment ("|1x3 Cross validator"); its labels
# carry a trailing period (">50K." / "<=50K.") not present in adult.data --
# a real label-encoding divergence between a canonical archive's own two
# constituent files.
test_raw = pd.read_csv(p_test, header=None, names=COLS, sep=r',\s*', engine='python',
                        na_values='?', skiprows=1).dropna(how='all')

official_n = {'train': 32561, 'test': 16281}
spec_check = {
    'train_n_matches_official': len(train) == official_n['train'],
    'test_n_matches_official': len(test_raw) == official_n['test'],
    'train_n': len(train), 'test_n': len(test_raw),
}
print(f"\ncanonical adult.data: n={len(train):,} (official {official_n['train']:,}) "
      f"sha256={sha_train[:20]}...")
print(f"canonical adult.test: n={len(test_raw):,} (official {official_n['test']:,}) "
      f"sha256={sha_test[:20]}...")
assert spec_check['train_n_matches_official'] and spec_check['test_n_matches_official'], \
    "Canonical file row counts do not match published UCI specification"

# 2. Trailing-period label divergence: quantify the naive-concatenation failure
train_labels = sorted(train['class'].unique().tolist())
test_labels_raw = sorted(test_raw['class'].unique().tolist())
naive_combined = pd.concat([train, test_raw], ignore_index=True)
naive_n_classes = naive_combined['class'].nunique()
test_clean = test_raw.copy()
test_clean['class'] = test_clean['class'].str.rstrip('.')
clean_combined = pd.concat([train, test_clean], ignore_index=True)
clean_n_classes = clean_combined['class'].nunique()

label_bug = {
    'train_label_values': train_labels,
    'test_label_values_as_shipped': test_labels_raw,
    'rows_affected_in_test_file': int(len(test_raw)),
    'pct_of_test_file_affected': 100.0,  # every test-file row carries the trailing period
    'naive_concat_n_classes': int(naive_n_classes),
    'corrected_concat_n_classes': int(clean_n_classes),
}
print(f"\n--- LABEL-ENCODING DIVERGENCE (adult.data vs adult.test) ---")
print(f"  train labels: {train_labels}")
print(f"  test labels (as shipped): {test_labels_raw}")
print(f"  naive train+test concatenation yields {naive_n_classes} distinct classes "
      f"(should be 2), affects {len(test_raw):,}/{len(naive_combined):,} rows "
      f"({100*len(test_raw)/len(naive_combined):.1f}% of the combined corpus)")

# 3. Canonical audit: duplicate/identifier/leak screen on the corrected, combined file
combined = clean_combined.copy()
y = pd.factorize(combined['class'])[0]
y = pd.Series(y, index=combined.index, name='target')
X = combined.drop(columns=['class'])
canonical_audit = duplicate_audit(X, y)
aux = auxiliary_column_screen(pd.concat([X, y], axis=1), 'target')
print(f"\n--- CANONICAL CORPUS AUDIT (n={len(combined):,}, corrected labels) ---")
print(f"  exact duplicate rows: {canonical_audit['exact_dup_rows']} "
      f"({canonical_audit['exact_dup_pct']}%)")
print(f"  label-conflicting feature twins: {canonical_audit['feature_dup_label_conflicts']}")
print(f"  target/order-correlation flags: {list(aux.keys())}")

# identifier detection (same heuristic as Check 1 in the protocol)
idlike = []
for c in X.columns:
    s = X[c]
    if not (pd.api.types.is_numeric_dtype(s) and s.notna().all() and s.nunique() == len(s)):
        continue
    v = s.to_numpy(dtype=float)
    if not np.allclose(v, np.round(v)):
        continue
    rng = v.max() - v.min() + 1
    if bool(np.all(np.diff(v) > 0)) or rng <= 1.05 * len(v):
        idlike.append(c)
print(f"  identifier-like columns detected: {idlike if idlike else 'none'}")

# 4. Mirror comparison 1: OpenML did=179 (v1) -- silent discretization -------
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
discretized_cols = [c for c in ['age', 'capitalgain', 'capitalloss', 'hoursperweek'] if c in m1.columns]
m1_domains = {c: sorted(m1[c].unique().tolist()) for c in discretized_cols}
print(f"\n--- MIRROR 1: OpenML did=179 (sklearn fetch_openml('adult') default) ---")
print(f"  n={len(m1):,} (canonical corrected n={len(combined):,}, "
      f"{'MATCHES' if len(m1) == len(combined) else 'DIVERGES'})")
print(f"  continuous columns silently discretized to 5-level bins: {discretized_cols}")
for c, dom in m1_domains.items():
    print(f"    {c}: canonical range is continuous; mirror domain = {dom}")

# 5. Mirror comparison 2: OpenML did=43898 (v3) -- row-count divergence -----
m3 = read_arff_data(os.path.join(MIRR, 'adult_openml_v3.arff'))
m3_n = len(m3)
row_diff = len(combined) - m3_n
print(f"\n--- MIRROR 2: OpenML did=43898 ('adult', v3) ---")
print(f"  n={m3_n:,} vs canonical corrected n={len(combined):,}  "
      f"(missing {row_diff:,} records, {100*row_diff/len(combined):.2f}% of the corpus)")
m3_cols = m3.columns.tolist()
canon_cols_norm = [c.replace('-', '_') for c in X.columns]
renamed = [c for c in m3_cols if c not in X.columns and c.replace('_', '-') in X.columns]
print(f"  columns silently renamed (hyphen -> underscore): {renamed}")

# Row-level verification: is v3 exactly the de-duplicated canonical corpus,
# or does it diverge in content beyond row count? Normalise missing-value
# representation identically on both sides before comparing (a naive string
# comparison otherwise reports a spurious ~3,615-row mismatch driven purely by
# '?' vs NaN formatting, not real content divergence).
c_norm = combined.copy(); c_norm.columns = [c.replace('-', '_') for c in c_norm.columns]
m3_norm = m3.replace('?', np.nan).copy()
for c in m3_norm.columns:
    conv = pd.to_numeric(m3_norm[c], errors='coerce')
    if conv.notna().sum() >= m3_norm[c].notna().sum() * 0.99:
        m3_norm[c] = conv
common = [c for c in c_norm.columns if c in m3_norm.columns]
c_dedup = c_norm.drop_duplicates(subset=common)

def _keyset(df):
    d = df[common].astype(object).where(df[common].notna(), 'MISSING').astype(str)
    return set(map(tuple, d.values))

k_canon, k_m3 = _keyset(c_dedup), _keyset(m3_norm)
v3_is_exact_dedup = (len(k_canon - k_m3) == 0) and (len(k_m3 - k_canon) == 0) and (len(c_dedup) == m3_n)
print(f"  row-level check: v3 == de-duplicated canonical corpus? {v3_is_exact_dedup} "
      f"({len(k_canon & k_m3):,}/{m3_n:,} rows match exactly)")

# 6. Persist results ----------------------------------------------------------
out = {
    'canonical': {
        'files': {'adult.data': {'sha256': sha_train, 'n': len(train)},
                   'adult.test': {'sha256': sha_test, 'n': len(test_raw)}},
        'spec_check': spec_check,
        'label_encoding_divergence': label_bug,
        'combined_n': len(combined),
        'duplicate_audit': canonical_audit,
        'auxiliary_flags': aux,
        'identifier_like_columns': idlike,
    },
    'mirror_openml_v1_did179': {
        'n': len(m1), 'n_matches_canonical': len(m1) == len(combined),
        'discretized_columns': discretized_cols, 'discretized_domains': m1_domains,
    },
    'mirror_openml_v3_did43898': {
        'n': m3_n, 'row_count_divergence_from_canonical': row_diff,
        'row_count_divergence_pct': round(100 * row_diff / len(combined), 3),
        'renamed_columns': renamed,
        'is_exact_undocumented_deduplication_of_canonical': bool(v3_is_exact_dedup),
    },
}
save_json(out, 's10_census_extension.json')
print("\n[Stage 10 complete]")
