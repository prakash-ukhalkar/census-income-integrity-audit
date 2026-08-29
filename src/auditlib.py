"""
auditlib.py - Shared data-integrity audit functions used by both pipeline
stages in this repository (s10_census_extension.py, s11_census_severity.py).
Domain-agnostic: operates on any [X, y] with a fixed schema.
"""
import hashlib, json, os
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RES, exist_ok=True)

def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

def duplicate_audit(X, y, round_sig=6):
    """Exact duplicates (all columns incl. label), feature-only duplicates
    (label-conflicting twins), near-duplicates after rounding, and a
    concatenation signature check."""
    full = pd.concat([X, y], axis=1)
    n_raw = len(full)
    n_unique = n_raw - full.duplicated().sum()
    counts = full.groupby(list(full.columns), dropna=False).size()
    rep_dist = counts.value_counts().sort_index().to_dict()
    factor = None
    dup_pct = 100 * (n_raw - n_unique) / n_raw
    if dup_pct > 10:
        for k in range(2, 30):
            reps = [c for c in rep_dist if c > 1]
            if n_raw % k == 0 and reps and all(c % k == 0 for c in reps):
                factor = k; break
    feat_dupd = X.duplicated().sum()
    conflict = int(feat_dupd - full.duplicated().sum())
    Xr = X.copy()
    num = Xr.select_dtypes(include=[np.number]).columns
    with np.errstate(all='ignore'):
        for c in num:
            v = Xr[c].to_numpy(dtype=float)
            mag = np.where(v == 0, 1, 10 ** np.floor(np.log10(np.abs(np.where(v == 0, 1, v)))))
            Xr[c] = np.round(v / mag, round_sig - 1) * mag
    near_dup = int(pd.concat([Xr, y], axis=1).duplicated().sum()) - int(full.duplicated().sum())
    return {'n_raw': int(n_raw), 'n_unique': int(n_unique),
            'exact_dup_rows': int(n_raw - n_unique),
            'exact_dup_pct': round(100 * (n_raw - n_unique) / n_raw, 3),
            'repeat_count_distribution': {int(k): int(v) for k, v in rep_dist.items() if k > 1},
            'concat_factor_signature': factor,
            'feature_dup_label_conflicts': conflict,
            'additional_near_dups_rounded': int(max(near_dup, 0))}

def auxiliary_column_screen(df_all, target_col='class'):
    """Flag columns whose |correlation| with the target or with row order is
    suspiciously high."""
    out = {}
    yv = pd.to_numeric(df_all[target_col], errors='coerce')
    order = pd.Series(np.arange(len(df_all)), index=df_all.index)
    for c in df_all.columns:
        if c == target_col:
            continue
        v = pd.to_numeric(df_all[c], errors='coerce')
        if v.notna().sum() < 10 or v.nunique() <= 1:
            continue
        r_t = v.corr(yv); r_o = v.corr(order)
        if (abs(r_t) > 0.95) or (abs(r_o) > 0.30):
            out[c] = {'corr_target': round(float(r_t), 3),
                      'corr_row_order': round(float(r_o), 3)}
    return out

def save_json(obj, name):
    with open(os.path.join(RES, name), 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] results/{name}")
