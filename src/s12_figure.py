"""Stage 12 - Generates Figure 1 (severity comparison across three models)
from the Stage 11 results. Reads results/s11_census_severity.json; produces
no new numbers of its own."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, 's11_census_severity.json')) as f:
    d = json.load(f)

MODEL_LABELS = {'logreg': 'Logistic\nregression', 'rf': 'Random\nForest', 'gb': 'Gradient\nBoosting'}
models = d['models']

dedup_delta = [d['deduplication_effect']['delta_mean_vs_baseline'][m] for m in models]
disc_delta = [d['discretization_effect']['delta_mean'][m] for m in models]

def std_of(section_key, subkey, m):
    return d[section_key][subkey][m]['std']

dedup_std = [std_of('deduplication_effect', 'auc_after_dedup', m) for m in models]
disc_std = [std_of('discretization_effect', 'mirror_discretized_auc', m) for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.bar(x - width/2, dedup_delta, width, yerr=dedup_std, capsize=4,
       label='De-duplication effect\n(canonical training data)', color='#4C72B0')
ax.bar(x + width/2, disc_delta, width, yerr=disc_std, capsize=4,
       label='Discretization effect\n(Mirror 1 vs. canonical)', color='#C44E52')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in models])
ax.set_ylabel(r'$\Delta$ AUC (mean of 5 seeds, error bars = std)')
ax.set_title('Severity of two defects, by model')
ax.legend(loc='lower left', fontsize=8, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
out_path = os.path.join(FIG, 'Figure_1.png')
fig.savefig(out_path, dpi=300)
print(f"[saved] {out_path}")
