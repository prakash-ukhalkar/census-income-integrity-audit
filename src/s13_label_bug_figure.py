"""Stage 13 - Generates Figure 2 (the label-encoding divergence's silent
failure): the spurious four-class test partition a naive stratified split
produces, next to what it should be once the trailing-period divergence is
corrected. Reads results/s11_census_severity.json; produces no new numbers
of its own."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, 's11_census_severity.json')) as f:
    d = json.load(f)

counts = d['label_bug_severity']['test_partition_class_counts']
naive_labels = ['<=50K', '<=50K.', '>50K', '>50K.']
naive_values = [counts[l] for l in naive_labels]
naive_colors = ['#4C72B0', '#8CA9D6', '#C44E52', '#DE9497']

corrected_labels = ['<=50K', '>50K']
corrected_values = [counts['<=50K'] + counts['<=50K.'], counts['>50K'] + counts['>50K.']]
corrected_colors = ['#4C72B0', '#C44E52']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

ax1.bar(naive_labels, naive_values, color=naive_colors)
ax1.set_title('Naive split\n(4 spurious classes)')
ax1.set_ylabel('Test-partition rows')
ax1.tick_params(axis='x', rotation=20)
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

ax2.bar(corrected_labels, corrected_values, color=corrected_colors)
ax2.set_title('Corrected split\n(true 2 classes)')
ax2.tick_params(axis='x', rotation=20)
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)

fig.suptitle('A standard stratified split accepts the corrupted labels without error')
fig.tight_layout()
out_path = os.path.join(FIG, 'Figure_2.png')
fig.savefig(out_path, dpi=300)
print(f"[saved] {out_path}")
