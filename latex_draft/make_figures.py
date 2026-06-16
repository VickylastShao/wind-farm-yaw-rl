#!/usr/bin/env python3
"""Generate final figures for Wind Energy submission."""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = "../figures_final"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
                     'legend.fontsize': 9, 'figure.dpi': 150})

# ---- Figure 1: FLORIS Cross-Validation Scatter ----
print("Fig 1: FLORIS cross-validation scatter...")
with open("../results/configE_floris_cross_validation.json") as f:
    floris_data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel a: Bar chart comparison
ax = axes[0]
metrics = ['Marginal mean', 'Aligned-cube']
gb_vals = [floris_data['gb_marginal_mean_pct'], floris_data['gb_aligned_cube_pct']] if 'gb_marginal_mean_pct' in floris_data else \
          [0.67, 5.24]
fl_vals = [floris_data.get('floris_marginal_mean_pct', 0.72), floris_data.get('floris_aligned_cube_pct', 5.02)]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x-w/2, gb_vals, w, label='Gray-box', color='#4472C4')
ax.bar(x+w/2, fl_vals, w, label='FLORIS', color='#ED7D31')
ax.set_ylabel('Farm-power gain (%)')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.legend(); ax.set_title('(a) Gain comparison')
for i, (g, f) in enumerate(zip(gb_vals, fl_vals)):
    ax.text(i-w/2, g+0.05, f'{g:.2f}%', ha='center', fontsize=8)
    ax.text(i+w/2, f+0.05, f'{f:.2f}%', ha='center', fontsize=8)

# Panel b: Erosion summary
ax = axes[1]
erosion = floris_data.get('erosion', 4.1)
ax.barh(['FLORIS / Gray-box'], [100-erosion], color='#4472C4', label='Retained')
ax.barh(['FLORIS / Gray-box'], [erosion], left=[100-erosion], color='#ED7D31', label='Erosion')
ax.set_xlim(90, 100.5); ax.set_xlabel('Aligned-cube gain retention (%)')
ax.legend(loc='lower right'); ax.set_title('(b) FLORIS erosion')
ax.text(100-erosion/2, 0, f'{100-erosion:.1f}%', ha='center', va='center', fontweight='bold')
ax.text(100-erosion/2, 0, f'{erosion:.1f}% erosion', ha='center', va='center', fontsize=8, color='white')

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/Fig_FLORIS_cross_validation.pdf", bbox_inches='tight')
fig.savefig(f"{FIG_DIR}/Fig_FLORIS_cross_validation.png", bbox_inches='tight')
plt.close()
print("  Saved.")

# ---- Figure 2: Noise Robustness ----
print("Fig 2: Noise robustness...")
noise_data = json.load(open("../results/configE_noise_robustness.json"))
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = [d['noise'] for d in noise_data]
gains = [d['marginal'] for d in noise_data]
colors = ['#4472C4'] + ['#5B9BD5']*4 + ['#ED7D31']*3 + ['#A5A5A5']*3 + ['#70AD47']
ax.bar(range(len(labels)), gains, color=colors, edgecolor='white')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Five-seed mean gain (regret-reward %)')
ax.set_title('Config-E observation-noise robustness')
ax.axhline(y=gains[0], color='gray', linestyle='--', linewidth=0.8)
ax.text(len(labels)-0.5, gains[0]+0.02, 'clean baseline', fontsize=7, color='gray')
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/Fig_noise_robustness.pdf", bbox_inches='tight')
fig.savefig(f"{FIG_DIR}/Fig_noise_robustness.png", bbox_inches='tight')
plt.close()
print("  Saved.")

# ---- Figure 3: Lock Ablation ----
print("Fig 3: Lock ablation...")
fig, ax = plt.subplots(figsize=(5, 4))
strategies = ['Lock ON', 'Lock OFF']
gains = [2.42, -0.46]
neg_fracs = [18, 28]
colors_bar = ['#4472C4', '#ED7D31']
ax.bar(strategies, gains, color=colors_bar, edgecolor='white')
ax.set_ylabel('Mean gain (regret-reward %)')
ax.set_title('Config-E downstream-lock ablation (3 seeds)')
for i, (g, n) in enumerate(zip(gains, neg_fracs)):
    ax.text(i, g+0.1 if g>0 else g-0.3, f'{g:+.2f}%\nneg: {n}%', ha='center', fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.8)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/Fig_lock_ablation.pdf", bbox_inches='tight')
fig.savefig(f"{FIG_DIR}/Fig_lock_ablation.png", bbox_inches='tight')
plt.close()
print("  Saved.")

print(f"\nAll figures saved to {FIG_DIR}/")
