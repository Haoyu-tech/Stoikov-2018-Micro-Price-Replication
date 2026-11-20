"""Plot results of bnb_stoikov_native.py (reads native.json, no recomputation).
Output results/microprice/stoikov_native/compare.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUT = Path(__file__).parent / 'results' / 'microprice' / 'stoikov_native'
J = json.load(open(OUT / 'native.json'))
S = J['summary']; G = J['G']
steps = sorted(int(k) for k in S)
NAMES = ['mid', 'weighted mid', r'$g^i$ (matched step)', r'$g^6$ (paper)', r'$g^\infty$ (limit)']
COL = [None, '#888888', '#1f77b4', '#d62728', '#2ca02c']

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for k, (key, title) in enumerate([('mse_gain', 'MSE improvement vs mid'), ('mae_gain', 'MAE improvement vs mid')]):
    a = ax[0, k]
    for j in range(1, 5):
        a.plot(steps, [100 * S[str(i)][key][j] for i in steps], 'o-', color=COL[j], label=NAMES[j])
    a.axhline(0, color='k', lw=0.6); a.set_xscale('log'); a.set_xticks(steps); a.set_xticklabels(steps)
    a.set_xlabel('i-th mid-price change'); a.set_ylabel('%'); a.set_title(title); a.grid(alpha=0.3); a.legend()

a = ax[1, 0]
b = np.arange(10)
for i, c in ((1, '#1f77b4'), (6, '#d62728'), (20, '#2ca02c')):
    a.plot(b, S[str(i)]['bucket_mean'][:10], 'o-', color=c, label=f'realized $y_{{{i}}}$')
    a.plot(b, G[str(i)][:10], 's--', color=c, alpha=0.7, label=f'predicted $g^{{{i}}}$')
a.axhline(0, color='k', lw=0.6)
a.set_xlabel('imbalance bucket (0 = ask-heavy, 9 = bid-heavy; spread 1 tick)'); a.set_ylabel('tick')
a.set_title('By imbalance bucket: realized mean move vs Stoikov prediction'); a.grid(alpha=0.3); a.legend(ncol=3, fontsize=9)

a = ax[1, 1]
for j in range(1, 5):
    a.plot(steps, [S[str(i)]['slope'][j] for i in steps], 'o-', color=COL[j], label=NAMES[j])
a.axhline(1, color='k', lw=0.8, ls=':'); a.text(steps[-1], 1.25, 'dotted 1 = correct magnitude', fontsize=9, ha='right')
a.set_xscale('log'); a.set_xticks(steps); a.set_xticklabels(steps)
a.set_xlabel('i-th mid-price change'); a.set_ylabel('slope of realized on predicted')
a.set_title('Calibration slope (>1 = prediction too small)'); a.grid(alpha=0.3); a.legend()

fig.suptitle('Stoikov micro-price: paper estimation, event-time target (BNBUSDT spot, October)', fontsize=14)
fig.tight_layout()
fig.savefig(OUT / 'compare.png', dpi=130)
print(OUT / 'compare.png')
