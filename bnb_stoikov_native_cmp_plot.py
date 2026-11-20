"""Compare Stoikov native estimate with K truncated (|dM|<=1 tick, paper) vs all jumps kept, on the paper event-time target.
Reads results/microprice/stoikov_native/native.json and stoikov_native_all/native.json. Output stoikov_native_all/compare_trunc_vs_all.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
R = Path(__file__).parent / 'results' / 'microprice'
A = json.load(open(R / 'stoikov_native' / 'native.json')); B = json.load(open(R / 'stoikov_native_all' / 'native.json'))
steps = sorted(int(k) for k in A['summary'])

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for k, (key, title) in enumerate([('mse_gain', 'MSE improvement vs mid'), ('mae_gain', 'MAE improvement vs mid')]):
    a = ax[0, k]
    for J, ls, tag in ((A, '--', 'truncated ≤1 tick'), (B, '-', 'all jumps')):
        a.plot(steps, [100 * J['summary'][str(i)][key][3] for i in steps], 'o' + ls, color='#d62728', label=f'$g^6$ {tag}')
        a.plot(steps, [100 * J['summary'][str(i)][key][2] for i in steps], 's' + ls, color='#1f77b4', label=f'$g^i$ {tag}')
    a.plot(steps, [100 * A['summary'][str(i)][key][1] for i in steps], 'o-', color='#999999', label='weighted mid')
    a.axhline(0, color='k', lw=0.6); a.set_xscale('log'); a.set_xticks(steps); a.set_xticklabels(steps)
    a.set_xlabel('i-th mid-price change'); a.set_ylabel('%'); a.set_title(title); a.grid(alpha=0.3); a.legend(fontsize=8)

a = ax[1, 0]; b = np.arange(10)
for i, c in ((1, '#1f77b4'), (6, '#d62728'), (20, '#2ca02c')):
    a.plot(b, A['summary'][str(i)]['bucket_mean'][:10], 'o-', color=c, lw=2, label=f'realized $y_{{{i}}}$')
    a.plot(b, A['G'][str(i)][:10], ':', color=c, label=f'$g^{{{i}}}$ truncated')
    a.plot(b, B['G'][str(i)][:10], 's--', color=c, ms=4, label=f'$g^{{{i}}}$ all jumps')
a.axhline(0, color='k', lw=0.6)
a.set_xlabel('imbalance bucket (spread 1 tick)'); a.set_ylabel('tick'); a.set_title('By imbalance bucket: realized vs two estimates')
a.grid(alpha=0.3); a.legend(ncol=3, fontsize=8)

a = ax[1, 1]
for J, ls, tag in ((A, '--', 'truncated'), (B, '-', 'all jumps')):
    a.plot(steps, [J['summary'][str(i)]['slope'][3] for i in steps], 'o' + ls, color='#d62728', label=f'$g^6$ {tag}')
    a.plot(steps, [J['summary'][str(i)]['slope'][2] for i in steps], 's' + ls, color='#1f77b4', label=f'$g^i$ {tag}')
a.axhline(1, color='k', lw=0.8, ls=':'); a.text(steps[-1], 1.15, 'dotted 1 = correct magnitude', fontsize=9, ha='right')
a.set_xscale('log'); a.set_xticks(steps); a.set_xticklabels(steps)
a.set_xlabel('i-th mid-price change'); a.set_ylabel('slope of realized on predicted'); a.set_title('Calibration slope (>1 = prediction too small)')
a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle('Stoikov paper method: jumps truncated at 1 tick vs all jumps kept (BNBUSDT spot, October, event-time target)', fontsize=14)
fig.tight_layout(); out = R / 'stoikov_native_all' / 'compare_trunc_vs_all.png'; fig.savefig(out, dpi=130); print(out)
