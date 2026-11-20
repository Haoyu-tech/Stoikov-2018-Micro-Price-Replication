"""Plot bnb_stoikov_memory.py results (reads memory.json). Usage: python bnb_stoikov_memory_plot.py [dir name, default stoikov_memory_age]"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUT = Path(__file__).parent / 'results' / 'microprice' / (sys.argv[1] if len(sys.argv) > 1 else 'stoikov_memory_age')
J = json.load(open(OUT / 'memory.json')); S = J['summary']; names = J['names']
steps = sorted(int(k) for k in S)
LAB = {'none': 'no memory (imbalance + spread)', 'last': '+ last change', 'cum3': '+ sum of last 3 changes',
       'last_age': '+ last change + time since change', 'cum3_age': '+ sum of last 3 + time since change'}
COL = {'none': '#999999', 'last': '#9ecae1', 'cum3': '#fcae91', 'last_age': '#1f77b4', 'cum3_age': '#d62728'}

NR = 3 if 'age_y' in S['1'] else 2
fig, ax = plt.subplots(NR, 3, figsize=(18, 5 * NR))
for k, (key, title) in enumerate([('mse_gain', 'MSE improvement vs mid'), ('mae_gain', 'MAE improvement vs mid'), ('slope', 'Calibration slope (1 = correct magnitude)')]):
    a = ax[0, k]
    for j, v in enumerate(names):
        a.plot(steps, [(100 if key != 'slope' else 1) * S[str(i)][key][j] for i in steps], 'o-', color=COL[v], label=LAB[v])
    a.axhline(1 if key == 'slope' else 0, color='k', lw=0.7, ls=':' if key == 'slope' else '-')
    a.set_xscale('log'); a.set_xticks(steps); a.set_xticklabels(steps); a.set_xlabel('i-th mid-price change (matched $g^i$)')
    a.set_title(title); a.grid(alpha=0.3); a.legend(fontsize=9)
ml = ['≤−2.5', '−2~−0.5', '0', '+0.5~+2', '≥+2.5']
for c, i in enumerate((1, 6, 20)):
    a = ax[1, c]; s = S[str(i)]; x = np.arange(10)
    a.plot(x, s['bucket_y'], 'ko-', lw=2, label=f'realized $y_{{{i}}}$')
    for j, v in enumerate(names):
        a.plot(x, s['bucket_p'][j], 's--', ms=4, color=COL[v], label=LAB[v])
    a.axhline(0, color='k', lw=0.5); a.set_xlabel('imbalance bucket (spread 1 tick)'); a.set_ylabel('tick')
    a.set_title(f'By imbalance bucket: change {i}, realized vs mean prediction'); a.grid(alpha=0.3); a.legend(fontsize=8)
    ins = a.inset_axes([0.62, 0.06, 0.36, 0.36]); xm = np.arange(5)
    ins.plot(xm, s['mem_y'], 'ko-', lw=1.5)
    for j, v in enumerate(names):
        ins.plot(xm, s['mem_p'][j], 's--', ms=3, color=COL[v])
    ins.set_xticks(xm); ins.set_xticklabels(ml, fontsize=6, rotation=30); ins.tick_params(labelsize=6)
    ins.set_title('by sum of last 3 changes', fontsize=7); ins.axhline(0, color='k', lw=0.4); ins.grid(alpha=0.3)
if 'age_y' in S['1']:
    al = ['0–4 rows', '5–29 rows', '30+ rows']
    for c, i in enumerate((1, 6, 20)):
        a = ax[2, c]; s = S[str(i)]; xa = np.arange(3); w = 0.8 / (len(names) + 1)
        a.bar(xa - 0.4 + w / 2, s['age_y'], w, color='k', label='realized')
        for j, v in enumerate(names):
            a.bar(xa - 0.4 + w / 2 + (j + 1) * w, s['age_p'][j], w, color=COL[v], label=LAB[v])
        tot = sum(s['age_n'])
        a.set_xticks(xa); a.set_xticklabels([f'{al[k]} ({s["age_n"][k]/tot:.0%})' for k in range(3)])
        a.set_xlabel('time since last mid change (quote-update rows; share of samples)'); a.set_ylabel(f'$y_{{{i}}}$ × sign of last change (tick)')
        a.set_title(f'Momentum at mid imbalance: change {i}'); a.axhline(0, color='k', lw=0.5); a.grid(alpha=0.3, axis='y'); a.legend(fontsize=7)
fig.suptitle('Stoikov micro-price with recent price changes in the state (paper method, all jumps; BNBUSDT spot, October, event-time target)', fontsize=14)
fig.tight_layout(); fig.savefig(OUT / 'compare.png', dpi=125); print(OUT / 'compare.png')
