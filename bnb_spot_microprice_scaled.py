"""Horizon-scaled micro-price on BNBUSDT spot (follow-up to bnb_spot_microprice.py).

For each horizon h, predictors of M_{t+h} - M_t (ticks), all fitted on TRAIN (Jul-Sep), scored on TEST (Oct):
  mid        : 0
  wmid       : (I-0.5)*S                     (no fit)
  micro      : g_full(I,S)                   (Stoikov, all jumps, unscaled; from g_n20_dt1_full.npz)
  micro_orig : g_orig(I,S)                   (Stoikov, |dM|<=1 tick; g_n20_dt1? -> uses n10 orig file)
  b_wmid     : beta_h * (I-0.5)*S            (OLS through origin on train)
  b_micro    : beta_h * g_full               (OLS through origin on train)
  table      : E_train[dM_h | bucket(I), S]  (direct conditional mean per horizon, symmetrized)
Per-day MSE kept for paired day-level comparisons.
"""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bnb_spot_microprice import CACHE, OUT, TRAIN, TEST, days, load

N = 20
HORIZONS_MS = (100, 250, 500, 1000, 2000, 5000, 10000, 30000)
G_FULL = np.load(OUT / f'g_n{N}_dt1_full.npz')['g']
G_ORIG = np.load(OUT / 'g_n10_dt1.npz')['g']
NAMES = ['mid', 'wmid', 'micro', 'micro_orig', 'b_wmid', 'b_micro', 'table']


def day_arrays(d):
    ts, bp, ap, bv, av = load(d)
    mid = (bp + ap) / 2.0
    spr = ap - bp
    imb = bv / (bv + av)
    ok = (spr == 1) | (spr == 2)
    s = np.clip(spr, 1, 2) - 1
    b20 = np.clip((imb * N).astype(int), 0, N - 1)
    b10 = np.clip((imb * 10).astype(int), 0, 9)
    x20, x10 = s * N + b20, s * 10 + b10
    wadj = (imb - 0.5) * spr
    out = {}
    for h in HORIZONS_MS:
        j = np.searchsorted(ts, ts + h * 1000, side='left')
        v = ok & (j < len(ts))
        out[h] = dict(dm=mid[np.minimum(j, len(ts) - 1)][v] - mid[v], x20=x20[v], x10=x10[v], w=wadj[v])
    return out


def mirror(x):
    return (x // N) * N + (N - 1 - x % N)


def fit():
    st = {h: dict(xy=np.zeros(2), xx=np.zeros(2), s=np.zeros(2 * N), c=np.zeros(2 * N)) for h in HORIZONS_MS}
    for d in days(*TRAIN):
        if not (CACHE / f'{d}.npz').exists():
            continue
        for h, a in day_arrays(d).items():
            t = st[h]; gm = G_FULL[a['x20']]
            t['xy'] += [(a['dm'] * a['w']).sum(), (a['dm'] * gm).sum()]
            t['xx'] += [(a['w'] ** 2).sum(), (gm ** 2).sum()]
            np.add.at(t['s'], a['x20'], a['dm']); np.add.at(t['c'], a['x20'], 1)
            np.add.at(t['s'], mirror(a['x20']), -a['dm']); np.add.at(t['c'], mirror(a['x20']), 1)
    par = {}
    for h, t in st.items():
        par[h] = dict(beta=t['xy'] / t['xx'], table=t['s'] / np.maximum(t['c'], 1), cnt=t['c'])
    return par


def evaluate(par):
    rows = []   # per day, per h: sse vector, n
    for d in days(*TEST):
        if not (CACHE / f'{d}.npz').exists():
            continue
        for h, a in day_arrays(d).items():
            p = par[h]
            preds = [np.zeros_like(a['dm']), a['w'], G_FULL[a['x20']], G_ORIG[a['x10']],
                     p['beta'][0] * a['w'], p['beta'][1] * G_FULL[a['x20']], p['table'][a['x20']]]
            sse = np.array([((a['dm'] - q) ** 2).sum() for q in preds])
            rows.append((d, h, len(a['dm']), sse))
    return rows


def main():
    par = fit()
    rows = evaluate(par)
    lines = ['# BNBUSDT spot: horizon-scaled micro-price', '',
             f'Fit {TRAIN[0]}~{TRAIN[1]} (beta and conditional-mean table), evaluate {TEST[0]}~{TEST[1]}; I in {N} buckets, S∈{{1,2}}. ', '',
             '## Training-period scale factors', '', '| horizon | beta_wmid | beta_micro(full) | table at I=0.025 (tick) | table at I=0.475 |', '|---|---|---|---|---|']
    for h in HORIZONS_MS:
        p = par[h]
        lines.append(f'| {h}ms | {p["beta"][0]:.3f} | {p["beta"][1]:.4f} | {p["table"][0]:+.3f} | {p["table"][N//2-1]:+.3f} |')
    lines += ['', '## October out-of-sample: MSE improvement vs mid (higher is better)', '',
              '| horizon | ' + ' | '.join(NAMES[1:]) + ' |', '|---|' + '---|' * (len(NAMES) - 1)]
    summ = {}
    for h in HORIZONS_MS:
        r = [x for x in rows if x[1] == h]
        tot = sum(x[3] for x in r); n = sum(x[2] for x in r)
        imp = 1 - tot / tot[0]
        # per-day improvements, paired diff of b_micro / table vs b_wmid
        dimp = np.array([1 - x[3] / x[3][0] for x in r])
        summ[h] = dict(n=int(n), improve=dict(zip(NAMES, imp.tolist())), daily=dimp.tolist())
        lines.append(f'| {h}ms | ' + ' | '.join(f'{v:+.2%}' for v in imp[1:]) + ' |')
    lines += ['', '## Daily paired: gain difference vs b_wmid (pp, 31-day mean ± 2 s.e., win rate)', '',
              '| horizon | b_micro − b_wmid | win rate | table − b_wmid | win rate |', '|---|---|---|---|---|']
    for h in HORIZONS_MS:
        dimp = np.array(summ[h]['daily'])
        out = []
        for k in (5, 6):
            dd = (dimp[:, k] - dimp[:, 4]) * 100
            out.append(f'{dd.mean():+.3f} ± {2*dd.std(ddof=1)/np.sqrt(len(dd)):.3f} | {np.mean(dd>0):.0%}')
        lines.append(f'| {h}ms | ' + ' | '.join(out) + ' |')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report_scaled.md').write_text('\n'.join(lines), encoding='utf-8')
    json.dump(dict(summary=summ, beta={h: par[h]['beta'].tolist() for h in HORIZONS_MS},
                   table={h: par[h]['table'].tolist() for h in HORIZONS_MS}), open(OUT / 'scaled.json', 'w'))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    c = (np.arange(N) + 0.5) / N
    for i, h in enumerate(HORIZONS_MS):
        col = plt.cm.viridis(i / (len(HORIZONS_MS) - 1))
        ax[0].plot(c, par[h]['table'][:N], '-', color=col, label=f'E[dM|I] {h}ms')
    ax[0].plot(c, G_FULL[:N], 'k--', lw=2, label='Stoikov g (full)')
    ax[0].plot(c, c - 0.5, 'r:', lw=2, label='wmid')
    ax[0].set_xlabel('imbalance I (S=1)'); ax[0].set_ylabel('ticks'); ax[0].legend(fontsize=7)
    ax[0].set_title('conditional mean move vs horizon')
    hs = np.array(HORIZONS_MS) / 1000
    for k in range(1, len(NAMES)):
        y = [summ[h]['improve'][NAMES[k]] * 100 for h in HORIZONS_MS]
        ax[1].plot(hs, np.clip(y, -5, None), 'o-', label=NAMES[k])
    ax[1].set_xscale('log'); ax[1].set_xlabel('horizon (s)'); ax[1].set_ylabel('MSE improvement vs mid (%, clipped at -5)')
    ax[1].axhline(0, c='k', lw=.5); ax[1].legend(fontsize=7); ax[1].set_title('Oct out-of-sample')
    fig.tight_layout(); fig.savefig(OUT / 'scaled.png', dpi=120)
    sys.stdout.reconfigure(encoding='utf-8'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
