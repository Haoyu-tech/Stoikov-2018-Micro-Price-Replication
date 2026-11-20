"""Stoikov micro-price on TICK-LEVEL BNBUSDT spot L1 (every book_ticker update), vs 100 ms snapshot version.

A) Event-time Stoikov g (step = next L1 update), I in N=20 buckets, S in {1,2} ticks, symmetrized by mirroring:
     paper : |dM| <= 2 ticks (paper's K with m=2 spread states)
     full  : |dM| <= 50 ticks
B) 100 ms clock grid (as-of L1 state). TRAIN Jul-Sep fits per-horizon: beta for wmid and g_full, and table
   E[dM_h | bucket, S] (symmetrized). TEST Oct scores MSE of future mid change (ticks) and per-day paired diffs.
   Also scores the snapshot-trained table from results/microprice/scaled.json where horizons overlap.
Cache: E:/HFT_Data/Research/microprice_tick/cache (bnb_tick_bbo_cache.py). Output: results/microprice/tick/
"""
from pathlib import Path
import sys, json, datetime as dtm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = Path('E:/HFT_Data/Research/microprice_tick/cache')
OUT = Path(__file__).parent / 'results' / 'microprice' / 'tick'
SNAP = Path(__file__).parent / 'results' / 'microprice' / 'scaled.json'
TRAIN = ('2026-07-01', '2026-09-30')
TEST = ('2026-10-01', '2026-10-31')
N = 20
NS = 2 * N
GRID_US = 100_000
HORIZONS_MS = (25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000, 300000)


def days(a, b):
    d0, d1 = dtm.date.fromisoformat(a), dtm.date.fromisoformat(b)
    return [(d0 + dtm.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]


def load(day):
    z = np.load(CACHE / f'{day}.npz')
    return z['ts'], z['bp'].astype(np.int64), z['ap'].astype(np.int64), z['bq'].astype(np.float64), z['aq'].astype(np.float64)


def state(bp, ap, bq, aq):
    spr = ap - bp
    imb = bq / (bq + aq)
    b = np.clip((imb * N).astype(np.int64), 0, N - 1)
    x = np.where((spr == 1) | (spr == 2), (np.clip(spr, 1, 2) - 1) * N + b, -1)
    return bp + ap, spr, imb, x            # mid2 = 2*mid in ticks


def mirror(x):
    return (x // N) * N + (N - 1 - x % N)


# ---------------- A) event-time Stoikov ----------------
def stoikov_counts(maxj):
    Q = np.zeros(NS * NS); R2 = np.zeros(NS * NS); J = np.zeros(NS); C = np.zeros(NS)
    jump = []
    for d in days(*TRAIN):
        ts, bp, ap, bq, aq = load(d)
        mid2, _, _, x = state(bp, ap, bq, aq)
        x0, x1 = x[:-1], x[1:]
        dM = (mid2[1:] - mid2[:-1]) / 2.0
        k = (x0 >= 0) & (x1 >= 0) & (np.abs(dM) <= maxj)
        x0, x1, dM = x0[k], x1[k], dM[k]
        for a0, a1, dd in ((x0, x1, dM), (mirror(x0), mirror(x1), -dM)):
            z = dd == 0
            C += np.bincount(a0, minlength=NS)
            Q += np.bincount(a0[z] * NS + a1[z], minlength=NS * NS)
            R2 += np.bincount(a0[~z] * NS + a1[~z], minlength=NS * NS)
            J += np.bincount(a0[~z], weights=dd[~z], minlength=NS)
    Cd = np.maximum(C, 1)
    Qm = Q.reshape(NS, NS) / Cd[:, None]; R2m = R2.reshape(NS, NS) / Cd[:, None]
    inv = np.linalg.inv(np.eye(NS) - Qm)
    g1 = inv @ (J / Cd); B = inv @ R2m
    g, gi = g1.copy(), g1.copy()
    for _ in range(50):
        gi = B @ gi; g = g + gi
    return g, C


def jump_stats():
    a = []
    for d in days(*TRAIN)[::7]:
        _, bp, ap, _, _ = load(d)
        dm = np.diff(bp + ap) / 2.0
        a.append(dm[dm != 0])
    a = np.abs(np.concatenate(a))
    return dict(le_half=float((a <= 0.5).mean()), le1=float((a <= 1).mean()), le2=float((a <= 2).mean()),
                mean=float(a.mean()), p99=float(np.quantile(a, .99)))


# ---------------- B) clock-grid evaluation ----------------
def grid_day(d):
    ts, bp, ap, bq, aq = load(d)
    mid2, spr, imb, x = state(bp, ap, bq, aq)
    g0 = (ts[0] // GRID_US + 1) * GRID_US
    grid = np.arange(g0, ts[-1], GRID_US)
    i = np.searchsorted(ts, grid, side='right') - 1
    mid = mid2[i] / 2.0
    out = dict(x=x[i], w=(imb[i] - 0.5) * spr[i], mid=mid, fut={})
    for h in HORIZONS_MS:
        j = np.searchsorted(ts, grid + h * 1000, side='right') - 1
        v = (grid + h * 1000 <= ts[-1]) & (x[i] >= 0)
        out['fut'][h] = (v, mid2[j] / 2.0 - mid)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    js = jump_stats()
    g_paper, Cp = stoikov_counts(2)
    g_full, Cf = stoikov_counts(50)
    print('stoikov done', flush=True)

    st = {h: dict(xy=np.zeros(2), xx=np.zeros(2), s=np.zeros(NS), c=np.zeros(NS)) for h in HORIZONS_MS}
    for d in days(*TRAIN):
        a = grid_day(d)
        for h in HORIZONS_MS:
            v, dm = a['fut'][h]; x = a['x'][v]; dm = dm[v]; w = a['w'][v]; gm = g_full[x]
            t = st[h]
            t['xy'] += [(dm * w).sum(), (dm * gm).sum()]; t['xx'] += [(w * w).sum(), (gm * gm).sum()]
            t['s'] += np.bincount(x, weights=dm, minlength=NS) - np.bincount(mirror(x), weights=dm, minlength=NS)
            t['c'] += np.bincount(x, minlength=NS) + np.bincount(mirror(x), minlength=NS)
    par = {h: dict(beta=st[h]['xy'] / st[h]['xx'], table=st[h]['s'] / np.maximum(st[h]['c'], 1)) for h in HORIZONS_MS}
    print('fit done', flush=True)

    snap = json.load(open(SNAP)) if SNAP.exists() else None
    names = ['wmid', 'g_paper', 'g_full', 'b_wmid', 'b_gfull', 'table', 'table_snap']
    rows = {h: [] for h in HORIZONS_MS}
    midbp = []
    for d in days(*TEST):
        a = grid_day(d)
        midbp.append(a['mid'].mean())
        for h in HORIZONS_MS:
            v, dm = a['fut'][h]; x = a['x'][v]; dm = dm[v]; w = a['w'][v]; p = par[h]
            ts_tab = np.array(snap['table'][str(h)]) if snap and str(h) in snap['table'] else None
            preds = [np.zeros_like(dm), w, g_paper[x], g_full[x], p['beta'][0] * w, p['beta'][1] * g_full[x],
                     p['table'][x], ts_tab[x] if ts_tab is not None else np.full_like(dm, np.nan)]
            sse = np.array([((dm - q) ** 2).sum() for q in preds])
            # IC of table prediction
            tb = p['table'][x]
            ic = np.corrcoef(tb, dm)[0, 1] if dm.std() > 0 else np.nan
            rows[h].append((d, len(dm), sse, ic))
    bp_per_tick = 1e4 / np.mean(midbp)          # mid is in ticks

    L = [f'# BNBUSDT spot: micro-price on tick-level L1 book', '',
         f'Tick-level book_ticker ({TRAIN[0]}~{TEST[1]}, ~250M updates). Estimation {TRAIN[0]}~{TRAIN[1]}, evaluation {TEST[0]}~{TEST[1]} (used in earlier research, not a pristine hold-out).',
         f'I in {N} buckets, S∈{{1,2}} tick; evaluated on a 100 ms clock grid using the latest L1 state. 1 tick ≈ {bp_per_tick:.3f} bp (October average price).', '',
         '## Tick-level mid jump sizes (training sample, non-zero jumps)', '',
         f'|dM|≤0.5 tick {js["le_half"]:.1%}, ≤1 tick {js["le1"]:.1%}, ≤2 tick {js["le2"]:.1%}, mean {js["mean"]:.2f} tick, 99th pct {js["p99"]:.1f} tick. ', '',
         '## Event-time Stoikov g (S=1, ticks)', '', '| I bucket mid | g_paper(|dM|≤2) | g_full | wmid |', '|---|---|---|---|']
    for b in (0, 1, 2, 4, 6, 8, 9):
        c = (b + .5) / N
        L.append(f'| {c:.3f} | {g_paper[b]:+.3f} | {g_full[b]:+.3f} | {c-.5:+.3f} |')
    L += ['', '## October out-of-sample: MSE improvement vs mid', '',
          '| horizon | ' + ' | '.join(names) + ' | table IC (daily mean) | table extreme value tick / bp |', '|---|' + '---|' * (len(names) + 2)]
    summ = {}
    for h in HORIZONS_MS:
        r = rows[h]; tot = sum(x[2] for x in r)
        imp = 1 - tot[1:] / tot[0]
        ic = np.nanmean([x[3] for x in r])
        ext = par[h]['table'][0]
        summ[h] = dict(improve=dict(zip(names, imp.tolist())), ic=float(ic), beta=par[h]['beta'].tolist(), ext_tick=float(ext))
        L.append(f'| {h}ms | ' + ' | '.join('—' if np.isnan(v) else f'{v:+.2%}' for v in imp) +
                 f' | {ic:.3f} | {ext:+.2f} / {ext*bp_per_tick:+.2f} |')
    L += ['', '## Daily paired (31 days, pp, mean ± 2 s.e., win rate)', '',
          '| horizon | table − b_wmid | win rate | table(tick-trained) − table(snapshot-trained) | win rate |', '|---|---|---|---|---|']
    for h in HORIZONS_MS:
        dimp = np.array([1 - x[2][1:] / x[2][0] for x in rows[h]]) * 100
        c = []
        for k, base in ((5, 3), (5, 6)):
            dd = dimp[:, k] - dimp[:, base]
            c.append('— | —' if np.isnan(dd).any() else f'{dd.mean():+.3f} ± {2*dd.std(ddof=1)/np.sqrt(len(dd)):.3f} | {np.mean(dd>0):.0%}')
        L.append(f'| {h}ms | ' + ' | '.join(c) + ' |')
    (OUT / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    json.dump(dict(summary={str(k): v for k, v in summ.items()}, jumps=js, g_paper=g_paper.tolist(), g_full=g_full.tolist(),
                   table={str(h): par[h]['table'].tolist() for h in HORIZONS_MS}, bp_per_tick=bp_per_tick),
              open(OUT / 'tick.json', 'w'))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    c = (np.arange(N) + .5) / N
    for i, h in enumerate(HORIZONS_MS):
        ax[0].plot(c, par[h]['table'][:N], color=plt.cm.viridis(i / (len(HORIZONS_MS) - 1)), label=f'{h}ms')
    ax[0].plot(c, g_full[:N], 'k--', lw=2, label='Stoikov g full'); ax[0].plot(c, g_paper[:N], 'r--', lw=1.5, label='Stoikov g |dM|<=2')
    ax[0].set_xlabel('imbalance I (S=1)'); ax[0].set_ylabel('E[dM] ticks'); ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title('tick-level conditional mean move')
    hs = np.array(HORIZONS_MS) / 1000
    for nm in ('wmid', 'b_wmid', 'b_gfull', 'table', 'table_snap'):
        y = [summ[h]['improve'][nm] * 100 for h in HORIZONS_MS]
        ax[1].plot(hs, y, 'o-', label=nm)
    ax[1].set_xscale('log'); ax[1].axhline(0, c='k', lw=.5); ax[1].set_xlabel('horizon (s)')
    ax[1].set_ylabel('MSE improvement vs mid (%)'); ax[1].legend(fontsize=7); ax[1].set_title('Oct out-of-sample, 100ms grid')
    fig.tight_layout(); fig.savefig(OUT / 'tick.png', dpi=120)
    sys.stdout.reconfigure(encoding='utf-8'); print('\n'.join(L))


if __name__ == '__main__':
    main()
