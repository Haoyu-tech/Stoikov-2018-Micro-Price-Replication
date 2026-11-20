"""Stoikov (2018) micro-price on BNBUSDT spot.

Data: E:/HFT_Data/Research/bnb_mm/cache/{day}.npz (book_snapshot_25 rows, prices in 0.01 ticks).
Estimate g(I,S) on TRAIN days (dt = 1 row), evaluate out-of-sample on TEST days:
    predictors of M_{t+h}: mid, weighted mid, micro-price (h in clock time via ts).
Usage: python bnb_spot_microprice.py [n_imb] [dt_rows]
"""
from pathlib import Path
import sys, json, datetime as dtm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = Path('E:/HFT_Data/Research/bnb_mm/cache')
OUT = Path(__file__).parent / 'results' / 'microprice'
TRAIN = ('2026-07-01', '2026-09-30')
TEST = ('2026-10-01', '2026-10-31')
N_IMB = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DT = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FULL = len(sys.argv) > 3 and sys.argv[3] == 'full'   # keep all jumps |dM|<=MAXJ (not only +-0.5/1 tick)
MAXJ = 50
SPREADS = (1, 2)
K = np.array([-1.0, -0.5, 0.5, 1.0])
HORIZONS_MS = (100, 1000, 5000, 30000)


def days(a, b):
    d0, d1 = dtm.date.fromisoformat(a), dtm.date.fromisoformat(b)
    return [(d0 + dtm.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]


def load(day):
    z = np.load(CACHE / f'{day}.npz')
    bp, ap = z['bp'][:, 0].astype(np.int64), z['ap'][:, 0].astype(np.int64)
    bv, av = z['bv'][:, 0].astype(np.float64), z['av'][:, 0].astype(np.float64)
    return z['ts'], bp, ap, bv, av


def state(bp, ap, bv, av):
    mid2 = bp + ap                       # 2*mid in ticks (integer)
    spr = ap - bp
    imb = bv / (bv + av)
    ib = np.clip((imb * N_IMB).astype(int), 0, N_IMB - 1)
    ok = np.isin(spr, SPREADS)
    x = np.where(ok, (np.clip(spr, 1, 2) - 1) * N_IMB + ib, -1)
    return mid2, spr, imb, x


def counts_day(day, Qc, R1c, R2c, Nc, Jc):
    _, bp, ap, bv, av = load(day)
    mid2, _, _, x = state(bp, ap, bv, av)
    x0, x1 = x[:-DT], x[DT:]
    dM = (mid2[DT:] - mid2[:-DT]) / 2.0
    keep = (x0 >= 0) & (x1 >= 0) & ((np.abs(dM) <= MAXJ) if FULL else np.isin(dM, [-1, -0.5, 0, 0.5, 1]))
    x0, x1, dM = x0[keep], x1[keep], dM[keep]
    # symmetrize: I -> 1-I  (bucket b -> N-1-b), dM -> -dM
    def mirror(s):
        sp, b = s // N_IMB, s % N_IMB
        return sp * N_IMB + (N_IMB - 1 - b)
    for a0, a1, d in ((x0, x1, dM), (mirror(x0), mirror(x1), -dM)):
        np.add.at(Nc, a0, 1)
        z = d == 0
        np.add.at(Qc, (a0[z], a1[z]), 1)
        nz = ~z
        np.add.at(R2c, (a0[nz], a1[nz]), 1)
        np.add.at(Jc, a0[nz], d[nz])
        if not FULL:
            np.add.at(R1c, (a0[nz], np.searchsorted(K, d[nz])), 1)
    return int(keep.sum())


def estimate():
    ns = len(SPREADS) * N_IMB
    Qc, R2c = np.zeros((ns, ns)), np.zeros((ns, ns))
    R1c, Nc, Jc = np.zeros((ns, len(K))), np.zeros(ns), np.zeros(ns)
    n = sum(counts_day(d, Qc, R1c, R2c, Nc, Jc) for d in days(*TRAIN) if (CACHE / f'{d}.npz').exists())
    den = np.maximum(Nc, 1)[:, None]
    Q, R1, R2 = Qc / den, R1c / den, R2c / den
    inv = np.linalg.inv(np.eye(ns) - Q)
    g1 = inv @ (Jc / np.maximum(Nc, 1))   # == inv @ R1 @ K when not FULL
    B = inv @ R2
    g, gi, path = g1.copy(), g1.copy(), [g1.copy()]
    for _ in range(12):
        gi = B @ gi; g = g + gi; path.append(g.copy())
    return dict(g=g, g1=g1, path=np.array(path), Nc=Nc, n=n, B=B)


def evaluate(g):
    # accumulate SSE of (M_{t+h} - P_t) for P in {mid, wmid, micro}, plus OLS of dM on adj
    acc = {h: dict(n=0, sse=np.zeros(3), xy=np.zeros(2), xx=np.zeros(2)) for h in HORIZONS_MS}
    per_day = []
    for d in days(*TEST):
        if not (CACHE / f'{d}.npz').exists():
            continue
        ts, bp, ap, bv, av = load(d)
        mid2, spr, imb, x = state(bp, ap, bv, av)
        mid = mid2 / 2.0
        wadj = (imb - 0.5) * spr               # weighted mid - mid (ticks)
        madj = np.where(x >= 0, g[np.maximum(x, 0)], 0.0)
        row = {'day': d}
        for h in HORIZONS_MS:
            j = np.searchsorted(ts, ts + h * 1000, side='left')   # ts in microseconds
            v = (j < len(ts)) & (x >= 0)
            fut = mid[np.minimum(j, len(ts) - 1)][v] - mid[v]
            e = np.stack([fut, fut - wadj[v], fut - madj[v]])
            a = acc[h]; a['n'] += v.sum(); a['sse'] += (e ** 2).sum(1)
            a['xy'] += [(fut * wadj[v]).sum(), (fut * madj[v]).sum()]
            a['xx'] += [(wadj[v] ** 2).sum(), (madj[v] ** 2).sum()]
            se = (e ** 2).mean(1)
            row[h] = (1 - se[1] / se[0], 1 - se[2] / se[0])
        per_day.append(row)
    return acc, per_day


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f'n{N_IMB}_dt{DT}' + ('_full' if FULL else '')
    est = estimate()
    g = est['g']
    acc, per_day = evaluate(g)

    lines = [f'# BNBUSDT spot micro-price ({tag})', '',
             f'Train {TRAIN[0]}~{TRAIN[1]}, sample pairs {est["n"]:,} (before symmetrization); test {TEST[0]}~{TEST[1]}. ', '',
             '## g(I,S) (ticks; weighted-mid offset = (I-0.5)*S)', '',
             '| I bucket mid | S=1 g | S=1 wmid | S=1 n | S=2 g | S=2 wmid | S=2 n |', '|---|---|---|---|---|---|---|']
    for b in range(N_IMB):
        c = (b + 0.5) / N_IMB
        lines.append(f'| {c:.2f} | {g[b]:+.4f} | {(c-0.5):+.3f} | {int(est["Nc"][b]):,} | '
                     f'{g[N_IMB+b]:+.4f} | {2*(c-0.5):+.3f} | {int(est["Nc"][N_IMB+b]):,} |')
    lines += ['', '## Out-of-sample: MSE (tick²) predicting future mid, and improvement vs mid', '',
              '| horizon | n | MSE mid | MSE wmid | MSE micro | wmid gain | micro gain | OLS coef wmid | OLS coef micro | daily win rate micro>wmid |',
              '|---|---|---|---|---|---|---|---|---|---|']
    summary = {}
    for h in HORIZONS_MS:
        a = acc[h]; m = a['sse'] / a['n']
        beta = a['xy'] / a['xx']
        win = np.mean([r[h][1] > r[h][0] for r in per_day])
        summary[h] = dict(mse=m.tolist(), beta=beta.tolist(), win=win)
        lines.append(f'| {h}ms | {a["n"]:,} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} | '
                     f'{1-m[1]/m[0]:+.2%} | {1-m[2]/m[0]:+.2%} | {beta[0]:.3f} | {beta[1]:.3f} | {win:.0%} |')
    (OUT / f'report_{tag}.md').write_text('\n'.join(lines), encoding='utf-8')
    np.savez(OUT / f'g_{tag}.npz', g=g, g1=est['g1'], path=est['path'], Nc=est['Nc'])
    json.dump(summary, open(OUT / f'summary_{tag}.json', 'w'), indent=1)

    c = (np.arange(N_IMB) + 0.5) / N_IMB
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in range(2):
        ax[0].plot(c, g[s*N_IMB:(s+1)*N_IMB], 'o-', label=f'micro g, S={s+1}')
        ax[0].plot(c, (s+1)*(c-0.5), '--', label=f'wmid, S={s+1}')
    ax[0].axhline(0, c='k', lw=.5); ax[0].set_xlabel('imbalance I'); ax[0].set_ylabel('adjustment (ticks)')
    ax[0].set_title('BNBUSDT spot g(I,S)'); ax[0].legend(fontsize=8)
    for i in range(est['path'].shape[0]):
        ax[1].plot(c, est['path'][i, :N_IMB], color=plt.cm.viridis(i / est['path'].shape[0]), lw=1)
    ax[1].set_title('convergence of sum g^i (S=1)'); ax[1].set_xlabel('imbalance I')
    fig.tight_layout(); fig.savefig(OUT / f'g_{tag}.png', dpi=120)
    sys.stdout.reconfigure(encoding='utf-8'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
