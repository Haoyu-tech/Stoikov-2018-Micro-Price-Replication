"""Stoikov (2018) micro-price, estimated AND evaluated the paper's own way, on BNBUSDT spot tick L1 (book_ticker).

Estimation (paper settings, following its public estimation procedure as reconstructed here):
  * one row = one L1 quote update; step dt = 1 row
  * state = (imbalance bucket, spread); imbalance I = Qb/(Qb+Qa) in n_imb = 10 EQUAL-FREQUENCY buckets (quantiles of the
    symmetrized train imbalance); spread S in {1, 2} ticks (rows with wider spreads excluded)
  * symmetrization: every transition is added mirrored (I -> 1-I, dM -> -dM)
  * K = {-1, -0.5, +0.5, +1} tick: transitions whose next mid change is larger than 1 tick are DROPPED
  * Q = no-mid-change transitions, R1 = absorbing jump K, R2 = transitions with a mid change
    g1 = (1-Q)^-1 R1 K,  B = (1-Q)^-1 R2,  g^i = sum_{j<i} B^j g1;  paper micro-price = mid + g^6; limit g^inf (200 terms)
Train Jul-Sep 2026, test Oct 2026 (reused month).
Evaluation on the paper target (EVENT time, not clock time): for each Oct quote-update row t with a valid state,
  y_i = mid(tau_i) - mid(t), tau_i = i-th subsequent mid change (actual size, any jump), i in 1,2,3,4,5,6,10,20, same day.
Predictors: mid (0), weighted mid (I-0.5)*S, g^i (matching step i), g^6 (paper micro-price), g^inf.
Metrics: MSE, MAE, calibration slope, IC; calibration by imbalance bucket (spread 1) for i = 1, 6.
Output results/microprice/stoikov_native/report.md, native.json
(--all: keep every jump size, output results/microprice/stoikov_native_all/)
"""
import sys, json
from pathlib import Path
import numpy as np
import bnb_tick_microprice as T

ALLJ = '--all' in sys.argv                      # keep every jump size (no K truncation)
KMAX = np.inf if ALLJ else 1.0
OUT = Path(__file__).parent / 'results' / 'microprice' / ('stoikov_native_all' if ALLJ else 'stoikov_native')
NI = 10
NS = 2 * NI
STEPS = (1, 2, 3, 4, 5, 6, 10, 20)


def edges():
    s = []
    for d in T.days(*T.TRAIN)[::5]:
        _, bp, ap, bq, aq = T.load(d)
        i = bq / (bq + aq); s.append(i[::20]); s.append(1 - i[::20])
    q = np.quantile(np.concatenate(s), np.linspace(0, 1, NI + 1)[1:-1])
    return q


def state(bp, ap, bq, aq, E):
    spr = ap - bp; imb = bq / (bq + aq)
    b = np.searchsorted(E, imb, side='right')
    x = np.where((spr == 1) | (spr == 2), (np.clip(spr, 1, 2) - 1) * NI + b, -1)
    return bp + ap, spr, imb, x


def mirror(x):
    return (x // NI) * NI + (NI - 1 - x % NI)


def estimate(E):
    C = np.zeros(NS); Q = np.zeros(NS * NS); R2 = np.zeros(NS * NS); J = np.zeros(NS)
    drop = keep = 0
    for d in T.days(*T.TRAIN):
        _, bp, ap, bq, aq = T.load(d)
        mid2, _, _, x = state(bp, ap, bq, aq, E)
        x0, x1 = x[:-1], x[1:]; dM = (mid2[1:] - mid2[:-1]) / 2.0
        v = (x0 >= 0) & (x1 >= 0)
        k = v & (np.abs(dM) <= KMAX)
        drop += int((v & ~k).sum()); keep += int(k.sum())
        x0, x1, dM = x0[k], x1[k], dM[k]
        for a0, a1, dd in ((x0, x1, dM), (mirror(x0), mirror(x1), -dM)):
            z = dd == 0
            C += np.bincount(a0, minlength=NS)
            Q += np.bincount(a0[z] * NS + a1[z], minlength=NS * NS)
            R2 += np.bincount(a0[~z] * NS + a1[~z], minlength=NS * NS)
            J += np.bincount(a0[~z], weights=dd[~z], minlength=NS)
        print('train', d, flush=True)
    Cd = np.maximum(C, 1)
    Qm = Q.reshape(NS, NS) / Cd[:, None]; R2m = R2.reshape(NS, NS) / Cd[:, None]
    inv = np.linalg.inv(np.eye(NS) - Qm)
    g1 = inv @ (J / Cd); B = inv @ R2m
    G = {1: g1.copy()}; gi = g1.copy(); g = g1.copy()
    for i in range(2, 201):
        gi = B @ gi; g = g + gi
        G[i] = g.copy()
    return G, dict(rows_kept=keep, rows_dropped_bigjump=drop, drop_share=drop / (keep + drop), count=C.tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = edges(); G, est = estimate(E)
    names = ['mid', 'wmid', 'g_i', 'g6', 'g_inf']
    acc = {i: dict(n=0, se=np.zeros(5), ae=np.zeros(5), xy=np.zeros(5), xx=np.zeros(5), ic=[], cs=np.zeros(NS), cn=np.zeros(NS),
                   big=0) for i in STEPS}
    for d in T.days(*T.TEST):
        _, bp, ap, bq, aq = T.load(d)
        mid2, spr, imb, x = state(bp, ap, bq, aq, E)
        mid = mid2 / 2.0
        chg = np.flatnonzero(mid2[1:] != mid2[:-1]) + 1          # row index where a new mid appears
        big = np.abs(np.diff(mid)[chg - 1]) > 1.0                # jump bigger than 1 tick at that change
        rows = np.flatnonzero(x >= 0)
        p = np.searchsorted(chg, rows, side='right')             # index of first change strictly after row
        w = (imb - 0.5) * spr
        for i in STEPS:
            v = p + i - 1 < len(chg)
            r = rows[v]; tau = chg[p[v] + i - 1]
            y = mid[tau] - mid[r]; xs = x[r]
            P = np.array([np.zeros(len(y)), w[r], G[i][xs], G[6][xs], G[200][xs]])
            a = acc[i]
            a['n'] += len(y); a['se'] += ((y - P) ** 2).sum(1); a['ae'] += np.abs(y - P).sum(1)
            a['xy'] += (P * y).sum(1); a['xx'] += (P * P).sum(1)
            a['ic'].append([np.nan] + [np.corrcoef(q, y)[0, 1] for q in P[1:]])
            a['cs'] += np.bincount(xs, weights=y, minlength=NS); a['cn'] += np.bincount(xs, minlength=NS)
            cb = np.concatenate([[0], np.cumsum(big)])
            a['big'] += int((cb[p[v] + i] - cb[p[v]] > 0).sum())
        print('test', d, flush=True)

    L = ['# Stoikov (2018) micro-price: paper estimation, evaluated on the paper target' + (' (all jumps kept, K not truncated)' if ALLJ else ''), '',
         'Estimation: one row per L1 quote update, step = 1 row; 10 equal-frequency imbalance buckets, spread 1/2 tick; symmetrized; ' +
         ('all jump sizes kept (K not truncated; only transitions where both rows have spread 1/2 tick).' if ALLJ else 'only transitions whose next mid change is <= 1 tick are kept'),
         ('' if ALLJ else f' (K = ±0.5, ±1 tick; share of transitions dropped for jumps > 1 tick: {est["drop_share"]:.2%}). ') + 'g⁶ = the paper micro-price adjustment; g^∞ = limit (200 terms).',
         'Trained on Jul–Sep ticks, tested on October (a month used in earlier research).', '',
         'Target (event time): yᵢ = mid after the i-th mid change − current mid (actual jump sizes, not truncated); one sample per valid quote-update row.',
         'g_i = finite-step gⁱ matching the target step. Units: ticks.', '',
         '## MSE improvement / MAE improvement (vs mid, i.e. predicting 0)', '',
         '| i-th change | samples | share with a >1 tick jump in window | wmid | gⁱ | g⁶ (paper) | g^∞ |', '|---|---|---|---|---|---|---|']
    summ = {}
    for i in STEPS:
        a = acc[i]; n = a['n']
        mg = 1 - a['se'] / a['se'][0]; ag = 1 - a['ae'] / a['ae'][0]
        summ[i] = dict(n=n, mse=(a['se'] / n).tolist(), mae=(a['ae'] / n).tolist(), mse_gain=mg.tolist(), mae_gain=ag.tolist(),
                       slope=(a['xy'] / np.maximum(a['xx'], 1e-12)).tolist(), ic=np.nanmean(a['ic'], 0).tolist(),
                       big_share=a['big'] / n, bucket_mean=(a['cs'] / np.maximum(a['cn'], 1)).tolist())
        L.append(f'| {i} | {n:,} | {a["big"]/n:.1%} | ' + ' | '.join(f'{mg[j]:+.2%} / {ag[j]:+.2%}' for j in range(1, 5)) + ' |')
    L += ['', '## calibration slope (1 = correct magnitude) / daily IC', '', '| i-th change | wmid | gⁱ | g⁶ | g^∞ |', '|---|---|---|---|---|']
    for i in STEPS:
        s = summ[i]
        L.append(f'| {i} | ' + ' | '.join(f'{s["slope"][j]:.2f} / {s["ic"][j]:.3f}' for j in range(1, 5)) + ' |')
    L += ['', '## By bucket (spread 1 tick): realized mean yᵢ vs predicted gⁱ', '',
          '| imbalance bucket | bucket upper edge | realized y₁ | g¹ | realized y₆ | g⁶ | realized y₂₀ | g²⁰ |', '|---|---|---|---|---|---|---|---|']
    ub = list(E) + [1.0]
    for b in range(NI):
        L.append(f'| {b} | {ub[b]:.3f} | {summ[1]["bucket_mean"][b]:+.3f} | {G[1][b]:+.3f} | {summ[6]["bucket_mean"][b]:+.3f} | {G[6][b]:+.3f} | '
                 f'{summ[20]["bucket_mean"][b]:+.3f} | {G[20][b]:+.3f} |')
    L += ['', f'g⁶ range (spread 1): {G[6][:NI].min():+.3f} ~ {G[6][:NI].max():+.3f} tick; g^∞: {G[200][:NI].min():+.3f} ~ {G[200][:NI].max():+.3f} tick. ']
    (OUT / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    json.dump(dict(edges=E.tolist(), est={k: v for k, v in est.items() if k != 'count'}, G={str(k): G[k].tolist() for k in (1, 2, 3, 4, 5, 6, 10, 20, 200)},
                   summary={str(k): v for k, v in summ.items()}), open(OUT / 'native.json', 'w'), indent=1)
    sys.stdout.reconfigure(encoding='utf-8'); print('\n'.join(L))


if __name__ == '__main__':
    main()
