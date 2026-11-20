"""Stoikov (2018) micro-price with recent mid changes added to the Markov state. Paper estimation, ALL jump sizes kept.

State = (spread 1/2) x (10 equal-frequency imbalance buckets, same edges as stoikov_native) x memory category:
  none : no memory (= stoikov_native --all; consistency check)
  last : last mid change before/at the row, 4 cats: <=-1.5, -1..-0.5, +0.5..+1, >=+1.5 tick
  cum3 : sum of the last 3 mid changes, 5 cats: <=-2.5, -2..-0.5, 0, +0.5..+2, >=+2.5 tick
  *_age: same memory x rows since the last mid change (age), 3 cats: 0-4, 5-29, 30+ rows (age evolves inside Q)
Rows before enough mid changes on the day are invalid. Symmetrization mirrors imbalance, flips memory, negates dM.
g1 = (1-Q)^-1 R1 K, B = (1-Q)^-1 R2, g^i = sum_{j<i} B^j g1 (K = actual jump sizes). Train Jul-Sep, test Oct (reused).
Evaluation on the paper event-time target y_i = mid(tau_i) - mid(t), i in 1,2,3,4,5,6,10,20, every valid row (rows valid in
ALL variants, so samples are identical across variants). Predictor = matched g^i, also g^6.
Output results/microprice/stoikov_memory_age/report.md, memory.json (first run without *_age variants: stoikov_memory/)
"""
import sys, json
from pathlib import Path
import numpy as np
import bnb_tick_microprice as T

R = Path(__file__).parent / 'results' / 'microprice'
OUT = R / 'stoikov_memory_age'
E = np.array(json.load(open(R / 'stoikov_native' / 'native.json'))['edges'])
NI = 10
STEPS = (1, 2, 3, 4, 5, 6, 10, 20)
VAR = {'none': (1, None, 0, 1), 'last': (4, [-1.25, 0.0, 1.25], 1, 1), 'cum3': (5, [-2.25, -0.25, 0.25, 2.25], 3, 1),
       'last_age': (4, [-1.25, 0.0, 1.25], 1, 3), 'cum3_age': (5, [-2.25, -0.25, 0.25, 2.25], 3, 3)}  # NM, edges, n changes, NA
AGE_EDGES = [5, 30]


def states(bp, ap, bq, aq):
    mid2 = bp + ap; spr = ap - bp; imb = bq / (bq + aq)
    b = np.searchsorted(E, imb, side='right')
    base = np.where((spr == 1) | (spr == 2), (np.clip(spr, 1, 2) - 1) * NI + b, -1)
    chg = np.flatnonzero(mid2[1:] != mid2[:-1]) + 1
    dm = (mid2[chg] - mid2[chg - 1]) / 2.0; cs = np.concatenate([[0.0], np.cumsum(dm)])
    k = np.searchsorted(chg, np.arange(len(mid2)), side='right')          # number of changes at or before row
    rowi = np.arange(len(mid2)); age = rowi - chg[np.maximum(k - 1, 0)] if len(chg) else rowi
    acat = np.digitize(age, AGE_EDGES); lsign = np.sign(cs[k] - cs[np.maximum(k - 1, 0)])
    X = {}
    for v, (nm, ed, nc, na) in VAR.items():
        if nm == 1:
            X[v] = base.copy(); continue
        mem = cs[k] - cs[np.maximum(k - nc, 0)]
        cat = np.digitize(mem, ed)
        X[v] = np.where((base >= 0) & (k >= nc), (base * nm + cat) * na + (acat if na > 1 else 0), -1)
    valid_all = np.all([X[v] >= 0 for v in VAR], 0)
    return mid2, spr, imb, X, chg, valid_all, acat, lsign


def mirror(x, nm, na):
    s, r = divmod(x, NI * nm * na); b, r2 = divmod(r, nm * na); c, a = divmod(r2, na)
    return s * NI * nm * na + (NI - 1 - b) * nm * na + (nm - 1 - c) * na + a


def solve(C, Q, R2, J, ns):
    Cd = np.maximum(C, 1)
    Qm = Q.reshape(ns, ns) / Cd[:, None]; R2m = R2.reshape(ns, ns) / Cd[:, None]
    inv = np.linalg.pinv(np.eye(ns) - Qm)
    g1 = inv @ (J / Cd); B = inv @ R2m
    G = {1: g1.copy()}; gi = g1.copy(); g = g1.copy()
    for i in range(2, 201):
        gi = B @ gi; g = g + gi; G[i] = g.copy()
    return G


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    NS = {v: 2 * NI * VAR[v][0] * VAR[v][3] for v in VAR}
    acc = {v: dict(C=np.zeros(NS[v]), Q=np.zeros(NS[v] ** 2), R2=np.zeros(NS[v] ** 2), J=np.zeros(NS[v])) for v in VAR}
    for d in T.days(*T.TRAIN):
        _, bp, ap, bq, aq = T.load(d)
        mid2, _, _, X, _, _, _, _ = states(bp, ap, bq, aq)
        dM = (mid2[1:] - mid2[:-1]) / 2.0
        for v in VAR:
            nm, na, ns = VAR[v][0], VAR[v][3], NS[v]; x0, x1 = X[v][:-1], X[v][1:]
            k = (x0 >= 0) & (x1 >= 0); a0, a1, dd = x0[k], x1[k], dM[k]
            for p0, p1, q in ((a0, a1, dd), (mirror(a0, nm, na), mirror(a1, nm, na), -dd)):
                z = q == 0; A = acc[v]
                A['C'] += np.bincount(p0, minlength=ns)
                A['Q'] += np.bincount(p0[z] * ns + p1[z], minlength=ns * ns)
                A['R2'] += np.bincount(p0[~z] * ns + p1[~z], minlength=ns * ns)
                A['J'] += np.bincount(p0[~z], weights=q[~z], minlength=ns)
        print('train', d, flush=True)
    G = {v: solve(acc[v]['C'], acc[v]['Q'], acc[v]['R2'], acc[v]['J'], NS[v]) for v in VAR}
    mincount = {v: int(acc[v]['C'].min()) for v in VAR}

    names = list(VAR)
    ev = {i: dict(n=0, y2=0.0, ay=0.0, se=np.zeros(2 * len(names)), ae=np.zeros(2 * len(names)), xy=np.zeros(2 * len(names)),
                  xx=np.zeros(2 * len(names)), ic=[], bs=np.zeros(NI), bn=np.zeros(NI), bp=np.zeros((len(names), NI)),
                  ms=np.zeros(5), mn=np.zeros(5), mp=np.zeros((len(names), 5)),
                  as_=np.zeros(3), an=np.zeros(3), ap=np.zeros((len(names), 3))) for i in STEPS}
    for d in T.days(*T.TEST):
        _, bp, ap, bq, aq = T.load(d)
        mid2, spr, imb, X, chg, va, acat, lsign = states(bp, ap, bq, aq)
        mid = mid2 / 2.0; rows = np.flatnonzero(va)
        p = np.searchsorted(chg, rows, side='right')
        for i in STEPS:
            v = p + i - 1 < len(chg); r = rows[v]; y = mid[chg[p[v] + i - 1]] - mid[r]
            P = np.array([G[nm][i][X[nm][r]] for nm in names] + [G[nm][6][X[nm][r]] for nm in names])
            e = ev[i]; e['n'] += len(y); e['y2'] += (y * y).sum(); e['ay'] += np.abs(y).sum()
            e['se'] += ((y - P) ** 2).sum(1); e['ae'] += np.abs(y - P).sum(1); e['xy'] += (P * y).sum(1); e['xx'] += (P * P).sum(1)
            e['ic'].append([np.corrcoef(q, y)[0, 1] for q in P])
            s1 = X['none'][r] < NI                                              # spread 1 rows
            bb = X['none'][r][s1]
            e['bs'] += np.bincount(bb, y[s1], NI); e['bn'] += np.bincount(bb, minlength=NI)
            for j in range(len(names)):
                e['bp'][j] += np.bincount(bb, P[j][s1], NI)
            bal = s1 & ((X['none'][r] % NI == 4) | (X['none'][r] % NI == 5))   # balanced book: momentum continuation by age
            ac = acat[r][bal]; sg = lsign[r][bal]
            e['as_'] += np.bincount(ac, y[bal] * sg, 3); e['an'] += np.bincount(ac, minlength=3)
            for j in range(len(names)):
                e['ap'][j] += np.bincount(ac, P[j][bal] * sg, 3)
            mc = X['cum3'][r][s1] % 5                                            # memory category (cum3) for spread-1 rows
            e['ms'] += np.bincount(mc, y[s1], 5); e['mn'] += np.bincount(mc, minlength=5)
            for j in range(len(names)):
                e['mp'][j] += np.bincount(mc, P[j][s1], 5)
        print('test', d, flush=True)

    L = ['# Stoikov micro-price + recent price changes (memory states)', '',
         'Paper estimation, all jumps kept; state = (spread 1/2) × (10 equal-frequency imbalance buckets, same edges as stoikov_native) × memory category:',
         '- none: no memory (= all-jumps version, as control)',
         '- last: last mid change, 4 classes: ≤−1.5, −1~−0.5, +0.5~+1, ≥+1.5 tick',
         '- cum3: sum of the last 3 mid changes, 5 classes: ≤−2.5, −2~−0.5, 0, +0.5~+2, ≥+2.5 tick',
         '- last_age / cum3_age: as above × quote-update rows since the last mid change, 3 classes: 0–4, 5–29, 30+ rows (age advances inside no-change transitions)',
         'Symmetrization: mirror imbalance, flip memory, negate jumps. Trained on Jul–Sep ticks, tested on October (a month used in earlier research).',
         'Evaluation: the paper event-time target yᵢ = mid after the i-th mid change − current mid; all variants use identical samples (rows valid in every variant).',
         f'Minimum training samples per state: ' + ', '.join(f'{v} {mincount[v]:,}' for v in names) + '. Units: ticks.', '',
         '## Matched-step gⁱ: MSE improvement / MAE improvement (vs mid)', '', '| i-th change | samples | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    summ = {}
    for i in STEPS:
        e = ev[i]; n = e['n']
        mg = 1 - e['se'] / e['y2']; ag = 1 - e['ae'] / e['ay']
        summ[i] = dict(n=n, mse_gain=mg.tolist(), mae_gain=ag.tolist(), slope=(e['xy'] / np.maximum(e['xx'], 1e-12)).tolist(),
                       ic=np.nanmean(e['ic'], 0).tolist(), bucket_y=(e['bs'] / np.maximum(e['bn'], 1)).tolist(),
                       bucket_p=(e['bp'] / np.maximum(e['bn'], 1)).tolist(), mem_y=(e['ms'] / np.maximum(e['mn'], 1)).tolist(),
                       mem_p=(e['mp'] / np.maximum(e['mn'], 1)).tolist(), mem_n=e['mn'].tolist(),
                       age_y=(e['as_'] / np.maximum(e['an'], 1)).tolist(), age_p=(e['ap'] / np.maximum(e['an'], 1)).tolist(), age_n=e['an'].tolist())
        L.append(f'| {i} | {n:,} | ' + ' | '.join(f'{mg[j]:+.2%} / {ag[j]:+.2%}' for j in range(len(names))) + ' |')
    L += ['', '## Paper g⁶ (same for all i): MSE improvement / MAE improvement', '', '| i-th change | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for i in STEPS:
        s = summ[i]; k = len(names)
        L.append(f'| {i} | ' + ' | '.join(f'{s["mse_gain"][k+j]:+.2%} / {s["mae_gain"][k+j]:+.2%}' for j in range(k)) + ' |')
    L += ['', '## Matched-step gⁱ: calibration slope / daily IC', '', '| i-th change | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for i in STEPS:
        s = summ[i]
        L.append(f'| {i} | ' + ' | '.join(f'{s["slope"][j]:.2f} / {s["ic"][j]:.3f}' for j in range(len(names))) + ' |')
    for i in (1, 6, 20):
        s = summ[i]
        L += ['', f'## By imbalance bucket (spread 1): realized mean y{i} vs mean predicted g^{i}', '', '| bucket | realized | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
        for b in range(NI):
            L.append(f'| {b} | {s["bucket_y"][b]:+.3f} | ' + ' | '.join(f'{s["bucket_p"][j][b]:+.3f}' for j in range(len(names))) + ' |')
    ml = ['≤−2.5', '−2~−0.5', '0', '+0.5~+2', '≥+2.5']
    for i in (1, 6, 20):
        s = summ[i]
        L += ['', f'## By sum of last 3 changes (spread 1): realized mean y{i} vs mean predicted g^{i}', '', '| sum of last 3 | sample share | realized | ' + ' | '.join(names) + ' |',
              '|---|---|---|' + '---|' * len(names)]
        tot = sum(s['mem_n'])
        for c in range(5):
            L.append(f'| {ml[c]} | {s["mem_n"][c]/tot:.1%} | {s["mem_y"][c]:+.3f} | ' + ' | '.join(f'{s["mem_p"][j][c]:+.3f}' for j in range(len(names))) + ' |')
    al = ['0–4 rows', '5–29 rows', '30+ rows']
    for i in (1, 6, 20):
        s = summ[i]
        L += ['', f'## Momentum by time since last change (spread 1, imbalance buckets 4/5): realized vs predicted y{i} × sign of last change', '',
              '| time since change | sample share | realized | ' + ' | '.join(names) + ' |', '|---|---|---|' + '---|' * len(names)]
        tot = sum(s['age_n'])
        for c in range(3):
            L.append(f'| {al[c]} | {s["age_n"][c]/tot:.1%} | {s["age_y"][c]:+.3f} | ' + ' | '.join(f'{s["age_p"][j][c]:+.3f}' for j in range(len(names))) + ' |')
    (OUT / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    json.dump(dict(names=names, mincount=mincount, G={v: {str(k): G[v][k].tolist() for k in (1, 6, 20, 200)} for v in names},
                   summary={str(k): v for k, v in summ.items()}), open(OUT / 'memory.json', 'w'), indent=1)
    sys.stdout.reconfigure(encoding='utf-8'); print('\n'.join(L))


if __name__ == '__main__':
    main()
