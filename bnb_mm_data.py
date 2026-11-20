"""BNBUSDT spot per-day arrays for the tick-level market-making simulator (bnb_mm_sim.py).

Rows = book_snapshot_25 rows (~100 ms).  Prices stored as integer ticks (0.01 USDT).
book: bp/bv/ap/av (N,10) top-10 levels.  Trades attached to row i if ts[i-1] < trade_ts <= ts[i]
(book row i already reflects them), CSR by row: off (N+1), tp (ticks), tv (base qty), tsell (taker sell).
Output: E:/HFT_Data/Research/bnb_mm/cache/{day}.npz
"""
from pathlib import Path
import sys, json, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SYMBOL = 'BNBUSDT'
SRC = Path('E:/HFT_Data/raw/binance') / SYMBOL
CACHE = Path('E:/HFT_Data/Research/bnb_mm/cache')
LV = 10
TICK = 0.01


def col(t, name):
    return t.column(name).cast(pa.float64()).to_numpy()


def build(day, force=False):
    out = CACHE / f'{day}.npz'
    if out.exists() and not force:
        return out
    t0 = time.time()
    cols = ['timestamp'] + [f'{s}[{i}].{k}' for i in range(LV) for s in ('asks', 'bids') for k in ('price', 'amount')]
    b = pq.read_table(SRC / 'book_snapshot_25' / f'{day}.parquet', columns=cols)
    ts = b.column('timestamp').to_numpy().astype(np.int64)
    order = np.argsort(ts, kind='stable'); ts = ts[order]
    bp = np.stack([np.rint(col(b, f'bids[{i}].price') / TICK) for i in range(LV)], 1)[order].astype(np.int32)
    ap = np.stack([np.rint(col(b, f'asks[{i}].price') / TICK) for i in range(LV)], 1)[order].astype(np.int32)
    bv = np.stack([col(b, f'bids[{i}].amount') for i in range(LV)], 1)[order].astype(np.float32)
    av = np.stack([col(b, f'asks[{i}].amount') for i in range(LV)], 1)[order].astype(np.float32)
    del b
    t = pq.read_table(SRC / 'trades' / f'{day}.parquet', columns=['timestamp', 'side', 'price', 'amount'])
    tts = t.column('timestamp').to_numpy().astype(np.int64)
    tsell = pc.equal(t.column('side'), 'sell').to_numpy(zero_copy_only=False)
    tp = np.rint(col(t, 'price') / TICK).astype(np.int32); tv = col(t, 'amount').astype(np.float32)
    o = np.argsort(tts, kind='stable'); tts, tsell, tp, tv = tts[o], tsell[o], tp[o], tv[o]
    row = np.searchsorted(ts, tts, side='left')
    keep = (row >= 1) & (row < len(ts))
    row, tp, tv, tsell = row[keep], tp[keep], tv[keep], tsell[keep]
    off = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=len(ts)))]).astype(np.int64)
    assert (ap[:, 0] > bp[:, 0]).all() and np.all(np.diff(row) >= 0)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, ts=ts, bp=bp, bv=bv, ap=ap, av=av, off=off, tp=tp, tv=tv, tsell=tsell)
    meta = dict(day=day, rows=len(ts), trades=int(keep.sum()), dropped=int((~keep).sum()),
                mid=float((bp[:, 0] + ap[:, 0]).mean() * TICK / 2), spread1=float(((ap[:, 0] - bp[:, 0]) == 1).mean()),
                build_s=round(time.time() - t0, 1))
    (CACHE / f'{day}.json').write_text(json.dumps(meta))
    print(meta, flush=True)
    return out


def days(start, end):
    import datetime as dt
    d0 = dt.date.fromisoformat(start); d1 = dt.date.fromisoformat(end)
    return [(d0 + dt.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    a = sys.argv[1:]
    ds = days(a[0], a[1]) if len(a) == 2 else a
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(build, ds))
