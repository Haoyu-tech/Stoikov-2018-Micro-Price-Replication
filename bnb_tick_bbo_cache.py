"""Tick-level (every book_ticker update) BNBUSDT spot L1 cache for micro-price work.

Source: E:/HFT_Data/raw/binance/BNBUSDT/book_ticker/{day}.parquet (timestamp == local_timestamp for Binance spot).
Output: E:/HFT_Data/Research/microprice_tick/cache/{day}.npz  ts(int64 us), bp/ap (int32, 0.01 ticks), bq/aq (float32)
Usage: python bnb_tick_bbo_cache.py 2026-07-01 2026-10-31
"""
from pathlib import Path
import sys, json, datetime as dtm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path('E:/HFT_Data/raw/binance/BNBUSDT/book_ticker')
CACHE = Path('E:/HFT_Data/Research/microprice_tick/cache')
TICK = 0.01


def build(day):
    out = CACHE / f'{day}.npz'
    if out.exists():
        return day
    t = pq.read_table(SRC / f'{day}.parquet', columns=['timestamp', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount'])
    f = lambda c: t.column(c).cast(pa.float64()).to_numpy()
    ts = t.column('timestamp').to_numpy().astype(np.int64)
    o = np.argsort(ts, kind='stable')
    bp = np.rint(f('bid_price') / TICK).astype(np.int32)[o]; ap = np.rint(f('ask_price') / TICK).astype(np.int32)[o]
    bq = f('bid_amount').astype(np.float32)[o]; aq = f('ask_amount').astype(np.float32)[o]
    ts = ts[o]
    keep = (ap > bp) & (bq > 0) & (aq > 0)
    ts, bp, ap, bq, aq = ts[keep], bp[keep], ap[keep], bq[keep], aq[keep]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, ts=ts, bp=bp, ap=ap, bq=bq, aq=aq)
    (CACHE / f'{day}.json').write_text(json.dumps(dict(day=day, rows=int(len(ts)), dropped=int((~keep).sum()),
                                                       spread1=float(((ap - bp) == 1).mean()))))
    return day


def days(a, b):
    d0, d1 = dtm.date.fromisoformat(a), dtm.date.fromisoformat(b)
    return [(d0 + dtm.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    ds = [d for d in days(sys.argv[1], sys.argv[2]) if (SRC / f'{d}.parquet').exists()]
    with ProcessPoolExecutor(6) as ex:
        for d in ex.map(build, ds):
            pass
    print('built', len(ds))
