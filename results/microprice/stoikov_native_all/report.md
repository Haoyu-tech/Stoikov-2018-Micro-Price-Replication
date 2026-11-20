# Stoikov (2018) micro-price: paper estimation, evaluated on the paper's own target (all jumps kept, K not truncated)

Estimation: one row per L1 quote update, step = 1 row; 10 equal-frequency imbalance buckets, spread 1/2 tick; symmetrized; all jump sizes kept (K not truncated; only transitions where both rows have spread 1/2 tick).
g⁶ = the paper's micro-price adjustment; g^∞ = limit (200 terms).
Trained on Jul–Sep ticks, tested on October (a month used in earlier research).

Target (event time): yᵢ = mid after the i-th mid change − current mid (actual jump sizes, not truncated); one sample per valid quote-update row.
g_i = finite-step gⁱ matching the target step. Units: ticks.

## MSE improvement / MAE improvement (vs mid, i.e. predicting 0)

| i-th change | samples | share with a >1 tick jump in window | wmid | gⁱ | g⁶ (paper) | g^∞ |
|---|---|---|---|---|---|---|
| 1 | 61,486,966 | 45.4% | +6.96% / +8.67% | +12.84% / +14.89% | +5.59% / +1.64% | +3.68% / -0.51% |
| 2 | 61,485,304 | 60.1% | +5.65% / +4.90% | +14.32% / +14.14% | +14.80% / +14.25% | +14.50% / +13.89% |
| 3 | 61,484,385 | 72.4% | +4.63% / +3.56% | +14.48% / +13.94% | +15.40% / +15.31% | +15.55% / +15.57% |
| 4 | 61,483,714 | 80.4% | +3.87% / +2.71% | +13.65% / +12.04% | +14.23% / +12.88% | +14.52% / +13.34% |
| 5 | 61,483,017 | 85.6% | +3.29% / +2.19% | +12.57% / +10.37% | +12.79% / +10.65% | +13.11% / +11.10% |
| 6 | 61,481,874 | 89.2% | +2.87% / +1.81% | +11.55% / +8.86% | +11.55% / +8.86% | +11.88% / +9.25% |
| 10 | 61,478,489 | 96.1% | +1.84% / +1.04% | +8.16% / +5.25% | +7.92% / +5.06% | +8.20% / +5.29% |
| 20 | 61,466,732 | 99.5% | +0.92% / +0.47% | +4.26% / +2.34% | +4.10% / +2.25% | +4.26% / +2.34% |

## calibration slope (1 = correct magnitude) / daily IC

| i-th change | wmid | gⁱ | g⁶ | g^∞ |
|---|---|---|---|---|
| 1 | 3.04 / 0.370 | 1.17 / 0.376 | 0.57 / 0.374 | 0.54 / 0.374 |
| 2 | 4.72 / 0.396 | 1.30 / 0.398 | 0.88 / 0.397 | 0.84 / 0.397 |
| 3 | 6.17 / 0.401 | 1.38 / 0.402 | 1.15 / 0.402 | 1.09 / 0.402 |
| 4 | 7.35 / 0.396 | 1.49 / 0.396 | 1.36 / 0.396 | 1.30 / 0.396 |
| 5 | 8.30 / 0.385 | 1.59 / 0.385 | 1.54 / 0.385 | 1.47 / 0.385 |
| 6 | 9.11 / 0.374 | 1.69 / 0.374 | 1.69 / 0.374 | 1.61 / 0.374 |
| 10 | 11.32 / 0.330 | 2.01 / 0.330 | 2.10 / 0.330 | 1.99 / 0.330 |
| 20 | 13.45 / 0.251 | 2.36 / 0.251 | 2.49 / 0.251 | 2.36 / 0.251 |

## By bucket (spread 1 tick): realized mean yᵢ vs predicted gⁱ

| imbalance bucket | bucket upper edge | realized y₁ | g¹ | realized y₆ | g⁶ | realized y₂₀ | g²⁰ |
|---|---|---|---|---|---|---|---|
| 0 | 0.003 | -1.846 | -1.375 | -5.406 | -2.797 | -8.012 | -2.940 |
| 1 | 0.068 | -1.694 | -1.273 | -4.848 | -2.542 | -7.141 | -2.671 |
| 2 | 0.183 | -1.102 | -1.019 | -3.362 | -2.075 | -4.910 | -2.183 |
| 3 | 0.324 | -0.682 | -0.721 | -2.072 | -1.474 | -3.055 | -1.551 |
| 4 | 0.500 | -0.287 | -0.275 | -0.915 | -0.563 | -1.331 | -0.593 |
| 5 | 0.676 | +0.088 | +0.275 | +0.340 | +0.563 | +0.525 | +0.593 |
| 6 | 0.817 | +0.510 | +0.721 | +1.630 | +1.474 | +2.417 | +1.551 |
| 7 | 0.932 | +0.973 | +1.019 | +2.934 | +2.075 | +4.267 | +2.183 |
| 8 | 0.997 | +1.610 | +1.273 | +4.589 | +2.542 | +6.726 | +2.671 |
| 9 | 1.000 | +1.830 | +1.375 | +5.183 | +2.797 | +7.510 | +2.940 |

g⁶ range (spread 1): -2.797 ~ +2.797 tick; g^∞: -2.940 ~ +2.940 tick. 