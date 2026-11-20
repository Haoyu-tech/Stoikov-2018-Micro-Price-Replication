# Stoikov (2018) micro-price: paper estimation, evaluated on the paper's own target

Estimation: one row per L1 quote update, step = 1 row; 10 equal-frequency imbalance buckets, spread 1/2 tick; symmetrized; only transitions whose next mid change is <= 1 tick are kept
 (K = ±0.5, ±1 tick; share of transitions dropped for jumps > 1 tick: 0.53%). g⁶ = the paper's micro-price adjustment; g^∞ = limit (200 terms).
Trained on Jul–Sep ticks, tested on October (a month used in earlier research).

Target (event time): yᵢ = mid after the i-th mid change − current mid (actual jump sizes, not truncated); one sample per valid quote-update row.
g_i = finite-step gⁱ matching the target step. Units: ticks.

## MSE improvement / MAE improvement (vs mid, i.e. predicting 0)

| i-th change | samples | share with a >1 tick jump in window | wmid | gⁱ | g⁶ (paper) | g^∞ |
|---|---|---|---|---|---|---|
| 1 | 61,486,966 | 45.4% | +6.96% / +8.67% | +7.42% / +9.42% | +12.60% / +14.66% | +12.83% / +14.12% |
| 2 | 61,485,304 | 60.1% | +5.65% / +4.90% | +8.47% / +7.80% | +11.98% / +11.71% | +12.64% / +12.37% |
| 3 | 61,484,385 | 72.4% | +4.63% / +3.56% | +8.59% / +7.12% | +10.38% / +8.96% | +11.07% / +9.73% |
| 4 | 61,483,714 | 80.4% | +3.87% / +2.71% | +8.06% / +6.11% | +8.91% / +6.87% | +9.55% / +7.46% |
| 5 | 61,483,017 | 85.6% | +3.29% / +2.19% | +7.40% / +5.29% | +7.69% / +5.53% | +8.27% / +6.01% |
| 6 | 61,481,874 | 89.2% | +2.87% / +1.81% | +6.78% / +4.58% | +6.78% / +4.58% | +7.30% / +4.98% |
| 10 | 61,478,489 | 96.1% | +1.84% / +1.04% | +4.73% / +2.81% | +4.44% / +2.63% | +4.80% / +2.85% |
| 20 | 61,466,732 | 99.5% | +0.92% / +0.47% | +2.42% / +1.28% | +2.24% / +1.18% | +2.42% / +1.28% |

## calibration slope (1 = correct magnitude) / daily IC

| i-th change | wmid | gⁱ | g⁶ | g^∞ |
|---|---|---|---|---|
| 1 | 3.04 / 0.370 | 2.91 / 0.375 | 1.18 / 0.373 | 1.08 / 0.373 |
| 2 | 4.72 / 0.396 | 2.96 / 0.397 | 1.82 / 0.397 | 1.66 / 0.397 |
| 3 | 6.17 / 0.401 | 3.04 / 0.402 | 2.38 / 0.402 | 2.17 / 0.402 |
| 4 | 7.35 / 0.396 | 3.20 / 0.396 | 2.83 / 0.396 | 2.58 / 0.396 |
| 5 | 8.30 / 0.385 | 3.35 / 0.385 | 3.19 / 0.385 | 2.92 / 0.385 |
| 6 | 9.11 / 0.374 | 3.50 / 0.374 | 3.50 / 0.374 | 3.20 / 0.374 |
| 10 | 11.32 / 0.330 | 4.04 / 0.330 | 4.34 / 0.330 | 3.97 / 0.330 |
| 20 | 13.45 / 0.251 | 4.71 / 0.251 | 5.16 / 0.251 | 4.71 / 0.251 |

## By bucket (spread 1 tick): realized mean yᵢ vs predicted gⁱ

| imbalance bucket | bucket upper edge | realized y₁ | g¹ | realized y₆ | g⁶ | realized y₂₀ | g²⁰ |
|---|---|---|---|---|---|---|---|
| 0 | 0.003 | -1.846 | -0.553 | -5.406 | -1.344 | -8.012 | -1.469 |
| 1 | 0.068 | -1.694 | -0.501 | -4.848 | -1.217 | -7.141 | -1.331 |
| 2 | 0.183 | -1.102 | -0.409 | -3.362 | -1.001 | -4.910 | -1.096 |
| 3 | 0.324 | -0.682 | -0.290 | -2.072 | -0.713 | -3.055 | -0.780 |
| 4 | 0.500 | -0.287 | -0.111 | -0.915 | -0.273 | -1.331 | -0.298 |
| 5 | 0.676 | +0.088 | +0.111 | +0.340 | +0.273 | +0.525 | +0.298 |
| 6 | 0.817 | +0.510 | +0.290 | +1.630 | +0.713 | +2.417 | +0.780 |
| 7 | 0.932 | +0.973 | +0.409 | +2.934 | +1.001 | +4.267 | +1.096 |
| 8 | 0.997 | +1.610 | +0.501 | +4.589 | +1.217 | +6.726 | +1.331 |
| 9 | 1.000 | +1.830 | +0.553 | +5.183 | +1.344 | +7.510 | +1.469 |

g⁶ range (spread 1): -1.344 ~ +1.344 tick; g^∞: -1.469 ~ +1.469 tick. 