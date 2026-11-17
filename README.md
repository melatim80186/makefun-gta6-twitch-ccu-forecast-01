# GTA VI Twitch-viewer forecaster — make.fun `gta6-twitch-ccu-forecast-01`

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour (2–24) of its release day, 2026-11-19 UTC, as five quantiles per hour,
from the case's own comparables: past big game launches' release days.

## Method

`main.py` (Python 3.13, standard library only, deterministic, offline) reads
one case file and:

1. Anchors on hour 1's print `p1`.
2. For every comparable launch with a usable release day (hour 1 plus at
   least 12 scored hours), forms the log-ratio curve `r(h) = ln(v_h / v_1)`.
3. Weights each comparable by how close its hour-1 scale is to the target's,
   with a Gaussian kernel on log viewers (bandwidth 1.0 in log units). A
   launch the size of GTA VI's hour 1 counts most; tiny categories that
   explode 200x after a late unlock count almost nothing. Pooling every
   launch equally is far worse — the small launches' wild curves would
   swamp the forecast.
4. Per hour, takes the weighted quantiles of `r(h)` at 0.05 / 0.25 / 0.5 /
   0.75 / 0.95, shrinks the outer quantiles 10% toward the median (with
   ~20 effective pool members the raw 5%/95% quantiles are the extremes,
   slightly over-wide), and forecasts `q = p1 * exp(r)`.
5. Falls back to the house baseline (persistence with a widening
   random-walk spread, σ = 0.15) for any hour with no pool data, and for a
   case with no usable comparables at all. A pool of fewer than 3 launches
   for an hour blends toward the baseline instead of trusting a collapsed
   spread.

## Scores (measured locally, leave-one-out on the kit's comparables)

Scored with the contest's pinball loss on the log scale, one pseudo-case per
comparable launch (forecast its release day from data up to the end of hour
1, score against its actual hours 2–24):

| model | mean pinball, big launches (hour 1 > 50k, n=20) | mean pinball, all launches (n=37) |
|---|---|---|
| house baseline (persistence) | 0.2509 | 0.9297 |
| **this entry** | **0.2067 (−17.6%)** | **0.4048 (−56.5%)** |
| this entry, wins vs baseline | 15 / 20 | 32 / 37 |

The house's published practice baseline is ≈ 0.387 on the four real
practice launches (server-graded; this entry's practice score shows on the
board after submission). The pseudo-case numbers above are the local proxy:
same recipe, same scoring, run through the shipped `main.py` end to end.

Why the kernel matters: without scale weighting the pooled quantiles are
dominated by small launches whose hour-1 print is pre-unlock and whose
later hours run up to 200x higher — the pooled forecaster scores 0.3621 on
big launches, worse than the baseline. With the kernel, GTA VI's hour 1
(whatever it is — the biggest launch in the set) is matched against the
biggest past launches: ELDEN RING, Cyberpunk 2077, The Last of Us Part II.

## Checks

- Deterministic: two runs over all four practice cases write byte-identical
  outputs. Runtime ≈ 0.3 s for all four cases (limit 8 s per case).
- Output format verified on all practice cases: 23 rows, hours 2–24,
  five finite quantiles each, non-decreasing, each within (0, 1e9].
- Edge cases tested: no comparables (baseline fallback), a single
  comparable (baseline fallback, no collapsed spread), malformed
  comparables skipped, hour-1 print of 0 (clamped to 1), hour-1 print of
  2,000,000 beyond the pool (nearest big launches dominate), alternate
  5-level set.
- No network, no randomness, no third-party packages.
