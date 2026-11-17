"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2..24 of release day, as five quantiles per hour, from the case's own
comparables: past big game launches' release days.

Method (tuned by leave-one-out cross-validation on the 37 comparable
launches in the kit, scored with the contest's pinball loss on the log
scale; see README.md for numbers):

  * Anchor on hour 1's print p1. For each comparable launch with a usable
    release day, form the log-ratio curve r(h) = ln(v_h / v_1), h = 2..24.
  * Weight each comparable by how close its hour-1 scale is to the target's,
    with a Gaussian kernel on log viewers (bandwidth 1.0 in log units):
    a launch the size of GTA VI's hour 1 counts most; tiny categories that
    explode 200x after a late unlock count almost nothing. This is what
    keeps the forecast honest: pooling every launch equally is far worse.
  * Per hour, take the weighted quantiles of r(h) at 0.05/0.25/0.5/0.75/0.95,
    shrink the outer quantiles 10% toward the median (with ~20 effective
    pool members the raw 5%/95% quantiles are the extremes, slightly
    over-wide), and forecast q = p1 * exp(r).
  * Fallbacks keep every case valid: an hour with no pool data, or a case
    with no usable comparables at all, falls back to the house baseline
    (persistence with a widening random-walk spread, sigma 0.15).

Deterministic, offline, no randomness, no network. Runs in well under a
second per case (limit 8 s).
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed here, but keep the
# line harmless and identical in spirit to the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (house baseline only).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# House baseline spread, in natural-log units per square-root hour.
BASE_SIGMA = 0.15
# Gaussian kernel bandwidth on log hour-1 viewers (tuned: 0.7-1.0 flat optimum).
KERNEL_BW = 1.0
# Tail shrinkage toward the median (10%): raw outer quantiles are extremes.
TAIL_SHRINK = 0.9
# A comparable counts as usable with at least this many scored release-day
# hours besides hour 1.
MIN_SCORED_HOURS = 12


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def wquantile(pairs, q):
    """Weighted quantile of [(value, weight), ...]; weights need not sum to 1."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        return None
    threshold = q * total
    acc = 0.0
    for value, w in pairs:
        acc += w
        if acc >= threshold:
            return value
    return pairs[-1][0]


def release_day(hourly, day_start):
    """Split a comparable's hourly points into release-day hours and the
    pre-release median. Returns (p1, {hour: viewers}, pre_median) or None
    when hour 1 is missing or too few release-day hours are scored."""
    by_hour = {}
    pre = []
    for pt in hourly:
        try:
            t = _parse_time(pt["from"])
            v = max(1.0, float(pt["viewers"]))
        except (KeyError, TypeError, ValueError):
            continue
        dt_h = (t - day_start).total_seconds() // 3600
        if 0 <= dt_h < 24:
            by_hour[int(dt_h) + 1] = v
        elif -168 <= dt_h < 0:
            pre.append(v)
    if 1 not in by_hour:
        return None
    actual = {h: v for h, v in by_hour.items() if h >= 2}
    if len(actual) < MIN_SCORED_HOURS:
        return None
    pre_med = statistics.median(pre) if pre else 1.0
    return by_hour[1], actual, pre_med


def _parse_time(text):
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, |err| < 1.2e-9).

    Used only for the baseline fallback so the entry works for any level
    set a case asks for; the contest's levels are the usual five.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def baseline_row(p1, h, levels=LEVELS):
    """House baseline quantiles for hour h (persistence + widening spread).

    For the contest's standard levels the starter's rounded z table is used
    so the fallback reproduces the house baseline number for number; other
    level sets use the inverse-normal approximation.
    """
    zs = Z if list(levels) == LEVELS else [_norm_ppf(tau) for tau in levels]
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)))
            for z in zs]


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    if (not isinstance(levels, list) or len(levels) != 5
            or any(not (0 < t < 1) for t in levels)):
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    try:
        p1 = max(1.0, float(first["viewers"])) if first is not None else 1.0
    except (KeyError, TypeError, ValueError):
        # Missing/null/malformed hour-1 print: per RULES the final is not
        # graded without a valid hour-1 read, but fall back to p1 = 1
        # (baseline path) instead of crashing with no output file.
        p1 = 1.0
    log_p1 = math.log(p1)

    # Build the pool: every comparable with a usable release day.
    pool = []  # (weight, {hour: log-ratio})
    for comp in case.get("comparables") or []:
        try:
            day_start = _parse_time(comp["day_start"])
            got = release_day(comp.get("hourly") or [], day_start)
        except (KeyError, TypeError, ValueError):
            continue
        if got is None:
            continue
        comp_p1, actual, _pre = got
        dlog = (math.log(comp_p1) - log_p1) / KERNEL_BW
        weight = math.exp(-0.5 * dlog * dlog)
        if weight <= 0.0:
            continue
        ratios = {h: math.log(v / comp_p1) for h, v in actual.items()}
        pool.append((weight, ratios))

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        pairs_by_level = None
        if pool:
            pairs = []
            for weight, ratios in pool:
                if h in ratios:
                    pairs.append((ratios[h], weight))
            if pairs:
                median = wquantile(pairs, 0.5)
                if median is not None:
                    qs = []
                    for level in levels:
                        raw = wquantile(pairs, level)
                        if raw is None:
                            raw = median
                        qs.append(median + TAIL_SHRINK * (raw - median))
                    quantiles = [round2(max(1.0, p1 * math.exp(x))) for x in qs]
                    # A tiny pool cannot support an honest spread: with
                    # fewer than 3 launches the empirical quantiles collapse
                    # toward one curve, so blend toward the baseline in log
                    # space (3-4 launches: half-way; under 3: baseline).
                    if len(pairs) < 3:
                        pairs_by_level = baseline_row(p1, h, levels)
                    else:
                        if len(pairs) < 5:
                            base = baseline_row(p1, h, levels)
                            quantiles = [
                                round2(math.exp(0.5 * math.log(max(1.0, q))
                                                + 0.5 * math.log(max(1.0, b))))
                                for q, b in zip(quantiles, base)
                            ]
                        # Keep the row non-decreasing after rounding.
                        for i in range(1, len(quantiles)):
                            if quantiles[i] < quantiles[i - 1]:
                                quantiles[i] = quantiles[i - 1]
                        pairs_by_level = quantiles
        if pairs_by_level is None:
            pairs_by_level = baseline_row(p1, h, levels)
        hours.append({"hour": h, "q": pairs_by_level})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
