"""Peak detection, dynamic trough-to-trough baselining, and net AUC calculation."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class PeakAnalysis:
    peak_number: int
    peak_index: int
    peak_x: float
    peak_y: float
    start_index: int
    end_index: int
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    auc: float
    segment_x: np.ndarray = field(repr=False)
    segment_y: np.ndarray = field(repr=False)
    baseline_y: np.ndarray = field(repr=False)
    shaded_y: np.ndarray = field(repr=False)


def detect_peaks_and_troughs(
    y: np.ndarray, prominence: float, distance: int
) -> tuple[np.ndarray, np.ndarray]:
    """Detect local maxima (peaks) and local minima (troughs) in the signal."""
    distance = max(1, int(distance))
    peak_idx, _ = find_peaks(y, prominence=prominence, distance=distance)
    trough_idx, _ = find_peaks(-y, prominence=prominence, distance=distance)
    return peak_idx, trough_idx


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _lower_hull_baseline(segment_x: np.ndarray, segment_y: np.ndarray) -> np.ndarray:
    """Piecewise-linear baseline along the lower convex hull of the segment's points.

    Anchored at the two endpoints (the bounding troughs), same as a straight
    trough-to-trough line, but bent down to any interior point that would
    otherwise poke below a straight segment - so y_baseline(x) <= y(x) holds
    everywhere in the window, by construction, not just at the endpoints.
    """
    n = len(segment_x)
    if n <= 2:
        if n == 2 and segment_x[-1] != segment_x[0]:
            slope = (segment_y[-1] - segment_y[0]) / (segment_x[-1] - segment_x[0])
            return segment_y[0] + slope * (segment_x - segment_x[0])
        return np.full_like(segment_x, segment_y[0], dtype=float)

    hull_idx = [0]
    for i in range(1, n):
        point = (segment_x[i], segment_y[i])
        while len(hull_idx) >= 2:
            o = (segment_x[hull_idx[-2]], segment_y[hull_idx[-2]])
            a = (segment_x[hull_idx[-1]], segment_y[hull_idx[-1]])
            if _cross(o, a, point) <= 0:
                hull_idx.pop()
            else:
                break
        hull_idx.append(i)

    hull_x = segment_x[hull_idx]
    hull_y = segment_y[hull_idx]
    return np.interp(segment_x, hull_x, hull_y)


def _walk_to_boundary(
    x: np.ndarray,
    y: np.ndarray,
    apex: int,
    outer: int,
    step: int,
    threshold_frac: float,
    slope_sensitivity: float = 0.10,
    min_flat_window: int = 5,
) -> int:
    """Walk outward from the peak apex toward `outer` and stop at the peak's
    actual foot - not at the far trough.

    Stops at the first point where either:
      - the signal has decayed to within `threshold_frac` of the local floor
        (the value at `outer`), or
      - a moving-average of the first derivative dy/dx has stayed within a
        `slope_sensitivity`-wide tolerance band of zero for at least
        `min_flat_window` consecutive samples, i.e. the signal has genuinely
        settled onto baseline rather than just touching zero slope once.
    whichever comes first. `outer` is only the outer search limit (expanded
    by the caller so the flatness window always has room to be evaluated),
    not the target itself - the actual foot is wherever the signal decays
    back to near-baseline within that limit.
    """
    if outer == apex:
        return outer

    floor = float(y[outer])
    prominence = max(float(y[apex]) - floor, 1e-12)
    cutoff = floor + threshold_frac * prominence

    idxs = np.arange(apex, outer + step, step)
    if len(idxs) < 2:
        return outer

    side_x = x[idxs]
    side_y = y[idxs]

    dydx = np.gradient(side_y, side_x)

    # Characteristic slope for this side: the *average* rate of decay needed
    # to return fully to baseline over its span. Scaling the sensitivity
    # slider against this (rather than this side's single steepest sample -
    # almost always the point right after the apex) keeps the slider's
    # effect stable and proportional: low % => only a near-perfectly flat
    # run qualifies; high % => a visibly sloped run still counts as "flat".
    span = max(abs(side_x[-1] - side_x[0]), 1e-12)
    reference_slope = prominence / span
    flat_eps = max(slope_sensitivity, 1e-6) * reference_slope

    # Smooth the derivative with a small moving average first, so that raw
    # sample-to-sample noise can't prevent a genuinely-flattened run from
    # ever registering as flat.
    smooth_window = max(2, min(3, len(dydx)))
    dydx_smooth = (
        pd.Series(dydx)
        .rolling(smooth_window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )

    # Slope-flatness is the primary criterion - it's checked across the full
    # search horizon rather than racing the intensity cutoff for whichever
    # triggers at the smaller index. The intensity cutoff (typically reached
    # after only ~threshold_frac of the decay) would otherwise almost always
    # fire first and silently override the slope/window sliders, which is
    # exactly why they previously had little to no visible effect.
    required = max(1, int(min_flat_window))
    consecutive = 0
    intensity_hit = None
    for j in range(1, len(idxs)):
        idx = int(idxs[j])
        if intensity_hit is None and y[idx] <= cutoff:
            intensity_hit = idx
        if abs(dydx_smooth[j]) <= flat_eps:
            consecutive += 1
            if consecutive >= required:
                return int(idxs[j - required + 1])
        else:
            consecutive = 0

    # Slope never settled within the search horizon - fall back to the
    # intensity cutoff if it was reached, else give up at the outer limit.
    return intensity_hit if intensity_hit is not None else outer


def analyze_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: np.ndarray,
    trough_idx: np.ndarray,
    boundary_threshold: float = 0.10,
    slope_sensitivity: float = 0.10,
    min_flat_window: int = 5,
) -> list[PeakAnalysis]:
    """Build baseline, shaded region, and net AUC for every detected peak."""
    results: list[PeakAnalysis] = []
    n = len(y)
    sorted_troughs = np.sort(trough_idx)

    # The flatness-duration slider needs at least this many samples of room
    # to evaluate a consecutive run; a trough sitting closer than that would
    # otherwise cap the search before the slider can have any effect at all.
    min_span = max(int(min_flat_window) + 3, 3)

    for peak_number, k in enumerate(sorted(peak_idx), start=1):
        preceding = sorted_troughs[sorted_troughs < k]
        succeeding = sorted_troughs[sorted_troughs > k]

        outer_left = int(preceding[-1]) if len(preceding) else 0
        outer_right = int(succeeding[0]) if len(succeeding) else n - 1

        # Troughs only cap how far the scan is *allowed* to look; expand
        # that ceiling so it never falls short of the slider-driven minimum
        # search horizon (the actual stopping point is still decided by the
        # slope/intensity criteria inside _walk_to_boundary).
        outer_left = min(outer_left, max(0, k - min_span))
        outer_right = max(outer_right, min(n - 1, k + min_span))

        if outer_left >= k:
            outer_left = max(0, k - 1)
        if outer_right <= k:
            outer_right = min(n - 1, k + 1)
        if outer_right <= outer_left:
            continue

        # Outer troughs only cap how far we're allowed to look; the actual
        # peak foot is wherever the signal decays back to near-baseline.
        i_start = _walk_to_boundary(
            x, y, k, outer_left, -1, boundary_threshold, slope_sensitivity, min_flat_window
        )
        i_end = _walk_to_boundary(
            x, y, k, outer_right, 1, boundary_threshold, slope_sensitivity, min_flat_window
        )
        if i_end <= i_start:
            continue

        x_start, y_start = x[i_start], y[i_start]
        x_end, y_end = x[i_end], y[i_end]

        segment_x = x[i_start : i_end + 1]
        segment_y = y[i_start : i_end + 1]
        baseline_y = _lower_hull_baseline(segment_x, segment_y)

        diff = np.clip(segment_y - baseline_y, a_min=0.0, a_max=None)
        auc = float(np.sum(0.5 * np.diff(segment_x) * (diff[1:] + diff[:-1])))

        shaded_y = baseline_y + diff  # equals max(segment_y, baseline_y)

        results.append(
            PeakAnalysis(
                peak_number=peak_number,
                peak_index=int(k),
                peak_x=float(x[k]),
                peak_y=float(y[k]),
                start_index=i_start,
                end_index=i_end,
                x_start=float(x_start),
                y_start=float(y_start),
                x_end=float(x_end),
                y_end=float(y_end),
                auc=auc,
                segment_x=segment_x,
                segment_y=segment_y,
                baseline_y=baseline_y,
                shaded_y=shaded_y,
            )
        )
    return results
