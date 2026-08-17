"""Peak detection, dynamic trough-to-trough baselining, and net AUC calculation."""

from dataclasses import dataclass, field

import numpy as np
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
    x: np.ndarray, y: np.ndarray, apex: int, outer: int, step: int, threshold_frac: float
) -> int:
    """Walk outward from the peak apex toward `outer` and stop at the peak's
    actual foot - not at the far trough.

    Stops at the first point where the signal has decayed to within
    `threshold_frac` of the local floor (the value at `outer`), or where the
    local slope has flattened to a small fraction of its steepest point on
    this side, whichever comes first. This keeps flat/resting tails out of
    the integration window instead of always running out to the nearest
    detected trough.
    """
    if outer == apex:
        return outer

    floor = float(y[outer])
    prominence = max(float(y[apex]) - floor, 1e-12)
    cutoff = floor + threshold_frac * prominence

    idxs = np.arange(apex, outer + step, step)
    if len(idxs) < 2:
        return outer

    side_y = y[idxs]
    side_x = x[idxs]
    slopes = np.abs(np.diff(side_y) / np.diff(side_x))
    slope_scale = float(np.max(slopes)) if len(slopes) else 0.0
    flat_eps = 0.1 * slope_scale

    for j in range(1, len(idxs)):
        idx = int(idxs[j])
        if y[idx] <= cutoff:
            return idx
        if j >= 2 and slopes[j - 1] < flat_eps:
            return idx
    return outer


def analyze_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: np.ndarray,
    trough_idx: np.ndarray,
    boundary_threshold: float = 0.10,
) -> list[PeakAnalysis]:
    """Build baseline, shaded region, and net AUC for every detected peak."""
    results: list[PeakAnalysis] = []
    n = len(y)
    sorted_troughs = np.sort(trough_idx)

    for peak_number, k in enumerate(sorted(peak_idx), start=1):
        preceding = sorted_troughs[sorted_troughs < k]
        succeeding = sorted_troughs[sorted_troughs > k]

        outer_left = int(preceding[-1]) if len(preceding) else 0
        outer_right = int(succeeding[0]) if len(succeeding) else n - 1

        if outer_left >= k:
            outer_left = max(0, k - 1)
        if outer_right <= k:
            outer_right = min(n - 1, k + 1)
        if outer_right <= outer_left:
            continue

        # Outer troughs only cap how far we're allowed to look; the actual
        # peak foot is wherever the signal decays back to near-baseline.
        i_start = _walk_to_boundary(x, y, k, outer_left, -1, boundary_threshold)
        i_end = _walk_to_boundary(x, y, k, outer_right, 1, boundary_threshold)
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
