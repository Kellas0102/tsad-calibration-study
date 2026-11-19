"""Correct window->timeline aggregation for (n_windows, win) score matrices.

The previous code flattened the (n_windows, win) error matrix to a 1-D vector and fed
it to heldout_common.timeline, which treats each ELEMENT as a whole window. That both
broadcasts a scalar where a length-win vector belongs and walks the window index past
the end of the series, so the timeline was shifted and the coverage profile was wrong
(it stayed at win everywhere instead of decaying to 3, 2, 1 at the tail).

This module provides the intended semantics:
    window i, position j  ->  time index  i*stride + j
with explicit structural assertions and an independent reference implementation used
to cross-check it on small random inputs.
"""
from __future__ import annotations

import numpy as np


def timeline2d(win_scores, n_times: int, win: int, stride: int = 1,
               check_tail: bool = True):
    """Aggregate a (n_windows, win) matrix onto the point timeline.

    Returns (values, coverage) where values is NaN outside covered points.
    """
    W = np.asarray(win_scores, dtype=np.float64)
    if W.ndim != 2:
        raise AssertionError(f"expected a 2-D (n_windows, win) matrix, got ndim={W.ndim}")
    if W.shape[1] != win:
        raise AssertionError(f"expected {win} positions per window, got {W.shape[1]}")

    starts = np.arange(0, n_times - win + 1, stride)
    if W.shape[0] != len(starts):
        raise AssertionError(
            f"window count mismatch: matrix has {W.shape[0]}, expected {len(starts)} "
            f"for n_times={n_times}, win={win}, stride={stride}")

    acc = np.zeros(n_times, dtype=np.float64)
    cnt = np.zeros(n_times, dtype=np.int64)
    for i, s in enumerate(starts):
        acc[s:s + win] += W[i]
        cnt[s:s + win] += 1

    if (cnt == 0).any():
        raise AssertionError("uncovered points exist; aggregation would need zero-fill")

    # structural checks on the coverage profile
    if check_tail and n_times > win and win >= 3 and stride == 1:
        tail = cnt[-3:].tolist()
        if tail != [3, 2, 1]:
            raise AssertionError(f"tail coverage must decay to [3, 2, 1], got {tail}")
        if cnt.max() != win:
            raise AssertionError(f"peak coverage must equal win={win}, got {cnt.max()}")

    return acc / cnt, cnt


def reference_slow(win_scores, n_times: int, win: int, stride: int = 1):
    """Deliberately naive O(n_windows*win) reference, written independently."""
    W = np.asarray(win_scores, dtype=np.float64)
    acc = np.zeros(n_times, dtype=np.float64)
    cnt = np.zeros(n_times, dtype=np.int64)
    starts = list(range(0, n_times - win + 1, stride))
    for i, s in enumerate(starts):
        for j in range(win):
            acc[s + j] += W[i, j]
            cnt[s + j] += 1
    out = np.full(n_times, np.nan)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out, cnt


def self_test(verbose: bool = True) -> bool:
    rng = np.random.default_rng(0)
    ok = True
    cases = [(120, 10, 1), (257, 100, 1), (300, 32, 1), (500, 16, 2)]
    for n_times, win, stride in cases:
        nwin = len(range(0, n_times - win + 1, stride))
        W = rng.normal(size=(nwin, win))
        a, ca = timeline2d(W, n_times, win, stride)
        b, cb = reference_slow(W, n_times, win, stride)
        same_vals = bool(np.allclose(a, b, equal_nan=True))
        same_cov = bool((ca == cb).all())
        tail_ok = True
        if stride == 1 and win >= 3:
            tail_ok = ca[-3:].tolist() == [3, 2, 1]
        if verbose:
            print(f"  n_times={n_times:4d} win={win:3d} stride={stride}: "
                  f"values_match={same_vals} coverage_match={same_cov} tail={ca[-3:].tolist()}")
        ok = ok and same_vals and same_cov and tail_ok

    # negative controls: the old buggy path must be caught
    W = rng.normal(size=(5, 10))
    try:
        timeline2d(W.reshape(-1), 60, 10, 1)
        ok = False
        print("  [FAIL] flattened input was accepted")
    except AssertionError as e:
        if verbose:
            print(f"  negative control (flattened): rejected -> {str(e)[:60]}")
    try:
        timeline2d(W, 60, 10, 1)   # 5 windows given, 51 expected
        ok = False
        print("  [FAIL] wrong window count was accepted")
    except AssertionError as e:
        if verbose:
            print(f"  negative control (count): rejected -> {str(e)[:60]}")
    return ok


if __name__ == "__main__":
    print("timeline2d self-test:")
    print("PASS" if self_test() else "FAIL")
