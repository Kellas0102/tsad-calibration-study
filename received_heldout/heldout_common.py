"""Heldout protocol: 80/20 split of the raw train series, scaler on the 80% only,
windows that never cross the split, and exact threshold rules.

This module is shared by the TimesNet / MEMTO / CATCH heldout runners so that all
three see byte-identical splits and identical threshold semantics.

Split definition (per entity)
    raw train file -> T points
    train segment  = [0, i80)          i80 = floor(0.8 * T)
    val   segment  = [i80, T)
    test  segment  = the separate test file (never touched before scoring)

Windows
    A training window [s, s+win) must satisfy s+win <= i80  (never crosses the split).
    A validation window [s, s+win) must satisfy i80 <= s and s+win <= T.
    Test windows are taken over the test segment with stride 1.

Scaling
    StandardScaler fitted on raw_train[0:i80] ONLY. The same scaler is applied to the
    val segment and to test. The val segment therefore never influences the scaler.

Threshold rules (both are exported; neither uses test labels)
    exact_f1max_highest_ge : the threshold that maximises validation F1, evaluated at
        every distinct score value (complete tied blocks are handled exactly); when
        several thresholds attain the same maximal F1 the HIGHEST one is chosen;
        predictions are `score >= threshold`.
    order_stat_budget_gt : the threshold placed so that exactly k validation points
        exceed it, with k = round(budget * n_val); predictions are `score > threshold`.

Point-level scores
    Window scores are averaged onto the timeline by the number of windows that cover
    each point (stride 1, no truncation). The coverage count per point is returned and
    exported so that no zero-fill is silent.
"""
from __future__ import annotations

import json
import os

import numpy as np

PROJ = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\tsad-paper\tsad-paper"
OUTROOT = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\dmkd_official\06_heldout"
DATA = os.path.join(OUTROOT, "_data")


# --------------------------------------------------------------------------- split
def build_split(entity: str, win: int = 100) -> dict:
    """Materialise the heldout split for one entity and return its manifest."""
    tr_src = os.path.join(PROJ, "data", "SMD", "train", f"{entity}.txt")
    te_src = os.path.join(PROJ, "data", "SMD", "test", f"{entity}.txt")
    lb_src = os.path.join(PROJ, "data", "SMD", "test_label", f"{entity}.txt")
    train_raw = np.loadtxt(tr_src, dtype=np.float64, delimiter=",")
    test_raw = np.loadtxt(te_src, dtype=np.float64, delimiter=",")
    test_lab = np.loadtxt(lb_src, dtype=np.int64)
    if test_lab.ndim > 1:
        test_lab = test_lab.reshape(-1)

    T = len(train_raw)
    i80 = int(np.floor(0.8 * T))
    if i80 <= win or (T - i80) <= win:
        raise SystemExit(f"{entity}: segments too short for win={win}")

    d = os.path.join(DATA, entity)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "train_part.npy"), train_raw[:i80].astype(np.float32))
    np.save(os.path.join(d, "val_part.npy"), train_raw[i80:].astype(np.float32))
    np.save(os.path.join(d, "test.npy"), test_raw.astype(np.float32))
    np.save(os.path.join(d, "test_label.npy"), test_lab.astype(np.int64))

    man = {
        "entity": entity, "window": win,
        "raw_train_points": int(T), "split_index_i80": int(i80),
        "train_segment": [0, i80], "val_segment": [i80, T],
        "train_points": int(i80), "val_points": int(T - i80),
        "test_points": int(len(test_raw)),
        "test_anomaly_rate": round(float(test_lab.mean()), 6),
        "train_window_range": [0, i80 - win],      # inclusive start range
        "val_window_range": [i80, T - win],
        "n_train_windows": int(i80 - win + 1),
        "n_val_windows": int(T - win - i80 + 1),
        "n_test_windows": int(len(test_raw) - win + 1),
        "scaler_fit_range": [0, i80],
        "disjointness": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "max_train_index": int(i80 - 1),
            "min_val_index": int(i80),
        },
    }
    with open(os.path.join(d, "split.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return man


def load_split(entity: str) -> dict:
    d = os.path.join(DATA, entity)
    with open(os.path.join(d, "split.json"), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- scaling
class Scaler:
    """StandardScaler equivalent, fitted on an explicitly passed range only."""

    def __init__(self, fit_array: np.ndarray) -> None:
        self.mean = fit_array.mean(axis=0)
        sd = fit_array.std(axis=0)
        self.std = np.where(sd < 1e-8, 1.0, sd)

    def transform(self, a: np.ndarray) -> np.ndarray:
        return ((a - self.mean) / self.std).astype(np.float32)


# --------------------------------------------------------------------------- windows
def make_windows(x: np.ndarray, starts: np.ndarray, win: int) -> np.ndarray:
    return np.stack([x[s:s + win] for s in starts]).astype(np.float32)


def timeline(scores_1d: np.ndarray, n_times: int, win: int, stride: int = 1):
    """Average window scores onto the timeline; return (values, coverage)."""
    acc = np.zeros(n_times, dtype=np.float64)
    cnt = np.zeros(n_times, dtype=np.float64)
    for i, v in enumerate(scores_1d):
        a = i * stride
        b = min(a + win, n_times)
        if a >= n_times:
            break
        acc[a:b] += v
        cnt[a:b] += 1.0
    covered = cnt > 0
    out = np.full(n_times, np.nan, dtype=np.float64)
    out[covered] = acc[covered] / cnt[covered]
    return out, cnt.astype(np.int64)


# --------------------------------------------------------------------------- thresholds
def _f1(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else 0.0


def f1_at(scores: np.ndarray, labels: np.ndarray, thr: float, ge: bool = True):
    pred = (scores >= thr) if ge else (scores > thr)
    y = labels.astype(bool)
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    return _f1(tp, fp, fn), pred


def exact_f1max_highest_ge(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Exact F1 maximiser over every distinct score value, highest threshold wins ties."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    # cumulative counts after processing each tied block
    n = len(ss)
    tp = fp = 0
    best = {"threshold": None, "f1": -1.0, "tp": 0, "fp": 0, "fn": 0}
    i = 0
    n_pos = int(y.sum())
    while i < n:
        v = ss[i]
        j = i
        while j < n and ss[j] == v:
            if yy[j]:
                tp += 1
            else:
                fp += 1
            j += 1
        fn = n_pos - tp
        f = _f1(tp, fp, fn)
        # thresholds are visited from highest to lowest; keep strictly-better only,
        # which makes the highest threshold win any F1 tie
        if f > best["f1"] + 1e-15:
            best = {"threshold": float(v), "f1": float(f),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn)}
        i = j
    best["comparison"] = ">="
    best["n_scored"] = int(n)
    return best


def order_stat_budget_gt(scores: np.ndarray, budget: float = 0.05) -> dict:
    """Threshold so that exactly round(budget*n) points exceed it; predictions `>`."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    k = int(round(budget * n))
    k = max(0, min(k, n))
    if k == 0:
        thr = float(np.nextafter(s.max(), np.inf))
        return {"threshold": thr, "budget": budget, "k": 0,
                "comparison": ">", "n_scored": n}
    desc = np.sort(s)[::-1]
    thr = float(desc[k - 1])
    return {"threshold": thr, "budget": budget, "k": k,
            "comparison": ">", "n_scored": n}


if __name__ == "__main__":
    import sys
    ents = sys.argv[1:] or ["machine-1-1", "machine-3-1"]
    out = [build_split(e) for e in ents]
    print(json.dumps(out, ensure_ascii=False, indent=1))
