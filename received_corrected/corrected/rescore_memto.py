"""Rescore MEMTO heldout runs from their saved checkpoints. No retraining.

Why rescoring is enough for MEMTO: `_work/ckpt_second.pth` and
`_work/memory_item/<rid>_memory_item.pth` are on disk, so the exact trained model can be
reloaded. Only the window->timeline aggregation was wrong, so we recompute the 2-D window
score matrices and aggregate with timeline2d.

State pairing that is checked and reported (not assumed):
    * weights      : best normal-validation reconstruction loss (early stopping)
    * memory items : the tensor at the END of phase-2 training, saved separately by the
                     runner. Official MEMTO returns the same object from train(), so this
                     pairing mirrors upstream; it is NOT "memory at the best epoch".
    * validation   : MemoryModule.update() runs whenever phase_type != 'test', so the
                     validation loop itself mutates the memory. Measured below.

Writes corrected/<run_id>/ scores.npz + metrics.json, leaving the quarantined runs intact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
MEMTO = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\dmkd_official\02_repos\MEMTO"
PROJ_CODE = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\tsad-paper\tsad-paper\code"
sys.path.insert(0, MEMTO)
sys.path.insert(0, HERE)
import heldout_common as HC          # noqa: E402
from timeline2d import timeline2d    # noqa: E402
from model.Transformer import TransformerVar                    # noqa: E402
from model.loss_functions import GatheringLoss                  # noqa: E402

QDIR = os.path.join(HC.OUTROOT)          # quarantined runs live here
OUT = os.path.join(HC.OUTROOT, "corrected")


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--score-batch", type=int, default=32,
                    help="batch size for scoring only; scores are batch-independent here")
    ap.add_argument("--inject-budget", type=float, default=0.05)
    a = ap.parse_args()

    rid = f"memto_heldout_{a.entity}_seed{a.seed}"
    src = os.path.join(QDIR, rid)
    outc = os.path.join(OUT, rid)
    os.makedirs(outc, exist_ok=True)
    work = os.path.join(src, "_work")
    ck2 = os.path.join(work, "ckpt_second.pth")
    memp = os.path.join(work, "memory_item", f"{rid}_memory_item.pth")
    if not (os.path.isfile(ck2) and os.path.isfile(memp)):
        raise SystemExit(f"[{rid}] missing checkpoint or memory item")

    cfgq = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    cap = cfgq["capacity"]
    d_model, n_memory = cap["d_model"], cap["n_memory"]
    n_heads, e_layers, d_ff = cap["n_heads"], cap["e_layers"], cap["d_ff"]

    man = HC.load_split(a.entity)
    d = os.path.join(HC.DATA, a.entity)
    tr_raw = np.load(os.path.join(d, "train_part.npy"))
    va_raw = np.load(os.path.join(d, "val_part.npy"))
    te_raw = np.load(os.path.join(d, "test.npy"))
    te_lab = np.load(os.path.join(d, "test_label.npy"))
    win, i80 = man["window"], man["split_index_i80"]
    sc = HC.Scaler(tr_raw)
    va, te = sc.transform(va_raw), sc.transform(te_raw)
    enc = va.shape[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cwd0 = os.getcwd()
    os.chdir(work)      # MemoryModule(phase_type='test') loads ./memory_item/<dataset>.pth
    model = TransformerVar(win_size=win, enc_in=enc, c_out=enc, n_memory=n_memory,
                           d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
                           dropout=0.0, device=device, memory_init_embedding=None,
                           memory_initial=False, phase_type="test", dataset_name=rid)
    model = model.float().to(device)
    model.load_state_dict(torch.load(ck2, map_location=device, weights_only=False))
    model.eval()
    mem_loaded = model.mem_module.mem.detach().clone()

    # --- does validation update memory? probe with the phase-2 model semantics ---
    val_updates_memory = None
    try:
        probe = TransformerVar(win_size=win, enc_in=enc, c_out=enc, n_memory=n_memory,
                               d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
                               dropout=0.0, device=device,
                               memory_init_embedding=mem_loaded,
                               memory_initial=False, phase_type="second_train",
                               dataset_name=rid)
        probe = probe.float().to(device)
        probe.load_state_dict(torch.load(ck2, map_location=device, weights_only=False))
        probe.eval()                      # eval mode, exactly as the validation loop runs
        before = probe.mem_module.mem.detach().clone()
        xb = torch.from_numpy(va[:win].reshape(1, win, enc)).float().to(device)
        with torch.no_grad():
            _ = probe(xb)
        after = probe.mem_module.mem.detach()
        val_updates_memory = bool(not torch.allclose(before, after))
    except Exception as e:
        val_updates_memory = f"probe_failed:{type(e).__name__}:{e}"

    crit_el = nn.MSELoss(reduce=False)
    gather = GatheringLoss(reduce=False)
    temperature = cfgq["training"]["temperature"]

    def score_2d(arr):
        starts = np.arange(0, len(arr) - win + 1, 1)
        rows = []
        with torch.no_grad():
            for i in range(0, len(starts), a.score_batch):
                ch = starts[i:i + a.score_batch]
                xb = np.stack([arr[s:s + win] for s in ch])
                xbt = torch.from_numpy(xb).float().to(device)
                od = model(xbt)
                rec = torch.mean(crit_el(xbt, od["out"]), dim=-1)
                lat = torch.softmax(gather(od["queries"], od["mem"]) / temperature, dim=-1)
                rows.append((lat * rec).cpu().numpy())
        W = np.concatenate(rows, axis=0)
        return W, starts

    spec = importlib.util.spec_from_file_location("pjd_data", os.path.join(PROJ_CODE, "data.py"))
    pjd = importlib.util.module_from_spec(spec)
    sys.modules["pjd_data"] = pjd
    spec.loader.exec_module(pjd)
    va_inj, va_inj_lab = pjd.inject_anomalies(va.astype(np.float64),
                                              np.zeros(len(va), dtype=int),
                                              ratio=a.inject_budget, seed=a.seed)

    Wv, sv = score_2d(va)
    Wj, sj = score_2d(va_inj)
    Wt, st = score_2d(te)
    os.chdir(cwd0)

    vs, vc = timeline2d(Wv, len(va), win, 1)
    js, jc = timeline2d(Wj, len(va), win, 1)
    ts, tc = timeline2d(Wt, len(te), win, 1)

    ok = np.isfinite(js) & (jc > 0)
    thr_f1 = HC.exact_f1max_highest_ge(js[ok], va_inj_lab[:len(js)][ok])
    f1_point, _ = HC.f1_at(ts, te_lab, thr_f1["threshold"], ge=True)
    okv = np.isfinite(vs) & (vc > 0)
    budgets = {}
    for al in (0.005, 0.01, 0.05):
        m_pts = int(okv.sum())
        k = int(np.ceil((1 - al) * m_pts))
        k = min(max(k, 1), m_pts)
        thr = float(np.sort(vs[okv])[k - 1])
        f1, _ = HC.f1_at(ts, te_lab, thr, ge=False)
        budgets[f"alpha_{al}"] = {"k_rank": k, "m": m_pts, "threshold": thr,
                                  "comparison": ">", "f1_test": round(float(f1), 8)}

    np.savez_compressed(
        os.path.join(outc, "scores.npz"),
        test_scores=ts, test_labels=te_lab, test_coverage=tc,
        val_scores=vs, val_labels=np.zeros(len(va), dtype=np.int64), val_coverage=vc,
        val_injected_scores=js, val_injected_labels=va_inj_lab, val_injected_coverage=jc,
        split_index=np.array([i80, len(tr_raw)]),
        window_scores_test=Wt.astype(np.float32),
        window_index_test=st.astype(np.int32),
    )
    met = {
        "run_id": rid, "entity": a.entity, "seed": a.seed,
        "rescored_from_checkpoint": True, "retrained": False,
        "n_test_points": int(len(te_lab)),
        "coverage_profile": {"first5": tc[:5].tolist(), "middle": int(tc[win]),
                             "tail5": tc[-5:].tolist(),
                             "peak": int(tc.max()), "min": int(tc.min())},
        "threshold_exact_f1max": thr_f1["threshold"],
        "f1_test": round(float(f1_point), 8),
        "budget_strategies": budgets,
        "state_pairing": {
            "weights": "best normal-validation reconstruction loss (early stopping)",
            "memory_items": "tensor at END of phase-2 training, saved separately",
            "official_memto_does_same": True,
            "memory_at_best_epoch_available": False,
            "validation_updates_memory": val_updates_memory,
        },
        "memory_sha256_loaded": sha256(memp),
        "score_batch": a.score_batch,
        "quarantined_source": rid,
    }
    json.dump(met, open(os.path.join(outc, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    cfg["rescored"] = {"from_checkpoint": True, "score_batch": a.score_batch,
                       "aggregation": "timeline2d (window i, position j -> time i*stride+j)",
                       "note": "original 1-D aggregation was wrong and is retained in "
                               "the quarantined run for audit"}
    json.dump(cfg, open(os.path.join(outc, "config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(met, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
