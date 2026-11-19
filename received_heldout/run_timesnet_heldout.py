"""TimesNet heldout runner (official TSLib model + official training semantics).

Divergences from TSLib's Exp_Anomaly_Detection, all recorded in config.json:
  D1 split      : train = raw train[0:i80), val = raw train[i80:), windows never cross.
  D2 scaler     : fitted on the 80% training segment only; val never influences it.
  D3 no test-in-training : TSLib computes a test loss every epoch for logging; removed.
  D4 val labels : TSLib's SMDSegLoader returns test_labels[0:win] as "val labels"
                  (a defect); here the normal val segment carries all-zero labels and a
                  separate injected copy is used only for post-hoc threshold choice.
  D5 threshold  : TSLib uses a percentile of test energies; here thresholds come from
                  the injected validation segment only, with exact tied-block F1-max.
  D6 scoring    : point-level scores are stride-1 window-score means with explicit
                  coverage counts (no zero-fill, no truncation).
Kept official: model, Adam(lr=1e-4), MSE criterion, early stopping on val recon loss,
type1 LR halving, 10 epochs, checkpoint-on-best-val.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TSLIB = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\dmkd_official\02_repos\Time-Series-Library"
PROJ_CODE = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\tsad-paper\tsad-paper\code"
sys.path.insert(0, HERE)
sys.path.insert(0, TSLIB)

import heldout_common as HC  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_args(seq_len: int, enc: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=seq_len, label_len=0, pred_len=0, d_model=64, d_ff=64, e_layers=2,
        top_k=5, num_kernels=6, enc_in=enc, dec_in=enc, c_out=enc, factor=1,
        dropout=0.1, embed="timeF", freq="h", output_attention=False, features="M",
        learning_rate=1e-4, train_epochs=10, patience=3, lradj="type1",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--inject-budget", type=float, default=0.05)
    ap.add_argument("--budget-strategy", type=float, default=0.05)
    args_in = ap.parse_args()

    torch.manual_seed(args_in.seed)
    np.random.seed(args_in.seed)
    torch.cuda.manual_seed_all(args_in.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rid = f"timesnet_heldout_{args_in.entity}_seed{args_in.seed}"
    outdir = os.path.join(HC.OUTROOT, rid)
    os.makedirs(outdir, exist_ok=True)

    man = HC.load_split(args_in.entity)
    d = os.path.join(HC.DATA, args_in.entity)
    tr_raw = np.load(os.path.join(d, "train_part.npy"))
    va_raw = np.load(os.path.join(d, "val_part.npy"))
    te_raw = np.load(os.path.join(d, "test.npy"))
    te_lab = np.load(os.path.join(d, "test_label.npy"))
    win = man["window"]
    i80 = man["split_index_i80"]

    # D2: scaler on the 80% segment only
    sc = HC.Scaler(tr_raw)
    tr = sc.transform(tr_raw)
    va = sc.transform(va_raw)
    te = sc.transform(te_raw)

    a = build_args(win, tr.shape[1])
    a.train_epochs = args_in.epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from models.TimesNet import Model as TimesNet  # official model
    model = TimesNet(a).float().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    crit = nn.MSELoss()

    # D1: windows strictly inside each segment
    tr_starts = np.arange(0, len(tr) - win + 1)
    va_starts = np.arange(0, len(va) - win + 1)
    assert tr_starts[-1] + win <= len(tr)
    assert va_starts[-1] + win <= len(va)
    tr_win = HC.make_windows(tr, tr_starts, win)
    va_win = HC.make_windows(va, va_starts, win)

    ckpt = os.path.join(outdir, "checkpoint.pth")
    best = float("inf")
    bad = 0
    epoch_log = []
    t0 = time.time()

    def val_recon_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_win), args_in.batch):
                xb = torch.from_numpy(va_win[i:i + args_in.batch]).float().to(device)
                out = model(xb, None, None, None)
                tot += float(crit(out, xb).item()) * len(xb)
                n += len(xb)
        model.train()
        return tot / max(n, 1)

    for epoch in range(a.train_epochs):
        model.train()
        perm = np.random.permutation(len(tr_win))
        tl, nb = 0.0, 0
        for i in range(0, len(perm), args_in.batch):
            idx = perm[i:i + args_in.batch]
            xb = torch.from_numpy(tr_win[idx]).float().to(device)
            opt.zero_grad()
            out = model(xb, None, None, None)
            loss = crit(out, xb)
            loss.backward()
            opt.step()
            tl += float(loss.item()); nb += 1
        vloss = val_recon_loss()
        improved = vloss < best - 1e-12
        if improved:
            best = vloss
            torch.save(model.state_dict(), ckpt)
            bad = 0
        else:
            bad += 1
        epoch_log.append({"epoch": epoch + 1, "train_loss": round(tl / max(nb, 1), 8),
                          "val_recon_loss": round(vloss, 8), "improved": improved,
                          "lr": opt.param_groups[0]["lr"]})
        print(f"[{rid}] epoch {epoch+1}/{a.train_epochs} train={tl/max(nb,1):.6f} "
              f"val={vloss:.6f} best={best:.6f} bad={bad}", flush=True)
        if bad >= a.patience:
            print("[early stop]", flush=True)
            break
        if a.lradj == "type1":
            lr = a.learning_rate * (0.5 ** (epoch + 1))
            for g in opt.param_groups:
                g["lr"] = lr

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()
    crit_el = nn.MSELoss(reduce=False)   # TSLib anomaly scoring criterion (per element)

    def point_scores(arr: np.ndarray, batch: int = 256):
        starts = np.arange(0, len(arr) - win + 1)
        rows = []
        with torch.no_grad():
            for i in range(0, len(starts), batch):
                chunk = starts[i:i + batch]
                xb = np.stack([arr[s:s + win] for s in chunk])
                xbt = torch.from_numpy(xb).float().to(device)
                out = model(xbt, None, None, None)
                loss = torch.mean(crit_el(out, xbt), dim=-1).cpu().numpy()
                rows.append(loss)
        ws = np.concatenate(rows, axis=0)
        return HC.timeline(ws, len(arr), win, 1)

    # normal validation (all-zero labels) kept for the record
    va_scores, va_cov = point_scores(va)
    va_labels_norm = np.zeros(len(va), dtype=np.int64)

    # injected validation: calibration only, never training / early stopping
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("pjd_data", os.path.join(PROJ_CODE, "data.py"))
    pjd = ilu.module_from_spec(spec)
    sys.modules["pjd_data"] = pjd
    spec.loader.exec_module(pjd)
    va_inj, va_inj_labels = pjd.inject_anomalies(
        va.astype(np.float64), np.zeros(len(va), dtype=int),
        ratio=args_in.inject_budget, seed=args_in.seed)   # injected AFTER scaling,
                                                          # as in the project's data.py
    va_inj_scores, va_inj_cov = point_scores(va_inj)

    ok = np.isfinite(va_inj_scores) & (va_inj_cov > 0)
    thr_f1 = HC.exact_f1max_highest_ge(va_inj_scores[ok], va_inj_labels[:len(va_inj_scores)][ok])
    thr_b = HC.order_stat_budget_gt(va_inj_scores[ok], budget=args_in.budget_strategy)

    te_scores, te_cov = point_scores(te)
    f1_point, _ = HC.f1_at(te_scores, te_lab, thr_f1["threshold"], ge=True)
    f1_budget, _ = HC.f1_at(te_scores, te_lab, thr_b["threshold"], ge=False)

    np.savez_compressed(
        os.path.join(outdir, "scores.npz"),
        test_scores=te_scores, test_labels=te_lab, test_coverage=te_cov,
        val_scores=va_scores, val_labels=va_labels_norm, val_coverage=va_cov,
        val_injected_scores=va_inj_scores, val_injected_labels=va_inj_labels,
        val_injected_coverage=va_inj_cov,
        split_index=np.array([man["split_index_i80"], len(tr_raw)]),
    )
    json.dump(epoch_log, open(os.path.join(outdir, "epoch_log.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)

    cfg = {
        "run_id": rid, "group": "heldout", "model": "TimesNet",
        "entity": args_in.entity, "seed": args_in.seed,
        "official_repo": "thuml/Time-Series-Library",
        "official_commit": "4e938a1767106324dd753b2a44832bf870a0252e",
        "split": man,
        "scaler": {"fit_on": "raw_train[0:i80)", "n_fit_points": int(len(tr_raw))},
        "model_args": {k: v for k, v in vars(a).items()},
        "training": {"optimizer": "Adam", "lr": a.learning_rate, "epochs_requested": a.train_epochs,
                     "epochs_run": len(epoch_log), "patience": a.patience,
                     "batch": args_in.batch, "criterion": "MSELoss",
                     "lr_schedule": "type1 halving", "checkpoint": "best val recon loss"},
        "thresholds": {"exact_f1max_highest_ge": thr_f1,
                       "order_stat_budget_gt": thr_b},
        "test_metrics": {"f1_exact_f1max_thr": round(float(f1_point), 8),
                         "f1_order_stat_budget_thr": round(float(f1_budget), 8),
                         "pa_applied": False},
        "divergences": ["D1 split 80/20", "D2 scaler on 80% only",
                        "D3 no test access during training",
                        "D4 val labels fixed (normal zeros; injected separate)",
                        "D5 thresholds from injected validation only",
                        "D6 stride-1 point scores with coverage, no truncation"],
        "compat_patches": ["models imported directly (no registry/CWD dependence)"],
        "data_hashes": {f: sha256(os.path.join(d, f)) for f in
                        ("train_part.npy", "val_part.npy", "test.npy", "test_label.npy")},
        "wall_time_s": round(time.time() - t0, 1),
    }
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    met = {"run_id": rid, "entity": args_in.entity, "seed": args_in.seed,
           "n_test_points": int(len(te_lab)),
           "coverage_min": int(te_cov.min()), "coverage_max": int(te_cov.max()),
           "thr_f1": thr_f1["threshold"], "thr_f1_value_on_val": round(thr_f1["f1"], 8),
           "f1_test": round(float(f1_point), 8),
           "f1_test_budget_strategy": round(float(f1_budget), 8),
           "epochs_run": len(epoch_log), "best_val_recon": round(best, 8),
           "scores_finite": bool(np.isfinite(te_scores).all())}
    json.dump(met, open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(met, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
