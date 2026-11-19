"""CATCH heldout runner v2 - corrected scoring and explicit schedule labelling.

Fixes over the quarantined version
    F1 scoring: the (n_windows, win) error matrix is aggregated with timeline2d, so
       window i position j contributes only to time i*stride+j. The previous version
       flattened the matrix and produced a shifted timeline with a wrong coverage profile.
    F2 checkpoint: the best-val weights are written to disk, so the run can be rescored
       later without retraining.
    F3 mask update cadence: the official code computes
       step = min(len(train_data_loader)//10, 100) over BATCHES. The quarantined run used
       the number of WINDOWS, capping at 100 instead of 17. Here the cadence is computed
       from the real batch count for the official schedule.
    F4 per-RNG artefacts: for every scoring RNG we store normal-val, injected-val and test
       scores plus coverage, and each RNG calibrates its own threshold on its own
       injected-val scores. RNG 0 is the pre-declared main analysis.
    F5 replay check: test is scored twice with the same RNG and the two score vectors are
       compared, to separate "seed control works" from "the method is stochastic".

Schedules
    --schedule official      epochs=3 (CATCH.py default), mask cadence from batch count
    --schedule modified10    epochs=10, mask cadence from window count (the quarantined
                             configuration, retrained because no checkpoint was saved)

The mask is never frozen: freezing it would change the method.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
C = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\dmkd_official\02_repos\CATCH"
BASE = C + r"\ts_benchmark\baselines\catch"
PROJ_CODE = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\tsad-paper\tsad-paper\code"
sys.path.insert(0, HERE)
import heldout_common as HC          # noqa: E402
from timeline2d import timeline2d    # noqa: E402

# NumPy 2.0 compat shim: CATCH's official tools.py uses np.Inf (removed in NumPy 2.0).
for _n, _v in (("Inf", float("inf")), ("Infinity", float("inf")), ("NaN", float("nan"))):
    if not hasattr(np, _n):
        setattr(np, _n, _v)


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_official():
    for name, path in [
        ("ts_benchmark", C + r"\ts_benchmark"),
        ("ts_benchmark.baselines", C + r"\ts_benchmark\baselines"),
        ("ts_benchmark.baselines.catch", BASE),
        ("ts_benchmark.baselines.catch.layers", BASE + r"\layers"),
        ("ts_benchmark.baselines.catch.models", BASE + r"\models"),
        ("ts_benchmark.baselines.catch.utils", BASE + r"\utils"),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    files = [
        ("ts_benchmark.baselines.catch.layers.RevIN", BASE + r"\layers\RevIN.py"),
        ("ts_benchmark.baselines.catch.layers.channel_mask", BASE + r"\layers\channel_mask.py"),
        ("ts_benchmark.baselines.catch.layers.cross_channel_Transformer",
         BASE + r"\layers\cross_channel_Transformer.py"),
        ("ts_benchmark.baselines.catch.utils.ch_discover_loss", BASE + r"\utils\ch_discover_loss.py"),
        ("ts_benchmark.baselines.catch.utils.fre_rec_loss", BASE + r"\utils\fre_rec_loss.py"),
        ("ts_benchmark.baselines.catch.utils.tools", BASE + r"\utils\tools.py"),
        ("ts_benchmark.baselines.catch.models.CATCH_model", BASE + r"\models\CATCH_model.py"),
    ]
    for name, path in files:
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return (sys.modules["ts_benchmark.baselines.catch.models.CATCH_model"].CATCHModel,
            sys.modules["ts_benchmark.baselines.catch.utils.fre_rec_loss"].frequency_loss,
            sys.modules["ts_benchmark.baselines.catch.utils.fre_rec_loss"].frequency_criterion,
            sys.modules["ts_benchmark.baselines.catch.utils.tools"].EarlyStopping,
            sys.modules["ts_benchmark.baselines.catch.utils.tools"].adjust_learning_rate)


OFFICIAL = dict(lr=1e-4, Mlr=1e-5, e_layers=3, n_heads=2, cf_dim=64, d_ff=256, d_model=128,
                head_dim=64, individual=0, dropout=0.2, head_dropout=0.1,
                auxi_mode="fft", auxi_type="complex", auxi_loss="MAE",
                auxi_lambda=0.005, score_lambda=0.05, regular_lambda=0.5, temperature=0.07,
                patch_stride=8, patch_size=16, inference_patch_stride=1,
                inference_patch_size=32, dc_lambda=0.005, module_first=True, mask=False,
                num_epochs=3, batch_size=128, patience=3, seq_len=192, pct_start=0.3,
                revin=1, affine=0, subtract_last=0, lradj="type1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--schedule", choices=["official", "modified10"], default="official")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rng-repeats", type=int, default=5)
    ap.add_argument("--inject-budget", type=float, default=0.05)
    ap.add_argument("--score-batch", type=int, default=128)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tag = "official" if a.schedule == "official" else "modified-schedule-10epoch"
    rid = f"catch_{tag}_{a.entity}_seed{a.seed}"
    outdir = os.path.join(HC.OUTROOT, "corrected", rid)
    os.makedirs(outdir, exist_ok=True)

    man = HC.load_split(a.entity)
    d = os.path.join(HC.DATA, a.entity)
    tr_raw = np.load(os.path.join(d, "train_part.npy"))
    va_raw = np.load(os.path.join(d, "val_part.npy"))
    te_raw = np.load(os.path.join(d, "test.npy"))
    te_lab = np.load(os.path.join(d, "test_label.npy"))
    win, i80 = man["window"], man["split_index_i80"]
    sc = HC.Scaler(tr_raw)
    tr, va, te = sc.transform(tr_raw), sc.transform(va_raw), sc.transform(te_raw)
    enc = tr.shape[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (CATCHModel, frequency_loss, frequency_criterion,
     EarlyStopping, adjust_learning_rate) = load_official()

    cfg = dict(OFFICIAL)
    epochs = 3 if a.schedule == "official" else 10
    cfg.update(seq_len=win, c_in=enc, c_out=enc, num_epochs=epochs, batch_size=a.batch)
    cfg["learning_rate"] = cfg["lr"]
    configs = types.SimpleNamespace(**cfg)

    torch.manual_seed(a.seed)
    model = CATCHModel(configs).float().to(device)

    def windows(arr):
        starts = np.arange(0, len(arr) - win + 1)
        return np.stack([arr[s:s + win] for s in starts]).astype(np.float32)

    tr_w, va_w = windows(tr), windows(va)
    n_batches = int(np.ceil(len(tr_w) / configs.batch_size))
    # F3: official cadence is over BATCHES
    mask_step = min(n_batches // 10, 100) if a.schedule == "official" else min(len(tr_w) // 10, 100)
    mask_step = max(mask_step, 1)

    main_params = [p for n, p in model.named_parameters() if "mask_generator" not in n]
    opt = torch.optim.Adam(main_params, lr=configs.lr)
    optM = torch.optim.Adam(model.mask_generator.parameters(), lr=configs.Mlr)
    from torch.optim import lr_scheduler
    sch = lr_scheduler.OneCycleLR(opt, steps_per_epoch=n_batches, pct_start=configs.pct_start,
                                  epochs=configs.num_epochs, max_lr=configs.lr)
    schM = lr_scheduler.OneCycleLR(optM, steps_per_epoch=n_batches,
                                   pct_start=configs.pct_start,
                                   epochs=configs.num_epochs, max_lr=configs.Mlr)

    crit = nn.MSELoss(); aux = frequency_loss(configs)
    es = EarlyStopping(patience=configs.patience, verbose=False)
    log = []; t0 = time.time()

    for ep in range(configs.num_epochs):
        model.train()
        perm = np.random.permutation(len(tr_w)); tl, nb = 0.0, 0
        for i in range(0, len(perm), configs.batch_size):
            sel = perm[i:i + configs.batch_size]
            xb = torch.from_numpy(tr_w[sel]).float().to(device)
            opt.zero_grad()
            out, out_c, dc = model(xb)
            rec = crit(out, xb)
            auxl = aux(out_c, model.revin_layer(xb, "transform"))
            loss = rec + configs.dc_lambda * dc + configs.auxi_lambda * auxl
            nb += 1
            if nb % mask_step == 0:
                optM.step(); optM.zero_grad()
            loss.backward(); opt.step()
            tl += float(rec.item())
        model.eval(); vl, vn = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(va_w), configs.batch_size):
                xb = torch.from_numpy(va_w[j:j + configs.batch_size]).float().to(device)
                o, _, _ = model(xb)
                vl += float(crit(o, xb).item()); vn += 1
        vloss = vl / max(vn, 1)
        es(vloss, model)
        log.append({"epoch": ep + 1, "train_rec": round(tl / max(nb, 1), 8),
                    "val_rec": round(vloss, 8), "mask_step": mask_step,
                    "n_batches": n_batches})
        print(f"[{rid}] ep{ep+1}/{configs.num_epochs} train={tl/max(nb,1):.6f} "
              f"val={vloss:.6f} mask_step={mask_step}", flush=True)
        if es.early_stop:
            break
        adjust_learning_rate(opt, sch, ep + 1, configs)
        adjust_learning_rate(optM, schM, ep + 1, configs, printout=False)

    model.load_state_dict(es.check_point)
    torch.save(model.state_dict(), os.path.join(outdir, "checkpoint.pth"))   # F2

    fcrit = frequency_criterion(configs)
    crit_el = nn.MSELoss(reduce=False)

    def score_2d(arr, rng):
        torch.manual_seed(rng)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng)
        model.eval()
        starts = np.arange(0, len(arr) - win + 1)
        rows = []
        with torch.no_grad():
            for j in range(0, len(starts), a.score_batch):
                ch = starts[j:j + a.score_batch]
                xb = np.stack([arr[s:s + win] for s in ch])
                xbt = torch.from_numpy(xb).float().to(device)
                o, _, _ = model(xbt)
                temp = torch.mean(crit_el(xbt, o), dim=-1)
                freq = torch.mean(fcrit(xbt, o), dim=-1)
                rows.append((temp + configs.score_lambda * freq).cpu().numpy())
        return np.concatenate(rows, axis=0)

    spec = importlib.util.spec_from_file_location("pjd_data", os.path.join(PROJ_CODE, "data.py"))
    pjd = importlib.util.module_from_spec(spec); sys.modules["pjd_data"] = pjd
    spec.loader.exec_module(pjd)
    va_inj, va_inj_lab = pjd.inject_anomalies(va.astype(np.float64),
                                              np.zeros(len(va), dtype=int),
                                              ratio=a.inject_budget, seed=a.seed)

    per_rng = []
    store = {"test_labels": te_lab, "val_labels": np.zeros(len(va), dtype=np.int64),
             "val_injected_labels": va_inj_lab, "split_index": np.array([i80, len(tr_raw)]),
             "rng_list": np.arange(a.rng_repeats)}
    for r in range(a.rng_repeats):
        Wv = score_2d(va, r); Wj = score_2d(va_inj, r); Wt = score_2d(te, r)
        vs, vc = timeline2d(Wv, len(va), win, 1)
        js, jc = timeline2d(Wj, len(va), win, 1)
        ts, tc = timeline2d(Wt, len(te), win, 1)
        okv = np.isfinite(vs) & (vc > 0)
        okj = np.isfinite(js) & (jc > 0)
        thr = HC.exact_f1max_highest_ge(js[okj], va_inj_lab[:len(js)][okj])
        f1, _ = HC.f1_at(ts, te_lab, thr["threshold"], ge=True)
        budgets = {}
        for al in (0.005, 0.01, 0.05):
            m_pts = int(okv.sum()); k = min(max(int(np.ceil((1 - al) * m_pts)), 1), m_pts)
            bthr = float(np.sort(vs[okv])[k - 1])
            bf1, _ = HC.f1_at(ts, te_lab, bthr, ge=False)
            budgets[f"alpha_{al}"] = {"k_rank": k, "m": m_pts, "threshold": bthr,
                                      "comparison": ">", "f1_test": round(float(bf1), 8)}
        # F5 replay: one seed-control check is enough; run it for rng 0 only
        if r == 0:
            Wt2 = score_2d(te, r); ts2, _ = timeline2d(Wt2, len(te), win, 1)
            replay_identical = bool(sha256_arr(ts) == sha256_arr(ts2))
        else:
            replay_identical = None
        per_rng.append({"rng": r, "threshold": thr["threshold"], "val_f1": thr["f1"],
                        "f1_test": round(float(f1), 8),
                        "score_sha256": sha256_arr(ts)[:32],
                        "replay_identical": replay_identical,
                        "budget_strategies": budgets,
                        "coverage_tail": tc[-3:].tolist()})
        store[f"test_scores_rng{r}"] = ts; store[f"test_coverage_rng{r}"] = tc
        store[f"val_scores_rng{r}"] = vs; store[f"val_coverage_rng{r}"] = vc
        store[f"val_injected_scores_rng{r}"] = js
        store[f"val_injected_coverage_rng{r}"] = jc
        print(f"[{rid}] rng{r} f1={f1:.6f} replay={replay_identical}", flush=True)

    np.savez_compressed(os.path.join(outdir, "scores.npz"), **store)
    json.dump(log, open(os.path.join(outdir, "epoch_log.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    f1s = [x["f1_test"] for x in per_rng]
    cfgout = {"run_id": rid, "group": "corrected", "model": "CATCH",
              "schedule": tag, "entity": a.entity, "seed": a.seed,
              "official_repo": "decisionintelligence/CATCH",
              "official_commit": "3647c69be5eb56649b072596cf89098e689e20c3",
              "split": man, "hyper_params": cfg,
              "schedule_facts": {"epochs": configs.num_epochs, "n_batches": n_batches,
                                 "n_windows": len(tr_w), "mask_step": mask_step,
                                 "cadence_basis": ("batches (official)" if a.schedule == "official"
                                                   else "windows (quarantined config)")},
              "scoring": {"rngs": list(range(a.rng_repeats)), "main_rng": 0,
                          "mask_frozen": False, "per_rng_own_threshold": True,
                          "aggregation": "timeline2d"},
              "checkpoint_saved": True,
              "patches": ["F1 timeline2d aggregation", "F2 checkpoint persisted",
                          "F3 mask cadence from batch count (official schedule)",
                          "F4 per-RNG val/injected/test scores+coverage",
                          "F5 same-RNG replay check", "P6 numpy2 np.Inf alias",
                          "P7 official modules loaded by path (no reformer_pytorch)"],
              "wall_time_s": round(time.time() - t0, 1)}
    json.dump(cfgout, open(os.path.join(outdir, "config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    met = {"run_id": rid, "schedule": tag, "entity": a.entity, "seed": a.seed,
           "n_test_points": int(len(te_lab)),
           "f1_rng0": f1s[0], "f1_mean_over_rng": round(float(np.mean(f1s)), 8),
           "f1_std_over_rng": round(float(np.std(f1s, ddof=1)), 8) if len(f1s) > 1 else None,
           "f1_range_over_rng": round(float(max(f1s) - min(f1s)), 8),
           "replay_all_identical": all(x["replay_identical"] for x in per_rng if x["replay_identical"] is not None),
           "distinct_score_hashes": len({x["score_sha256"] for x in per_rng}),
           "mask_step": mask_step, "n_batches": n_batches, "epochs": len(log),
           "coverage_tail": per_rng[0]["coverage_tail"],
           "per_rng": per_rng, "wall_time_s": cfgout["wall_time_s"]}
    json.dump(met, open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: met[k] for k in ("run_id", "f1_rng0", "f1_std_over_rng",
                                          "replay_all_identical", "coverage_tail",
                                          "mask_step", "n_batches")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
