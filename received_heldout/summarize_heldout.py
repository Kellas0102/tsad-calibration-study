import glob
import json
import os

import numpy as np

H = r"C:\Users\98048\Documents\deepseek-harness\default-workspace\dmkd_official\06_heldout"

rows = []
for d in sorted(glob.glob(os.path.join(H, "timesnet_heldout_*"))):
    m = os.path.join(d, "metrics.json")
    if os.path.isfile(m):
        r = json.load(open(m, encoding="utf-8"))
        r["dir"] = os.path.basename(d)
        rows.append(r)

print("completed runs:", len(rows))
by = {}
for r in rows:
    by.setdefault(r["entity"], []).append(r)

summary = {}
for ent, sub in sorted(by.items()):
    f1 = [x["f1_test"] for x in sub]
    fb = [x["f1_test_budget_strategy"] for x in sub]
    summary[ent] = {
        "n": len(sub),
        "F1_mean": round(float(np.mean(f1)), 4),
        "F1_std": round(float(np.std(f1, ddof=1)), 4) if len(f1) > 1 else None,
        "F1_budget_mean": round(float(np.mean(fb)), 4),
        "per_seed": {x["seed"]: round(x["f1_test"], 4) for x in sub},
        "epochs_run": sorted({x["epochs_run"] for x in sub}),
        "coverage_min": sorted({x["coverage_min"] for x in sub}),
        "scores_finite_all": all(x["scores_finite"] for x in sub),
        "n_test_points": sub[0]["n_test_points"],
    }

out = {"completed": len(rows), "expected": 10, "summary": summary, "runs": rows}
json.dump(out, open(os.path.join(H, "heldout_timesnet_summary.json"), "w",
                    encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))

# split proof
splits = {}
for ent in ("machine-1-1", "machine-3-1"):
    p = os.path.join(H, "_data", ent, "split.json")
    if os.path.isfile(p):
        splits[ent] = json.load(open(p, encoding="utf-8"))
proof = {e: {"split_index_i80": s["split_index_i80"],
             "train_window_range": s["train_window_range"],
             "val_window_range": s["val_window_range"],
             "max_train_index": s["disjointness"]["max_train_index"],
             "min_val_index": s["disjointness"]["min_val_index"],
             "n_train_windows": s["n_train_windows"],
             "n_val_windows": s["n_val_windows"],
             "scaler_fit_range": s["scaler_fit_range"]}
        for e, s in splits.items()}
json.dump(proof, open(os.path.join(H, "split_proof.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(proof, ensure_ascii=False, indent=1))
