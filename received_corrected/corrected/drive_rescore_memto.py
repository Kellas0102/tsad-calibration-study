"""Driver: rescore all quarantined MEMTO heldout runs from their checkpoints."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "rescore_memto.py")
ENTITIES = ["machine-1-1", "machine-3-1"]
SEEDS = [7, 42, 123, 2024, 2026]
OUT = os.path.join(HERE, "corrected")

rows = []
for ent in ENTITIES:
    for seed in SEEDS:
        rid = f"memto_heldout_{ent}_seed{seed}"
        mpath = os.path.join(OUT, rid, "metrics.json")
        if os.path.exists(mpath):
            rows.append({"run_id": rid, "status": "skipped"})
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, RUNNER, "--entity", ent, "--seed", str(seed)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = round(time.time() - t0, 1)
        ok = os.path.exists(mpath)
        rows.append({"run_id": rid, "status": "ok" if ok else "failed",
                     "seconds": dt, "rc": p.returncode})
        if not ok:
            with open(os.path.join(OUT, "rescore_failures.log"), "a", encoding="utf-8") as f:
                f.write(f"{rid}\n{(p.stdout or '')[-1500:]}\n{(p.stderr or '')[-1500:]}\n\n")
        print(("OK  " if ok else "FAIL") + f" {rid} {dt}s", flush=True)

json.dump(rows, open(os.path.join(OUT, "rescore_status.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(rows, ensure_ascii=False, indent=1))
