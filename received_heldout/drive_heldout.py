"""Driver for the heldout group (TimesNet first; MEMTO/CATCH added when wired)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENTITIES = ["machine-1-1", "machine-3-1"]
SEEDS = [7, 42, 123, 2024, 2026]
EPOCHS = 10

MODELS = {
    "timesnet": os.path.join(HERE, "run_timesnet_heldout.py"),
    "memto": os.path.join(HERE, "run_memto_heldout.py"),
    "catch": os.path.join(HERE, "run_catch_heldout.py"),
}

LOG = os.path.join(HERE, "driver_heldout.log")


def main() -> int:
    which = sys.argv[1:] or ["timesnet"]
    rows = []
    with open(LOG, "a", encoding="utf-8") as log:
        for model in which:
            runner = MODELS.get(model)
            if not runner or not os.path.isfile(runner):
                log.write(f"SKIP {model}: runner missing ({runner})\n"); log.flush()
                rows.append({"model": model, "status": "runner_missing"})
                continue
            for ent in ENTITIES:
                for seed in SEEDS:
                    rid = f"{model}_heldout_{ent}_seed{seed}"
                    mpath = os.path.join(HERE, rid, "metrics.json")
                    if os.path.exists(mpath):
                        log.write(f"SKIP {rid}\n"); log.flush()
                        rows.append({"run_id": rid, "status": "skipped"})
                        continue
                    t0 = time.time()
                    p = subprocess.run([sys.executable, runner, "--entity", ent,
                                        "--seed", str(seed), "--epochs", str(EPOCHS)],
                                       capture_output=True, text=True, encoding="utf-8",
                                       errors="replace")
                    dt = round(time.time() - t0, 1)
                    ok = os.path.exists(mpath)
                    log.write(f"{'OK ' if ok else 'FAIL'} {rid} {dt}s rc={p.returncode}\n")
                    if not ok:
                        log.write((p.stdout or "")[-1500:] + "\n" + (p.stderr or "")[-1500:] + "\n")
                    else:
                        log.write((p.stdout or "")[-800:] + "\n")
                    log.flush()
                    rows.append({"run_id": rid, "status": "ok" if ok else "failed",
                                 "seconds": dt, "rc": p.returncode})
    json.dump(rows, open(os.path.join(HERE, "driver_status.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
