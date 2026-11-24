"""Config diff of M4 against M1 (blocked fold 1, stage B; stage A), from the pipeline's own dry-run output (nothing is launched).
Writes reports/v3_tables/m4_config_diff.txt. Run: python scripts/v3_m4_config_diff.py"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
env = dict(os.environ, DRY_RUN="1", GPUS="0 1")


def dry(phase):
    out = subprocess.run(["bash", "scripts/attribution_pipeline.sh", phase], cwd=REPO, env=env, capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if "tva_flood.hydro.train" in l]


def parse(cmd):
    t = shlex.split(cmd.split(">>")[0])
    a, d, k = t[t.index("tva_flood.hydro.train") + 1 :], {}, None
    for x in a:
        if x.startswith("--"):
            k = x
            d[k] = []
        else:
            d[k].append(x)
    return d


m1 = parse([c for c in dry("blocked_m1") if "S1_M1_s1_b1" in c][0])
m4_cmds = dry("m4")
m4 = parse([c for c in m4_cmds if "S1_M4_s1_b1" in c][0])
lines = ["M4 against M1, stage B, blocked fold 1 (settings that differ):"]
lines += [f"  {k}: {m1.get(k)} -> {m4.get(k)}" for k in sorted(set(m1) | set(m4)) if m1.get(k) != m4.get(k)]
lines += ["", "Settings identical: " + ", ".join(sorted(k for k in set(m1) & set(m4) if m1[k] == m4[k])), ""]
a_m4 = parse([c for c in m4_cmds if "instant_A" in c and "--stage A" in c][0])
lines += ["M4 stage A (instant_A) arguments: " + " ".join(f"{k} {' '.join(v)}" for k, v in a_m4.items() if k != "--out"),
          "M1 stage A (manning_A) arguments:  --stage A --router manning --updates 5000 --lr 5e-4 --warmup 500 --validate-every 1000 --resume --graph-layers 3 (from the pipeline's MANNING_A)",
          "Both: 5,000 updates of 8 samples, same learning-rate schedule, 3 graph layers; only the router differs."]
out = REPO / "reports" / "v3_tables" / "m4_config_diff.txt"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
