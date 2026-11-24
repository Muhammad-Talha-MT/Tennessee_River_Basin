"""Phase 1.2 support: the model configs and training arguments of M0, M1, M2 (and the LSTM) read from the saved files.

Writes reports/v3_tables/config_audit.csv: one row per setting, one column per model; settings equal in all models are marked.
Run: python scripts/v3_config_audit.py
"""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import pandas as pd
import torch
from v3_common import A, OUT

OUT.mkdir(parents=True, exist_ok=True)
rows = {}
for name in ("M0", "M1", "M2"):
    d = A / f"{name}_s1"
    state = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    args = json.loads((d / "arguments.json").read_text())
    r = {f"config.{k}": v for k, v in cfg.items() if k not in ("groups", "delay")}
    r.update({f"args.{k}": v for k, v in args.items() if k not in ("out",)})
    r["parameters"] = sum(p.numel() for p in state["model"].values())
    rows[name] = r
df = pd.DataFrame(rows).fillna("(absent)").astype(str)
df["same_in_M0_M1_M2"] = (df["M0"] == df["M1"]) & (df["M1"] == df["M2"])
df.to_csv(OUT / "config_audit.csv")
print(df[~df.same_in_M0_M1_M2].drop(columns="same_in_M0_M1_M2").to_string())
print("\nsettings identical in all three:", int(df.same_in_M0_M1_M2.sum()), "of", len(df))
print(df[df.same_in_M0_M1_M2].index.tolist())
