"""Prepare a held-out test period after the validation year, for checkpoints trained as-is.

The test preparation is a separate directory holding the same graph and the same
byte-identical statics and training-only normalization as the source preparation.
Its manifest lists the test months (plus the month before, for the 30-day
history) under a ``test`` split and records the source's identity as
``parent_identity``, which is what checkpoints compare. Nothing is refit, the
source preparation and its array cache stay valid, and no model sees these
months in training or selection (training ends 2019-01-01; 2019 is validation).
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tva_flood.common import atomic_json

from .prepare import scan_archive
from .schema import HISTORY, HORIZON

FIRST_TEST = pd.Timestamp("2020-01-01")  # 2019 is the validation year
COPIED = ("graph.npz", "static_normalization.json", "dynamic_normalization.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path, help="Source (default processed/pretraining_full)")
    p.add_argument("--out", type=Path, help="Default: <root>/processed/test_preparation")
    p.add_argument("--start", default="2020-01-01", help="First test month (month start)")
    p.add_argument("--stop", default="2023-02-01", help="Exclusive month boundary")
    a = p.parse_args()
    start, stop = map(pd.Timestamp, (a.start, a.stop))
    if any(t.tzinfo is not None or t != t.to_period("M").start_time for t in (start, stop)):
        p.error("Dates must be timezone-naive month starts")
    if not FIRST_TEST <= start < stop:
        p.error("The test period must start in 2020 or later (2019 is the validation year)")
    root = a.root.resolve()
    source = (a.preparation or root / "processed/pretraining_full").resolve()
    out = (a.out or root / "processed/test_preparation").resolve()
    if out == source:
        p.error("--out must differ from the source preparation")
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    if not manifest["training_ready"]:
        p.error("Source preparation is not training-ready")
    if "test" in manifest["splits"]:
        p.error("Source is already a test preparation")
    parent = manifest["identity"]
    with np.load(source / "graph.npz") as graph:
        ids = graph["feature_id"]
    # Every month must hold exactly the graph's COMIDs (their order may differ).
    records = scan_archive(root, ids, start - pd.offsets.MonthBegin(1), stop)
    new = hashlib.sha256(
        json.dumps(
            dict(parent=parent, months=records, test=[str(start), str(stop)]), sort_keys=True
        ).encode()
    ).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "dataset_manifest.json"
    if existing.exists() and json.loads(existing.read_text()).get("identity") != new:
        p.error(f"{out} holds a different preparation; choose a new --out")
    for name in COPIED:
        shutil.copyfile(source / name, out / name)
    steps = sum(r["steps"] for r in records if pd.Timestamp(r["start"]) >= start)
    for key in ("train_windows", "validation_windows"):
        manifest.pop(key)
    manifest.update(
        identity=new,
        parent_identity=parent,
        months=records,
        splits=dict(test=[str(start), str(stop)]),
        test_windows=max(0, steps - HORIZON + 1),
        history=HISTORY,
        training_ready=True,
        note="Held-out test period; training, normalization and selection never used it.",
    )
    atomic_json(out / "dataset_manifest.json", manifest)
    print(
        f"Test preparation: {out}; months {records[0]['start'][:7]} to {records[-1]['start'][:7]}; "
        f"test issues about {manifest['test_windows']:,}; parent {parent[:12]}"
    )


if __name__ == "__main__":
    main()
