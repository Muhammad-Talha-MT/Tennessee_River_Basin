"""NOAA NWM medium-range streamflow at every graph reach, 00Z issues (for the comparison
with the NWM retrospective at all reaches, retrospective_comparison.py).

The benchmark download (download-nwm-forecasts) keeps the catalog gauges only; this one
reads the same archive files (deterministic member, channel_rt_1, leads 3-240 h) and
keeps all 58,834 reaches, as six-hour means over [t+6k, t+6k+6) (six_hour_flow). No rain.
One file per month: <out>/YYYY-MM.npz with issue [day], flow [day, 40, reach] (float32,
NaN where missing), reach (COMID order of the test preparation's graph).

  PYTHONPATH=src python scripts/nwm_all_reaches.py --start 2020-01-01 --stop 2023-01-21
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tva_flood.downloads.nwm_forecasts import FLOW_LEADS, WAIT, flow_task, six_hour_flow

ROOT = Path("/data/tva_data")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--stop", default="2023-01-21")
    p.add_argument("--out", type=Path, default=ROOT / "processed/benchmarks/nwm_medium_range_reaches")
    p.add_argument("--preparation", type=Path, default=ROOT / "processed/test_preparation_2020_2025")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.preparation / "graph.npz") as graph:
        ids = graph["feature_id"].copy()
    days = pd.date_range(a.start, a.stop, freq="D", inclusive="left")
    print(f"NWM medium range at {len(ids)} reaches, {len(days)} issue days", flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=context) as pool:
        for period, group in pd.Series(days).groupby(days.to_period("M")):
            target = a.out / f"{period}.npz"
            if target.exists():
                continue
            began = time.perf_counter()
            group = list(group)
            flow = np.full((len(group), len(FLOW_LEADS), len(ids)), np.nan, np.float32)
            missing, versions = [], [""] * len(group)
            tasks = {
                pool.submit(flow_task, day, int(lead), ids): (d, i)
                for d, day in enumerate(group)
                for i, lead in enumerate(FLOW_LEADS)
            }
            for future, (d, i) in tasks.items():
                values, version, url = future.result(timeout=WAIT)
                if values is None:
                    missing.append(url)
                    continue
                flow[d, i] = values
                versions[d] = version or versions[d]
            partial = target.with_name(target.stem + ".part.npz")
            np.savez(
                partial,
                issue=np.array([str(d.date()) for d in group]),
                flow=six_hour_flow(flow).astype(np.float32),
                version=np.array(versions),
                reach=ids,
            )
            partial.rename(target)
            target.with_name(target.stem + ".missing.json").write_text(json.dumps(missing, indent=1))
            print(
                f"{period}: {len(group)} days, versions {sorted(set(versions))}, "
                f"{len(missing)} missing files, {time.perf_counter() - began:.0f} s",
                flush=True,
            )
    print(f"Done: {a.out}", flush=True)
    os._exit(0)  # as download-nwm-forecasts: skip native-library cleanup at exit


if __name__ == "__main__":
    main()
