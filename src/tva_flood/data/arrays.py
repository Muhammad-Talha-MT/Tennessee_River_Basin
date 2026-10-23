"""Decoded, normalized monthly arrays as memory maps: no NetCDF work during training.

``build-array-cache`` stores exactly what ``CachedWindows.month`` returns
(normalized weather, weather validity, normalized log flow, flow validity) for
every prepared month. Ranks then read only the slices they need through the
shared operating-system page cache instead of each decoding whole months.
The cache is tied to the dataset identity, normalization file, COMID order and
each monthly source's size/mtime; any change makes the reader refuse it.
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tva_flood.common import atomic_json, sha

NAMES = ("weather", "weather_valid", "flow", "flow_valid")
FORMAT = "tva-flood-monthly-arrays-v1"


def identity(data, preparation):
    return dict(
        format=FORMAT,
        dataset_identity=data.manifest["identity"],
        dynamic_normalization_sha256=sha(Path(preparation) / "dynamic_normalization.json"),
        precip_transform=data.precip_transform,
        node_hash=data.manifest["node_hash"],
        nodes=len(data.ids),
        months=[
            {k: r[k] for k in ("path", "start", "steps", "bytes", "mtime_ns", "ordered_ids_sha256")}
            for r in data.records
        ],
    )


def any_split(preparation):
    """A split name of this preparation; months are the same whichever is used."""
    manifest = json.loads((Path(preparation) / "dataset_manifest.json").read_text())
    return next(iter(manifest["splits"]))


class ArrayMonths:
    """Read-only memory-mapped months for an ``EncoderWindows`` dataset."""

    def __init__(self, directory, data):
        self.directory = Path(directory)
        index = self.directory / "array_cache.json"
        if not index.exists():
            raise ValueError(f"Incomplete array cache (no {index.name}); rerun build-array-cache")
        stored = json.loads(index.read_text())
        if len(data.ids) != stored["nodes"] or data.nodes.tolist() != list(range(stored["nodes"])):
            raise ValueError("Array cache needs the loader to use the full graph node order")
        expected = identity(data, data.preparation)
        if stored != expected:
            raise ValueError("Array cache belongs to different prepared data; rebuild it")

    def month(self, number):
        folder = self.directory / f"{number:04d}"
        return tuple(np.load(folder / f"{name}.npy", mmap_mode="r") for name in NAMES)


_worker = None


def _initialize(root, preparation):
    global _worker
    from .dataset import CachedWindows

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _worker = CachedWindows(root, preparation, any_split(preparation), cache_months=1)


def _write(number, out):
    final = out / f"{number:04d}"
    if (final / "done.json").exists():
        return number, 0.0
    started = time.perf_counter()
    arrays = _worker.month(number)
    temp = out / f".{number:04d}.part"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    for name, array in zip(NAMES, arrays):
        with open(temp / f"{name}.npy", "wb") as handle:
            np.save(handle, np.ascontiguousarray(array))
            handle.flush()
            os.fsync(handle.fileno())
    record = _worker.records[number]
    atomic_json(
        temp / "done.json",
        dict(month=number, path=record["path"], shapes=[list(a.shape) for a in arrays]),
    )
    shutil.rmtree(final, ignore_errors=True)
    os.replace(temp, final)
    return number, time.perf_counter() - started


def main():
    """Write every prepared month once as normalized arrays for fast encoder training."""
    from .dataset import CachedWindows

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument("--out", type=Path, help="Default: <preparation>_arrays")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.workers < 1:
        p.error("Need at least one worker")
    root = a.root.resolve()
    prep = (a.preparation or root / "processed/pretraining_full").resolve()
    out = (a.out or prep.with_name(prep.name + "_arrays")).resolve()
    data = CachedWindows(root, prep, any_split(prep), cache_months=1)
    data.preparation = prep
    meta = identity(data, prep)
    out.mkdir(parents=True, exist_ok=True)
    (out / "array_cache.json").unlink(missing_ok=True)
    pending = [i for i in range(len(data.records)) if not (out / f"{i:04d}" / "done.json").exists()]
    steps = sum(r["steps"] for r in data.records)
    gib = steps * len(data.ids) * (8 * 4 + 8 + 4 + 1) / 1024**3
    print(
        f"Array cache: {out}; months={len(data.records)} pending={len(pending)}; "
        f"total size about {gib:.0f} GiB",
        flush=True,
    )
    began = time.perf_counter()
    with ProcessPoolExecutor(a.workers, initializer=_initialize, initargs=(root, prep)) as pool:
        for done, (number, seconds) in enumerate(pool.map(_write, pending, [out] * len(pending))):
            if done % 10 == 0 or done + 1 == len(pending):
                print(
                    f"month {number:04d} written in {seconds:.1f}s; {done + 1}/{len(pending)}; "
                    f"elapsed {(time.perf_counter() - began) / 60:.1f} min",
                    flush=True,
                )
    missing = [i for i in range(len(data.records)) if not (out / f"{i:04d}" / "done.json").exists()]
    if missing:
        raise RuntimeError(f"Months not written: {missing[:10]}")
    atomic_json(out / "array_cache.json", meta)
    print(f"ARRAY CACHE COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
