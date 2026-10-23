"""Public rain forecasts per reach, for training and selection with real forecast rain.

``download-gefs`` (and ``download-nwm-forecasts``) write monthly files holding
six-hour totals ``rain`` [day, step, reach] (mm, float16, NaN where missing) for
00Z issues listed in ``issue``. ``RainForecasts`` gathers them once into
``<folder>/cache/rain.npy`` and opens it memory-mapped, so several trainers share
one copy through the page cache. The cache is rebuilt when the monthly files change.
"""

import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd


class RainForecasts:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        files = sorted(f for f in self.folder.glob("*.npz") if not f.name.endswith(".part.npz"))
        if not files:
            raise ValueError(f"No rain-forecast files in {self.folder}")
        listing = [[f.name, f.stat().st_size] for f in files]
        cache = self.folder / "cache"
        cache.mkdir(exist_ok=True)
        with open(cache / "lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # one process builds, the others wait
            manifest = cache / "files.json"
            if not manifest.exists() or json.loads(manifest.read_text()) != listing:
                self._build(files, cache)
                manifest.write_text(json.dumps(listing))
        self.days = pd.DatetimeIndex(np.load(cache / "issues.npy").astype("datetime64[ns]"))
        self.rain = np.load(cache / "rain.npy", mmap_mode="r")
        self.position = {day: k for k, day in enumerate(self.days)}

    @staticmethod
    def _build(files, cache):
        days, shape = [], None
        for f in files:
            with np.load(f) as data:
                days += [str(d) for d in data["issue"]]
                shape = data["rain"].shape[1:]
        partial = cache / "rain.part.npy"
        rain = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float16, shape=(len(days), *shape)
        )
        start = 0
        for f in files:
            with np.load(f) as data:
                block = data["rain"]
                rain[start : start + len(block)] = block
                start += len(block)
        rain.flush()
        del rain
        np.save(cache / "issues.npy", np.array(days, dtype="datetime64[D]"))
        partial.rename(cache / "rain.npy")

    def __contains__(self, day):
        return pd.Timestamp(day).normalize() in self.position

    def get(self, day):
        """Six-hour rain [step, reach] (mm, float32) of the 00Z issue on ``day``."""
        return np.asarray(self.rain[self.position[pd.Timestamp(day).normalize()]], np.float32)
