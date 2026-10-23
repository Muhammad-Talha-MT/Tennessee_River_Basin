"""Atomic metadata writes and streaming source fingerprints."""

import hashlib
import json
import os

import numpy as np


def atomic_json(path, data):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def hash_array(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
