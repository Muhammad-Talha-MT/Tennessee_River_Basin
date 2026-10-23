"""Shared distributed loop for the graph-size-independent temporal/graph encoder.

Used by ``ssl-pretrain`` and ``train-forecast``. Each GPU process trains on
``--batch-size`` different dates of one connected river patch (or the full
graph), so ranks see different places and times. Encoder outputs at the patch
core are exactly their full-graph values (see ``data/subgraph.py``); losses use
core reaches only. Validation always runs on the full graph.

The learning-rate schedule spans the requested update budget, every evaluated
model is kept under ``checkpoints/``, and an interrupted run resumes exactly
(optimizer, scheduler, data position and each rank's RNG). Source files are
snapshotted on the first run; edits are recorded and reported on resume rather
than refused, because numerical equivalence is checked by tests, not file bytes.
"""

import argparse
import fcntl
import json
import math
import os
import random
import shutil
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tva_flood.common import atomic_json, sha
from tva_flood.data.river import River
from tva_flood.data.ssl import EncoderWindows
from tva_flood.data.subgraph import SubgraphSampler
from tva_flood.models.sparse import sparse_supports
from tva_flood.models.ssl import TemporalGraphEncoder

from .runtime import capture_rng, reduced, restore_rng, sample_windows, save_atomic

PACKAGE = Path(__file__).parents[1]
SOURCE_FILES = (
    "data/arrays.py",
    "data/dataset.py",
    "data/river.py",
    "data/schema.py",
    "data/ssl.py",
    "data/statics.py",
    "data/subgraph.py",
    "data/transforms.py",
    "evaluation/metrics.py",
    "models/sparse.py",
    "models/ssl.py",
    "training/encoder_loop.py",
    "training/forecast.py",
    "training/ssl_pretrain.py",
)
ARCHITECTURE_DEFAULTS = dict(
    hidden=32,
    temporal_layers=6,
    graph_layers=4,
    dropout=0.0,
    direction="both",
    aggregation="mean",
    basin_inputs=False,
)
# Validation masks and windows never depend on --seed, so runs remain comparable.
VALIDATION_SEED = 20190101


def sources():
    return {name: sha(PACKAGE / name) for name in SOURCE_FILES}


def add_arguments(p, updates, batch_size, subgraph_nodes, lr):
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--preparation", type=Path)
    p.add_argument(
        "--array-cache", type=Path, help="Directory written by build-array-cache (recommended)"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--updates", type=int, default=updates, help="Optimizer-update budget; sets the schedule"
    )
    p.add_argument("--batch-size", type=int, default=batch_size, help="Dates per GPU per update")
    p.add_argument(
        "--subgraph-nodes",
        type=int,
        default=subgraph_nodes,
        help="Core reaches in each training patch; 0 trains on the full graph",
    )
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-updates", type=int, default=200)
    p.add_argument("--min-lr-factor", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    for name, kind in (
        ("hidden", int),
        ("temporal-layers", int),
        ("graph-layers", int),
        ("dropout", float),
    ):
        default = ARCHITECTURE_DEFAULTS[name.replace("-", "_")]
        p.add_argument(f"--{name}", type=kind, help=f"Encoder architecture (default {default})")
    p.add_argument(
        "--direction",
        choices=["both", "upstream"],
        help="Graph messages in both river directions (default) or from upstream only. "
        "upstream trains on whole basins (needs build-routing) and allows the basin options",
    )
    p.add_argument(
        "--aggregation",
        choices=["mean", "attention"],
        help="Graph blocks average their neighbours (default) or attend to them",
    )
    p.add_argument(
        "--basin-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add basin-mean weather history over each reach's whole upstream basin",
    )
    p.add_argument(
        "--checkpoint-blocks", action="store_true", help="Recompute activations to save memory"
    )
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument(
        "--shuffle",
        choices=["monthly", "global"],
        default="monthly",
        help="monthly keeps NetCDF cache locality; global mixes seasons (use with --array-cache)",
    )
    p.add_argument("--cache-months", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument(
        "--validation-windows",
        type=int,
        default=64,
        help="Evenly spaced 2019 windows per evaluation; 0 uses every window",
    )
    p.add_argument("--early-stop-patience", type=int, default=0, help="Evaluations; 0 disables")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop safely after this many updates; rerun to resume",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        help="Only score this checkpoint of the same command on validation; no training",
    )
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")


def check_arguments(p, a):
    if (
        min(a.updates, a.batch_size, a.cache_months, a.eval_every, a.checkpoint_every, a.log_every)
        < 1
    ):
        p.error("Updates, batch size, cache, evaluation, checkpoint and log intervals must be >= 1")
    if (
        min(
            a.subgraph_nodes,
            a.warmup_updates,
            a.validation_windows,
            a.early_stop_patience,
            a.max_steps,
            a.seed,
        )
        < 0
    ):
        p.error("Subgraph, warmup, validation, patience, max-steps and seed must be >= 0")
    if a.lr <= 0 or a.clip <= 0 or a.weight_decay < 0 or not 0 < a.min_lr_factor <= 1:
        p.error("Positive learning rate and clip, nonnegative decay, min LR factor in (0,1]")


def resolve_architecture(p, a, inherited=None):
    explicit = dict(
        hidden=a.hidden,
        temporal_layers=a.temporal_layers,
        graph_layers=a.graph_layers,
        dropout=a.dropout,
        direction=getattr(a, "direction", None),
        aggregation=getattr(a, "aggregation", None),
        basin_inputs=getattr(a, "basin_inputs", None),
    )
    # Checkpoints from before the river options hold the defaults for them.
    base = {**ARCHITECTURE_DEFAULTS, **(inherited or {})}
    for key, value in explicit.items():
        if value is None:
            continue
        if inherited is not None and value != base[key]:
            p.error(f"--{key.replace('_', '-')} differs from the inherited architecture")
        base[key] = value
    if base["hidden"] < 2 or base["temporal_layers"] < 1 or base["graph_layers"] < 0:
        p.error("Invalid encoder architecture")
    if not 0 <= base["dropout"] < 1:
        p.error("Dropout must be in [0,1)")
    if base["basin_inputs"] and base["direction"] != "upstream":
        p.error("--basin-inputs needs --direction upstream")
    if base["aggregation"] == "attention" and base["hidden"] % 4:
        p.error("--aggregation attention needs --hidden divisible by 4")
    return base


def setup():
    rank, world, local = [
        int(os.environ.get(key, default))
        for key, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    ]
    return rank, world, local


def daily_order(data, seed, epoch, shuffle):
    """One random issue per calendar day; months or all days in random order."""
    times = data.times[data.issues]
    days = times.normalize().asi8
    _, starts, counts = np.unique(days, return_index=True, return_counts=True)
    rng = np.random.default_rng([seed, epoch])
    selected = starts + (rng.random(len(starts)) * counts).astype(np.int64)
    if shuffle == "global":
        return rng.permutation(selected)
    labels = times.to_period("M").astype(str).to_numpy()[selected]
    return np.concatenate(
        [rng.permutation(selected[labels == month]) for month in rng.permutation(np.unique(labels))]
    )


def derived_seed(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


class Run:
    """Distributed setup, model/optimizer state, checkpoints and the update loop."""

    def __init__(self, task, a, p):
        self.task, self.a, self.p = task, a, p
        self.rank, self.world, local = setup()
        self.device = torch.device(f"cuda:{local}" if a.device == "cuda" else "cpu")
        if a.device == "cuda":
            if not torch.cuda.is_available():
                p.error("CUDA unavailable; use --device cpu for small tests")
            torch.cuda.set_device(self.device)
        if self.world > 1:
            kwargs = dict(
                backend="nccl" if a.device == "cuda" else "gloo", timeout=timedelta(minutes=60)
            )
            if a.device == "cuda":
                kwargs["device_id"] = self.device
            dist.init_process_group(**kwargs)

    def barrier(self):
        if self.world > 1:
            dist.barrier(device_ids=[self.device.index] if self.device.type == "cuda" else None)

    def autocast(self):
        if self.a.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def log(self, row, name="training_log.jsonl"):
        if self.rank == 0:
            print(json.dumps(row), flush=True)
            with open(self.out / name, "a") as handle:
                handle.write(json.dumps(row) + "\n")

    def prepare(self):
        a, task = self.a, self.task
        self.root = a.root.resolve()
        self.prep = (a.preparation or self.root / "processed/pretraining_full").resolve()
        self.out = a.out.resolve()
        if self.out in (self.root, self.prep):
            self.p.error("Use a separate experiment directory for --out")
        if self.rank == 0:
            self.out.mkdir(parents=True, exist_ok=True)
            self.lock = open(self.out / ".run.lock", "w")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another run is using {self.out}") from None
        self.barrier()
        array_cache = a.array_cache.resolve() if a.array_cache else None
        future = getattr(task, "future_rain", False)
        self.train = EncoderWindows(
            self.root,
            self.prep,
            "train",
            task.horizon,
            a.cache_months,
            array_cache,
            future,
        )
        self.validation = EncoderWindows(
            self.root,
            self.prep,
            "validation",
            task.horizon,
            a.cache_months,
            array_cache,
            future,
        )
        self.validation.cache = self.train.cache
        with np.load(self.prep / "graph.npz") as graph:
            self.ids = graph["feature_id"].copy()
            self.edges = np.column_stack([graph["source"], graph["target"]])
        if not np.array_equal(self.ids, self.train.ids):
            raise ValueError("Loader node order differs from graph.npz")
        if a.evaluate_checkpoint:
            self.evaluated = torch.load(
                a.evaluate_checkpoint, map_location="cpu", weights_only=False
            )
            if self.evaluated.get("format") != task.format:
                self.p.error(f"--evaluate-checkpoint is not a {task.format} checkpoint")
            inherited = self.evaluated["config"]["architecture"]
            self.architecture = resolve_architecture(self.p, a, inherited)
        else:
            self.architecture = task.architecture(self.p, a)
        supports = sparse_supports(len(self.ids), self.edges)
        # Every graph block, in the encoder or a decoder, needs one halo hop; with
        # upstream-only messages, patches are whole basins instead (data.river).
        halo = getattr(task, "halo", lambda arch: arch["graph_layers"])(self.architecture)
        self.halo = halo
        self.river = (
            River.load(self.root, self.ids, self.edges)
            if self.architecture["direction"] == "upstream"
            else None
        )
        self.sampler = SubgraphSampler(len(self.ids), self.edges, supports, halo, self.river)
        self.full_graph = self.sampler.full()
        self.full_graph["supports"] = [s.to(self.device) for s in self.full_graph["supports"]]
        self.full_static = torch.as_tensor(
            self.train.static, dtype=torch.float32, device=self.device
        )
        torch.manual_seed(a.seed)
        encoder = TemporalGraphEncoder(
            [s.to(self.device) for s in supports],
            checkpoint_blocks=a.checkpoint_blocks,
            **self.architecture,
        )
        self.model = task.build(encoder).to(self.device)
        if self.river is not None:
            encoder.full_routing = self.river.operator(np.arange(len(self.ids))).to(self.device)
        task.initialize(self.model, self)
        self.items = sample_windows(
            len(self.validation), a.validation_windows or len(self.validation)
        )
        self.mean, self.std = float(self.train.mean[8]), float(self.train.std[8])

    def configuration(self):
        a = self.a
        per_epoch = len(daily_order(self.train, a.seed, 0, a.shuffle)) // (
            self.world * a.batch_size
        )
        if per_epoch < 1:
            raise ValueError("Too few training days for world size x batch size")
        return dict(
            format=self.task.format,
            dataset_identity=self.train.manifest["identity"],
            normalization_sha256=sha(self.prep / "dynamic_normalization.json"),
            static_normalization_sha256=sha(self.prep / "static_normalization.json"),
            array_cache=str(a.array_cache.resolve()) if a.array_cache else None,
            architecture=self.architecture,
            parameters=sum(p.numel() for p in self.model.parameters()),
            world_size=self.world,
            batch_size_per_process=a.batch_size,
            windows_per_update=self.world * a.batch_size,
            updates_per_epoch=per_epoch,
            subgraph_core_nodes=a.subgraph_nodes,
            subgraph_halo_hops=None if self.river is not None else self.halo,
            subgraph="whole upstream basins" if self.river is not None else "patch plus halo",
            river_routing=str(self.river.path) if self.river is not None else None,
            sampling="one random issue per day; " + a.shuffle + " shuffle",
            updates=a.updates,
            lr=a.lr,
            weight_decay=a.weight_decay,
            warmup_updates=min(a.warmup_updates, a.updates // 10),
            min_lr_factor=a.min_lr_factor,
            clip=a.clip,
            precision=a.precision,
            eval_every=a.eval_every,
            validation_windows=len(self.items),
            validation_window_indices=self.items.tolist(),
            early_stop_patience=a.early_stop_patience,
            seed=a.seed,
            **self.task.config(a),
        )

    def schedule(self, warmup):
        a = self.a

        def factor(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            fraction = min(1.0, max(0.0, (step - warmup) / max(a.updates - warmup, 1)))
            return a.min_lr_factor + (1 - a.min_lr_factor) * (1 + math.cos(math.pi * fraction)) / 2

        return factor

    def snapshot(self, config):
        current = sources()
        if self.rank == 0:
            atomic_json(self.out / "run_config.json", config)
            for name in SOURCE_FILES:
                target = self.out / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(PACKAGE / name, target)
            atomic_json(self.out / "source_hashes.json", current)

    def record_segment(self):
        if self.rank != 0:
            return
        current = sources()
        original = json.loads((self.out / "source_hashes.json").read_text())
        changed = sorted(k for k in current if original.get(k) != current[k])
        if changed:
            print(
                f"WARNING: source files changed since this run started: {changed}. "
                f"The original copies are in {self.out / 'source'}.",
                flush=True,
            )
        with open(self.out / "segments.jsonl", "a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        update=self.state["update"],
                        changed_sources=changed,
                        source_hashes=current,
                    )
                )
                + "\n"
            )

    def save(self, path):
        rng = capture_rng(self.device)
        if self.world > 1:
            ranks = [None] * self.world
            dist.all_gather_object(ranks, rng)
        else:
            ranks = [rng]
        if self.rank == 0:
            save_atomic(
                dict(
                    format=self.task.format,
                    config=self.config,
                    progress=dict(self.state),
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict(),
                    rank_rng=ranks,
                ),
                path,
            )
        self.barrier()

    def evaluate(self):
        update = self.state["update"]
        folder = self.out / "evaluations" / f"update_{update:07d}"
        self.model.eval()
        with torch.inference_mode(), self.autocast():
            result = self.task.validate(self)
        self.model.train()
        improved = self.state["best"] is None or result["selection"] < self.state["best"]
        if improved:
            self.state.update(best=result["selection"], best_update=update, bad_evaluations=0)
        else:
            self.state["bad_evaluations"] += 1
        if self.rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            for name, frame in result.pop("tables", {}).items():
                frame.to_csv(folder / f"{name}.csv", index=False)
            atomic_json(folder / "metrics.json", dict(update=update, best=improved, **result))
            row = dict(
                update=update,
                epoch=self.state["epoch"] + 1,
                best=improved,
                **{k: v for k, v in result.items() if not isinstance(v, (dict, list))},
            )
            self.history = [r for r in self.history if r["update"] != update] + [row]
            pd.DataFrame(self.history).to_csv(self.out / "validation_progress.csv", index=False)
            print(json.dumps(row), flush=True)
            (self.out / "checkpoints").mkdir(exist_ok=True)
            target = self.out / "checkpoints" / f"update_{update:07d}.pt"
            save_atomic(
                dict(
                    format=self.task.format,
                    config=self.config,
                    update=update,
                    model=self.model.state_dict(),
                    metrics=result,
                ),
                target,
            )
            if improved:
                shutil.copyfile(target, self.out / "best.pt.part")
                os.replace(self.out / "best.pt.part", self.out / "best.pt")
        self.state["last_evaluated"] = update
        self.barrier()

    def batch(self, epoch, step, order):
        """Runs in the loader thread: patch, windows and task-specific masking on CPU."""
        a = self.a
        first = (step * self.world + self.rank) * a.batch_size
        indices = order[first : first + a.batch_size]
        seed = derived_seed(a.seed, epoch, step, self.rank)
        rng = np.random.default_rng(seed)
        graph = self.sampler.sample(rng, a.subgraph_nodes) if a.subgraph_nodes else None
        nodes = None if graph is None else graph["nodes"]
        started = time.perf_counter()
        samples = [self.train.load(int(i), nodes) for i in indices]
        load_seconds = time.perf_counter() - started
        batch = {k: torch.stack([s[k] for s in samples]) for k in samples[0] if k != "issue"}
        if graph is None:
            graph = dict(
                nodes=None,
                core=np.ones(len(self.ids), dtype=bool),
                adjacency=self.full_graph["adjacency"],
                supports=None,
            )
        batch["static"] = torch.as_tensor(
            self.train.static if nodes is None else self.train.static[nodes], dtype=torch.float32
        )
        batch["core"] = torch.as_tensor(graph["core"])
        self.task.prepare(batch, graph, seed, a)
        if self.device.type == "cuda":
            batch = {k: v.pin_memory() for k, v in batch.items()}
        return graph, batch, load_seconds

    def evaluate_only(self):
        """Score one saved model on the full graph; the optimizer is never created."""
        a = self.a
        if self.evaluated["config"]["dataset_identity"] != self.config["dataset_identity"]:
            raise ValueError("Checkpoint was trained on different prepared data")
        if (self.out / "experiment_manifest.json").exists() or (self.out / "last.pt").exists():
            raise ValueError("--out already holds a run; choose a new directory")
        self.model.load_state_dict(self.evaluated["model"], strict=True)
        self.config.update(
            evaluated_checkpoint=str(a.evaluate_checkpoint.resolve()),
            evaluated_sha256=sha(a.evaluate_checkpoint),
        )
        self.snapshot(self.config)
        self.state = dict(update=0, epoch=0, best=None, best_update=None, bad_evaluations=0)
        self.history = []
        self.evaluate()
        if self.rank == 0:
            atomic_json(
                self.out / "experiment_manifest.json",
                dict(
                    completed=True,
                    evaluation_only=True,
                    optimizer_steps=0,
                    evaluated_checkpoint=self.config["evaluated_checkpoint"],
                    selection=self.state["best"],
                ),
            )
            print(f"EVALUATION COMPLETE: {self.out / 'evaluations/update_0000000'}", flush=True)

    def run(self):
        a, task = self.a, self.task
        self.prepare()
        self.config = self.configuration()
        if a.evaluate_checkpoint:
            return self.evaluate_only()
        self.optimizer = torch.optim.AdamW(
            task.parameter_groups(self.model, a), weight_decay=a.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.schedule(self.config["warmup_updates"])
        )
        self.state = dict(
            update=0,
            epoch=0,
            cursor=0,
            best=None,
            best_update=None,
            bad_evaluations=0,
            last_evaluated=None,
            completed=False,
        )
        last = self.out / "last.pt"
        saved = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
        if saved:
            stored = saved.get("config", {})
            # The array cache holds values identical to NetCDF decoding, so it may be switched.
            keys = sorted(
                k
                for k in set(stored) | set(self.config)
                if k != "array_cache" and stored.get(k) != self.config.get(k)
            )
            if saved.get("format") != task.format or keys:
                raise ValueError(
                    f"Existing run in --out has a different configuration ({keys}); use a new --out"
                )
            self.model.load_state_dict(saved["model"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])
            self.state = saved["progress"]
        else:
            self.snapshot(self.config)
        self.record_segment()
        distributed = (
            DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            if self.world > 1
            else self.model
        )
        random.seed(a.seed + self.rank)
        np.random.seed(a.seed + self.rank)
        torch.manual_seed(a.seed + self.rank)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(a.seed + self.rank)
        if saved:
            restore_rng(saved["rank_rng"][self.rank], self.device)
        del saved
        history = self.out / "validation_progress.csv"
        self.history = pd.read_csv(history).to_dict("records") if history.exists() else []
        if self.state["completed"]:
            if self.rank == 0:
                print(f"Run already complete: {self.out}", flush=True)
            return
        stop_requested = [False]

        def stop(signum, frame):
            stop_requested[0] = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if self.rank == 0:
            c = self.config
            print(
                f"{task.format}: processes={self.world}; windows/update={c['windows_per_update']}"
                f"; core reaches/patch={a.subgraph_nodes or 'full graph'}; parameters="
                f"{c['parameters']:,}; updates/epoch={c['updates_per_epoch']}; resume update="
                f"{self.state['update']}/{a.updates}",
                flush=True,
            )
        if self.state["last_evaluated"] is None:
            self.evaluate()
            self.save(last)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        invocation, began = 0, time.perf_counter()
        self.model.train()
        while self.state["update"] < a.updates:
            epoch = self.state["epoch"]
            order = daily_order(self.train, a.seed, epoch, a.shuffle)
            steps = self.config["updates_per_epoch"]
            if self.state["cursor"] >= steps:
                self.state.update(epoch=epoch + 1, cursor=0)
                continue
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.batch, epoch, self.state["cursor"], order)
                for step in range(self.state["cursor"], steps):
                    waited = time.perf_counter()
                    graph, batch, load_seconds = future.result()
                    waited = time.perf_counter() - waited
                    if step + 1 < steps and self.state["update"] + 1 < a.updates:
                        future = pool.submit(self.batch, epoch, step + 1, order)
                    started = time.perf_counter()
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    supports = (
                        None
                        if graph["supports"] is None
                        else [s.to(self.device) for s in graph["supports"]]
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    with self.autocast():
                        sse, count, extra = task.loss(distributed, batch, supports, self)
                    statistics = reduced(
                        torch.stack(
                            [
                                sse.detach().double(),
                                count.double(),
                                *[e.double() for e in extra],
                                torch.tensor(
                                    float(stop_requested[0]),
                                    device=self.device,
                                    dtype=torch.float64,
                                ),
                            ]
                        ),
                        self.world,
                    )
                    total = statistics[1].item()
                    (sse * (self.world / max(total, 1.0))).backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        trainable, a.clip, error_if_nonfinite=True
                    )
                    if total:
                        self.optimizer.step()
                    self.scheduler.step()
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    seconds = time.perf_counter() - started
                    self.state.update(cursor=step + 1, update=self.state["update"] + 1)
                    invocation += 1
                    update = self.state["update"]
                    halt = statistics[-1].item() > 0 or bool(
                        a.max_steps and invocation >= a.max_steps
                    )
                    if update % a.log_every == 0 or halt or update == 1:
                        row = dict(
                            update=update,
                            epoch=epoch + 1,
                            targets=int(total),
                            loss=statistics[0].item() / max(total, 1),
                            **task.log_values(statistics, total),
                            gradient_norm=float(norm),
                            lr=self.optimizer.param_groups[0]["lr"],
                            data_wait_seconds=waited,
                            load_seconds=load_seconds,
                            step_seconds=seconds,
                            nodes_per_patch=len(graph["core"]),
                            elapsed_hours=(time.perf_counter() - began) / 3600,
                        )
                        if self.device.type == "cuda":
                            row["peak_gib"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                        self.log(row)
                    del batch, sse, count, extra, statistics
                    finished = update >= a.updates
                    evaluated = False
                    if update % a.eval_every == 0 or finished:
                        self.evaluate()
                        evaluated = True
                    patience = a.early_stop_patience
                    early = bool(
                        evaluated and patience and self.state["bad_evaluations"] >= patience
                    )
                    if early or finished:
                        self.state["completed"] = True
                    if evaluated or update % a.checkpoint_every == 0 or halt or finished:
                        self.save(last)
                    if self.state["completed"]:
                        break
                    if halt:
                        if self.rank == 0:
                            print("Stopped safely; rerun the same command to resume.", flush=True)
                        return
            if self.state["completed"]:
                break
        if not self.state["completed"]:
            self.state["completed"] = True
            self.save(last)
        if self.rank == 0:
            atomic_json(
                self.out / "experiment_manifest.json",
                dict(
                    completed=True,
                    updates=self.state["update"],
                    best_update=self.state["best_update"],
                    best_selection=self.state["best"],
                    early_stopped=self.state["update"] < a.updates,
                    elapsed_seconds_last_segment=time.perf_counter() - began,
                ),
            )
            print(
                f"COMPLETE: {self.out}; best update={self.state['best_update']}; "
                f"selection={self.state['best']}",
                flush=True,
            )


def main(task, description):
    p = argparse.ArgumentParser(description=description)
    task.arguments(p)
    a = p.parse_args()
    check_arguments(p, a)
    task.check(p, a)
    run = Run(task, a, p)
    try:
        run.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
