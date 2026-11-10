#!/usr/bin/env python3
"""Measure full-network GWNet training memory on one GPU; no checkpoints saved.

Synthetic mode works while normalization is running. --real-data uses script
15's ready dataset. Each measured step includes forward, masked loss, backward,
gradient clipping and AdamW. Sparse adjacency alone is not a memory guarantee.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def preparation_module():
    path = Path(__file__).with_name("15_prepare_full_pretraining.py")
    spec = importlib.util.spec_from_file_location("full_pretraining", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forecast(model, x):
    raw = model(x)
    if raw.shape != (x.shape[0], 1, x.shape[2], 40):
        raise ValueError(f"Unexpected model forecast shape: {tuple(raw.shape)}")
    return raw[:, 0].permute(0, 2, 1).unsqueeze(-1)


def synthetic_inputs(batch_size, nodes, device):
    # Statics are constant in time, as in the real loader. Values are already
    # standardized; masks are ones. Valid target shape is B x 40 x N x 1.
    x = torch.empty(batch_size, 120, nodes, 64, device=device)
    x[..., :8].normal_()
    x[..., 8:16] = 1
    static = torch.randn(batch_size, 1, nodes, 24, device=device)
    x[..., 16:40] = static
    x[..., 40:64] = 1
    y = torch.randn(batch_size, 40, nodes, 1, device=device)
    mask = torch.ones_like(y, dtype=torch.bool)
    return x, y, mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    p.add_argument("--repo", type=Path, required=True, help="Existing GraphRivers directory")
    p.add_argument("--preparation", type=Path)
    p.add_argument("--real-data", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if min(args.batch_size, args.steps) < 1 or args.warmup < 0:
        p.error("batch-size and steps must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        p.error("CUDA is required for this memory benchmark")
    root = args.root.resolve()
    prep = args.preparation or root / "processed/pretraining_full"
    mode = "real" if args.real_data else "synthetic"
    report_path = args.report or root / "logs" / f"16_full_graph_{mode}_benchmark.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    module = preparation_module()
    if args.real_data:
        dataset = module.FullNetworkWindows(root, prep, split="train")
        with np.load(prep / "graph.npz") as graph:
            ids = graph["feature_id"].copy()
            edges = np.column_stack([graph["source"], graph["target"]])
        if not np.array_equal(ids, dataset.ids):
            raise ValueError("Loader node ordering differs from graph")
        if len(dataset) < args.batch_size * (args.warmup + args.steps):
            raise ValueError("Not enough training windows for requested benchmark")
    else:
        ids, edges = module.read_graph(root)
        dataset = None
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    report = dict(status="running", mode=mode, precision="float32", nodes=len(ids),
                  edges=len(edges), batch_size=args.batch_size, warmup=args.warmup,
                  gpu=properties.name, gpu_total_gib=total/1024**3,
                  gpu_free_at_start_gib=free/1024**3, torch_version=torch.__version__,
                  training_steps=[], checkpoints_saved=False)
    stage, step = "model_construction", -1
    try:
        model = module.make_sparse_gwnet(ids, edges, args.repo, device=device, in_dim=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        model.train()
        print(f"FULL GRAPH: {len(ids):,} nodes; batch={args.batch_size}; {mode} inputs; float32", flush=True)
        print(f"GPU: {properties.name}; free={free/1024**3:.2f} GiB", flush=True)
        cursor = 0
        for step in range(args.warmup + args.steps):
            stage = "input_loading"
            load_start = time.perf_counter()
            if dataset is None:
                x, y, mask = synthetic_inputs(args.batch_size, len(ids), device)
            else:
                samples = [dataset[cursor+i] for i in range(args.batch_size)]
                cursor += args.batch_size
                x, y, mask = [torch.stack([s[key] for s in samples]).to(device) for key in ("x", "y", "mask")]
                del samples
            torch.cuda.synchronize(device)
            load_seconds = time.perf_counter() - load_start
            valid_count = int(mask.sum().item())
            if not valid_count:
                raise ValueError("Benchmark window has no valid targets; choose another window")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            stage = "forward"
            prediction = forecast(model, x)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite model predictions")
            loss = (prediction[mask] - y[mask]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite masked loss")
            stage = "backward"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            stage = "optimizer"
            optimizer.step()
            torch.cuda.synchronize(device)
            record = dict(step=step, warmup=step<args.warmup, loss=float(loss.detach()),
                          valid_targets=valid_count, input_shape=list(x.shape),
                          forecast_shape=list(prediction.shape), load_seconds=load_seconds,
                          training_seconds=time.perf_counter()-started,
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
            report["training_steps"].append(record)
            print(json.dumps(record), flush=True)
            # Release activations and GPU input before the next load.
            del prediction, loss, x, y, mask
        measured = [r for r in report["training_steps"] if not r["warmup"]]
        report.update(status="passed", peak_allocated_gib=max(r["peak_allocated_gib"] for r in report["training_steps"]),
                      peak_reserved_gib=max(r["peak_reserved_gib"] for r in report["training_steps"]),
                      mean_training_seconds=float(np.mean([r["training_seconds"] for r in measured])),
                      mean_load_seconds=float(np.mean([r["load_seconds"] for r in measured])))
        print("FULL-GRAPH TRAINING-STEP CHECK PASSED. Check real data before starting pretraining.", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="out_of_memory", failure_stage=stage, failure_step=step,
                      error=str(exc), peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
        print(f"OUT OF MEMORY during {stage}. Report saved; model weights are disposable.", flush=True)
    except Exception as exc:
        report.update(status="error", failure_stage=stage, failure_step=step,
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        module.atomic_json(report_path, report)
        print(f"Report: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
