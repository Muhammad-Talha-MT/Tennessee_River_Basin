#!/usr/bin/env python3
"""Full-network GWNet pretraining; one complete river graph per GPU/window.

Run with torchrun --standalone --nproc_per_node=4. Scripts 14 and 15 must be
alongside this file. Monthly normalized arrays are cached in bounded host RAM;
one CPU thread prepares the next window during GPU computation. All training
windows are used once per epoch; months and windows within months are shuffled.
Validation uses every 2019 window exactly once. No later test data is opened.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import xarray as xr


def preparation_module():
    spec = importlib.util.spec_from_file_location("full_prep", Path(__file__).with_name("15_prepare_full_pretraining.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


PREP = preparation_module()


class CachedWindows(PREP.FullNetworkWindows):
    def __init__(self, root, out, split, cache_months=6):
        super().__init__(root, out, split=split)
        self.cache, self.cache_months = OrderedDict(), cache_months
        self.mean = np.asarray(self.normalization["mean"])
        self.std = np.asarray(self.normalization["std"])

    def month(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        r = self.records[number]; path = self.root / r["path"]
        if (path.stat().st_size, path.stat().st_mtime_ns) != (r["bytes"], r["mtime_ns"]):
            raise ValueError(f"Monthly source changed: {path}")
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            index = pd.Index(ds.feature_id.values)
            if not index.is_unique or PREP.hash_array(index.to_numpy(dtype="<i8")) != r["ordered_ids_sha256"]:
                raise ValueError(f"Monthly COMID order changed: {path}")
            positions = index.get_indexer(self.ids)
            if (positions < 0).any():
                raise ValueError("Graph reach missing in monthly source")
            selected = ds.isel(feature_id=positions)
            shape = (r["steps"], len(self.ids))
            x, xm = np.empty(shape+(8,), dtype=np.float32), np.empty(shape+(8,), dtype=bool)
            for i, variable in enumerate(PREP.WEATHER):
                a = selected[variable].values.astype(np.float64)
                valid = np.isfinite(a) & (selected[variable+"_valid_hours"].values == 6)
                if i == 0:
                    a = PREP.transform_precipitation(a, valid, self.precip_transform)
                x[...,i] = np.where(valid, (a-self.mean[i])/self.std[i], 0).astype(np.float32)
                xm[...,i] = valid
            q = selected.streamflow.values.astype(np.float64)
            qm = np.isfinite(q) & (selected.streamflow_valid_hours.values == 6)
            if (q[qm] < 0).any():
                raise ValueError("Negative valid streamflow")
            q = np.where(qm, (np.log1p(np.where(qm,q,0))-self.mean[8])/self.std[8], 0).astype(np.float32)
        result = x, xm, q, qm
        self.cache[number] = result
        while len(self.cache) > self.cache_months:
            self.cache.popitem(last=False)
        return result

    def window(self, start, stop):
        parts = [[],[],[],[]]; cursor = start
        while cursor < stop:
            number = int(np.searchsorted(self.offsets,cursor,side="right")-1)
            end = min(stop,int(self.offsets[number+1]))
            left, right = cursor-self.offsets[number], end-self.offsets[number]
            for p,a in zip(parts,self.month(number)):
                p.append(a[left:right])
            cursor = end
        return tuple(np.concatenate(p,axis=0) for p in parts)

    def __getitem__(self, item):
        issue = int(self.issues[item])
        # Read one contiguous range so context and targets share cached months.
        x,xm,q,qm = self.window(issue-PREP.HISTORY,issue+PREP.HORIZON)
        inputs = np.concatenate([x[:PREP.HISTORY],xm[:PREP.HISTORY].astype(np.float32),
            np.broadcast_to(self.static,(PREP.HISTORY,)+self.static.shape)],axis=-1)
        return dict(x=torch.from_numpy(inputs), y=torch.from_numpy(q[PREP.HISTORY:,...,None]),
                    mask=torch.from_numpy(qm[PREP.HISTORY:,...,None]),
                    last_q=torch.from_numpy(q[PREP.HISTORY-1,:,None].copy()),
                    last_mask=torch.from_numpy(qm[PREP.HISTORY-1,:,None].copy()))


def epoch_order(dataset, seed, epoch, world):
    months = dataset.times[dataset.issues].to_period("M").astype(str).to_numpy()
    rng = np.random.default_rng(seed+epoch)
    groups = [np.flatnonzero(months==month) for month in rng.permutation(np.unique(months))]
    order = np.concatenate([rng.permutation(g) for g in groups])
    active = np.ones(len(order),dtype=bool)
    padding = (-len(order)) % world
    if padding:
        order = np.r_[order,np.repeat(order[-1],padding)]
        active = np.r_[active,np.zeros(padding,dtype=bool)]
    return order.reshape(-1,world), active.reshape(-1,world)


def prediction(model, x):
    raw = model(x)
    if raw.shape != (x.shape[0],1,x.shape[2],PREP.HORIZON):
        raise ValueError(f"Unexpected forecast shape {tuple(raw.shape)}")
    return raw[:,0].permute(0,2,1).unsqueeze(-1)


def save_atomic(obj, path):
    temp = path.with_name(path.name+".part")
    with open(temp,"wb") as f:
        torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def capture_rng(device):
    result = dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if device.type == "cuda": result["cuda"] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(state,device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if device.type == "cuda": torch.cuda.set_rng_state(state["cuda"],device)


def reduced(tensor, world):
    if world > 1: dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
    return tensor


def validate(model, dataset, rank, world, device):
    # Validation calls the underlying module: uneven rank lengths must not invoke
    # DDP forward collectives. Synchronize rank-zero BN buffers once beforehand.
    if world > 1:
        for buffer in model.buffers(): dist.broadcast(buffer,src=0)
    model.eval(); totals = torch.zeros(6,dtype=torch.float64,device=device)
    with torch.no_grad():
        for item in range(rank,len(dataset),world):
            b = dataset[item]
            x,y,mask = [b[k].unsqueeze(0).to(device) for k in ("x","y","mask")]
            pred = prediction(model,x)
            if not torch.isfinite(pred).all(): raise FloatingPointError("Nonfinite validation predictions")
            error = pred[mask]-y[mask]
            totals[0] += error.square().sum(dtype=torch.float64); totals[1] += mask.sum()
            last_q = b["last_q"].to(device)[None,None]
            last_mask = b["last_mask"].to(device)[None,None]
            matched = mask & last_mask
            baseline = last_q.expand_as(y)
            totals[2] += (pred[matched]-y[matched]).square().sum(dtype=torch.float64)
            totals[3] += (baseline[matched]-y[matched]).square().sum(dtype=torch.float64)
            totals[4] += matched.sum(); totals[5] += 1
    values = reduced(totals,world).cpu().tolist()
    if values[1] == 0: raise ValueError("Validation has no valid targets")
    return dict(val_mse=values[0]/values[1], val_valid_targets=int(values[1]),
                matched_gnn_mse=values[2]/values[4] if values[4] else None,
                matched_persistence_mse=values[3]/values[4] if values[4] else None,
                matched_targets=int(values[4]), validation_windows=int(values[5]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/data/tva_data"))
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--preparation",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--cache-months",type=int,default=6)
    p.add_argument("--checkpoint-every",type=int,default=100)
    p.add_argument("--log-every",type=int,default=25)
    p.add_argument("--early-stop-patience",type=int,default=5)
    p.add_argument("--max-steps",type=int,default=0,help="Stop and checkpoint after this many steps in this invocation; 0 means unlimited")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    args = p.parse_args()
    if min(args.epochs,args.cache_months,args.checkpoint_every,args.log_every)<1 or args.lr<=0 or min(args.max_steps,args.early_stop_patience)<0:
        p.error("Invalid epoch, cache, learning-rate or interval settings")
    rank,world,local = [int(os.environ.get(k,d)) for k,d in (("RANK",0),("WORLD_SIZE",1),("LOCAL_RANK",0))]
    device = torch.device(f"cuda:{local}" if args.device=="cuda" else "cpu")
    if args.device=="cuda":
        if not torch.cuda.is_available(): p.error("CUDA unavailable")
        torch.cuda.set_device(device)
    if world>1: dist.init_process_group("nccl" if device.type=="cuda" else "gloo",timeout=timedelta(minutes=30))
    root = args.root.resolve(); prep = (args.preparation or root/"processed/pretraining_full").resolve()
    out = (args.out or root/"models/full_network_pretraining").resolve(); out.mkdir(parents=True,exist_ok=True)
    lock = None
    if rank==0:
        lock = open(out/".run.lock","w"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if world>1: dist.barrier()
    train = CachedWindows(root,prep,"train",args.cache_months)
    if train.precip_transform!="log1p": raise ValueError("Expected the log1p precipitation preparation; rerun script 15")
    val = CachedWindows(root,prep,"validation",args.cache_months)
    # One shared cache per process; training prefetch ends before validation.
    val.cache = train.cache
    with np.load(prep/"graph.npz") as graph:
        ids = graph["feature_id"].copy(); edges = np.column_stack([graph["source"],graph["target"]])
    config = dict(dataset_identity=train.manifest["identity"], world_size=world, lr=args.lr, seed=args.seed,
                  model="GWNet64/40,4blocks,5layers,8/8/32/64,sparse,no_adaptive",precision="float32",
                  window_order="shuffled_months_then_shuffled_windows",per_gpu_batch_size=1,
                  normalization_hashes={name:hashlib.sha256((prep/name).read_bytes()).hexdigest() for name in
                    ("dynamic_normalization.json","static_normalization.json")},
                  source_hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                    [("gwnetmodel",args.repo/"gwnetmodel.py"),("statics",Path(__file__).with_name("14_prepare_static_attributes.py")),("preparation",Path(__file__).with_name("15_prepare_full_pretraining.py")),("trainer",Path(__file__))]})
    random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed)
    if device.type=="cuda": torch.cuda.manual_seed_all(args.seed)
    model = PREP.make_sparse_gwnet(ids,edges,args.repo,device=device,in_dim=64)
    # The legacy model builds residual convolutions that its graph-convolution
    # branch never calls. Exclude these parameters so DDP reduces only used layers.
    model.residual_convs.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-4)
    state = dict(epoch=0,next_step=0,global_step=0,epoch_sse=0.,epoch_count=0,best_val=None,bad_epochs=0)
    last = out/"last.pt"
    saved = torch.load(last,map_location="cpu",weights_only=False) if last.exists() else None
    if saved:
        if saved["config"]!=config: raise ValueError("Checkpoint configuration/source differs; use a different --out")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value): values[key]=value.to(device)
        state = saved["progress"]
    distributed = DDP(model,device_ids=[local] if device.type=="cuda" else None,
                      broadcast_buffers=True,find_unused_parameters=True) if world>1 else model
    # Different dropout streams after identical model initialization/DDP broadcast.
    torch.manual_seed(args.seed+rank)
    if device.type=="cuda": torch.cuda.manual_seed(args.seed+rank)
    if saved: restore_rng(saved["rank_rng"][rank],device)
    del saved
    requested_stop = [False]
    def stop_handler(signum,frame): requested_stop[0]=True
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGINT,stop_handler)

    def checkpoint():
        rng = capture_rng(device)
        if world>1:
            all_rng=[None]*world; dist.all_gather_object(all_rng,rng)
        else: all_rng=[rng]
        if rank==0:
            save_atomic(dict(config=config,progress=dict(state),model=model.state_dict(),
                             optimizer=optimizer.state_dict(),rank_rng=all_rng,
                             dynamic_normalization=train.normalization,
                             static_normalization=json.loads((prep/"static_normalization.json").read_text())),last)
        if world>1: dist.barrier()

    if rank==0:
        PREP.atomic_json(out/"run_config.json",config)
        print(f"FULL PRETRAINING: nodes={len(ids):,}; GPUs/processes={world}; effective batch={world}; train windows={len(train):,}; validation={len(val):,}",flush=True)
        print(f"Resume: epoch={state['epoch']+1}, next global batch={state['next_step']}; checkpoint={last}",flush=True)
    checkpoint(); invocation_steps=0; began=time.perf_counter()
    for epoch in range(state["epoch"],args.epochs):
        order,active=epoch_order(train,args.seed,epoch,world)
        first=state["next_step"]; distributed.train()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(train.__getitem__,int(order[first,rank])) if first<len(order) else None
            for step in range(first,len(order)):
                load_start=time.perf_counter(); batch=future.result()
                load_wait=time.perf_counter()-load_start
                if step+1<len(order): future=pool.submit(train.__getitem__,int(order[step+1,rank]))
                transfer_start=time.perf_counter()
                x,y,mask=[batch[k].unsqueeze(0).to(device) for k in ("x","y","mask")]; del batch
                if not active[step,rank]: mask=mask & False
                if device.type=="cuda": torch.cuda.synchronize(device)
                transfer_seconds=time.perf_counter()-transfer_start
                optimizer.zero_grad(set_to_none=True); compute_start=time.perf_counter()
                pred=prediction(distributed,x)
                if not torch.isfinite(pred).all(): raise FloatingPointError("Nonfinite training predictions")
                error=pred[mask]-y[mask]
                sse=error.square().sum(dtype=torch.float64)
                stats=torch.stack([sse.detach(),mask.sum().to(torch.float64),
                                   torch.tensor(float(requested_stop[0]),device=device,dtype=torch.float64)])
                reduced(stats,world); global_count=int(stats[1].item())
                loss=sse*(world/max(global_count,1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if global_count: optimizer.step()
                if device.type=="cuda": torch.cuda.synchronize(device)
                compute_seconds=time.perf_counter()-compute_start
                state["epoch_sse"]+=float(stats[0].item()); state["epoch_count"]+=global_count
                state["next_step"]=step+1; state["global_step"]+=1; invocation_steps+=1
                stop=stats[2].item()>0 or (args.max_steps and invocation_steps>=args.max_steps)
                if rank==0 and (step%args.log_every==0 or stop):
                    print(json.dumps(dict(epoch=epoch+1,batch=step+1,batches=len(order),global_step=state["global_step"],
                        train_mse=state["epoch_sse"]/max(state["epoch_count"],1),load_wait_seconds=load_wait,
                        transfer_seconds=transfer_seconds,compute_seconds=compute_seconds,
                        elapsed_hours=(time.perf_counter()-began)/3600)),flush=True)
                del x,y,mask,pred,error,sse,loss,stats
                if state["global_step"]%args.checkpoint_every==0 or stop: checkpoint()
                if stop:
                    if rank==0: print("STOPPED SAFELY. Rerun the same command to resume; remove --max-steps for the full run.",flush=True)
                    if world>1: dist.destroy_process_group()
                    return
        # Save finished training before validation, which can be rerun on resume.
        checkpoint(); metrics=validate(model,val,rank,world,device)
        improved=state["best_val"] is None or metrics["val_mse"]<state["best_val"]
        state["bad_epochs"]=0 if improved else state["bad_epochs"]+1
        if improved: state["best_val"]=metrics["val_mse"]
        if rank==0:
            report=dict(epoch=epoch+1,train_mse=state["epoch_sse"]/max(state["epoch_count"],1),best=improved,**metrics)
            PREP.atomic_json(out/f"epoch_{epoch+1:03d}.json",report); print(json.dumps(report),flush=True)
            if improved: save_atomic(dict(config=config,epoch=epoch+1,model=model.state_dict(),
                val_mse=metrics["val_mse"],dynamic_normalization=train.normalization,
                static_normalization=json.loads((prep/"static_normalization.json").read_text())),out/"best.pt")
        state.update(epoch=epoch+1,next_step=0,epoch_sse=0.,epoch_count=0)
        checkpoint()
        if args.early_stop_patience and state["bad_epochs"]>=args.early_stop_patience: break
    if rank==0: print(f"PRETRAINING COMPLETE. Best validation MSE={state['best_val']}; outputs={out}",flush=True)
    if world>1: dist.destroy_process_group()


if __name__=="__main__": main()
