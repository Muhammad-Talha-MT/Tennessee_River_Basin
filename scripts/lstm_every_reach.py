"""The tuned LSTM at every reach, as the new model is tested there (torchrun, all GPUs).

The LSTM is lumped: it reads the basin means of weather, static attributes and forecast
rain of one gauge's upstream basin. Any reach has such a basin, and the graph's upstream
area weights (``Network``: a_j f_ij / A_i, the rows of ``data.river``) give the same means
for all 58,834 of them, so one LSTM forecasts every reach with no change to its weights.

As in ``tva_flood.hydro.evaluate`` (section 3): the same issues, the same retrospective
targets, masks and metric, the GEFSv12 operational rain (and, as a diagnostic, realized
rain), and the same model choice: a reach in a gauge's basin takes the LSTM trained with
that gauge's fold withheld, every other reach the all-gauge LSTM. A reach has no flow of its
own, so the flow history is hidden everywhere (the LSTM was trained with 10% hidden), also
at the few gauge reaches of the all-gauge model that the new model sees.

  torchrun --standalone --nproc_per_node 4 scripts/lstm_every_reach.py
Output: models/gefs_lstm_reaches/metrics.npz (nse/gefs, nse/aorc [lead, reach], issues).
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from tva_flood.evaluation.benchmark import load_nwm
from tva_flood.evaluation.hydrology import Accumulator, reach_metrics
from tva_flood.hydro import paths as P
from tva_flood.hydro.evaluate import FOLDS, FUTURE, NWMReaches, reach_folds, test_issues
from tva_flood.hydro.samples import Samples
from tva_flood.models.lstm import ForecastLSTM
from tva_flood.training.forecast import MAX_LOG_FLOW

LSTM_HISTORY = 120  # steps of weather the LSTM reads before the issue
PRECIP = 0
CHUNK = 8192


def basin_mean(network, device):
    """Sparse [reach, reach] area-weighted basin mean over each reach's upstream basin."""
    rows = np.repeat(np.arange(network.n), np.diff(network.indptr))
    values = network.share * network.area[network.members] / network.basin_area[rows]
    return torch.sparse_csr_tensor(
        torch.as_tensor(network.indptr, dtype=torch.int64),
        torch.as_tensor(network.members, dtype=torch.int64),
        torch.as_tensor(values, dtype=torch.float32),
        size=(network.n, network.n),
    ).to(device)


def load_lstm(path, k, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    c = state["config"]
    withheld = (c.get("arguments", {}).get("holdout_fold") or 0)
    if withheld != k:
        raise ValueError(f"{path} was trained with fold {withheld} withheld, not {k}")
    model = ForecastLSTM(hidden=c["hidden"], dropout=c["dropout"], horizon=P.HORIZON)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), c["flow_mean"], c["flow_std"], c["rain_hours"] // 6


def raw(z, mean, std):
    return torch.expm1((z.double() * std + mean).clamp(max=MAX_LOG_FLOW)).clamp_min(0)


class Inputs:
    """Basin-mean LSTM inputs of every reach at one issue, on the device."""

    def __init__(self, s, device):
        self.s, self.device = s, device
        self.mean = basin_mean(s.network, device)
        static = torch.as_tensor(s.network.static, dtype=torch.float32, device=device)
        self.static = self.mean @ static  # [reach, 48]
        self.rain_mean, self.rain_std = float(s.arrays.mean[PRECIP]), float(s.arrays.std[PRECIP])

    def history(self, weather, valid):
        """[reach, 120, 18]: basin-mean weather and validity, flow hidden (zero, flag 0)."""
        w = torch.as_tensor(weather[P.HISTORY - LSTM_HISTORY : P.HISTORY], device=self.device).float()
        v = torch.as_tensor(valid[P.HISTORY - LSTM_HISTORY : P.HISTORY], device=self.device).float()
        x = torch.cat([w, v], dim=-1).permute(1, 0, 2).reshape(self.s.network.n, -1)
        x = (self.mean @ x).view(self.s.network.n, LSTM_HISTORY, 16)
        return torch.cat([x, torch.zeros(*x.shape[:2], 2, device=self.device)], dim=-1)

    def rain(self, weather, valid, forecast=None):
        """[reach, 40, 2] basin-mean future rain and its known share: realized AORC, or the
        forecast ``forecast`` [lead, reach] (mm) where it is finite and AORC covers the reach."""
        n = self.s.network.n
        covered = torch.as_tensor(valid[P.HISTORY : P.WINDOW, :, PRECIP], device=self.device).float() > 0
        if forecast is None:
            z = torch.as_tensor(weather[P.HISTORY : P.WINDOW, :, PRECIP], device=self.device).float()
            known = torch.as_tensor(valid[P.HISTORY : P.WINDOW, :, PRECIP], device=self.device).float()
        else:
            mm = torch.as_tensor(forecast, dtype=torch.float32, device=self.device)
            ok = torch.isfinite(mm) & (mm >= 0) & covered
            z = torch.where(ok, (torch.log1p(torch.nan_to_num(mm).clamp_min(0)) - self.rain_mean) / self.rain_std, 0.0)
            known = ok.float()
        both = torch.stack([z, known], dim=-1).permute(1, 0, 2).reshape(n, -1)
        return (self.mean @ both).view(n, P.HORIZON, 2)


def forecast(model, mean, std, history, static, rain, index):
    """Raw flow [lead, len(index)] of the reaches ``index`` from one LSTM."""
    out = []
    for part in index.split(CHUNK):
        z = model(history[part], static[part], rain[part])
        out.append(raw(z.T, mean, std))
    return torch.cat(out, dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(P.MODELS / "gefs_lstm{fold}/best.pt"))
    p.add_argument("--out", type=Path, default=P.MODELS / "gefs_lstm_reaches")
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="First n issues only (testing)")
    a = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=120), device_id=device)
    try:
        run(a, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(a, rank, world, device):
    s = Samples()
    network, arrays = s.network, s.arrays
    models = {
        k: load_lstm(a.model.format(fold="" if k == 0 else f"_f{k}"), k, device) for k in range(FOLDS + 1)
    }
    reach_fold, _ = reach_folds(s)
    which = np.maximum(reach_fold, 0)
    index = {k: torch.as_tensor(np.flatnonzero(which == k), device=device) for k in models}
    nwm, _ = load_nwm(P.NWM_GAUGES)
    reach_nwm = NWMReaches()
    issues, every_reach = test_issues(s, nwm, reach_nwm.days(), a.limit)
    mine = sorted(every_reach)[rank::world]
    inputs = Inputs(s, device)
    tags = ("gefs", "aorc")
    acc = {t: Accumulator(network.n, device) for t in tags}
    began = time.perf_counter()
    with torch.inference_mode():
        for position, step in enumerate(mine):
            day = arrays.times[step].normalize()
            sample = s.build(step, "aorc")
            retro = torch.as_tensor(arrays.flow_raw(sample["nwm"][FUTURE]), device=device)
            ok = torch.as_tensor(sample["nwm_valid"][FUTURE], device=device).bool()
            latest = torch.as_tensor(sample["nwm_valid"][P.INNOVATION - 1], device=device).bool()
            nwm_reach = torch.as_tensor(reach_nwm.get(day), device=device).double()
            mask = ok & latest[None] & torch.isfinite(nwm_reach)
            history = inputs.history(sample["weather"], sample["valid"])
            rains = {
                "gefs": inputs.rain(sample["weather"], sample["valid"], s.rain.get("gefs", day)),
                "aorc": inputs.rain(sample["weather"], sample["valid"]),
            }
            for tag in tags:
                composite = torch.zeros(P.HORIZON, network.n, dtype=torch.float64, device=device)
                for k, (model, mean, std, steps) in models.items():
                    if steps != P.HORIZON:
                        raise ValueError("The LSTM was trained with rain to another lead")
                    composite[:, index[k]] = forecast(model, mean, std, history, inputs.static, rains[tag], index[k])
                acc[tag].add(composite, torch.nan_to_num(retro).double(), mask)
            if rank == 0 and position % 10 == 0:
                print(f"issue {position + 1}/{len(mine)} per process; {(time.perf_counter() - began) / 60:.1f} min", flush=True)
    for accumulator in acc.values():
        accumulator.reduce(world)
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
        nse = {t: reach_metrics(**acc[t].numpy(), min_count=a.min_count)["nse"] for t in tags}
        if not a.limit:
            np.savez_compressed(a.out / "metrics.npz", comid=network.ids, issues=len(every_reach), **{f"nse/{t}": v for t, v in nse.items()})
        print(f"LSTM every reach COMPLETE: {a.out} ({len(every_reach)} issues)", flush=True)


if __name__ == "__main__":
    main()
