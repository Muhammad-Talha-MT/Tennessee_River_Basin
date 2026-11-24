"""Phase 1.2: what M0, M1, M2 and the LSTM receive at issue time and during the forecast window, read from the code constants and checkpoints.

Writes reports/v3_tables/inputs_audit.csv (input | M0, M1, M2 | LSTM | matched?). Numbers come from `tva_flood.hydro.model` / `paths` constants,
the checkpoint configs and arguments, and `Network`; text cites the source files.  Run: python scripts/v3_inputs_audit.py
"""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import pandas as pd
import torch
from v3_common import A, OUT
from tva_flood.hydro import paths as P
from tva_flood.hydro.channel import CHANNEL_INPUTS
from tva_flood.hydro.data import Network
from tva_flood.hydro.model import REACH_INPUTS, STEP_INPUTS

OUT.mkdir(parents=True, exist_ok=True)
net = Network()
l4 = torch.load(A / "ladder_L4_s1" / "best.pt", map_location="cpu", weights_only=False)
lc, la = l4["config"], l4["config"]["arguments"]
m0 = json.loads((A / "M0_s1" / "arguments.json").read_text())
m0cfg = torch.load(A / "M0_s1" / "best.pt", map_location="cpu", weights_only=False)["config"]
n_params = {m: sum(v.numel() for v in torch.load(A / d / "best.pt", map_location="cpu", weights_only=False)["model"].values())
            for m, d in (("M0", "M0_s1"), ("M1", "M1_s1"), ("M2", "M2_s1"), ("LSTM", "ladder_L4_s1"))}
stage_a = json.loads((A / "manning_A" / "arguments.json").read_text())
CH = dict(weather=8, validity=8, future=1, season=2, own=2, owner=2)  # the decomposition in model.py (STEP_INPUTS)
assert sum(CH.values()) == STEP_INPUTS and CH["weather"] == CH["validity"]
NW = CH["weather"]
days = lambda steps: f"{steps * 6 / 24:g} days"  # noqa: E731
GNN = "M0, M1, M2"
rows = [
    ("Weather before the issue", f"{P.HISTORY} steps ({days(P.HISTORY)}) of {NW} AORC variables + {NW} validity flags per reach (reach scale)",
     f"{lc['history']} steps ({days(lc['history'])}) of the same {NW} variables + {NW} validity flags, basin-mean over the gauge's upstream basin",
     "same variables and source; different window length and spatial aggregation"),
    ("Rain after the issue", f"GEFSv12 control precipitation to {P.HORIZON * 6} h ({P.HORIZON} steps) per reach; the other {NW - 1} weather variables are unknown and flagged invalid; known-rain flag per step",
     f"GEFSv12 control rain + known-share, basin-mean, {lc['horizon']} leads (2 channels); reforecast in training, operational at test",
     "same forecast source and horizon (samples.py:build; lstm_baseline.py:batch)"),
    ("Static attributes", "48 per reach (REACH_INPUTS)", f"48 basin-mean (static_dim {lc['static_dim']} = 48 + 7)", "same attribute set, reach vs basin mean"),
    ("Antecedent storage", "7 per reach (REACH_INPUTS)", "7 basin-mean, appended to the attributes (--antecedent)", "same features"),
    ("Season", f"sin and cos of the day of year ({CH['season']} step channels)", "none (no explicit season input)", "differs"),
    ("Own flow of the target gauge when it is withheld", "not available: gauge_view sets it to NaN and visible False, at every time (tests/test_hydro.py)",
     "not available: history zeroed and flag 0 (fold models trained with that gauge removed)", "matched"),
    ("Own flow of a visible gauge", f"log flow + shown flag at the gauge's own reach ({CH['own']} step channels), history only; the gauge head reads the last 120 steps (30 days; HydroModel.forward)",
     f"flow + flag in the {lc['history']}-step history", "matched"),
    ("Flow of other gauges", f"visible gauges only, before the issue: the owner gauge's specific discharge + validity ({CH['owner']} step channels) and the area ratio + flag (2 reach inputs) on every reach inside the owner's basin; "
     "carried by the graph layers (M0, M1: 3; M2: 0) and, in M0 and M2 only, by insertion",
     "none", "DIFFERS: the network models see other gauges, the LSTM does not (by design)"),
    ("Network structure", f"river graph with {len(net.edges):,} links over {net.n:,} reaches; Manning router inputs per reach (CHANNEL_INPUTS = {CHANNEL_INPUTS})",
     "none (lumped basin mean)", "differs by design"),
    ("NWM", f"stage A pretraining on NWM retrospective flows at every reach ({stage_a['updates']} updates), plus a prior term (nwm_weight {m0['nwm_weight']}) in stage B; not an input at test",
     "not used (no NWM pretraining, no NWM input)", "DIFFERS: the LSTM has no NWM pretraining"),
    ("Output", "area-scaled runoff routed down the network; a visible gauge's forecast is the gauge head's; any other reach is routed flow (+ insertion in M0, M2)",
     f"specific discharge x basin area / 21.6 (area_output = {lc['area_output']})", "matched at the gauge (same units: flow in m3/s)"),
    ("Training loss", f"{m0['loss']}: NSE* on gauge and routed forecasts + pinball tail (weight {m0['tail_weight']}) + NWM prior (weight {m0['nwm_weight']})",
     f"{lc['loss']} on gauge flow; high-flow weight {la['high_flow_weight']}, raw weight {la['raw_weight']}", "differs"),
    ("Gauge visibility in training", "15% of samples show no gauge; otherwise each eligible gauge is shown with probability 0.5 (train.py:107-108)",
     f"own history hidden in {la['history_hide']:.1%} of samples", "comparable intent, different mechanism"),
    ("Training volume (stage B)", f"{m0['updates']} updates x 8 whole-network samples, lr {m0['lr']}", f"{la['updates']:,} updates x {la['batch_size']} (issue, gauge) pairs, lr {la['lr']}", "differs"),
    ("Model selection", "2019 validation: mean of gauged and hidden median NSE at training gauges, leads 24-120 h", lc["selection"].split(", forecast rain")[0] + "; GEFS reforecast rain", "similar (validation year 2019, median NSE over gauges)"),
    ("Recurrent size", f"LSTM hidden {m0cfg['hidden']} -> {m0cfg['channels']} channels", f"LSTM hidden {lc['hidden']}", "differs"),
    ("Parameters", f"M0 {n_params['M0']:,}; M1 {n_params['M1']:,}; M2 {n_params['M2']:,}", f"{n_params['LSTM']:,}", "differs"),
    ("Step inputs / reach inputs (network models)", f"{STEP_INPUTS} step channels, {REACH_INPUTS} reach inputs", f"{2 * NW + 2} dynamic channels + {lc['static_dim']} static", "-"),
]
df = pd.DataFrame(rows, columns=["input", GNN, "LSTM", "matched"])
df.to_csv(OUT / "inputs_audit.csv", index=False)
print(df.to_string(index=False, max_colwidth=110))
