"""Shared loaders for the Section 3 facts sheet and figures (scripts/section3/). Reuses the repository's own loaders:
hydro.data.Network / Gauges / Arrays, hydro.splits (folds, blocked basins), hydro.evaluate.regulation (dam categories) and the
geometry files used by scripts/headwater_map.py. Every input path is listed in INPUTS.
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from tva_flood.hydro import paths as P  # noqa: E402
from tva_flood.hydro import splits  # noqa: E402
from tva_flood.hydro.data import Arrays, Gauges, Network, Rain  # noqa: E402
from tva_flood.hydro.evaluate import reach_folds, regulation  # noqa: E402

FIG = REPO / "figures" / "section3"
FACTS = REPO / "reports" / "section3_facts.md"
EVAL = P.MODELS / "attr"
INPUTS = dict(
    nodes=P.PROCESSED / "graph_candidate/nodes.parquet",
    catchments=P.PROCESSED / "graph_candidate/catchments_clean.parquet",
    flowlines=P.ROOT / "raw/geography/nhdplus_v21/extracted/NHDPlusV21_MS_06_NHDSnapshot_09/NHDPlusMS/NHDPlus06/NHDSnapshot/Hydrography/NHDFlowline.shp",
    huc2=P.ROOT / "raw/geography/tennessee_huc2.geojson",
    states=P.ROOT / "raw/geography/us_states/cb_2022_us_state_5m.shp",
    gauges=P.GAUGES,
    gauges_csv=P.PROCESSED / "gauges_public/gauges.csv",
    gauges_json=P.PROCESSED / "gauges_public/gauges.json",
    catalog_csv=P.PROCESSED / "gauges/gauges.csv",
    catalog_json=P.PROCESSED / "gauges/gauges.json",
    excluded=P.PROCESSED / "gauges/excluded.csv",
    camelsh_meta=P.ROOT / "raw/camelsh/metadata/tennessee_camelsh_gauges.csv",
    dams_npz=P.HYDRO / "dams.npz",
    dams_csv=P.HYDRO / "dams.csv",
    routing=P.HYDRO / "routing.npz",
    static_norm=P.TRAIN_PREP / "static_normalization.json",
    dataset_manifest=P.TRAIN_PREP / "dataset_manifest.json",
    static_manifest=P.PROCESSED / "static/static_manifest.json",
    aorc_mapping=P.PROCESSED / "aorc_mapping/mapping_report.json",
    eval_gauged=EVAL / "eval_M0_s1/forecasts_gauged.npz",
    eval_heldout=EVAL / "eval_M0_s1/forecasts_ungauged.npz",
    eval_blocked=EVAL / "eval_S1_M0_s1/forecasts_ungauged.npz",
    eval_reach=EVAL / "eval_M0_reach/reach_metrics.npz",
    eval_manifest=EVAL / "eval_M0_s1/manifest.json",
)
# one color per category, used in every figure (Okabe-Ito, colorblind-safe)
DAM_COLORS = {"headwater, no dam": "#0072B2", "below a gauge, no dam": "#E69F00", "dam in the basin": "#CC79A7"}
DAM_MARKERS = {"headwater, no dam": "o", "below a gauge, no dam": "s", "dam in the basin": "D"}
FOLD_COLORS = {1: "#332288", 2: "#88CCEE", 3: "#117733", 4: "#DDCC77", 5: "#882255"}  # Paul Tol, distinct from the dam colors
BLOCK_COLORS = {1: "#D55E00", 2: "#009E73", 3: "#0072B2"}
DAM_MARKER_COLOR = "#000000"


def scores_n(npz):
    """Per gauge: number of scorable (finite forecast and truth) pairs at the 24 h lead, and whether the gauge is scored by the evaluation's rule
    (>= 100 pairs, positive mean, std >= 1e-3), from a saved evaluation file (method 0 = GEFS forecast)."""
    d = np.load(npz, allow_pickle=True)
    t, v = d["truth"][:, 3].astype(float), d["values"][:, 0, 3].astype(float)
    ok = np.isfinite(t) & np.isfinite(v)
    n = ok.sum(0)
    o = np.where(ok, t, 0)
    mean = o.sum(0) / np.maximum(n, 1)
    std = np.sqrt(np.maximum((o * o).sum(0) / np.maximum(n, 1) - mean**2, 0))
    return n, (n >= 100) & (mean > 0) & (std >= 1e-3), d["stations"].astype(str)


def load():
    network = Network()
    arrays = Arrays()
    gauges = Gauges(network, arrays.times)
    regulated, groups = regulation(network, gauges)
    nodes = pd.read_parquet(INPUTS["nodes"]).set_index("comid").reindex(network.ids)
    names = np.array(gauges.names).astype(str)
    group = np.full(gauges.count, "", object)
    for g_name, m in groups:
        group[m] = g_name
    fold_random = splits.gauge_folds("random", gauges, network)
    fold_blocked = splits.gauge_folds("blocked", gauges, network)
    blocks = splits.blocks(gauges, network)  # [(outlet gauge, basin reaches)]
    in_block = np.zeros(gauges.count, int)
    for k, (_, basin) in enumerate(blocks, start=1):
        in_block[np.isin(gauges.reach, basin)] = k
    scored = {}
    npairs = {}
    for proto, key in (("gauged", "eval_gauged"), ("heldout", "eval_heldout"), ("blocked", "eval_blocked")):
        n, ok, st = scores_n(INPUTS[key])
        assert (st == names).all()
        scored[proto], npairs[proto] = ok, n
    s = SimpleNamespace(gauges=gauges, network=network)
    reach_fold, _ = reach_folds(s, "random")
    return SimpleNamespace(network=network, arrays=arrays, gauges=gauges, regulated=regulated, groups=groups, group=group, nodes=nodes,
                           names=names, fold_random=fold_random, fold_blocked=fold_blocked, blocks=blocks, in_block=in_block,
                           scored=scored, npairs=npairs, reach_fold=reach_fold, rain=Rain())
