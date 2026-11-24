"""Shared code for the v3 test comparison (models M0, M1, M2, LSTM, NWM; no A0).

Every number in ``reports/test_comparison_v3.md`` comes from scripts that import this module.

* ``load_protocol(protocol)`` -> {model: (truth, values)} [issue, lead, gauge] on the 128 catalog gauges (float32, NaN
  where a gauge is not scored by that model in that protocol). "LSTM" is the area-scaled LSTM (L4).
* ``scores(truth, values)`` -> per-gauge NSE, KGE (Gupta et al. 2009: alpha = std ratio, beta = mean ratio) and its
  components, with the evaluation's rule for a scored gauge (>= 100 values, positive mean, std >= 1e-3).
* ``Context`` -> gauge names, areas, dam groups, fold of every gauge per protocol, and the upstream-gauge structure.
* statistics: gauge-level bootstrap, paired Wilcoxon, paired TOST equivalence, Holm correction.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

A = Path("/data/tva_data/models/attr")
OUT = REPO / "reports" / "v3_tables"
FIG = REPO / "figures" / "paper"
LEADS_H = (6, 24, 72, 120, 240)
DAYS15_H = (24, 48, 72, 96, 120)
DAYS15 = [h // 6 - 1 for h in DAYS15_H]
MODELS = ("M0", "M1", "M2", "M4", "LSTM")  # M4: held-out and blocked only (no all-gauge model); loaded only if its evaluation exists
PROTOCOLS = ("gauged", "heldout", "blocked")
N_GAUGES = {"gauged": 101, "heldout": 98, "blocked": 29}

HYDRO_FILES = {
    "gauged": {"M0": "eval_M0_s1", "M1": "eval_M1_s1", "M2": "eval_M2_s1"},
    "heldout": {"M0": "eval_M0_s1", "M1": "eval_M1_s1", "M2": "eval_M2_s1", "M4": "eval_M4_s1"},
    "blocked": {"M0": "eval_S1_M0_s1", "M1": "eval_S1_M1_s1", "M2": "eval_S1_M2_s1", "M4": "eval_S1_M4_s1"},
}
LSTM_FILES = {
    "gauged": ["ladder_L4_s1_bench"],
    "heldout": [f"ladder_L4_s1_bench_f{k}" for k in range(1, 6)],
    "blocked": [f"ladder_L4_s1_bench_b{k}" for k in range(1, 4)],
}


def _stations():
    return np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["stations"].astype(str)


STATIONS = _stations()


def _hydro(path):
    d = np.load(path, allow_pickle=True)
    assert (d["stations"].astype(str) == STATIONS).all()
    return d["truth"].astype(np.float32), d["values"][:, 0].astype(np.float32)  # method 0 is the GEFS forecast


def _lstm(dirs):
    """LSTM (L4) benchmark forecasts merged by station name: a fold file scores only that fold's gauges."""
    truth = values = None
    for name in dirs:
        d = np.load(A / name / "forecasts.npz", allow_pickle=True)
        names = list(d["methods"].astype(str))
        v = d["values"][:, names.index("lstm+gefs")].astype(np.float32)
        t = d["truth"].astype(np.float32)
        if truth is None:
            truth = np.full((t.shape[0], t.shape[1], len(STATIONS)), np.nan, np.float32)
            values = truth.copy()
        for i, st in enumerate(d["stations"].astype(str)):
            j = np.flatnonzero(STATIONS == st)
            if len(j) == 0:
                continue
            fin = np.isfinite(v[:, :, i])
            if fin.any():
                values[:, :, j[0]] = np.where(fin, v[:, :, i], values[:, :, j[0]])
                truth[:, :, j[0]] = np.where(fin, t[:, :, i], truth[:, :, j[0]])
    return truth, values


def load_protocol(protocol, models=MODELS):
    out = {}
    for m in models:
        if m == "LSTM":
            out[m] = _lstm(LSTM_FILES[protocol])
        elif m in HYDRO_FILES[protocol]:
            f = "forecasts_gauged.npz" if protocol == "gauged" else "forecasts_ungauged.npz"
            path = A / HYDRO_FILES[protocol][m] / f
            if path.exists():  # a model whose evaluation has not been run is left out (reported as MISSING by the callers)
                out[m] = _hydro(path)
    return out


def load_nwm():
    """NWM forecast at the gauges (the same in every protocol: it is not a held-out model)."""
    d = np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)
    names = list(d["methods"].astype(str))
    return d["truth"].astype(np.float32), d["values"][:, names.index("nwm")].astype(np.float32)


def scores(truth, values, min_count=100):
    """NSE, KGE, r, alpha, beta, n: each [lead, gauge]; NaN where the gauge is not scored (as the evaluation script)."""
    ok = np.isfinite(values) & np.isfinite(truth)
    n = ok.sum(0)
    o = np.where(ok, truth, 0.0).astype(np.float64)
    f = np.where(ok, values, 0.0).astype(np.float64)
    m = np.maximum(n, 1)
    mo, mf = o.sum(0) / m, f.sum(0) / m
    var_o = np.maximum((o * o).sum(0) / m - mo**2, 0)
    var_f = np.maximum((f * f).sum(0) / m - mf**2, 0)
    cov = (o * f).sum(0) / m - mo * mf
    mse = ((f - o) ** 2).sum(0) / m
    good = (n >= min_count) & (mo > 0) & (np.sqrt(var_o) >= 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        nse = 1 - mse / var_o
        r = np.where(var_f > 0, cov / np.sqrt(var_f * var_o), np.nan)
        alpha = np.sqrt(var_f / var_o)
        beta = mf / mo
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nan = np.nan
    return {k: np.where(good, v, nan) for k, v in dict(nse=nse, kge=kge, r=r, alpha=alpha, beta=beta).items()} | {"n": n}


def days15(x):
    """Per-gauge mean over the leads 24-120 h of a [lead, gauge] array."""
    with np.errstate(all="ignore"):
        return np.nanmean(x[DAYS15], axis=0)


def median_by_lead(x, gauges):
    with np.errstate(all="ignore"):
        return np.nanmedian(x[:, gauges], axis=1)


def days15_of_median(x, gauges):
    return float(np.mean(median_by_lead(x, gauges)[DAYS15]))


# ---- gauge context --------------------------------------------------------------------------------------------------
class Context:
    """Static facts about the 128 catalog gauges and the network, and what each protocol withholds."""

    def __init__(self):
        from tva_flood.hydro import splits
        from tva_flood.hydro.data import Arrays, Gauges, Network
        from tva_flood.hydro.evaluate import regulation

        self.network = Network()
        self.arrays = Arrays()
        self.gauges = Gauges(self.network, self.arrays.times)
        g = self.gauges
        assert list(g.names) == list(STATIONS)
        self.names = np.array(g.names).astype(str)
        self.area = g.area.astype(float)  # km2 of each gauge's basin
        self._regulation = lambda: regulation(self.network, g)  # dam groups: computed only when a script asks for them
        self.splits = splits
        self.fold = {
            "gauged": np.zeros(g.count, int),
            "heldout": splits.gauge_folds("random", g, self.network),
            "blocked": splits.gauge_folds("blocked", g, self.network),
        }
        self.scheme = {"heldout": "random", "blocked": "blocked"}

    @property
    def groups(self):
        return self._regulation()[1]

    @property
    def regulated(self):
        return self._regulation()[0]

    @property
    def group(self):
        return {n: m for n, m in self.groups}

    def withheld(self, protocol, fold):
        if protocol == "gauged" or fold == 0:
            return np.zeros(self.gauges.count, bool)
        return self.splits.withheld(self.gauges, self.network, fold, self.scheme[protocol])

    def upstream_gauges(self, g):
        """Catalog gauges (other than g) whose reach lies in the upstream basin of gauge g."""
        inside = np.isin(self.gauges.reach, self.gauges.basins[g])
        inside[g] = False
        return np.flatnonzero(inside)

    def upstream_visible(self, protocol):
        """For every scored gauge of ``protocol``: the upstream gauges that stay visible to the model scoring it."""
        out = {}
        for g in np.flatnonzero(self.fold[protocol] > 0) if protocol != "gauged" else np.flatnonzero(self.gauges.camelsh):
            w = self.withheld(protocol, self.fold[protocol][g])
            up = self.upstream_gauges(g)
            out[g] = up[~w[up]]
        return out

    def coverage(self, protocol):
        """Share of each scored gauge's basin area (sum of reach areas) lying upstream of a visible upstream gauge."""
        net, area = self.network, self.network.area
        out = {}
        for g, vis in self.upstream_visible(protocol).items():
            basin = self.gauges.basins[g]
            if len(vis) == 0:
                out[g] = 0.0
                continue
            covered = np.unique(np.concatenate([self.gauges.basins[h] for h in vis]))
            out[g] = float(area[covered].sum() / area[basin].sum())
        return out


# ---- statistics ---------------------------------------------------------------------------------------------------
def boot_mean(d, n=10000, seed=0, q=(5, 95)):
    """Gauge-level bootstrap of the mean paired difference: (mean, lo, hi)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, q[0])), float(np.percentile(means, q[1]))


def boot_median_diff(a, b, n=10000, seed=0, q=(2.5, 97.5)):
    """Gauge-level bootstrap of median(a) - median(b) (the same resample for both models)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = np.median(a[idx], 1) - np.median(b[idx], 1)
    return float(np.median(a) - np.median(b)), float(np.percentile(diffs, q[0])), float(np.percentile(diffs, q[1]))


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    return float(stats.wilcoxon(d).pvalue) if len(d) > 5 else np.nan


def tost(d, margin):
    """Paired TOST on the mean difference with equivalence bounds +-margin: the larger of the two one-sided t-test p-values."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_low = (d.mean() + margin) / se  # H0: mean <= -margin
    t_high = (d.mean() - margin) / se  # H0: mean >= +margin
    return float(max(stats.t.sf(t_low, n - 1), stats.t.cdf(t_high, n - 1)))


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj
