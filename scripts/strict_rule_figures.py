"""Figures of the graph model (models/gnn_B and folds) under the strict forecast rule.

    python scripts/strict_rule_figures.py --out /data/tva_data/figures/strict_rule

Every forecast uses observed data before the 00Z issue and GEFSv12 rain after it; "realized
rain" panels are a perfect-rain diagnostic. Methods share issues, gauges and values (the
evaluation's mask with every method present). Sources:
  models/gnn_evaluation         forecasts at gauges and held-out gauges, metrics at every reach
  models/gefs_benchmark(_f1..5) the GEFS-rule LSTM (all gauges and with each fold withheld)
  models/gnn_strength           tail quantiles, history hidden, and Hurricane Helene at every reach
                                (scripts/strength_forecasts.py)
Maps are in the Albers equal-area projection (EPSG:5070) at true aspect.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

from tva_flood.data.folds import holdout_folds  # noqa: E402
from tva_flood.data.gauges import Catalog  # noqa: E402
from tva_flood.evaluation.hydrology import reach_metrics  # noqa: E402
from tva_flood.hydro import paths as P  # noqa: E402
from tva_flood.hydro.data import Arrays, Gauges, Network  # noqa: E402
from tva_flood.hydro.evaluate import AREA_CLASSES, regulation  # noqa: E402

NAVY, BLUE, GREEN, ORANGE, GRAY, PURPLE, LIGHT = "#14213D", "#2A78D6", "#1BAF7A", "#EB6834", "#52514E", "#6450D2", "#F4F6F9"
COLOR = {"GNN": BLUE, "LSTM": ORANGE, "NWM": GRAY, "persistence": "#A9A9A9"}
LEAD_H = np.arange(6, 241, 6)
D15 = [h // 6 - 1 for h in (24, 48, 72, 96, 120)]
CRS = "EPSG:5070"
RNG = np.random.default_rng(0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRAY,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


# ------------------------------------------------------------------ scoring
def sums(f, o):
    """reach_metrics inputs over issues for forecasts f and truth o [issue, lead, site] (NaN missing)."""
    ok = np.isfinite(f) & np.isfinite(o)
    w = ok.astype(np.float64)
    s, t = np.where(ok, f, 0.0).astype(np.float64), np.where(ok, o, 0.0).astype(np.float64)
    tm = np.where(ok, o, -np.inf)
    at_peak = np.take_along_axis(np.where(ok, f, 0.0), tm.argmax(0)[None], 0)[0]
    return dict(sums=np.stack([w.sum(0), s.sum(0), t.sum(0), (s * s).sum(0), (t * t).sum(0), (s * t).sum(0)], axis=-1),
                target_peak=tm.max(0), forecast_peak=np.where(ok, f, -np.inf).max(0), forecast_at_target_peak=at_peak)


def metrics(f, o, min_count=100):
    """The benchmark's per-site metrics [lead, site] (evaluation.hydrology.reach_metrics)."""
    return reach_metrics(**sums(f, o), min_count=min_count)


def nse(f, o, min_count=100):
    return metrics(f, o, min_count)["nse"]


def days15(x):
    """Per-site mean over 24-120 h of a [lead, site] score."""
    return np.nanmean(x[D15], axis=0)


def median_band(x, n=1000):
    """Median over sites per lead [lead, site] and its 95% bootstrap interval."""
    keep = np.isfinite(x).all(axis=0)
    x = x[:, keep]
    boot = np.array([np.median(x[:, RNG.integers(0, x.shape[1], x.shape[1])], axis=1) for _ in range(n)])
    return np.median(x, axis=1), *np.percentile(boot, [2.5, 97.5], axis=0)


# ------------------------------------------------------------------ data
class Data:
    def __init__(self):
        self.cat = Catalog(P.GAUGES)
        self.names = pd.Index(self.cat.names)
        self.fold = holdout_folds(P.ROOT, self.cat, 5)
        E = P.MODELS / "gnn_evaluation"
        g, u = (np.load(E / f"forecasts_{k}.npz", allow_pickle=True) for k in ("gauged", "ungauged"))
        self.times = pd.DatetimeIndex(g["time"])
        assert (pd.DatetimeIndex(u["time"]) == self.times).all() and list(g["stations"]) == list(self.names)
        lstm = self.lstm(P.MODELS / "gefs_benchmark/forecasts.npz")
        lstm_u = {r: np.full_like(lstm["gefs"], np.nan) for r in ("gefs", "aorc")}
        for k in range(1, 6):
            part = self.lstm(P.MODELS / f"gefs_benchmark_f{k}/forecasts.npz")
            for r in lstm_u:
                lstm_u[r][..., self.fold == k] = part[r][..., self.fold == k]
        self.gauged = {"GNN": g["values"][:, 0], "LSTM": lstm["gefs"], "NWM": g["values"][:, 2], "persistence": g["values"][:, 3]}
        self.gauged_aorc = {"GNN": g["values"][:, 1], "LSTM": lstm["aorc"]}
        self.ungauged = {"GNN": u["values"][:, 0], "LSTM": lstm_u["gefs"], "NWM": g["values"][:, 2]}
        self.ungauged_aorc = {"GNN": u["values"][:, 1], "LSTM": lstm_u["aorc"]}
        self.truth_g = self.common(g["truth"], self.gauged | self.gauged_aorc)
        self.truth_u = self.common(u["truth"], self.ungauged | self.ungauged_aorc)
        self.area = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]).set_index("comid").totdasqkm
        self.gauge_area = self.area.reindex(self.cat.comid).to_numpy()
        _, self.groups = regulation(Network(), Gauges(Network(), Arrays().times))
        self.reach = np.load(E / "reach_metrics.npz")
        self.reach_area = self.area.reindex(self.reach["comid"]).to_numpy()
        self.regulated = np.load(P.HYDRO / "dams.npz")["regulated"]
        self.strength = np.load(P.MODELS / "gnn_strength/gauges.npz", allow_pickle=True)
        assert (pd.DatetimeIndex(self.strength["time"]) == self.times).all()
        self.helene = np.load(P.MODELS / "gnn_strength/helene.npz", allow_pickle=True)
        train = self.cat.times < P.TRAIN_UNTIL
        with np.errstate(all="ignore"):
            self.p90 = np.nanquantile(self.cat.observed[train], 0.90, axis=0)
            self.p99 = np.nanquantile(self.cat.observed[train], 0.99, axis=0)
        self.observed = pd.DataFrame(self.cat.observed, index=self.cat.times, columns=self.names)
        self.river = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "gnis_name"]).set_index("comid").gnis_name

    def lstm(self, path):
        """GEFS-rule LSTM forecasts {rain: [issue, lead, gauge]} on the evaluation's issues and catalog order."""
        s = np.load(path, allow_pickle=True)
        rows = pd.DatetimeIndex(s["time"]).get_indexer(self.times)
        cols = self.names.get_indexer(s["stations"].astype(str))
        methods = list(s["methods"].astype(str))
        out = {}
        for rain in ("gefs", "aorc"):
            v = np.full((len(self.times), 40, len(self.names)), np.nan, np.float32)
            v[np.ix_(rows >= 0, np.arange(40), cols)] = s["values"][rows[rows >= 0]][:, methods.index(f"lstm+{rain}")]
            out[rain] = v
        return out

    @staticmethod
    def common(truth, series):
        truth = truth.astype(np.float64)
        ok = np.isfinite(truth) & np.all([np.isfinite(v) for v in series.values()], axis=0)
        return np.where(ok, truth, np.nan)

    def label(self, gi):
        river = self.river.get(self.cat.comid[gi])
        return f"{self.names[gi]}" + (f", {river}" if isinstance(river, str) else "") + f" ({self.gauge_area[gi]:,.0f} km2)"


class Maps:
    """Basin outline and NHDPlus flowlines in EPSG:5070."""

    def __init__(self, comids):
        fl = gpd.read_file(P.ROOT / "raw/geography/nhdplus_v21/extracted/NHDPlusV21_MS_06_NHDSnapshot_09/NHDPlusMS/NHDPlus06/NHDSnapshot/Hydrography/NHDFlowline.shp", columns=["COMID"])
        self.lines = fl[fl.COMID.isin(comids)].set_index("COMID").to_crs(CRS)
        self.basin = gpd.read_file(P.ROOT / "raw/geography/tennessee_huc2.geojson").to_crs(CRS)
        x0, y0, x1, y1 = self.basin.total_bounds
        self.ratio = (x1 - x0) / (y1 - y0)  # width / height at true scale

    def figure(self, rows=1, cols=1, width=11.0, colorbar=True):
        """A figure whose map panels keep the basin's true aspect."""
        panel_w = width / cols * (0.84 if colorbar else 1.0)
        height = rows * (panel_w / self.ratio + 0.45)
        fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, layout="constrained")
        return fig, axes

    def frame(self, ax):
        self.basin.plot(ax=ax, color=LIGHT, edgecolor=NAVY, lw=1.0)
        ax.set_aspect("equal")
        ax.set_axis_off()

    def rivers(self, ax, values, cmap, norm_, area):
        """Reach lines coloured by values (Series by comid), wider for larger rivers."""
        frame = self.lines.assign(v=values.reindex(self.lines.index).to_numpy(), a=area.reindex(self.lines.index).to_numpy()).dropna(subset=["v"])
        for lw, lo, hi in ((0.22, 0, 50), (0.6, 50, 1500), (1.6, 1500, 1e9)):
            part = frame[(frame.a >= lo) & (frame.a < hi)]
            if len(part):
                part.plot(ax=ax, column="v", cmap=cmap, norm=norm_, lw=lw)
        ax.set_aspect("equal")

    def main_rivers(self, ax, area):
        big = self.lines[area.reindex(self.lines.index).to_numpy() > 1500]
        big.plot(ax=ax, color="#9DB4CF", lw=0.8)
        ax.set_aspect("equal")

    def points(self, comids):
        pts = []
        for c in comids:
            geom = self.lines.geometry.get(c)
            if geom is None:
                pts.append((np.nan, np.nan))
                continue
            geom = geom.geoms[-1] if geom.geom_type == "MultiLineString" else geom
            pts.append(geom.coords[-1][:2])
        return np.array(pts)


def save(fig, out, name):
    fig.savefig(out / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name, flush=True)


def lead_axis(ax):
    ax.set_xlabel("lead (h)")
    ax.set_xticks([0, 48, 96, 144, 192, 240])
    ax.set_xlim(0, 246)


# ------------------------------------------------------------------ gauged and ungauged
def by_lead(d, out):
    """g1 / u1: median NSE by lead with 95% intervals; realized rain dashed."""
    for tag, series, aorc, truth, refs in (("g1_gauged_nse_by_lead.png", d.gauged, d.gauged_aorc, d.truth_g, ("NWM", "persistence")),
                                            ("u1_ungauged_nse_by_lead.png", d.ungauged, d.ungauged_aorc, d.truth_u, ("NWM",))):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for name in ("GNN", "LSTM") + refs:
            med, lo, hi = median_band(nse(series[name], truth))
            ax.fill_between(LEAD_H, lo, hi, color=COLOR[name], alpha=0.15, lw=0)
            ax.plot(LEAD_H, med, color=COLOR[name], lw=2.3, label=f"{name} ({np.mean(med[D15]):.3f} days 1-5)")
        for name in ("GNN", "LSTM"):
            med = np.nanmedian(nse(aorc[name], truth), axis=1)
            ax.plot(LEAD_H, med, "--", color=COLOR[name], lw=1.4, label=f"{name}, realized rain (diagnostic)")
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_ylim(-0.6 if "persistence" in refs else -0.4, 1)
        lead_axis(ax)
        ax.set_ylabel("median NSE over gauges")
        what = "Gauged: recent flow shown" if tag.startswith("g") else "Ungauged: each gauge forecast by the model that never saw it"
        ax.set_title(f"{what} ({int(np.isfinite(nse(series['GNN'], truth)[3]).sum())} gauges, 1,917 issues)", color=NAVY, fontsize=11)
        ax.legend(fontsize=9, ncol=2)
        save(fig, out, tag)


def summary(d, out):
    """g2: days 1-5 median NSE with 95% intervals, gauged and ungauged, by dam group."""
    rows = []
    for protocol, series, truth in (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u)):
        scores = {m: days15(nse(series[m], truth)) for m in ("GNN", "LSTM", "NWM")}
        for scope, sel in (("all gauges", np.ones(len(d.names), bool)),) + tuple(d.groups):
            for m, v in scores.items():
                x = v[sel & np.isfinite(v)]
                boot = [np.median(x[RNG.integers(0, len(x), len(x))]) for _ in range(1000)]
                rows.append(dict(protocol=protocol, gauges=f"{scope} ({len(x)})", method=m, median=np.median(x), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
    t = pd.DataFrame(rows)
    t.round(3).to_csv(out / "g2_days_1_5.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, protocol in zip(axes, ("gauged", "ungauged")):
        p = t[t.protocol == protocol]
        groups = list(dict.fromkeys(p.gauges))
        for j, m in enumerate(("GNN", "LSTM", "NWM")):
            q = p[p.method == m].set_index("gauges").reindex(groups)
            x = np.arange(len(groups)) + (j - 1) * 0.26
            ax.bar(x, q["median"], width=0.25, color=COLOR[m], label=m)
            ax.errorbar(x, q["median"], yerr=[q["median"] - q.lo, q.hi - q["median"]], fmt="none", ecolor=NAVY, lw=1, capsize=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace(", ", ",\n").replace(" (", "\n(") for g in groups], fontsize=9)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_title(protocol.capitalize(), color=NAVY, fontweight="bold")
    axes[0].set_ylabel("median NSE, days 1-5 (95% interval)")
    axes[0].legend()
    save(fig, out, "g2_days_1_5.png")


def gain_and_wins(d, out):
    """g3: paired median NSE difference and share of gauges won, by lead, with 95% intervals."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, (protocol, series, truth) in enumerate((("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        gnn = nse(series["GNN"], truth)
        for other in ("LSTM", "NWM"):
            diff = gnn - nse(series[other], truth)
            keep = np.isfinite(diff).all(axis=0)
            diff = diff[:, keep]
            boot = RNG.integers(0, diff.shape[1], (1000, diff.shape[1]))
            med = np.median(diff, axis=1)
            mb = np.array([np.median(diff[:, b], axis=1) for b in boot])
            win = (diff > 0).mean(axis=1) * 100
            wb = np.array([(diff[:, b] > 0).mean(axis=1) * 100 for b in boot])
            for row, (y, yb) in enumerate(((med, mb), (win, wb))):
                ax = axes[row, col]
                ax.fill_between(LEAD_H, *np.percentile(yb, [2.5, 97.5], axis=0), color=COLOR[other], alpha=0.18, lw=0)
                ax.plot(LEAD_H, y, color=COLOR[other], lw=2.2, label=f"GNN vs {other}")
        axes[0, col].axhline(0, color=GRAY, lw=1, ls="--")
        axes[1, col].axhline(50, color=GRAY, lw=1, ls="--")
        axes[1, col].set_ylim(0, 100)
        axes[0, col].set_title(f"{protocol.capitalize()} ({int(keep.sum())} gauges)", color=NAVY, fontweight="bold")
        lead_axis(axes[1, col])
    axes[0, 0].set_ylabel("median of GNN NSE minus other\n(paired by gauge)")
    axes[1, 0].set_ylabel("% of gauges where the GNN\nhas the higher NSE")
    axes[0, 1].legend()
    save(fig, out, "g3_gain_and_wins_by_lead.png")


def scatter(d, out):
    """g4: per-gauge days 1-5 NSE, GNN against each baseline."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    panels = [("gauged", "LSTM", d.gauged, d.truth_g), ("gauged", "NWM", d.gauged, d.truth_g),
              ("ungauged", "LSTM", d.ungauged, d.truth_u), ("ungauged", "NWM", d.ungauged, d.truth_u)]
    for ax, (protocol, other, series, truth) in zip(axes, panels):
        x, y = days15(nse(series[other], truth)), days15(nse(series["GNN"], truth))
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(np.clip(x[ok], -1, 1), np.clip(y[ok], -1, 1), s=22, color=BLUE, alpha=0.8, edgecolor="white", lw=0.4)
        ax.plot([-1, 1], [-1, 1], color=GRAY, lw=1)
        ax.set_xlim(-1.02, 1); ax.set_ylim(-1.02, 1); ax.set_aspect("equal")
        ax.set_xlabel(f"{other} NSE, days 1-5"); ax.set_ylabel("GNN NSE, days 1-5")
        ax.set_title(f"{protocol.capitalize()}: GNN above the line\nat {int((y[ok] > x[ok]).sum())} of {int(ok.sum())} gauges", color=NAVY, fontsize=10.5)
    save(fig, out, "g4_gauge_scatter.png")


def distribution(d, out):
    """g5: distribution of per-gauge days 1-5 NSE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (protocol, series, truth) in zip(axes, (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        for m in ("GNN", "LSTM", "NWM"):
            v = days15(nse(series[m], truth))
            v = np.sort(v[np.isfinite(v)])
            ax.plot(np.clip(v, -1, 1), np.arange(1, len(v) + 1) / len(v) * 100, color=COLOR[m], lw=2.2, label=f"{m} (median {np.median(v):.2f})")
        ax.set_xlim(-1, 1); ax.set_xlabel("NSE, days 1-5, per gauge (clipped at -1)")
        ax.set_title(protocol.capitalize(), color=NAVY, fontweight="bold"); ax.legend(loc="upper left", fontsize=9.5)
    axes[0].set_ylabel("% of gauges at or below")
    save(fig, out, "g5_nse_distribution.png")


def year_season(d, out):
    """g6: days 1-5 median NSE by test year and by season (issues in that period only)."""
    seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey="row")
    for col, (protocol, series, truth) in enumerate((("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        for row, (labels, masks) in enumerate((
            ([str(y) for y in range(2021, 2026)], [d.times.year == y for y in range(2021, 2026)]),
            (list(seasons), [d.times.month.isin(m) for m in seasons.values()]),
        )):
            ax = axes[row, col]
            for j, m in enumerate(("GNN", "LSTM", "NWM")):
                vals = [np.nanmedian(days15(nse(series[m][k], truth[k], min_count=60))) for k in masks]
                ax.bar(np.arange(len(labels)) + (j - 1) * 0.26, vals, width=0.25, color=COLOR[m], label=m)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
            ax.axhline(0, color=GRAY, lw=0.8)
            if row == 0:
                ax.set_title(protocol.capitalize(), color=NAVY, fontweight="bold")
    axes[0, 0].set_ylabel("median NSE, days 1-5\nby test year"); axes[1, 0].set_ylabel("median NSE, days 1-5\nby season")
    axes[0, 0].legend(ncol=3, fontsize=9)
    save(fig, out, "g6_by_year_and_season.png")


def gain_vs_area(d, out):
    """g7: per-gauge days 1-5 gain over the LSTM against drainage area, with a running median."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (protocol, series, truth) in zip(axes, (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        gain = days15(nse(series["GNN"], truth)) - days15(nse(series["LSTM"], truth))
        ok = np.isfinite(gain)
        a, gn = d.gauge_area[ok], np.clip(gain[ok], -1, 1.5)
        ax.scatter(a, gn, s=24, color=BLUE, alpha=0.8, edgecolor="white", lw=0.4)
        edges = np.array([10, 100, 200, 400, 1000, 2000, 10000])
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (a >= lo) & (a < hi)
            if sel.sum() >= 5:
                ax.plot([lo, hi], [np.median(gn[sel])] * 2, color=NAVY, lw=2.4)
                ax.text(np.sqrt(lo * hi), np.median(gn[sel]) + 0.06, f"n={sel.sum()}", ha="center", fontsize=8, color=NAVY)
        ax.plot([], [], color=NAVY, lw=2.4, label="median within each area class")
        ax.set_xscale("log"); ax.axhline(0, color=GRAY, lw=1, ls="--"); ax.set_ylim(-1.05, 1.55)
        ax.set_xlabel("drainage area (km2)")
        ax.set_title(f"{protocol.capitalize()}: GNN minus LSTM", color=NAVY, fontweight="bold"); ax.legend(fontsize=9)
    axes[0].set_ylabel("NSE difference, days 1-5\n(clipped)")
    save(fig, out, "g7_gain_vs_area.png")


def flood_skill(d, out):
    """g8: critical success index above the 90th and 99th percentile, by lead."""
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (protocol, series, truth) in zip(axes, (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        for thr, ls, tag in ((d.p90, "-", "90th"), (d.p99, "--", "99th")):
            for m in ("GNN", "LSTM", "NWM"):
                o, f, ok = truth > thr, series[m] > thr, np.isfinite(truth)
                hit, miss, false = ((o & f & ok).sum((0, 2)), (o & ~f & ok).sum((0, 2)), (~o & f & ok).sum((0, 2)))
                csi = hit / np.maximum(hit + miss + false, 1)
                ax.plot(LEAD_H, csi, ls, color=COLOR[m], lw=2, label=f"{m}, above {tag} pct")
                rows += [dict(protocol=protocol, threshold=tag, method=m, lead_h=h, csi=csi[h // 6 - 1], pod=(hit / np.maximum(hit + miss, 1))[h // 6 - 1],
                              far=(false / np.maximum(hit + false, 1))[h // 6 - 1]) for h in (24, 72, 120, 240)]
        ax.set_title(f"{protocol.capitalize()}: flood detection", color=NAVY, fontweight="bold")
        lead_axis(ax)
    axes[0].set_ylabel("critical success index\nhits / (hits + misses + false alarms)")
    axes[1].legend(fontsize=8.5, ncol=2)
    pd.DataFrame(rows).round(3).to_csv(out / "g8_flood_skill.csv", index=False)
    save(fig, out, "g8_flood_skill.png")


def flood_events(d):
    """Independent observed test peaks above the gauge's 99th percentile (largest within +-7 days)."""
    events = []
    for gi, st in enumerate(d.names):
        if not d.cat.camelsh[gi] or not np.isfinite(d.p99[gi]):
            continue
        s = d.observed[st][d.times[0] : d.times[-1] + pd.Timedelta(days=10)]
        peaks = s[(s > d.p99[gi]) & (s == s.rolling(29, center=True, min_periods=1).max())]
        events += [(gi, t, float(q)) for t, q in peaks.items()]
    return events


def flood_peaks(d, out, events):
    """g9: forecast peak / observed peak - 1 and peak timing, issued 1, 3, 5 days before."""
    issue_index = {t: i for i, t in enumerate(d.times)}
    rows = []
    for protocol, series in (("gauged", d.gauged), ("ungauged", d.ungauged)):
        for gi, t, q in events:
            if protocol == "ungauged" and d.fold[gi] == 0:
                continue
            for days in (1, 3, 5):
                i = issue_index.get((t - pd.Timedelta(days=days)).normalize())
                if i is None:
                    continue
                target = int((t - d.times[i]) / pd.Timedelta(hours=6)) - 1
                lo, hi = max(target - 8, 0), min(target + 9, 40)
                fs = {m: series[m][i, lo:hi, gi] for m in ("GNN", "LSTM", "NWM")}
                if not all(np.isfinite(f).all() and len(f) for f in fs.values()):
                    continue
                for m, f in fs.items():
                    rows.append(dict(protocol=protocol, method=m, days=days, gauge=d.names[gi], peak=t, magnitude=float(f.max() / q - 1),
                                     timing_h=float((lo + int(np.argmax(f)) - target) * 6)))
    p = pd.DataFrame(rows)
    p.groupby(["protocol", "days", "method"]).agg(events=("magnitude", "size"), median_peak_error=("magnitude", "median"),
                                                  median_abs_timing_h=("timing_h", lambda x: x.abs().median())).round(3).to_csv(out / "g9_flood_peaks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, protocol in zip(axes, ("gauged", "ungauged")):
        q = p[p.protocol == protocol]
        for k, days in enumerate((1, 3, 5)):
            for j, m in enumerate(("GNN", "LSTM", "NWM")):
                bp = ax.boxplot(np.clip(q[(q.days == days) & (q.method == m)].magnitude * 100, -100, 200), positions=[k * 3 + j * 0.8],
                                widths=0.6, showfliers=False, patch_artist=True)
                for b in bp["boxes"]:
                    b.set(facecolor=COLOR[m], alpha=0.75, edgecolor=NAVY)
                for b in bp["medians"]:
                    b.set(color=NAVY)
        n = q[q.days == 1].groupby(["gauge", "peak"]).ngroups
        ax.set_xticks([0.8, 3.8, 6.8]); ax.set_xticklabels(["issued 1 day\nbefore the peak", "3 days", "5 days"])
        ax.axhline(0, color=GRAY, lw=1, ls="--")
        ax.set_title(f"{protocol.capitalize()}: {n:,} peaks above the 99th percentile", color=NAVY, fontsize=11)
    axes[0].set_ylabel("forecast peak / observed peak - 1 (%)")
    axes[0].set_ylim(-110, 130)
    axes[1].legend([plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in ("GNN", "LSTM", "NWM")], ["GNN", "LSTM", "NWM"], loc="upper right", ncol=3)
    save(fig, out, "g9_flood_peaks.png")


def hydrographs(d, out, events, protocol):
    """g10 / u6: the three largest floods (distinct gauges), forecasts issued 1, 3, 5 days before."""
    series = d.gauged if protocol == "gauged" else d.ungauged
    issue_index = {t: i for i, t in enumerate(d.times)}
    cand = pd.DataFrame(events, columns=["gi", "peak", "q"]).assign(ratio=lambda e: e.q / d.p99[e.gi.to_numpy()])
    if protocol == "ungauged":
        cand = cand[d.fold[cand.gi.to_numpy()] > 0]
    ok = [all((t - pd.Timedelta(days=k)).normalize() in issue_index for k in (1, 3, 5)) for t in cand.peak]
    cand = cand[ok].sort_values("ratio", ascending=False).drop_duplicates("gi").head(3)
    fig, axes = plt.subplots(len(cand), 3, figsize=(15, 3.3 * len(cand)), sharey="row", squeeze=False)
    for row, (_, e) in zip(axes, cand.iterrows()):
        gi, t = int(e.gi), e.peak
        for ax, days in zip(row, (1, 3, 5)):
            issue = (t - pd.Timedelta(days=days)).normalize()
            i = issue_index[issue]
            window = d.observed[d.names[gi]][issue - pd.Timedelta(days=3) : issue + pd.Timedelta(days=10)]
            ax.plot(window.index, window.values, color="black", lw=2, label="observed")
            for m in ("GNN", "LSTM", "NWM"):
                ax.plot(issue + pd.to_timedelta(LEAD_H, unit="h"), series[m][i, :, gi], color=COLOR[m], lw=1.8, label=m)
            ax.axvline(issue, color=ORANGE, lw=1, ls=":")
            ax.set_title(f"issued {issue:%Y-%m-%d} 00Z ({days} day{'s' if days > 1 else ''} before)", fontsize=10, color=NAVY)
            ax.tick_params(axis="x", labelsize=8, rotation=25)
        row[0].set_ylabel(d.label(gi).replace(", ", "\n", 1) + "\nm3/s", fontsize=9)
    axes[0, 0].legend(fontsize=8.5)
    what = "recent flow shown" if protocol == "gauged" else "each model forecasts without the gauge"
    fig.suptitle(f"Largest test floods, {protocol} ({what})", color=NAVY, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "g10_hydrographs_gauged.png" if protocol == "gauged" else "u6_hydrographs_ungauged.png")


def full_metrics(d, out):
    """g11: NSE, KGE and its parts, RMSE and |peak error| by lead."""
    keys = (("nse", "NSE"), ("kge", "KGE"), ("r", "correlation r"), ("alpha", "variability ratio"), ("beta", "bias ratio"), ("rmse", "RMSE (m3/s)"), ("peak_error", "|peak error|"))
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True)
    rows = []
    for col_off, (protocol, series, truth) in enumerate((("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        m = {name: metrics(series[name], truth) for name in ("GNN", "LSTM", "NWM")}
        scored = np.all([x["scored"][3] for x in m.values()], axis=0)
        for name, x in m.items():
            for k, _ in keys:
                rows += [dict(protocol=protocol, method=name, metric=k, lead_h=h, median=np.nanmedian(x[k][h // 6 - 1][scored])) for h in (24, 72, 120, 240)]
        for j, (k, label) in enumerate(keys[1:5] if col_off == 0 else keys[1:5]):
            ax = axes[col_off, j]
            for name, x in m.items():
                ax.plot(LEAD_H, np.nanmedian(x[k][:, scored], axis=1), color=COLOR[name], lw=2, label=name)
            if k in ("alpha", "beta"):
                ax.axhline(1, color=GRAY, lw=0.9, ls="--")
            ax.set_title(label if col_off == 0 else "", color=NAVY)
            if j == 0:
                ax.set_ylabel(f"{protocol}\nmedian over gauges")
            if col_off == 1:
                lead_axis(ax)
    axes[0, 0].legend()
    pd.DataFrame(rows).round(4).to_csv(out / "g11_full_metrics.csv", index=False)
    fig.suptitle("KGE and its parts (variability ratio and bias ratio: 1 is right)", color=NAVY, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "g11_kge_components.png")


def value_of_gauge(d, out):
    """g12: the same gauges with the recent flow shown, hidden, and never seen."""
    s = d.strength
    o = np.where(d.fold[None, None] > 0, s["truth"].astype(np.float64), np.nan)
    series = {"GNN, recent flow shown": (s["shown_mu"], BLUE, "-"), "GNN, no gauge's flow shown": (s["hidden_mu"], BLUE, "--"),
              "GNN, gauge never seen (fold model)": (s["fold_mu"], BLUE, ":"), "LSTM, recent flow shown": (d.gauged["LSTM"], ORANGE, "-"),
              "LSTM, gauge never seen": (d.ungauged["LSTM"], ORANGE, ":"), "NWM": (d.gauged["NWM"], GRAY, "-")}
    o = Data.common(o, {k: v for k, (v, _, _) in series.items()})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for label, (v, c, ls) in series.items():
        med = np.nanmedian(nse(v, o), axis=1)
        ax.plot(LEAD_H, med, ls, color=c, lw=2.2, label=f"{label} ({np.mean(med[D15]):.3f})")
    lead_axis(ax); ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_ylabel("median NSE over gauges")
    ax.set_title(f"What the gauge's own flow is worth ({int(np.isfinite(nse(series['NWM'][0], o)[3]).sum())} gauges; days 1-5 in brackets)", color=NAVY, fontsize=11)
    ax.legend(fontsize=9)
    save(fig, out, "g12_value_of_the_gauge.png")


def tail(d, out, events):
    """g13-g15: the tail head's quantiles (0.5, 0.9, 0.99): calibration and pinball loss, flood warnings, and peaks."""
    s = d.strength
    held = d.fold[None, None] > 0
    cases = (("gauged", s["shown_mu"], s["shown_q"], d.truth_g, d.gauged, "-"),
             ("ungauged", s["fold_mu"], s["fold_q"], np.where(held, d.truth_u, np.nan), d.ungauged, ":"))
    qs = (0.5, 0.9, 0.99)
    colors = (GREEN, BLUE, NAVY)
    # g13: calibration and pinball loss
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    shares = []
    for label, mu, q, truth, series, ls in cases:
        ok = np.isfinite(truth) & np.isfinite(mu) & np.isfinite(q).all(-1)
        for i, (qq, c) in enumerate(zip(qs, colors)):
            below = np.where(ok, truth <= q[..., i], 0).sum((0, 2)) / ok.sum((0, 2))
            shares.append(dict(protocol=label, quantile=qq, share_below_24_120h=float(np.mean(below[3:20])), **{f"share_below_{h}h": float(below[h // 6 - 1]) for h in (24, 72, 120, 240)}))
            axes[0].plot(LEAD_H, below * 100, ls, color=c, lw=2, label=f"q{int(qq * 100)}, {label}")
            axes[0].axhline(qq * 100, color=c, lw=0.7, alpha=0.5)
        scale = np.nanmean(np.where(ok, truth, np.nan), axis=(0, 1))
        for ax, i in zip(axes[1:], (1, 2)):
            for name, f, c in ((f"GNN q{int(qs[i] * 100)} (pinball)", q[..., i], BLUE), ("GNN mean", mu, PURPLE), ("LSTM", series["LSTM"], ORANGE), ("NWM", d.gauged["NWM"], GRAY)):
                miss = truth - f
                qv = qs[i]
                loss = np.where(ok, np.maximum(qv * miss, (qv - 1) * miss), np.nan)
                ax.plot(LEAD_H, np.nanmedian(np.nanmean(loss, axis=0) / scale[None], axis=1), ls, color=c, lw=2, label=f"{name}, {label}")
    lead_axis(axes[0]); axes[0].set_ylabel("% of observations below the quantile")
    axes[0].set_title("Calibration of the tail (thin lines: nominal)", color=NAVY, fontsize=11); axes[0].legend(fontsize=8, ncol=2)
    for ax, i in zip(axes[1:], (1, 2)):
        lead_axis(ax); ax.set_ylabel("pinball loss / mean flow (median over gauges)")
        ax.set_title(f"Pinball loss at the {int(qs[i] * 100)}th percentile (lower is better)", color=NAVY, fontsize=11); ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    pd.DataFrame(shares).round(4).to_csv(out / "g13_tail_calibration.csv", index=False)
    save(fig, out, "g13_tail_calibration.png")
    # g14: flood warnings from the mean and from the quantiles
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    rows = []
    for col, (label, mu, q, truth, series, ls) in enumerate(cases):
        ok0 = np.isfinite(truth) & np.isfinite(mu) & np.isfinite(q).all(-1)
        for row, (tag, thr) in enumerate((("90th", d.p90), ("99th", d.p99))):
            ok = ok0 & np.isfinite(thr)[None, None]
            ax = axes[row, col]
            for name, f, c, st in (("GNN mean", mu, BLUE, "-"), ("GNN q90", q[..., 1], PURPLE, "-"), ("GNN q99", q[..., 2], NAVY, "--"), ("LSTM", series["LSTM"], ORANGE, "-"), ("NWM", d.gauged["NWM"], GRAY, "-")):
                o, fl = truth > thr, f > thr
                hit, miss, fa = (o & fl & ok).sum((0, 2)), (o & ~fl & ok).sum((0, 2)), (~o & fl & ok).sum((0, 2))
                csi = hit / np.maximum(hit + miss + fa, 1)
                ax.plot(LEAD_H, csi, st, color=c, lw=2, label=name)
                rows += [dict(protocol=label, threshold=tag, warning=name, lead_h=h, csi=csi[h // 6 - 1], hit_rate=(hit / np.maximum(hit + miss, 1))[h // 6 - 1],
                              false_alarm_ratio=(fa / np.maximum(hit + fa, 1))[h // 6 - 1]) for h in (24, 72, 120, 240)]
            ax.set_title(f"{label.capitalize()}: floods above the {tag} percentile", color=NAVY, fontsize=11)
            if row == 1:
                lead_axis(ax)
            if col == 0:
                ax.set_ylabel("critical success index")
    axes[0, 0].legend(fontsize=9)
    pd.DataFrame(rows).round(3).to_csv(out / "g14_tail_warnings.csv", index=False)
    save(fig, out, "g14_tail_warnings.png")
    # g15: peaks from the mean and the quantiles
    issue_index = {t_: i for i, t_ in enumerate(d.times)}
    names = ("GNN mean", "GNN q90", "GNN q99", "LSTM", "NWM")
    pcolor = {"GNN mean": BLUE, "GNN q90": PURPLE, "GNN q99": NAVY, "LSTM": ORANGE, "NWM": GRAY}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    peak_errors = {}
    for ax, (label, mu, q, truth, series, ls) in zip(axes, cases):
        fc = {"GNN mean": mu, "GNN q90": q[..., 1], "GNN q99": q[..., 2], "LSTM": series["LSTM"], "NWM": d.gauged["NWM"]}
        errs = {(n, k): [] for n in names for k in (1, 3, 5)}
        for gi, t0, qv in events:
            if label == "ungauged" and d.fold[gi] == 0:
                continue
            for days in (1, 3, 5):
                i = issue_index.get((t0 - pd.Timedelta(days=days)).normalize())
                if i is None:
                    continue
                target = int((t0 - d.times[i]) / pd.Timedelta(hours=6)) - 1
                lo, hi = max(target - 8, 0), min(target + 9, 40)
                w = {n: v[i, lo:hi, gi] for n, v in fc.items()}
                if all(len(x) and np.isfinite(x).all() for x in w.values()):
                    for n, x in w.items():
                        errs[(n, days)].append(x.max() / qv - 1)
        peak_errors[label] = errs
        for k, days in enumerate((1, 3, 5)):
            for j, n in enumerate(names):
                bp = ax.boxplot(np.clip(np.array(errs[(n, days)]) * 100, -100, 400), positions=[k * 6.5 + j], widths=0.8, showfliers=False, patch_artist=True)
                for bx in bp["boxes"]:
                    bx.set(facecolor=pcolor[n], alpha=0.75, edgecolor=NAVY)
                for md in bp["medians"]:
                    md.set(color=NAVY)
        ax.set_xticks([2, 8.5, 15]); ax.set_xticklabels(["issued 1 day\nbefore the peak", "3 days", "5 days"])
        ax.axhline(0, color=GRAY, lw=1, ls="--")
        ax.set_title(f"{label.capitalize()}: peaks above the 99th percentile ({len(errs[('NWM', 1)]):,} floods)", color=NAVY, fontsize=11)
    pd.DataFrame([dict(protocol=lab, forecast=n, days_ahead=k, events=len(v), median_peak_error=float(np.median(v)))
                  for lab, errs_ in peak_errors.items() for (n, k), v in errs_.items() if v]).round(3).to_csv(out / "g15_tail_peaks.csv", index=False)
    axes[0].set_ylabel("forecast peak / observed peak - 1 (%)")
    axes[0].set_ylim(-110, 520)
    axes[1].legend([plt.Rectangle((0, 0), 1, 1, color=pcolor[n]) for n in names], names, loc="upper right", fontsize=9, ncol=5)
    save(fig, out, "g15_tail_peaks.png")


def skill_lost(d, out):
    """u3: what hiding the gauge costs, per gauge (gauged minus ungauged days 1-5 NSE)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for m in ("GNN", "LSTM"):
        lost = days15(nse(d.gauged[m], d.truth_g)) - days15(nse(d.ungauged[m], d.truth_u))
        lost = np.sort(lost[np.isfinite(lost)])
        ax.plot(np.clip(lost, -0.5, 2), np.arange(1, len(lost) + 1) / len(lost) * 100, color=COLOR[m], lw=2.2, label=f"{m} (median {np.median(lost):.2f})")
    ax.axvline(0, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("NSE lost when the gauge is never seen, days 1-5 (clipped at 2)")
    ax.set_ylabel("% of gauges at or below"); ax.legend()
    ax.set_title("Skill lost without the gauge", color=NAVY, fontweight="bold")
    save(fig, out, "u3_skill_lost.png")


def gauge_maps(d, out, maps):
    """u7: per-gauge days 1-5 difference, GNN minus LSTM and minus NWM, held-out gauges."""
    xy = maps.points(d.cat.comid)
    fig, axes = maps.figure(2, 1, width=10)
    for ax, other, lim in zip(axes[:, 0], ("LSTM", "NWM"), (0.8, 0.5)):
        maps.frame(ax)
        maps.main_rivers(ax, d.area)
        diff = days15(nse(d.ungauged["GNN"], d.truth_u)) - days15(nse(d.ungauged[other], d.truth_u))
        ok = np.isfinite(diff)
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=np.clip(diff[ok], -lim, lim), cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), s=55, edgecolor=NAVY, lw=0.6, zorder=3)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="NSE difference")
        ax.set_title(f"Held-out gauges, days 1-5: GNN minus {other} (blue: GNN better, at {int((diff[ok] > 0).sum())} of {int(ok.sum())})", color=NAVY, fontsize=11)
    save(fig, out, "u7_heldout_gain_maps.png")


# ------------------------------------------------------------------ every reach
REACH = {"GNN": "GNN+gefs", "LSTM": "LSTM, tuned+gefs", "NWM": "NWM forecast", "persistence": "retrospective persistence (oracle)"}


def reach_by_lead(d, out):
    """r1: median NSE against the NWM retrospective by lead, by scope."""
    r = d.reach
    held = r["reach_fold"] > 0
    scopes = (("all reaches", np.ones(len(held), bool)), ("held-out basins", held), ("outside every gauge basin", r["reach_fold"] == -1),
              ("behind dams (> 50%)", d.regulated > 0.5))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, (scope, sel) in zip(axes, scopes):
        for m, key in REACH.items():
            ax.plot(LEAD_H, np.nanmedian(r[f"{key}/nse"][:, sel], axis=1), color=COLOR[m], lw=2.2, label="retrospective persistence (oracle)" if m == "persistence" else m)
        ax.plot(LEAD_H, np.nanmedian(r["GNN+aorc/nse"][:, sel], axis=1), "--", color=BLUE, lw=1.4, label="GNN, realized rain")
        ax.axhline(0, color=GRAY, lw=0.8); ax.set_ylim(-1.5, 1)
        lead_axis(ax); ax.set_title(f"{scope} ({sel.sum():,})", color=NAVY, fontsize=11)
    axes[0].set_ylabel("median NSE against the\nNWM v3.0 retrospective"); axes[0].legend(fontsize=8.5)
    save(fig, out, "r1_reach_nse_by_lead.png")


def reach_by_area(d, out):
    """r2: median NSE and KGE parts by drainage area, held-out basins, 24 h and 72 h."""
    r = d.reach
    held = r["reach_fold"] > 0
    bins = np.logspace(-1, 5, 19)
    mid = np.sqrt(bins[:-1] * bins[1:])
    keys = (("nse", "NSE", (-1, 1)), ("kge", "KGE", (-1, 1)), ("r", "correlation r", (0, 1)), ("alpha", "variability ratio", (0, 2)), ("beta", "bias ratio", (0, 2)))
    rows = []
    fig, axes = plt.subplots(2, len(keys), figsize=(20, 7.2), sharex=True)
    for row, h in enumerate((24, 72)):
        for col, (k, label, lim) in enumerate(keys):
            ax = axes[row, col]
            for m in ("GNN", "LSTM", "NWM"):
                name = f"{REACH[m]}/{k}"
                if name not in r.files:
                    continue
                v = r[name][h // 6 - 1]
                med = [np.nanmedian(v[held & (d.reach_area >= lo) & (d.reach_area < hi)]) if (held & (d.reach_area >= lo) & (d.reach_area < hi)).sum() >= 30 else np.nan
                       for lo, hi in zip(bins[:-1], bins[1:])]
                ax.plot(mid, med, "o-", color=COLOR[m], lw=1.8, ms=3.5, label=m)
                rows += [dict(lead_h=h, metric=k, method=m, area_km2=round(a, 2), median=x) for a, x in zip(mid, med)]
            ax.set_xscale("log"); ax.set_ylim(*lim)
            ax.axhline(1 if k in ("alpha", "beta", "r") else 0, color=GRAY, lw=0.8, ls="--")
            if row == 0:
                ax.set_title(label, color=NAVY)
            if col == 0:
                ax.set_ylabel(f"{h} h, held-out basins\nmedian over reaches")
            if row == 1:
                ax.set_xlabel("drainage area (km2)")
    axes[0, 0].legend(fontsize=9)
    pd.DataFrame(rows).round(3).to_csv(out / "r2_reach_by_area.csv", index=False)
    save(fig, out, "r2_reach_by_area.png")


def reach_distribution(d, out):
    """r3: distribution of per-reach NSE (held-out basins)."""
    r = d.reach
    held = r["reach_fold"] > 0
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, h in zip(axes, (24, 72, 120)):
        for m in ("GNN", "LSTM", "NWM"):
            v = r[f"{REACH[m]}/nse"][h // 6 - 1][held]
            v = np.sort(v[np.isfinite(v)])
            ax.plot(np.clip(v, -2, 1), np.arange(1, len(v) + 1) / len(v) * 100, color=COLOR[m], lw=2.2, label=f"{m} (median {np.median(v):.2f})")
        ax.set_xlim(-2, 1); ax.axvline(0, color=GRAY, lw=0.8)
        ax.set_xlabel("NSE against the retrospective (clipped at -2)"); ax.set_title(f"{h} h", color=NAVY, fontweight="bold"); ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("% of reaches at or below\n(held-out basins)")
    save(fig, out, "r3_reach_nse_distribution.png")


def reach_maps(d, out, maps):
    """r4: NSE at every reach for the GNN, the LSTM and NWM, and the GNN's gain over NWM and over the LSTM, at 24 h and 72 h."""
    comid = d.reach["comid"]
    area = pd.Series(d.reach_area, index=comid)
    for h in (24, 72):
        nse_ = {m: pd.Series(d.reach[f"{REACH[m]}/nse"][h // 6 - 1], index=comid) for m in ("GNN", "LSTM", "NWM")}
        fig, axes = maps.figure(3, 1, width=10)
        for ax, (m, title) in zip(axes[:, 0], (("GNN", "GNN"), ("LSTM", "LSTM (run at every reach)"), ("NWM", "NWM forecast"))):
            nrm = plt.Normalize(-1, 1)
            maps.frame(ax)
            maps.rivers(ax, nse_[m].clip(-1, 1), "viridis", nrm, area)
            fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=nrm), ax=ax, shrink=0.8, label="NSE (clipped at -1)")
            ax.set_title(f"{h} h: {title}; median {np.nanmedian(nse_[m]):.3f}", color=NAVY, fontsize=11)
        save(fig, out, f"r4_reach_maps_{h}h.png")
        fig, axes = maps.figure(2, 1, width=10)
        for ax, (other, title, lim) in zip(axes[:, 0], (("NWM", "GNN minus NWM forecast", 0.5), ("LSTM", "GNN minus LSTM", 1.0))):
            diff = nse_["GNN"].clip(-1, 1) - nse_[other].clip(-1, 1)
            nrm = TwoSlopeNorm(0, -lim, lim)
            maps.frame(ax)
            maps.rivers(ax, diff, "RdBu", nrm, area)
            fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu", norm=nrm), ax=ax, shrink=0.8, label="NSE difference")
            better = np.nanmean((nse_["GNN"] - nse_[other]).to_numpy() > 0)
            ax.set_title(f"{h} h: {title} (blue: GNN better, at {better:.0%} of reaches)", color=NAVY, fontsize=11)
        save(fig, out, f"r4_reach_gain_maps_{h}h.png")


# ------------------------------------------------------------------ Hurricane Helene
def helene(d, out, maps):
    """h1: GEFS rain and forecast peaks at every reach issued 2024-09-25; h2: hydrographs."""
    h = d.helene
    ht = pd.DatetimeIndex(h["time"])
    comid = pd.Index(h["comid"])
    area = d.area.reindex(comid)
    area.index = comid
    mean_flow = np.load(P.HYDRO / "reach_stats.npz")["mean_raw"]
    mean_flow = np.where(mean_flow >= 0.05, mean_flow, np.nan)  # near-dry reaches: ratio undefined
    j = ht.get_loc(pd.Timestamp("2024-09-25"))
    first5 = slice(0, 20)
    panels = (
        (pd.Series(h["gefs_rain"][j, first5].sum(0), index=comid), "GEFS control rain forecast over 5 days (mm)", "Blues", None, "mm"),
        (pd.Series(np.nanmax(h["fold"][j, first5], axis=0) / mean_flow, index=comid), "GNN with GEFS rain (held-out basins without their gauge)", "magma_r", LogNorm(1, 100), "peak / long-term mean flow"),
        (pd.Series(np.nanmax(h["model_aorc"][j, first5], axis=0) / mean_flow, index=comid), "GNN with the rain that fell (diagnostic)", "magma_r", LogNorm(1, 100), "peak / long-term mean flow"),
    )
    fig, axes = maps.figure(3, 1, width=10)
    for ax, (v, title, cmap, nrm, unit) in zip(axes[:, 0], panels):
        nrm = nrm or plt.Normalize(0, float(np.nanpercentile(v, 99.5)))
        maps.frame(ax)
        maps.rivers(ax, v, cmap, nrm, area)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=nrm), ax=ax, shrink=0.8, label=unit)
        ax.set_title(title, color=NAVY, fontsize=11)
    fig.suptitle("Hurricane Helene: issued 2024-09-25 00Z, two days before the peak, at all 58,834 reaches", color=NAVY, fontweight="bold")
    save(fig, out, "h1_helene_maps.png")
    s = d.strength
    peak = (d.observed["2024-09-26":"2024-09-30"].max() / d.p99)[d.cat.camelsh & (d.fold > 0)].dropna().sort_values(ascending=False).head(4)
    issue_index = {t: i for i, t in enumerate(d.times)}
    reach_of = comid.get_indexer(d.cat.comid)
    fig, axes = plt.subplots(len(peak), 2, figsize=(14, 3.1 * len(peak)), sharey="row", squeeze=False)
    for row, st in zip(axes, peak.index):
        gi = d.names.get_loc(st)
        for ax, day in zip(row, ("2024-09-26", "2024-09-24")):
            t0 = pd.Timestamp(day)
            i, k = issue_index[t0], ht.get_loc(t0)
            lead_t = t0 + pd.to_timedelta(LEAD_H, unit="h")
            w = d.observed[st][t0 - pd.Timedelta(days=2) : t0 + pd.Timedelta(days=10)]
            ax.plot(w.index, w.values, color="black", lw=2.2, label="observed")
            ax.plot(lead_t, s["shown_mu"][i, :, gi], color=BLUE, lw=2, label="GNN, recent flow shown")
            ax.plot(lead_t, s["fold_mu"][i, :, gi], ":", color=BLUE, lw=2, label="GNN, gauge never seen")
            ax.plot(lead_t, d.gauged["LSTM"][i, :, gi], color=ORANGE, lw=1.8, label="LSTM, recent flow shown")
            ax.plot(lead_t, d.gauged["NWM"][i, :, gi], color=GRAY, lw=1.8, label="NWM")
            ax.plot(lead_t, h["model_aorc"][k, :, reach_of[gi]], "--", color=GREEN, lw=1.6, label="GNN with the rain that fell")
            ax.axvline(t0, color=ORANGE, lw=1, ls=":")
            ax.set_title(f"issued {t0:%Y-%m-%d} 00Z", fontsize=10, color=NAVY)
            ax.tick_params(axis="x", labelsize=8, rotation=25)
        row[0].set_ylabel(d.label(gi).replace(", ", "\n", 1) + "\nm3/s", fontsize=9)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Hurricane Helene at the hardest-hit held-out gauges", color=NAVY, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "h2_helene_hydrographs.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=P.ROOT / "figures/strict_rule")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = Data()
    maps = Maps(d.reach["comid"])
    by_lead(d, a.out)
    summary(d, a.out)
    gain_and_wins(d, a.out)
    scatter(d, a.out)
    distribution(d, a.out)
    year_season(d, a.out)
    gain_vs_area(d, a.out)
    flood_skill(d, a.out)
    events = flood_events(d)
    flood_peaks(d, a.out, events)
    hydrographs(d, a.out, events, "gauged")
    hydrographs(d, a.out, events, "ungauged")
    full_metrics(d, a.out)
    value_of_gauge(d, a.out)
    tail(d, a.out, events)
    skill_lost(d, a.out)
    gauge_maps(d, a.out, maps)
    reach_by_lead(d, a.out)
    reach_by_area(d, a.out)
    reach_distribution(d, a.out)
    reach_maps(d, a.out, maps)
    helene(d, a.out, maps)


if __name__ == "__main__":
    main()
