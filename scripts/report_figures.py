"""The results report's figures and tables: the GNN (models/gnn_B and folds) against the LSTM and NWM.

    python scripts/report_figures.py --out /data/tva_data/figures/final_report

A small set of figures, one message each, plus tables.json with every number the report quotes.
Data, scoring and masks are those of scripts/strict_rule_figures.py (imported): every method is
scored on the same issues, gauges and values under the forecast rule (observed data before the
00Z issue, GEFSv12 rain after it). "Days 1-5" here is each gauge's (or reach's) mean NSE over
24, 48, 72, 96 and 120 h, then the median over gauges (or reaches), with a 95% bootstrap interval.
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from strict_rule_figures import (BLUE, D15, GRAY, GREEN, LEAD_H, NAVY, ORANGE, PURPLE, REACH, Data, Maps, flood_events,
                                 lead_axis, metrics, nse, plt, save)
from tva_flood.data import tva
from tva_flood.hydro import paths as P

METHODS = ("GNN", "LSTM", "NWM")
COLOR = {"GNN": BLUE, "LSTM": ORANGE, "NWM": GRAY}
SKY, INK = "#8CB8EC", "#1F2A44"  # GNN upper quantiles
RNG = np.random.default_rng(0)
LEADS = (6, 24, 72, 120, 240)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12, "legend.fontsize": 10.5,
                     "axes.grid": True, "grid.color": "#E3E7EE", "grid.linewidth": 0.7, "axes.axisbelow": True})


def boot_median(x, n=2000):
    """Median of x and its 95% bootstrap interval."""
    x = x[np.isfinite(x)]
    b = np.median(x[RNG.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def days15(x):
    return np.nanmean(x[D15], axis=0)


def band(x, n=1000):
    """Per-lead median over sites [lead, site] and its 95% bootstrap interval (sites scored at every lead)."""
    x = x[:, np.isfinite(x).all(axis=0)]
    b = np.array([np.median(x[:, RNG.integers(0, x.shape[1], x.shape[1])], axis=1) for _ in range(n)])
    return np.median(x, axis=1), *np.percentile(b, [2.5, 97.5], axis=0)


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=10.5, color=color, fontweight="bold")


def top_legend(fig, ax, ncol=3):
    """One legend above the panels, taken from ax."""
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=ncol, frameon=False)


def dates(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", labelsize=9.5)


# ------------------------------------------------------------------ scores
class Scores:
    """Per-site NSE [lead, site] for each protocol and method, on the gauges every method scores."""

    def __init__(self, d):
        self.d = d
        self.nse = {}
        for protocol, series, truth in (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u)):
            self.nse[protocol] = {m: nse(series[m], truth) for m in METHODS}
        self.nse["gauged"]["persistence"] = nse(d.gauged["persistence"], d.truth_g)
        self.aorc = {p: {m: nse(s[m], t) for m in ("GNN", "LSTM")} for p, s, t in (("gauged", d.gauged_aorc, d.truth_g), ("ungauged", d.ungauged_aorc, d.truth_u))}
        self.d15 = {p: {m: days15(v) for m, v in x.items()} for p, x in self.nse.items()}

    def gauges(self, protocol):
        """Gauges scored by every method in this protocol (days 1-5)."""
        return np.all([np.isfinite(self.d15[protocol][m]) for m in METHODS], axis=0)


def reach_d15(d, m):
    return np.nanmean(d.reach[f"{REACH[m]}/nse"][D15], axis=0)


# ------------------------------------------------------------------ figures
def f01_scoreboard(d, s, out, t, hw=None, free=None, name="f01_scoreboard.png", key="scoreboard"):
    """The three tests side by side: days 1-5 median NSE with 95% intervals; optionally only the gauges in hw and the
    reaches in free."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.28))
    scoreboard = {}
    gauges = "headwater gauges" if hw is not None else "gauges"
    reach_title = "3. Reaches with no dam upstream\n" if free is not None else "3. Every reach\n"
    for ax, (test, title) in zip(axes, (("gauged", "1. Gauged\nrecent flow shown"), ("ungauged", "2. Ungauged\nheld-out gauges, never seen"),
                                         ("reach", reach_title + "against the NWM v3.0 retrospective"))):
        if test == "reach":
            vals = {m: reach_d15(d, m) for m in METHODS}
            keep = np.all([np.isfinite(v) for v in vals.values()], axis=0) & (True if free is None else free)
            n = f"{keep.sum():,} reaches scored by all three"
        else:
            keep = s.gauges(test) & (True if hw is None else hw)
            vals = s.d15[test]
            n = f"{keep.sum()} {gauges}"
        stats = {m: boot_median(vals[m][keep]) for m in METHODS}
        scoreboard[test] = dict(n=int(keep.sum()), **{m: dict(zip(("median", "lo", "hi"), stats[m])) for m in METHODS})
        lo_lim = -1.0 if test == "reach" else 0
        for i, m in enumerate(METHODS):
            med, lo, hi = stats[m]
            shown = max(med, lo_lim)
            ax.bar(i, shown, width=0.62, color=COLOR[m], edgecolor="white", lw=2)
            if med >= lo_lim:
                ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="none", ecolor=NAVY, lw=1.2, capsize=4)
                up = med >= 0
                ax.annotate(f"{med:.3f}", (i, max(hi, 0) if up else lo), xytext=(0, 4 if up else -4), textcoords="offset points", ha="center",
                            va="bottom" if up else "top", fontsize=11, color=NAVY, fontweight="bold")
            else:
                ax.annotate(f"{med:.2f}\n(off scale)", (i, lo_lim), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, color=NAVY, fontweight="bold")
        ax.set_xticks(range(3), METHODS)
        ax.set_ylim(lo_lim, 0.55 if test != "reach" else 0.3)
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.set_title(f"{title}\n({n})", color=NAVY, fontsize=11.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median NSE, days 1-5\n(95% bootstrap interval)")
    save(fig, out, name)
    t[key] = scoreboard


def f02_by_lead(d, s, out, t, hw=None, name="f02_skill_by_lead.png"):
    """Median NSE by lead, gauged and ungauged, with 95% intervals; optionally only the gauges in hw."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, protocol, title in zip(axes, ("gauged", "ungauged"), ("Gauged: recent flow shown", "Ungauged: held-out gauges, never seen")):
        keep = s.gauges(protocol) & (True if hw is None else hw)
        for m in METHODS:
            med, lo, hi = band(s.nse[protocol][m][:, keep])
            ax.fill_between(LEAD_H, lo, hi, color=COLOR[m], alpha=0.16, lw=0)
            ax.plot(LEAD_H, med, color=COLOR[m], lw=2.4, label=m)
            label_end(ax, 240, med[-1], m, COLOR[m], dy={"GNN": 6, "LSTM": -4, "NWM": 0}[m])
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.set_ylim(-0.45, 1)
        lead_axis(ax)
        ax.set_xlim(0, 270)
        ax.set_title(f"{title} ({keep.sum()} {'headwater gauges' if hw is not None else 'gauges'})", color=NAVY)
    axes[0].set_ylabel("median NSE over gauges")
    axes[0].legend(loc="lower left")
    save(fig, out, name)


def f03_wins(d, s, out, t, hw=None, name="f03_wins_by_lead.png", key="wins"):
    """Share of gauges where the GNN beats each baseline, by lead; optionally only the gauges in hw."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    wins = {}
    for ax, protocol in zip(axes, ("gauged", "ungauged")):
        keep = s.gauges(protocol) & (True if hw is None else hw)
        for other in ("LSTM", "NWM"):
            diff = (s.nse[protocol]["GNN"] - s.nse[protocol][other])[:, keep]
            diff = diff[:, np.isfinite(diff).all(axis=0)]
            w = (diff > 0).mean(axis=1) * 100
            b = np.array([(diff[:, RNG.integers(0, diff.shape[1], diff.shape[1])] > 0).mean(axis=1) * 100 for _ in range(1000)])
            ax.fill_between(LEAD_H, *np.percentile(b, [2.5, 97.5], axis=0), color=COLOR[other], alpha=0.16, lw=0)
            ax.plot(LEAD_H, w, color=COLOR[other], lw=2.4, label=f"GNN better than {other}")
            wins[f"{protocol}/{other}"] = {f"{h}h": float(w[h // 6 - 1]) for h in LEADS}
            d15 = (s.d15[protocol]["GNN"] - s.d15[protocol][other])[keep]
            wins[f"{protocol}/{other}"]["days 1-5"] = float((d15 > 0).mean() * 100)
            wins[f"{protocol}/{other}"]["n"] = int(np.isfinite(d15).sum())
        ax.axhline(50, color=NAVY, lw=1, ls="--")
        ax.text(244, 51.5, "even", color=NAVY, fontsize=9.5, ha="right")
        ax.set_ylim(0, 100)
        lead_axis(ax)
        ax.set_title(f"{protocol.capitalize()} ({keep.sum()} {'headwater gauges' if hw is not None else 'gauges'})", color=NAVY)
    axes[0].set_ylabel("% of gauges where the GNN\nhas the higher NSE")
    axes[1].legend(loc="lower right")
    save(fig, out, name)
    t[key] = wins


def f04_scatter(d, s, out, t):
    """Per-gauge days 1-5 NSE: GNN against each baseline."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw=dict(hspace=0.32, wspace=0.25))
    for row, protocol in zip(axes, ("gauged", "ungauged")):
        keep = s.gauges(protocol)
        for ax, other in zip(row, ("LSTM", "NWM")):
            x, y = s.d15[protocol][other][keep], s.d15[protocol]["GNN"][keep]
            above = y > x
            ax.fill_between([-1, 1], [-1, 1], [1, 1], color=BLUE, alpha=0.05, lw=0)
            ax.scatter(np.clip(x, -1, 1), np.clip(y, -1, 1), s=34, color=np.where(above, BLUE, ORANGE if other == "LSTM" else GRAY), edgecolor="white", lw=0.6, zorder=3)
            ax.plot([-1, 1], [-1, 1], color=NAVY, lw=1)
            ax.set_xlim(-1.02, 1); ax.set_ylim(-1.02, 1); ax.set_aspect("equal")
            ax.set_xlabel(f"{other} NSE, days 1-5"); ax.set_ylabel("GNN NSE, days 1-5")
            ax.text(-0.95, 0.9, "GNN better", color=BLUE, fontsize=10.5, fontweight="bold")
            ax.text(0.95, -0.92, f"{other} better", color=ORANGE if other == "LSTM" else GRAY, fontsize=10.5, fontweight="bold", ha="right")
            ax.set_title(f"{protocol.capitalize()}: GNN better at {above.sum()} of {keep.sum()} gauges", color=NAVY, fontsize=11.5)
    fig.text(0.5, 0.005, "Values below -1 are drawn at -1.", ha="center", fontsize=9.5, color=GRAY)
    save(fig, out, "f04_gauge_scatter.png")


def f05_groups(d, s, out, t):
    """Days 1-5 median NSE by gauge type (TVA dams), gauged and ungauged."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    table = {}
    groups = [("all gauges", np.ones(len(d.names), bool))] + list(d.groups)
    for ax, protocol in zip(axes, ("gauged", "ungauged")):
        keep = s.gauges(protocol)
        labels = []
        for gi, (g, sel) in enumerate(groups):
            sel = sel & keep
            labels.append(f"{g.replace(', ', chr(10))}\n({sel.sum()})".replace("dam in the basin", "dam in\nthe basin"))
            for j, m in enumerate(METHODS):
                med, lo, hi = boot_median(s.d15[protocol][m][sel])
                table[f"{protocol}/{g}/{m}"] = dict(n=int(sel.sum()), median=med, lo=lo, hi=hi)
                x = gi + (j - 1) * 0.26
                ax.bar(x, med, width=0.25, color=COLOR[m], edgecolor="white", lw=1.5, label=m if gi == 0 else None)
                ax.errorbar(x, med, yerr=[[med - max(lo, -0.3)], [hi - med]], fmt="none", ecolor=NAVY, lw=1, capsize=2)
                if lo < -0.3:
                    ax.annotate(f"to {lo:.2f}", (x, -0.3), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=NAVY, rotation=90)
        ax.set_xticks(range(len(groups)), labels, fontsize=10)
        ax.set_ylim(-0.3, 0.65)
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.set_title(protocol.capitalize(), color=NAVY, fontweight="bold")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median NSE, days 1-5\n(95% interval)")
    top_legend(fig, axes[0])
    save(fig, out, "f05_by_gauge_type.png")
    t["groups"] = table


def f06_year_season(d, s, out, t):
    """Days 1-5 median NSE by test year and by season (issues in that period)."""
    seasons = {"Dec-Feb": (12, 1, 2), "Mar-May": (3, 4, 5), "Jun-Aug": (6, 7, 8), "Sep-Nov": (9, 10, 11)}
    periods = [(str(y), d.times.year == y) for y in range(2021, 2026)] + [(k, d.times.month.isin(v)) for k, v in seasons.items()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True, gridspec_kw=dict(wspace=0.06))
    table = {}
    for ax, (protocol, series, truth) in zip(axes, (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        keep = s.gauges(protocol)
        for j, m in enumerate(METHODS):
            vals = [float(np.nanmedian(days15(nse(series[m][k], truth[k], min_count=60))[keep])) for _, k in periods]
            table.update({f"{protocol}/{p}/{m}": v for (p, _), v in zip(periods, vals)})
            ax.bar(np.arange(len(periods)) + (j - 1) * 0.27, vals, width=0.26, color=COLOR[m], edgecolor="white", lw=1.2, label=m)
        ax.axvline(4.5, color=GRAY, lw=0.8, ls=":")
        ax.set_xticks(range(len(periods)), [p for p, _ in periods], rotation=35, fontsize=10)
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.set_title(f"{protocol.capitalize()}: by test year (left) and season (right)", color=NAVY)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median NSE, days 1-5")
    top_legend(fig, axes[0])
    save(fig, out, "f06_year_season.png")
    t["year_season"] = table


def f07_reach_by_lead(d, out, t):
    """Every reach: median NSE against the NWM retrospective by lead, four scopes."""
    r = d.reach
    scopes = (("all reaches", np.ones(len(r["reach_fold"]), bool)), ("reaches in held-out gauge basins", r["reach_fold"] > 0),
              ("reaches outside every gauge basin", r["reach_fold"] == -1), ("reaches behind dams (> 50% of basin)", d.regulated > 0.5))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.22, wspace=0.06))
    table = {}
    for ax, (scope, sel) in zip(axes.flat, scopes):
        for m in METHODS:
            med = np.nanmedian(r[f"{REACH[m]}/nse"][:, sel], axis=1)
            table.update({f"{scope}/{m}/{h}h": float(med[h // 6 - 1]) for h in LEADS})
            table[f"{scope}/{m}/days 1-5"] = float(np.nanmedian(np.nanmean(r[f"{REACH[m]}/nse"][D15][:, sel], axis=0)))
            if m != "LSTM":
                ax.plot(LEAD_H, med, color=COLOR[m], lw=2.4, label=m)
        table[f"{scope}/n"] = int(sel.sum())
        lstm = table[f"{scope}/LSTM/24h"]
        ax.text(0.98, 0.96, f"LSTM below -1 at every lead\n(24 h median {lstm:.1f})", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=ORANGE)
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.set_ylim(-1.0, 0.8)
        ax.set_title(f"{scope} ({sel.sum():,})", color=NAVY, fontsize=11.5)
    for ax in axes[1]:
        lead_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("median NSE against the\nNWM v3.0 retrospective")
    axes[0, 0].legend(loc="upper left", ncol=2)
    save(fig, out, "f07_reach_by_lead.png")
    t["reach_scopes"] = table


def f08_reach_by_area(d, out, t):
    """Every reach (held-out basins): median NSE by drainage area at 24 h and 72 h."""
    r = d.reach
    held = r["reach_fold"] > 0
    bins = np.logspace(-1, 4.25, 16)
    mid = np.sqrt(bins[:-1] * bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    table = {}
    for ax, h in zip(axes, (24, 72)):
        for m in METHODS:
            v = r[f"{REACH[m]}/nse"][h // 6 - 1]
            med = np.array([np.nanmedian(v[held & (d.reach_area >= lo) & (d.reach_area < hi)]) if (held & (d.reach_area >= lo) & (d.reach_area < hi)).sum() >= 30 else np.nan
                            for lo, hi in zip(bins[:-1], bins[1:])])
            ax.plot(mid, med, "o-", color=COLOR[m], lw=2.2, ms=5, label=m)
        for lo, hi, name in ((0, 50, "< 50"), (50, 500, "50-500"), (500, 5000, "500-5,000"), (5000, 1e9, "> 5,000")):
            sel = held & (d.reach_area >= lo) & (d.reach_area < hi)
            for m in METHODS:
                table[f"{h}h/{name}/{m}"] = float(np.nanmedian(r[f"{REACH[m]}/nse"][h // 6 - 1][sel]))
            table[f"{h}h/{name}/n"] = int(sel.sum())
        ax.set_xscale("log"); ax.set_ylim(-1, 1)
        ax.axhline(0, color=GRAY, lw=0.9)
        ax.set_xlabel("drainage area (km$^2$)")
        ax.set_title(f"{h} h ahead, reaches in held-out gauge basins", color=NAVY)
    axes[0].set_ylabel("median NSE against the\nNWM v3.0 retrospective")
    axes[0].legend(loc="upper left")
    fig.text(0.5, -0.08, "Points below -1 are off the plot (the LSTM at small and very large rivers).", ha="center", fontsize=9.5, color=GRAY)
    save(fig, out, "f08_reach_by_area.png")
    t["reach_area"] = table


def f09_reach_maps(d, out, maps, t):
    """Every reach at 24 h: the GNN's NSE and its gain over NWM."""
    comid = d.reach["comid"]
    area = pd.Series(d.reach_area, index=comid)
    h = 24
    g = pd.Series(d.reach["GNN+gefs/nse"][h // 6 - 1], index=comid)
    n = pd.Series(d.reach["NWM forecast/nse"][h // 6 - 1], index=comid)
    fig, axes = maps.figure(2, 1, width=10)
    ax = axes[0, 0]
    nrm = plt.Normalize(-1, 1)
    maps.frame(ax); maps.rivers(ax, g.clip(-1, 1), "viridis", nrm, area)
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=nrm), ax=ax, shrink=0.8, label="NSE (clipped at -1)")
    ax.set_title(f"GNN, 24 h ahead: NSE against the NWM retrospective (median {np.nanmedian(g):.3f})", color=NAVY, fontsize=11.5)
    ax = axes[1, 0]
    diff = g.clip(-1, 1) - n.clip(-1, 1)
    both = np.isfinite(g.to_numpy()) & np.isfinite(n.to_numpy())
    better = float(((g - n).to_numpy()[both] > 0).mean())
    nrm = TwoSlopeNorm(0, -0.5, 0.5)
    maps.frame(ax); maps.rivers(ax, diff, "RdBu", nrm, area)
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu", norm=nrm), ax=ax, shrink=0.8, label="NSE difference")
    ax.set_title(f"GNN minus NWM forecast, 24 h (blue: GNN closer, at {better:.0%} of {both.sum():,} reaches)", color=NAVY, fontsize=11.5)
    fig.savefig(out / "f09_reach_maps_24h.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote f09_reach_maps_24h.png", flush=True)
    wins = {}
    for hh in (24, 72):
        g_ = d.reach["GNN+gefs/nse"][hh // 6 - 1]
        for other in ("NWM", "LSTM"):
            o_ = d.reach[f"{REACH[other]}/nse"][hh // 6 - 1]
            ok = np.isfinite(g_) & np.isfinite(o_)
            wins[f"{hh}h/{other}"] = dict(share=float((g_[ok] > o_[ok]).mean()), n=int(ok.sum()))
    t["reach_wins"] = wins


def f10_flood_detection(d, s, out, t):
    """Critical success index for flows above the 90th and 99th training percentile."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey="row", gridspec_kw=dict(hspace=0.2, wspace=0.06))
    table = {}
    for col, (protocol, series, truth) in enumerate((("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u))):
        for row, (tag, thr) in enumerate((("90th", d.p90), ("99th", d.p99))):
            ax = axes[row, col]
            ok = np.isfinite(truth) & np.isfinite(thr)[None, None]
            for m in METHODS:
                o, f = truth > thr, series[m] > thr
                hit, miss, fa = (o & f & ok).sum((0, 2)), (o & ~f & ok).sum((0, 2)), (~o & f & ok).sum((0, 2))
                csi = hit / np.maximum(hit + miss + fa, 1)
                pod = hit / np.maximum(hit + miss, 1)
                far = fa / np.maximum(hit + fa, 1)
                ax.plot(LEAD_H, csi, color=COLOR[m], lw=2.4, label=m)
                table.update({f"{protocol}/{tag}/{m}/{h}h": dict(csi=float(csi[h // 6 - 1]), pod=float(pod[h // 6 - 1]), far=float(far[h // 6 - 1])) for h in (24, 72, 120, 240)})
            ax.set_ylim(0, None)
            ax.set_title(f"{protocol.capitalize()}: flow above the {tag} percentile", color=NAVY)
            if row == 1:
                lead_axis(ax)
            if col == 0:
                ax.set_ylabel("critical success index\nhits / (hits + misses + false alarms)")
    axes[0, 0].legend()
    save(fig, out, "f10_flood_detection.png")
    t["flood_detection"] = table


def f11_peaks(d, out, events, t):
    """Median flood-peak error, issued 1, 3 and 5 days before the peak, from the mean forecast and the tail quantiles."""
    st = d.strength
    held = d.fold > 0
    cases = (("gauged", {"GNN": st["shown_mu"], "GNN q90": st["shown_q"][..., 1], "GNN q99": st["shown_q"][..., 2], "LSTM": d.gauged["LSTM"], "NWM": d.gauged["NWM"]}),
             ("ungauged", {"GNN": st["fold_mu"], "GNN q90": st["fold_q"][..., 1], "GNN q99": st["fold_q"][..., 2], "LSTM": d.ungauged["LSTM"], "NWM": d.gauged["NWM"]}))
    names = ("GNN", "GNN q90", "GNN q99", "LSTM", "NWM")
    pcolor = {"GNN": BLUE, "GNN q90": SKY, "GNN q99": INK, "LSTM": ORANGE, "NWM": GRAY}
    issue_index = {x: i for i, x in enumerate(d.times)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, gridspec_kw=dict(wspace=0.06))
    table = {}
    for ax, (protocol, fc) in zip(axes, cases):
        errs = {(n, k): [] for n in names for k in (1, 3, 5)}
        for gi, t0, q in events:
            if protocol == "ungauged" and not held[gi]:
                continue
            for k in (1, 3, 5):
                i = issue_index.get((t0 - pd.Timedelta(days=k)).normalize())
                if i is None:
                    continue
                target = int((t0 - d.times[i]) / pd.Timedelta(hours=6)) - 1
                lo, hi = max(target - 8, 0), min(target + 9, 40)
                w = {n: v[i, lo:hi, gi] for n, v in fc.items()}
                if all(len(x) and np.isfinite(x).all() for x in w.values()):
                    for n, x in w.items():
                        errs[(n, k)].append(x.max() / q - 1)
        for kk, k in enumerate((1, 3, 5)):
            for j, n in enumerate(names):
                v = np.median(errs[(n, k)]) * 100
                table[f"{protocol}/{n}/{k}d"] = float(v)
                x = kk * 6.2 + j
                ax.bar(x, v, width=0.88, color=pcolor[n], edgecolor="white", lw=1.2, label=n if kk == 0 else None)
                ax.annotate(f"{v:+.0f}", (x, v), xytext=(0, -12 if v < 0 else 3), textcoords="offset points", ha="center", fontsize=8.5, color=NAVY)
        table[f"{protocol}/floods"] = len(errs[("NWM", 1)])
        ax.set_xticks([2, 8.2, 14.4], ["issued 1 day\nbefore the peak", "3 days before", "5 days before"])
        ax.axhline(0, color=NAVY, lw=1.1)
        ax.set_ylim(-100, 45)
        ax.set_title(f"{protocol.capitalize()}: {len(errs[('NWM', 1)]):,} floods above the 99th percentile", color=NAVY)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median forecast peak error (%)\n0 = right size, -50 = half the peak")
    top_legend(fig, axes[0], ncol=5)
    save(fig, out, "f11_flood_peaks.png")
    t["peaks"] = table


def f12_hydrographs(d, out, events, t):
    """Three typical large floods (1.3-3 times the 99th percentile, basins over 300 km2, distinct held-out gauges), issued
    2 days before the peak, gauged and ungauged; the largest floods are flash floods every model misses."""
    st = d.strength
    issue_index = {x: i for i, x in enumerate(d.times)}
    cand = pd.DataFrame(events, columns=["gi", "peak", "q"]).assign(ratio=lambda e: e.q / d.p99[e.gi.to_numpy()])
    helene = (cand.peak >= "2024-09-20") & (cand.peak <= "2024-10-10")  # Helene has its own figure
    cand = cand[~helene & (d.fold[cand.gi.to_numpy()] > 0) & [(p - pd.Timedelta(days=2)).normalize() in issue_index for p in cand.peak]]
    cand = cand[(cand.ratio >= 1.3) & (cand.ratio <= 3) & (d.gauge_area[cand.gi.to_numpy()] > 300)]
    cand = cand.sort_values("ratio", ascending=False).drop_duplicates("gi").head(3)
    fig, axes = plt.subplots(len(cand), 2, figsize=(13, 3.4 * len(cand)), sharey="row", squeeze=False)
    for row, (_, e) in zip(axes, cand.iterrows()):
        gi, peak = int(e.gi), e.peak
        issue = (peak - pd.Timedelta(days=2)).normalize()
        i = issue_index[issue]
        lead_t = issue + pd.to_timedelta(LEAD_H, unit="h")
        obs = d.observed[d.names[gi]][issue - pd.Timedelta(days=3): issue + pd.Timedelta(days=10)]
        for ax, protocol, series, gnn in ((row[0], "gauged", d.gauged, st["shown_mu"]), (row[1], "ungauged", d.ungauged, st["fold_mu"])):
            ax.plot(obs.index, obs.values, color="black", lw=2.2, label="observed")
            ax.plot(lead_t, gnn[i, :, gi], color=BLUE, lw=2.2, label="GNN")
            ax.plot(lead_t, series["LSTM"][i, :, gi], color=ORANGE, lw=2, label="LSTM")
            ax.plot(lead_t, series["NWM"][i, :, gi], color=GRAY, lw=2, label="NWM")
            ax.axvline(issue, color=NAVY, lw=1, ls=":")
            dates(ax)
            ax.set_title(f"{protocol}, issued {issue:%Y-%m-%d} 00Z", fontsize=11, color=NAVY)
        row[0].set_ylabel(d.label(gi).replace(", ", "\n", 1).replace("km2", "km$^2$") + "\nflow (m$^3$/s)", fontsize=10)
    axes[0, 0].legend(fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "f12_flood_hydrographs.png")
    t["hydrograph_gauges"] = [f"{d.label(int(g))}, peak {p:%Y-%m-%d}" for g, p in zip(cand.gi, cand.peak)]


def event_scores(d, events):
    """Peak size error, peak timing error and detection (forecast above the 90th percentile within +-48 h of the peak)
    for each event, forecast issued 1, 3 and 5 days before, gauged and ungauged; one row per event, method and issue."""
    issue_index = {x: i for i, x in enumerate(d.times)}
    rows = []
    for protocol, series in (("gauged", d.gauged), ("ungauged", d.ungauged)):
        for gi, peak, q in events:
            if protocol == "ungauged" and d.fold[gi] == 0:
                continue
            for k in (1, 3, 5):
                i = issue_index.get((peak - pd.Timedelta(days=k)).normalize())
                if i is None:
                    continue
                target = int((peak - d.times[i]) / pd.Timedelta(hours=6)) - 1
                lo, hi = max(target - 8, 0), min(target + 9, 40)
                w = {m: series[m][i, lo:hi, gi] for m in METHODS}
                if not all(len(x) and np.isfinite(x).all() for x in w.values()):
                    continue
                for m, x in w.items():
                    rows.append(dict(protocol=protocol, method=m, days=k, gauge=gi, peak=peak, error=x.max() / q - 1,
                                     timing=abs(lo + int(np.argmax(x)) - target) * 6, detected=bool(x.max() > d.p90[gi])))
    return pd.DataFrame(rows)


def f13_helene(d, out, t):
    """Hurricane Helene at the two hardest-hit held-out gauges."""
    st, h = d.strength, d.helene
    ht = pd.DatetimeIndex(h["time"])
    comid = pd.Index(h["comid"])
    reach_of = comid.get_indexer(d.cat.comid)
    peak = (d.observed["2024-09-26":"2024-09-30"].max() / d.p99)[d.cat.camelsh & (d.fold > 0)].dropna().sort_values(ascending=False).head(2)
    issue_index = {x: i for i, x in enumerate(d.times)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharey="row")
    for row, stn in zip(axes, peak.index):
        gi = d.names.get_loc(stn)
        for ax, day in zip(row, ("2024-09-24", "2024-09-26")):
            t0 = pd.Timestamp(day)
            i, k = issue_index[t0], ht.get_loc(t0)
            lead_t = t0 + pd.to_timedelta(LEAD_H, unit="h")
            w = d.observed[stn][t0 - pd.Timedelta(days=2): t0 + pd.Timedelta(days=8)]
            ax.plot(w.index, w.values, color="black", lw=2.4, label="observed")
            ax.plot(lead_t, st["shown_mu"][i, :, gi], color=BLUE, lw=2.2, label="GNN, recent flow shown")
            ax.plot(lead_t, st["fold_mu"][i, :, gi], ":", color=BLUE, lw=2.2, label="GNN, gauge never seen")
            ax.plot(lead_t, st["shown_q"][i, :, gi, 2], color=INK, lw=1.4, ls="--", label="GNN 99th-percentile flow")
            ax.plot(lead_t, d.gauged["LSTM"][i, :, gi], color=ORANGE, lw=2, label="LSTM")
            ax.plot(lead_t, d.gauged["NWM"][i, :, gi], color=GRAY, lw=2, label="NWM")
            ax.plot(lead_t, h["model_aorc"][k, :, reach_of[gi]], "--", color=GREEN, lw=1.6, label="GNN with the rain that fell")
            ax.axvline(t0, color=NAVY, lw=1, ls=":")
            ax.set_xlim(w.index[0], t0 + pd.Timedelta(days=8))
            ax.set_title(f"issued {t0:%Y-%m-%d} 00Z", fontsize=11, color=NAVY)
            dates(ax)
        row[0].set_ylabel(d.label(gi).replace(", ", "\n", 1).replace("km2", "km$^2$") + "\nflow (m$^3$/s)", fontsize=10)
    axes[0, 1].legend(fontsize=9, loc="upper right")
    fig.suptitle("Hurricane Helene (peak 27-28 September 2024) at the hardest-hit held-out gauges", color=NAVY, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "f13_helene.png")
    t["helene_gauges"] = {d.label(d.names.get_loc(k)): float(v) for k, v in peak.items()}


# ------------------------------------------------------------------ headwater basins (no TVA dam upstream)
def add_subsets(d):
    """d.headwater: gauges with no TVA dam and no other gauge in the basin; d.free: reaches with no dam upstream."""
    d.headwater = dict(d.groups)["headwater, no dam"]
    d.free = d.regulated <= 0


def h1_dam_map(d, s, out, maps, t):
    """Where the TVA dams reach: reaches with a dam upstream, the dams, and the scored headwater and other gauges."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    comid = d.reach["comid"]
    area = pd.Series(d.reach_area, index=comid)
    fig, axes = maps.figure(1, 1, width=11, colorbar=False)
    ax = axes[0, 0]
    maps.frame(ax)
    maps.rivers(ax, pd.Series((~d.free).astype(float), index=comid), ListedColormap(["#9DB4CF", "#E8A27C"]), plt.Normalize(0, 1), area)
    with np.load(P.HYDRO / "dams.npz", allow_pickle=True) as z:
        dam_xy = maps.points(z["comid"])
    gauge_xy = maps.points(d.cat.comid)
    camels = s.gauges("gauged") | s.gauges("ungauged")  # gauges scored in either test
    other = camels & ~d.headwater
    ax.scatter(*gauge_xy[other].T, s=34, color="white", edgecolor=GRAY, lw=1, zorder=3)
    ax.scatter(*gauge_xy[camels & d.headwater].T, s=44, color=BLUE, edgecolor="white", lw=0.8, zorder=4)
    ax.scatter(*dam_xy.T, s=70, marker="^", color=NAVY, edgecolor="white", lw=0.6, zorder=5)
    handles = [Line2D([], [], color="#9DB4CF", lw=2.5, label=f"reach with no dam upstream ({d.free.sum():,})"),
               Line2D([], [], color="#E8A27C", lw=2.5, label=f"reach with a TVA dam upstream ({(~d.free).sum():,})"),
               Line2D([], [], marker="^", ls="", color=NAVY, ms=9, label=f"TVA dam ({len(dam_xy)})"),
               Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label=f"headwater gauge: no dam, no gauge upstream ({int((camels & d.headwater).sum())})"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, ms=8, label=f"other gauge, left out here ({int(other.sum())})")]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5, frameon=True)
    fig.savefig(out / "h1_dam_map.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote h1_dam_map.png", flush=True)
    t["dam_map"] = dict(free_reaches=int(d.free.sum()), dammed_reaches=int((~d.free).sum()), dams=len(dam_xy),
                        headwater_gauges=int((camels & d.headwater).sum()), other_gauges=int(other.sum()))


def headwater_tables(d, s, events, t):
    """The report's numbers for headwater gauges and dam-free reaches."""
    hw, r = d.headwater, d.reach
    lead, kge, csi = {}, {}, {}
    for protocol, series, truth in (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u)):
        keep = s.gauges(protocol) & hw
        for m in METHODS:
            x = s.nse[protocol][m]
            med = np.nanmedian(x[:, keep], axis=1)
            lead[f"{protocol}/{m}"] = {**{f"{h}h": float(med[h // 6 - 1]) for h in LEADS}, "days 1-5": float(np.nanmedian(days15(x)[keep])), "n": int(keep.sum())}
            k = metrics(series[m], truth)
            kge[f"{protocol}/{m}"] = {key: float(np.nanmedian(days15(k[key])[keep])) for key in ("nse", "kge", "r", "alpha", "beta")}
        sub = np.where(hw[None, None], truth, np.nan)
        for tag, thr in (("90th", d.p90), ("99th", d.p99)):
            ok = np.isfinite(sub) & np.isfinite(thr)[None, None]
            for m in METHODS:
                o, f_ = sub > thr, series[m] > thr
                hit, miss, fa = (o & f_ & ok).sum((0, 2)), (o & ~f_ & ok).sum((0, 2)), (~o & f_ & ok).sum((0, 2))
                for h in (24, 72):
                    csi[f"{protocol}/{tag}/{m}/{h}h"] = float(hit[h // 6 - 1] / max(hit[h // 6 - 1] + miss[h // 6 - 1] + fa[h // 6 - 1], 1))
    peaks = event_scores(d, [e for e in events if hw[e[0]]])
    pk = {f"{p}/{m}/{k}d": dict(error=float(g.error.median() * 100), timing=float(g.timing.median()), events=len(g))
          for (p, m, k), g in peaks.groupby(["protocol", "method", "days"])}
    reach = {}
    for h in (24, 72, 120):
        for scope, sel in (("all", d.free), ("< 50", d.free & (d.reach_area < 50)), ("50-500", d.free & (d.reach_area >= 50) & (d.reach_area < 500)),
                           ("> 500", d.free & (d.reach_area >= 500))):
            reach[f"{h}h/{scope}/n"] = int(sel.sum())
            for m in METHODS:
                reach[f"{h}h/{scope}/{m}"] = float(np.nanmedian(r[f"{REACH[m]}/nse"][h // 6 - 1][sel]))
        g_, n_ = r["GNN+gefs/nse"][h // 6 - 1], r["NWM forecast/nse"][h // 6 - 1]
        ok = d.free & np.isfinite(g_) & np.isfinite(n_)
        reach[f"{h}h/win_vs_NWM"] = float((g_[ok] > n_[ok]).mean())
    t["headwater"] = dict(by_lead=lead, kge=kge, csi=csi, peaks=pk, reach=reach)


# ------------------------------------------------------------------ TVA's own forecasts at its headwater gauges
TVA_COLOR = PURPLE


def tva_forecasts(d):
    """TVA's issued flow forecasts on the evaluation's issues [issue, lead, gauge] (NaN where TVA has none). A TVA value
    at lead L is the six-hour mean ending at issue + L, so TVA lead 6(k+1) h is model lead index k (data.tva)."""
    out = np.full((len(d.times), 40, len(d.names)), np.nan)
    with tempfile.TemporaryDirectory() as tmp:
        folder = tva.extract(P.ROOT / "raw/tva/TVA_headwater.zip", tmp)
        for f in sorted((folder / "forecast").glob("*.nc")):
            if f.stem not in d.names:
                continue
            with xr.open_dataset(f) as z:
                if "river_flow" not in z:
                    continue
                rows = d.times.get_indexer(pd.DatetimeIndex(z.forecast_time.values))
                leads = list(z.lead_time.values.astype(int))
                cols = [leads.index(6 * (k + 1)) for k in range(40)]
                flow = z.river_flow.values[:, cols]
            ok = rows >= 0
            if not ok.any():
                continue
            out[rows[ok], :, d.names.get_loc(f.stem)] = flow[ok]
    return out


RAIN_TVA, RAIN_BLEND, RAIN_TVA_NEVER = "GNN, TVA rain", "GNN, TVA rain to 72 h then GEFS", "GNN, gauge never seen, TVA rain"


def tva_rain_series(d):
    """The GNN run with TVA's rain at TVA gauges (models/gnn_tva_rain), on the evaluation's issues; NaN where TVA
    had no full rain forecast. Empty when the run is missing."""
    path = P.MODELS / "gnn_tva_rain/gauges.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    assert list(z["stations"]) == list(d.names)
    rows = pd.DatetimeIndex(z["time"]).get_indexer(d.times)
    out = {}
    for name, key in ((RAIN_TVA, "tva_shown"), (RAIN_BLEND, "tva_gefs_shown")):
        v = np.full((len(d.times), 40, len(d.names)), np.nan)
        v[rows >= 0] = np.where(z["rain_ok"][:, None], z[key], np.nan)[rows[rows >= 0]]
        out[name] = v
    return out


def tva_rain_figure(scores, keep, wins, color, out):
    """t3: the GNN with GEFS and with TVA's rain against TVA's forecast: skill by lead and gauges won."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.22))
    lines = (("TVA", "TVA forecast", "-"), ("GNN", "GNN, GEFS rain", "-"), (RAIN_TVA, RAIN_TVA, "-"), (RAIN_BLEND, RAIN_BLEND, "--"),
             )
    for m, label, ls in lines:
        med, lo, hi = band(scores[m][:, keep])
        if ls == "-":
            axes[0].fill_between(LEAD_H, lo, hi, color=color[m], alpha=0.14, lw=0)
        axes[0].plot(LEAD_H, med, ls, color=color[m], lw=2.4, label=label)
        if m != "TVA":
            axes[1].plot(LEAD_H, wins[m]["w"], ls, color=color[m], lw=2.4, label=label)
            if ls == "-":
                axes[1].fill_between(LEAD_H, *np.percentile(wins[m]["b"], [2.5, 97.5], axis=0), color=color[m], alpha=0.14, lw=0)
    for ax in axes:
        lead_axis(ax)
        ax.axvline(72, color=GRAY, lw=0.8, ls=":")
    axes[0].text(74, 0.93, "TVA rain\nends", fontsize=9, color=GRAY, va="top")
    axes[0].axhline(0, color=GRAY, lw=0.9); axes[0].set_ylim(-0.45, 1)
    axes[0].set_ylabel("median NSE over gauges"); axes[0].legend(loc="lower left", fontsize=9)
    axes[0].set_title(f"Skill at {keep.sum()} TVA headwater gauges", color=NAVY)
    axes[1].axhline(50, color=NAVY, lw=1, ls="--"); axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("% of TVA gauges where the GNN\nhas a higher NSE than TVA's forecast")
    axes[1].set_title("Gauge by gauge against TVA's forecast", color=NAVY)
    save(fig, out, "t3_tva_rain.png")


def tva_section(d, s, out, events, t):
    """TVA's forecast against the GNN, LSTM and NWM at TVA's headwater gauges (recent flow shown to every model), plus
    the GNN that never saw the gauge; same issues, gauges and values for every method."""
    tv = tva_forecasts(d)
    series = {"GNN": d.gauged["GNN"], "LSTM": d.gauged["LSTM"], "NWM": d.gauged["NWM"], "TVA": tv,
              "GNN, gauge never seen": d.ungauged["GNN"]}
    color = {"GNN": BLUE, "LSTM": ORANGE, "NWM": GRAY, "TVA": TVA_COLOR, "GNN, gauge never seen": BLUE}
    series.update(tva_rain_series(d))  # the GNN with TVA's rain (scripts/tva_rain_forecasts.py)
    color.update({RAIN_TVA: GREEN, RAIN_BLEND: NAVY})
    first = ("GNN", "LSTM", "NWM", "TVA", "GNN, gauge never seen")
    truth = Data.common(d.truth_g, series)
    scores = {m: nse(v, truth) for m, v in series.items()}
    keep = np.all([np.isfinite(days15(x)) for x in scores.values()], axis=0)
    # t1: skill by lead, and the share of gauges where each model beats TVA
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.22))
    ax = axes[0]
    for m in first:
        x = scores[m]
        med, lo, hi = band(x[:, keep])
        ls = ":" if "never" in m else "-"
        if "never" not in m:
            ax.fill_between(LEAD_H, lo, hi, color=color[m], alpha=0.14, lw=0)
        ax.plot(LEAD_H, med, ls, color=color[m], lw=2.4, label=m)
    ax.axhline(0, color=GRAY, lw=0.9); ax.set_ylim(-0.45, 1); lead_axis(ax)
    ax.set_ylabel("median NSE over gauges"); ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title(f"Skill at {keep.sum()} TVA headwater gauges", color=NAVY)
    ax = axes[1]
    raw, wins = {}, {}
    for m in [x for x in series if x != "TVA"]:
        diff = (scores[m] - scores["TVA"])[:, keep]
        w = (diff > 0).mean(axis=1) * 100
        b = np.array([(diff[:, RNG.integers(0, diff.shape[1], diff.shape[1])] > 0).mean(axis=1) * 100 for _ in range(1000)])
        raw[m] = dict(w=w, b=b)
        wins[m] = {**{f"{h}h": float(w[h // 6 - 1]) for h in LEADS}, "days 1-5": float(((days15(scores[m]) - days15(scores["TVA"]))[keep] > 0).mean() * 100)}
        if m not in first:
            continue
        if "never" not in m:
            ax.fill_between(LEAD_H, *np.percentile(b, [2.5, 97.5], axis=0), color=color[m], alpha=0.14, lw=0)
        ax.plot(LEAD_H, w, ":" if "never" in m else "-", color=color[m], lw=2.4, label=f"{m} better than TVA")
    ax.axhline(50, color=NAVY, lw=1, ls="--"); ax.set_ylim(0, 100); lead_axis(ax)
    ax.set_ylabel("% of TVA gauges where the model\nhas a higher NSE than TVA's forecast")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9.5)
    ax.set_title("Gauge by gauge against TVA's forecast", color=NAVY)
    save(fig, out, "t1_tva_skill.png")
    if RAIN_TVA in series:
        tva_rain_figure(scores, keep, raw, color, out)
    # tables
    table = dict(n=int(keep.sum()), wins=wins, groups={g: int((sel & keep).sum()) for g, sel in d.groups})
    table["by_lead"] = {m: {**{f"{h}h": float(np.nanmedian(x[h // 6 - 1][keep])) for h in LEADS},
                            "days 1-5": dict(zip(("median", "lo", "hi"), boot_median(days15(x)[keep])))} for m, x in scores.items()}
    aorc = nse(d.gauged_aorc["GNN"], truth)  # perfect-rain diagnostic on the same values
    table["by_lead"]["GNN, realized rain"] = {**{f"{h}h": float(np.nanmedian(aorc[h // 6 - 1][keep])) for h in LEADS},
                                              "days 1-5": dict(zip(("median", "lo", "hi"), boot_median(days15(aorc)[keep])))}
    table["kge"] = {}
    for m, v in series.items():
        k = metrics(v, truth)
        table["kge"][m] = {key: float(np.nanmedian(days15(k[key])[keep])) for key in ("nse", "kge", "r", "alpha", "beta")}
    sub = np.where(keep[None, None], truth, np.nan)
    table["csi"] = {}
    for tag, thr in (("90th", d.p90), ("99th", d.p99)):
        ok = np.isfinite(sub) & np.isfinite(thr)[None, None]
        for m, v in series.items():
            o, f_ = sub > thr, v > thr
            hit, miss, fa = (o & f_ & ok).sum((0, 2)), (o & ~f_ & ok).sum((0, 2)), (~o & f_ & ok).sum((0, 2))
            for h in (24, 72, 120):
                i = h // 6 - 1
                table["csi"][f"{tag}/{m}/{h}h"] = float(hit[i] / max(hit[i] + miss[i] + fa[i], 1))
    issue_index = {x: i for i, x in enumerate(d.times)}
    errs = {(m, k): [] for m in series for k in (1, 3, 5)}
    for gi, peak, q in events:
        if not keep[gi]:
            continue
        for k in (1, 3, 5):
            i = issue_index.get((peak - pd.Timedelta(days=k)).normalize())
            if i is None:
                continue
            target = int((peak - d.times[i]) / pd.Timedelta(hours=6)) - 1
            lo, hi = max(target - 8, 0), min(target + 9, 40)
            w = {m: v[i, lo:hi, gi] for m, v in series.items()}
            if all(len(x) and np.isfinite(x).all() for x in w.values()):
                for m, x in w.items():
                    errs[(m, k)].append(x.max() / q - 1)
    table["peaks"] = {f"{m}/{k}d": float(np.median(v) * 100) for (m, k), v in errs.items() if v}
    table["floods"] = len(errs[("TVA", 1)])
    # t2: the three largest test floods at TVA gauges (peaks 1.3-3 times the 99th percentile, outside Helene), issued 1 and 3 days before
    cand = pd.DataFrame([e for e in events if keep[e[0]]], columns=["gi", "peak", "q"]).assign(ratio=lambda e: e.q / d.p99[e.gi.to_numpy()])
    helene = (cand.peak >= "2024-09-20") & (cand.peak <= "2024-10-10")
    ok = [all((p - pd.Timedelta(days=k)).normalize() in issue_index for k in (1, 3)) for p in cand.peak]
    cand = cand[~helene & ok & (cand.ratio >= 1.3) & (cand.ratio <= 3)].sort_values("ratio", ascending=False).drop_duplicates("gi").head(3)
    fig, axes = plt.subplots(len(cand), 2, figsize=(13, 3.4 * len(cand)), sharey="row", squeeze=False)
    for row, (_, e) in zip(axes, cand.iterrows()):
        gi, peak = int(e.gi), e.peak
        for ax, k in zip(row, (1, 3)):
            issue = (peak - pd.Timedelta(days=k)).normalize()
            i = issue_index[issue]
            lead_t = issue + pd.to_timedelta(LEAD_H, unit="h")
            obs = d.observed[d.names[gi]][issue - pd.Timedelta(days=3): issue + pd.Timedelta(days=10)]
            ax.plot(obs.index, obs.values, color="black", lw=2.2, label="observed")
            for m in ("GNN", "LSTM", "NWM", "TVA") + ((RAIN_TVA,) if RAIN_TVA in series else ()):
                ax.plot(lead_t, series[m][i, :, gi], "--" if m == RAIN_TVA else "-", color=color[m], lw=2.2 if m in ("GNN", "TVA", RAIN_TVA) else 1.8,
                        label={"TVA": "TVA forecast", "GNN": "GNN (GEFS rain)"}.get(m, m))
            ax.axvline(issue, color=NAVY, lw=1, ls=":")
            dates(ax)
            ax.set_title(f"issued {issue:%Y-%m-%d} 00Z ({k} day{'s' if k > 1 else ''} before the peak)", fontsize=11, color=NAVY)
        row[0].set_ylabel(d.label(gi).replace(", ", "\n", 1).replace("km2", "km$^2$") + "\nflow (m$^3$/s)", fontsize=10)
    axes[0, 0].legend(fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "t2_tva_hydrographs.png")
    table["hydrograph_gauges"] = [f"{d.label(int(g))}, peak {p:%Y-%m-%d}" for g, p in zip(cand.gi, cand.peak)]
    t["tva"] = table


# ------------------------------------------------------------------ tables only
def tables(d, s, t):
    """Numbers quoted in the report that have no figure of their own."""
    lead = {}
    for protocol in ("gauged", "ungauged"):
        keep = s.gauges(protocol)
        for m, x in list(s.nse[protocol].items()) + [(f"{k}+realized rain", v) for k, v in s.aorc[protocol].items()]:
            med = np.nanmedian(x[:, keep], axis=1)
            lead[f"{protocol}/{m}"] = {**{f"{h}h": float(med[h // 6 - 1]) for h in LEADS}, "days 1-5": float(np.nanmedian(days15(x)[keep]))}
    t["by_lead"] = lead
    # KGE and its parts, median over gauges of the days 1-5 mean
    kge = {}
    for protocol, series, truth in (("gauged", d.gauged, d.truth_g), ("ungauged", d.ungauged, d.truth_u)):
        keep = s.gauges(protocol)
        for m in METHODS:
            x = metrics(series[m], truth)
            kge[f"{protocol}/{m}"] = {k: float(np.nanmedian(days15(x[k])[keep])) for k in ("nse", "kge", "r", "alpha", "beta")}
    t["kge"] = kge
    # the value of the gauge's own flow (held-out gauges, same values for every line)
    st = d.strength
    o = np.where(d.fold[None, None] > 0, st["truth"].astype(np.float64), np.nan)
    series = {"GNN, recent flow shown": st["shown_mu"], "GNN, no flow shown (all-gauge model)": st["hidden_mu"],
              "GNN, gauge never seen (fold model)": st["fold_mu"], "LSTM, recent flow shown": d.gauged["LSTM"],
              "LSTM, gauge never seen (fold model)": d.ungauged["LSTM"], "NWM": d.gauged["NWM"]}
    o = Data.common(o, series)
    t["value_of_gauge"] = {k: float(np.nanmedian(days15(nse(v, o)))) for k, v in series.items()}
    lost = {m: s.d15["gauged"][m] - s.d15["ungauged"][m] for m in ("GNN", "LSTM")}
    t["skill_lost"] = {m: float(np.nanmedian(v)) for m, v in lost.items()}
    # tail calibration: share of observations below each quantile, days 1-5
    cal = {}
    for protocol, q, truth in (("gauged", st["shown_q"], d.truth_g), ("ungauged", st["fold_q"], np.where(d.fold[None, None] > 0, d.truth_u, np.nan))):
        ok = np.isfinite(truth) & np.isfinite(q).all(-1)
        for i, qq in enumerate((0.5, 0.9, 0.99)):
            below = np.where(ok, truth <= q[..., i], 0).sum((0, 2)) / ok.sum((0, 2))
            cal[f"{protocol}/q{int(qq * 100)}"] = float(np.mean(below[3:20]))
    t["tail_calibration"] = cal
    t["counts"] = dict(issues=len(d.times), first=str(d.times[0].date()), last=str(d.times[-1].date()),
                       gauged=int(s.gauges("gauged").sum()), ungauged=int(s.gauges("ungauged").sum()), reaches=len(d.reach["comid"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=P.ROOT / "figures/final_report")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = Data()
    add_subsets(d)
    s = Scores(d)
    t = {}
    tables(d, s, t)
    f01_scoreboard(d, s, a.out, t)
    f02_by_lead(d, s, a.out, t)
    f03_wins(d, s, a.out, t)
    f04_scatter(d, s, a.out, t)
    f05_groups(d, s, a.out, t)
    f06_year_season(d, s, a.out, t)
    f07_reach_by_lead(d, a.out, t)
    f08_reach_by_area(d, a.out, t)
    f10_flood_detection(d, s, a.out, t)
    events = flood_events(d)
    f11_peaks(d, a.out, events, t)
    f12_hydrographs(d, a.out, events, t)
    f13_helene(d, a.out, t)
    headwater_tables(d, s, events, t)
    tva_section(d, s, a.out, events, t)
    f01_scoreboard(d, s, a.out, t, hw=d.headwater, free=d.free, name="h2_headwater_scoreboard.png", key="hw_scoreboard")
    f02_by_lead(d, s, a.out, t, hw=d.headwater, name="h3_headwater_by_lead.png")
    f03_wins(d, s, a.out, t, hw=d.headwater, name="h4_headwater_wins.png", key="hw_wins")
    (a.out / "tables.json").write_text(json.dumps(t, indent=1))
    maps = Maps(d.reach["comid"])
    f09_reach_maps(d, a.out, maps, t)
    h1_dam_map(d, s, a.out, maps, t)
    (a.out / "tables.json").write_text(json.dumps(t, indent=1))
    print("wrote tables.json")


if __name__ == "__main__":
    main()
