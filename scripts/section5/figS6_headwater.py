"""Figure S6 (a-e): headwater reaches (reaches with no reach upstream of them, 23,172 of 58,834; median drainage area 2.0 km2), skill against the NWM v3.0
retrospective (the reference the reach-scale models were pretrained on; the LSTM and the NWM forecast were not). Reads reports/section5/s5_perreach_nse.csv
and s5_headwater_lead_curves.csv. Per-reach NSE = days 1-5 mean. Models: M0, LSTM, NWM operational forecast (M1, M2, M4 have no every-reach run).
(a) distribution of per-reach NSE by drainage-area class; (b) median NSE against lead by reach class; (c) maps of the headwater reaches: M0 - LSTM and M0's NSE;
(d) NSE and the share of reaches with NSE > 0 against drainage area; (e) NSE by mean catchment elevation.
Run: python scripts/section5/figS6_headwater.py   Writes figures/figS6{a..e}_headwater_*.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "section3"))
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from style import COLOR, DIVERGING, FIG, LABEL, STYLE, WIDTH, read, save

from common import INPUTS  # noqa: E402  (study-area geometry paths only)

pr = read("s5_perreach_nse.csv")
hl = read("s5_headwater_lead_curves.csv")
hw = pr[pr.is_headwater].copy()
M = ["M0", "LSTM", "NWM"]
NAME = {"M0": "GNN", "LSTM": "LSTM", "NWM": "NWM forecast"}
MCOL = {m: COLOR[m] for m in M}

# (a) ECDF of per-reach NSE, headwater reaches by drainage-area class
bins = [(0, 1, "< 1 km$^2$"), (1, 3, "1-3 km$^2$"), (3, 1e9, "$\\geq$ 3 km$^2$")]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.3, top=0.9, wspace=0.07)
for k, (ax, (lo, hi, lab)) in enumerate(zip(axes, bins)):
    sub = hw[(hw.area_km2 >= lo) & (hw.area_km2 < hi)]
    for m in M:
        x = np.sort(sub[f"nse_{m}"].dropna().to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=MCOL[m], ls=STYLE[m], lw=WIDTH[m], label=NAME[m])
    ax.axvline(0, color="#000000", lw=0.8)
    ax.set_xlim(-3, 1)
    ax.set_xlabel("NSE (days 1-5)")
    ax.set_title(f"({'abc'[k]}) {lab}  (n = {int(sub.nse_M0.notna().sum()):,})")
axes[0].set_ylabel("Cumulative fraction of reaches")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.54, 0.0))
save(fig, "figS6a_headwater_ecdf", axes)

# (b) median NSE against lead by reach class
cls = list(dict.fromkeys(hl.reach_class))
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
fig.subplots_adjust(left=0.08, right=0.975, bottom=0.3, top=0.88, wspace=0.07)
for k, (ax, c) in enumerate(zip(axes, cls)):
    for m in M[::-1]:
        d = hl[(hl.reach_class == c) & (hl.model == m)]
        ax.plot(d.lead_h, d.median_nse, ls=STYLE[m], color=MCOL[m], lw=WIDTH[m], label=NAME[m])
    ax.axhline(0, color="#000000", lw=0.8)
    ax.set_xlim(6, 240)
    ax.set_xticks([48, 96, 144, 192, 240])
    ax.set_xlabel("Lead time (h)")
    n = int(hl[(hl.reach_class == c) & (hl.model == "M0")].reaches.iloc[0])
    pretty = c.replace("km2", "km$^2$").replace(">=", "$\\geq$")
    ax.set_title(f"({'abc'[k]}) {pretty}", fontsize=7.5)
    ax.text(0.03, 0.04, f"{n:,} reaches", transform=ax.transAxes, fontsize=7.5)
axes[0].set_ylabel("Median NSE vs. NWM retrospective")
axes[0].set_ylim(-2.5, 0.8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, bbox_to_anchor=(0.54, 0.0))
save(fig, "figS6b_headwater_leads", axes)

# (d) NSE and share of reaches with NSE > 0 against drainage area (headwater reaches), log bins
edges = np.logspace(np.log10(0.1), np.log10(60), 13)
mid = np.sqrt(edges[:-1] * edges[1:])
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.31, top=0.9, wspace=0.26)
for m in M:
    med, q25, q75, pos, n = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = hw[(hw.area_km2 >= lo) & (hw.area_km2 < hi)][f"nse_{m}"].dropna().to_numpy()
        n.append(len(x))
        med.append(np.median(x) if len(x) >= 30 else np.nan)
        q25.append(np.percentile(x, 25) if len(x) >= 30 else np.nan)
        q75.append(np.percentile(x, 75) if len(x) >= 30 else np.nan)
        pos.append((x > 0).mean() if len(x) >= 30 else np.nan)
    axa.plot(mid, med, ls=STYLE[m], marker="o", ms=3, color=MCOL[m], lw=WIDTH[m], label=NAME[m])
    if m == "M0":
        axa.fill_between(mid, q25, q75, color=MCOL[m], alpha=0.15, lw=0)
    axb.plot(mid, pos, ls=STYLE[m], marker="o", ms=3, color=MCOL[m], lw=WIDTH[m], label=NAME[m])
axa.axhline(0, color="#000000", lw=0.8)
for ax in (axa, axb):
    ax.set_xscale("log")
    ax.set_xlabel("Drainage area (km$^2$)")
    ax.set_xlim(0.1, 60)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.minorticks_off()
axa.set_ylim(-3, 0.6)
axa.set_ylabel("Median NSE (band: GNN interquartile)")
axa.set_title("(a) NSE")
axb.set_ylabel("Share of reaches with NSE > 0")
axb.set_ylim(0, 1)
axb.set_title("(b) Share of reaches with skill above zero")
h, l = axa.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.54, 0.0))
save(fig, "figS6d_headwater_area", [axa, axb])

# (e) by mean catchment elevation (headwater reaches)
eb = [(0, 250, "< 250"), (250, 500, "250-500"), (500, 1000, "500-1,000"), (1000, 1e9, "$\\geq$ 1,000")]
fig, ax = plt.subplots(figsize=(3.5, 3.1))
fig.subplots_adjust(left=0.2, right=0.97, bottom=0.34, top=0.96)
for k, m in enumerate(M):
    med, lo, hi = [], [], []
    for a, b, _ in eb:
        x = hw[(hw.elev_m >= a) & (hw.elev_m < b)][f"nse_{m}"].dropna().to_numpy()
        med.append(np.median(x)), lo.append(np.percentile(x, 25)), hi.append(np.percentile(x, 75))
    med, lo, hi = map(np.array, (med, lo, hi))
    xx = np.arange(len(eb)) + (k - 1) * 0.14
    ax.errorbar(xx, med, yerr=[med - np.maximum(lo, -3), hi - med], fmt="o", color=MCOL[m], ms=4, elinewidth=1.0, capsize=2, label=NAME[m])
ax.axhline(0, color="#000000", lw=0.8)
ax.set_xticks(range(len(eb)))
ax.set_xticklabels([f"{l}\n({int(((hw.elev_m >= a) & (hw.elev_m < b) & hw.nse_M0.notna()).sum()):,})" for a, b, l in eb])
ax.set_ylim(-3, 0.8)
ax.set_yticks([-3, -2, -1, 0])
ax.set_xlabel("Mean catchment elevation (m), (reaches)")
ax.set_ylabel("NSE (median and interquartile range)")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.58, 0.0), fontsize=7, columnspacing=0.8, handletextpad=0.3)
save(fig, "figS6e_headwater_elevation", [ax])

# (c) maps of the headwater reaches: M0 - LSTM and M0's NSE
CRS = "EPSG:5070"
lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(pr.comid)].to_crs(CRS)
p = pr.set_index("comid")
lines["is_hw"] = p.is_headwater.reindex(lines.COMID).to_numpy()
lines["diff"] = (p.nse_M0 - p.nse_LSTM).reindex(lines.COMID).to_numpy()
lines["m0"] = p.nse_M0.reindex(lines.COMID).to_numpy()
lines["a"] = p.area_km2.reindex(lines.COMID).to_numpy()
outline = gpd.read_file(INPUTS["huc2"]).to_crs(CRS)
x0, y0, x1, y1 = outline.total_bounds
pad = 0.015 * (x1 - x0)
other = lines[~lines.is_hw]
hwl = lines[lines.is_hw]
for tag, col, title, cmap, norm, label, ext, ticks in (
    ("c1", "diff", "Headwater reaches: GNN - LSTM", plt.get_cmap(DIVERGING), Normalize(-0.5, 0.5), "Difference in NSE, days 1-5 (GNN - LSTM)", "both", [-0.5, -0.25, 0, 0.25, 0.5]),
    ("c2", "m0", "Headwater reaches: GNN NSE", ListedColormap(plt.get_cmap("GnBu")(np.linspace(0.12, 1, 256))), Normalize(0, 0.6), "GNN NSE, days 1-5 (gray: NSE $\\leq$ 0 or not scored)", "max", [0, 0.2, 0.4, 0.6]),
):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.15, top=0.97)
    outline.plot(ax=ax, facecolor="#FAFAFA", edgecolor="#1F2937", lw=1.0, zorder=0)
    other.plot(ax=ax, color="#E3E3E3", linewidth=0.25, zorder=1, rasterized=True)
    v = hwl[col].to_numpy(float)
    ok = np.isfinite(v) & (v > 0 if col == "m0" else True)
    hwl[~ok].plot(ax=ax, color="#BDBDBD", linewidth=0.45, zorder=2, rasterized=True)
    hwl[ok].plot(ax=ax, color=cmap(norm(np.clip(v[ok], norm.vmin, norm.vmax))), linewidth=0.7, zorder=3, rasterized=True)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(0.01, 0.97, f"({tag[0]}) {title}  ({len(hwl):,} reaches; other reaches in light gray)", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    cax = fig.add_axes([0.3, 0.075, 0.4, 0.028])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend=ext)
    cb.set_ticks(ticks)
    cb.set_label(label, labelpad=2)
    cb.ax.grid(False)
    for e, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(FIG / f"figS6{tag}_headwater_map.{e}", facecolor="white", **kw)
    plt.close(fig)
    print(f"figS6{tag}: saved")
