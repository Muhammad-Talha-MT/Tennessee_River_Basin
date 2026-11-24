"""Figure 3 (redesigned): why a gauge-based evaluation is not representative. 12.5 x 13.2 cm, constrained layout, no legend boxes.

(a) Empirical CDFs of drainage area (NHDPlus totdasqkm): all 58,834 reaches (vermilion), the 101 evaluation gauges (blue), the 128 catalog gauges (faint
    dashed gray, explained in the caption); dots mark the medians.
(b) The 101 evaluation gauges by position in the network: headwater gauges (green) and nested gauges (reddish purple; another of the 128 catalog gauges
    lies upstream, as `regulation` in hydro/evaluate.py). One dot per gauge. Dots are laid out as a beeswarm computed in points on the rendered axes,
    so no two dots touch (checked, the script fails otherwise). The x-axis is shared with (a); black bars mark the group medians.

Colors (Okabe-Ito): vermilion #D55E00 reaches, blue #0072B2 all evaluation gauges (panel a), bluish green #009E73 headwater and reddish purple #CC79A7
nested gauges (panel b). Fails if text overlaps text, a line or a dot.
Run: python scripts/section3/fig3_representativeness.py   Writes figures/fig3_representativeness.{pdf,png} (and a copy in figures/section3/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import FIG, load

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
c = load()
g = c.gauges
reach_area = c.nodes.totdasqkm.to_numpy(float)
gauge_area = reach_area[g.reach]
ev = c.scored["gauged"]
REACH, EVAL, HEAD, NEST, GRAYC, GRID = "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#9A9A9A", "#E8E8E8"
XLIM, XTICKS = (10 ** -1, 10 ** 4.5), [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

# nested = another of the 128 catalog gauges in the gauge's upstream basin (no dam data is used)
at = np.zeros(c.network.n, bool)
at[g.reach] = True
nested = np.array([at[b_].sum() > 1 for b_ in g.basins])
GROUPS = (("Headwater", ev & ~nested, HEAD), ("Nested", ev & nested, NEST))

fig = plt.figure(figsize=(12.5 / 2.54, 13.2 / 2.54), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1], sharex=axa)

# ---- (a) CDFs -------------------------------------------------------------------------------------------------------------------
med = {}
for x, lab, col, ls, lw in ((reach_area, "reaches", REACH, "-", 1.6), (gauge_area, "catalog", GRAYC, "--", 0.8), (gauge_area[ev], "evaluation", EVAL, "-", 1.6)):
    xs = np.sort(x)
    axa.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=col, ls=ls, lw=lw, zorder=3)
    med[lab] = float(np.median(x))
for lab, col in (("reaches", REACH), ("evaluation", EVAL)):
    axa.plot([med[lab]], [0.5], "o", color=col, ms=4.5, mec="white", mew=0.6, zorder=5)
axa.text(0.14, 0.93, "All reaches", ha="left", va="top", color=REACH, fontweight="bold")
axa.text(10 ** 4.4, 0.27, "Evaluation\ngauges", ha="right", va="top", color=EVAL, fontweight="bold", linespacing=1.15)
axa.set_xscale("log")
axa.set_xlim(*XLIM)
axa.set_xticks(XTICKS)
axa.set_ylim(0, 1.0)
axa.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axa.set_ylabel("Cumulative fraction")
axa.tick_params(labelbottom=False)

# ---- (b) beeswarm: laid out in points on the rendered axes so that no two dots touch ---------------------------------------------
ROW_GAP = 1.0  # data units between the two row centres
axb.set_xscale("log")
axb.set_xlim(*XLIM)
axb.set_xticks(XTICKS)
axb.set_ylim(1.55, -0.55)
axb.set_yticks([0, ROW_GAP])
axb.set_yticklabels([n for n, _, _ in GROUPS])
axb.tick_params(axis="y", length=0)
axb.spines["left"].set_visible(False)
axb.set_xlabel("Drainage area (km$^2$)")
for a_ in (axa, axb):
    a_.set_axisbelow(True)
    a_.grid(axis="x", color=GRID, lw=0.6)
axa.grid(axis="y", color=GRID, lw=0.6)
for lab_, a_ in (("(a)", axa), ("(b)", axb)):
    a_.text(0, 1.03, lab_, transform=a_.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

fig.canvas.draw()  # fixes the axes geometry used for the layout below
DOT_S = 20.0  # marker area (points^2)
d_pt = np.sqrt(DOT_S)  # marker diameter in points
GAP = 0.8  # minimum clear gap between two dots (points)
to_pt = 72.0 / fig.dpi
y0_pt = axb.transData.transform((1.0, 0.0))[1] * to_pt
y1_pt = axb.transData.transform((1.0, ROW_GAP))[1] * to_pt
pt_per_row = abs(y1_pt - y0_pt)  # points between the two row centres
step = 0.35 * d_pt
cand = [0.0] + [s * k * step for k in range(1, 80) for s in (1, -1)]  # vertical offsets in points, smallest first
medb, nb, maxoff = {}, {}, 0.0
swarm = []
for i, (name, mask, col) in enumerate(GROUPS):
    a = np.sort(gauge_area[mask])
    xpt = axb.transData.transform(np.column_stack([a, np.zeros(len(a))]))[:, 0] * to_pt
    placed = []
    for xp in xpt:
        for o in cand:
            if all((xp - px) ** 2 + (o - po) ** 2 >= (d_pt + GAP) ** 2 for px, po in placed):
                placed.append((xp, o))
                break
    off = np.array([p[1] for p in placed])
    maxoff = max(maxoff, np.abs(off).max())
    y_data = i * ROW_GAP + off / pt_per_row * ROW_GAP  # points -> data units (row axis is inverted: positive offset = lower on the page, as drawn)
    swarm.append((a, y_data, col, off))
    axb.scatter(a, y_data, s=DOT_S, color=col, edgecolor="white", lw=0.5, zorder=3)
    medb[name], nb[name] = float(np.median(a)), len(a)
    axb.plot([medb[name]] * 2, [i * ROW_GAP - 0.40 * ROW_GAP, i * ROW_GAP + 0.40 * ROW_GAP], color="black", lw=1.8, zorder=5, solid_capstyle="butt")
assert maxoff < 0.46 * pt_per_row, f"rows too close for the beeswarm: max offset {maxoff:.1f} pt of {pt_per_row:.1f} pt row spacing"

# ---- checks: no two dots touch; no text overlaps text, a line or a dot ------------------------------------------------------------
fig.canvas.draw()
bad = []
for a, y_data, col, off in swarm:  # re-measure on the final geometry
    xy = axb.transData.transform(np.column_stack([a, y_data])) * to_pt
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) + np.eye(len(a)) * 1e9
    if dist.min() < d_pt:
        bad.append(("dots touch", f"min distance {dist.min():.2f} pt < {d_pt:.2f} pt", ""))
rend = fig.canvas.get_renderer()
items = []
for ax_ in (axa, axb):
    items += [(t, ax_) for t in ax_.texts] + [(t, ax_) for t in ax_.get_xticklabels() + ax_.get_yticklabels() if t.get_visible() and t.get_text()]
    items += [(ax_.xaxis.label, ax_), (ax_.yaxis.label, ax_)]
items = [(t, a_) for t, a_ in items if t.get_text()]
bbs = [t.get_window_extent(rend).expanded(1.02, 1.05) for t, _ in items]
bad += [("text-text", items[i][0].get_text()[:20], items[j][0].get_text()[:20]) for i in range(len(items)) for j in range(i + 1, len(items)) if bbs[i].overlaps(bbs[j])]
fb = fig.bbox
bad += [("outside figure", t.get_text()[:20], "") for (t, _), bb in zip(items, bbs) if bb.x0 < fb.x0 or bb.x1 > fb.x1 or bb.y0 < fb.y0 or bb.y1 > fb.y1]
for (t, ax_), bb in zip(items, bbs):
    if t in ax_.get_xticklabels() + ax_.get_yticklabels() or t in (ax_.xaxis.label, ax_.yaxis.label):
        continue
    for ln in ax_.lines:
        if len(ln.get_xdata()) < 2:
            continue
        pts = ln.get_transform().transform(ln.get_xydata())
        seg = np.concatenate([np.linspace(pts[k], pts[k + 1], 8) for k in range(len(pts) - 1)])
        if ((seg[:, 0] > bb.x0) & (seg[:, 0] < bb.x1) & (seg[:, 1] > bb.y0) & (seg[:, 1] < bb.y1)).any():
            bad.append(("text-line", t.get_text()[:20], str(ln.get_color())))
    for col_ in ax_.collections:
        off_ = ax_.transData.transform(col_.get_offsets())
        if ((off_[:, 0] > bb.x0 - 2) & (off_[:, 0] < bb.x1 + 2) & (off_[:, 1] > bb.y0 - 2) & (off_[:, 1] < bb.y1 + 2)).any():
            bad.append(("text-dot", t.get_text()[:20], ""))
print("overlaps:", bad or "none")

OUT = Path(FIG).parent  # figures/
for folder in (OUT, FIG):
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(folder / f"fig3_representativeness.{ext}", facecolor="white", **kw)
ga = gauge_area[ev]
print("medians km2:", {k: round(v, 1) for k, v in med.items()}, "| headwater / nested:", {k: (nb[k], round(v, 1)) for k, v in medb.items()})
print("reaches", len(reach_area), "catalog", g.count, "evaluation", int(ev.sum()), "| reaches < 0.1 km2:", int((reach_area < 0.1).sum()),
      "| reaches > 10^4.5:", int((reach_area > 10 ** 4.5).sum()), "| gauges outside range:", int(((ga < 0.1) | (ga > 10 ** 4.5)).sum()),
      "| beeswarm: dot diameter %.1f pt, max offset %.1f pt of %.1f pt row spacing" % (d_pt, maxoff, pt_per_row))
if bad:
    sys.exit("overlap found")
