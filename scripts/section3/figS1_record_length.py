"""Figure S1: observation availability of the 101 evaluation gauges (6.5 in wide, one panel). Fraction of six-hour values observed per gauge in the
training period (2000-01-01 to 2018-12-31, dashed gray-black) and the test period (2020-09-23 to 2025-12-31, blue); each curve is sorted on its own, so
the x-axis is a rank, not a gauge. Counts are in figures/section3/README.md. Fails if text overlaps text or a curve.
Run: python scripts/section3/figS1_record_length.py   Writes figures/figS1_record_length.{pdf,png} (and a copy in figures/section3/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIG, P, load

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
c = load()
g, times = c.gauges, c.arrays.times
ev = c.scored["gauged"]
BLUE, TRAIN = "#0072B2", "#333333"
train = (times >= pd.Timestamp("2000-01-01")) & (times < P.TRAIN_UNTIL)
test = (times >= pd.Timestamp("2020-09-23")) & (times < pd.Timestamp("2026-01-01"))
fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
fr = {}
for k, mask, ls, col in (("train", train, "--", TRAIN), ("test", test, "-", BLUE)):
    fr[k] = np.sort(np.isfinite(g.observed[mask][:, ev]).mean(0))[::-1]
    ax.plot(np.arange(1, len(fr[k]) + 1), fr[k], color=col, ls=ls, lw=1.5)
ax.text(2, 1.09, "Test, 2020-09-23 to 2025-12-31", ha="left", va="center", color=BLUE)
ax.text(70, 0.40, "Training, 2000-2018", ha="right", va="center", color=TRAIN)
ax.set_xlabel(f"Gauge rank (sorted separately per period, n = {int(ev.sum())})")
ax.set_ylabel("Fraction of values observed")
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(0, int(ev.sum()) + 1)
ax.set_xticks([0, 25, 50, 75, 100])
fig.canvas.draw()
rend = fig.canvas.get_renderer()
bad = []
items = [t for t in ax.texts if t.get_text()] + ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]
bbs = [t.get_window_extent(rend).expanded(1.02, 1.05) for t in items]
bad += [("text-text", items[i].get_text()[:20], items[j].get_text()[:20]) for i in range(len(items)) for j in range(i + 1, len(items)) if bbs[i].overlaps(bbs[j]) and items[i].get_text() and items[j].get_text()]
for t, bb in zip(ax.texts, bbs):
    for ln in ax.lines:
        pts = ln.get_transform().transform(ln.get_xydata())
        seg = np.concatenate([np.linspace(pts[k], pts[k + 1], 8) for k in range(len(pts) - 1)])
        if ((seg[:, 0] > bb.x0) & (seg[:, 0] < bb.x1) & (seg[:, 1] > bb.y0) & (seg[:, 1] < bb.y1)).any():
            bad.append(("text-line", t.get_text()[:20], str(ln.get_color())))
print("overlaps:", bad or "none")
print("training availability < 50%:", int((fr["train"] < 0.5).sum()), "| < 10%:", int((fr["train"] < 0.1).sum()), "| zero:", int((fr["train"] == 0).sum()))
OUT = Path(FIG).parent
for folder in (OUT, FIG):
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(folder / f"figS1_record_length.{ext}", facecolor="white", **kw)
if bad:
    sys.exit("overlap found")
