"""Figure 2: one forecast issue (full width, 17.4 cm, about 5 cm high). Inputs known at the 00Z issue time (observed forcing and discharge) end at day 0;
the forecast rain runs from day 0 to day +10 (240 h = 40 six-hourly steps). Days 1-5 (leads 24-120 h) are shaded. No dates are drawn: the data periods are in
reports/section3_facts.md (section 13).
Windows are read from the code: P.HISTORY (hydro/paths.py) = the weather and observed-discharge steps before the issue that the model reads
(`samples.py: Samples.build` and `Device.assemble`: `shown = finite & (step < HISTORY) & visible`), P.HORIZON = forecast steps, 6 h per step.
Observed discharge is assumed available up to the issue time (the last six-hour mean ends at 00Z); no reporting latency is modelled in the code.
Run: python scripts/section3/fig2_timeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIG, P

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
STEP_DAYS = 6 / 24
back = P.HISTORY * STEP_DAYS  # days of observed forcing and discharge the model reads before the issue
ahead = P.HORIZON * STEP_DAYS  # forecast horizon in days
HOURS = P.HORIZON * 6
KNOWN, FORECAST, BAND = "#0072B2", "#D55E00", "#EDEDED"  # Okabe-Ito blue and vermilion

W = 17.4 / 2.54
fig, ax = plt.subplots(figsize=(W, 5 / 2.54))
fig.subplots_adjust(left=0.19, right=0.985, top=0.80, bottom=0.30)
# short row labels on the left; what each bar covers is written inside the bar
rows = [
    ("Observed forcing (AORC)", "all reaches", -back, 0, KNOWN),
    ("Observed discharge", "visible gauges only", -back, 0, KNOWN),
    ("Forecast rain (GEFSv12)", "all reaches", 0, ahead, FORECAST),
]
n = len(rows)
ax.axvspan(1, 5, color=BAND, lw=0, zorder=0)  # days 1-5
for i, (label, inside, a, b, col) in enumerate(rows):
    y = n - 1 - i
    ax.barh(y, b - a, left=a, height=0.58, color=col, lw=0, zorder=2)
    ax.text(-back - 0.9, y, label, ha="right", va="center", fontsize=8)
    ax.text((a + b) / 2, y, inside, ha="center", va="center", fontsize=8, color="white", zorder=3)
ax.axvline(0, color="black", lw=1.0, ymin=0, ymax=1.10, clip_on=False, zorder=4)
ax.text(0, n - 0.5 + 0.10 * n, "Issue time (00Z)", ha="center", va="bottom", fontsize=8, fontweight="bold")
# horizon bracket above the forecast bar: its ends are set in from the issue line and from day 10, so nothing touches the issue line
gap = 0.35
x0, x1 = gap, ahead - gap
by, tick = 0.78, 0.09
ax.plot([x0, x0, x1, x1], [by - tick, by, by, by - tick], color="black", lw=0.9, zorder=3)
ax.text(x0 + 0.5, by + 0.08, f"{HOURS} h\n({P.HORIZON} six-hourly steps)", ha="left", va="bottom", fontsize=8, linespacing=1.15)  # starts well right of the issue line
ax.set_xlim(-back - 0.5, ahead + 0.5)
ax.set_ylim(-0.5, n - 0.5)
ax.set_yticks([])
ax.set_xticks([-30, -20, -10, 0, 5, 10])
ax.tick_params(axis="x", labelsize=8, length=3)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.set_xlabel("Days relative to the issue time", fontsize=8, labelpad=3, loc="left")
ax.annotate(f"Days 1-5 (24-{5 * 24} h)", xy=(3, 0), xycoords=("data", "axes fraction"), xytext=(3, -0.33), textcoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=8)
for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
    fig.savefig(FIG / f"fig2_timeline.{ext}", bbox_inches="tight", facecolor="white", **kw)
print(f"look-back {back:g} days ({P.HISTORY} steps, hydro/paths.py HISTORY) for forcing and discharge; horizon {ahead:g} days ({P.HORIZON} steps = {HOURS} h)")
