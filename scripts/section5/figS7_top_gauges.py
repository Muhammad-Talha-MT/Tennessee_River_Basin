"""Figure S7: the six top-scoring gauges of the gauged protocol (all 101 gauges) (rule in s5_top_gauges.py: highest GNN mean NSE at leads 24-120 h). One row per gauge, two columns
(forecasts valid 24 h and 72 h after the daily 00Z issue; one value per issue, so each line joins forecasts made on different days): observed discharge (black dots)
against the GNN, the LSTM and the NWM over 60 days around the gauge's largest observed test-period flow. Panel titles give the gauge, its drainage area and the NSE of
each model at that lead over the whole test period.
Reads reports/section5/s5_10_top_gauges.csv and s5_10_top_gauge_series.csv. Run: python scripts/section5/figS7_top_gauges.py   Writes figures/figS7_top_gauges.{pdf,png}.
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from style import COLOR, DISPLAY, STYLE, WIDTH, read, save

top = read("s5_10_top_gauges.csv")
ser = read("s5_10_top_gauge_series.csv")
ser["valid_time"] = pd.to_datetime(ser.valid_time)
LEADS = (24, 72)
n = len(top)
fig, axes = plt.subplots(n, 2, figsize=(7.0, 1.62 * n + 0.55), sharey="row")
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.07, top=0.965, wspace=0.12, hspace=0.85)
for r, row in top.iterrows():
    for c, h in enumerate(LEADS):
        ax = axes[r, c]
        d = ser[(ser["rank"] == row["rank"]) & (ser.lead_h == h)].sort_values("valid_time")
        for m in ("NWM", "LSTM", "M0"):
            ax.plot(d.valid_time, d[m], ls=STYLE[m], color=COLOR[m], lw=WIDTH[m], label=DISPLAY[m])
        ax.plot(d.valid_time, d.observed, "o", color="#000000", ms=2.4, label="Observed", zorder=5)
        ax.set_xlim(d.valid_time.min() - pd.Timedelta(hours=12), d.valid_time.max() + pd.Timedelta(hours=12))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(bottom=0)
        yrs = sorted({d.valid_time.min().year, d.valid_time.max().year})
        ax.set_xlabel("–".join(str(y) for y in yrs), fontsize=6.5, labelpad=1)
        gid = f"USGS {int(row['usgs_id']):08d}"
        nse = "   ".join(f"{DISPLAY[m]} {row[f'nse_{m}_{h}h']:.2f}" for m in ("M0", "LSTM", "NWM"))
        letter = "abcdefghijkl"[2 * r + c]
        head = f"({letter}) {gid}, {row['area_km2']:,.0f} km$^2$, {h} h" if c == 0 else f"({letter}) {h} h"
        ax.set_title(f"{head}\nNSE: {nse}", fontsize=6.8, loc="left", pad=3)
        if c == 0:
            ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)", fontsize=7)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=4, bbox_to_anchor=(0.54, 0.0), columnspacing=1.5)
save(fig, "figS7_top_gauges", list(axes.ravel()), lines=False)
