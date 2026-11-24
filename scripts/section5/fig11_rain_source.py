"""Figure 11: forecast rain against realized rain, GNN (M0) and LSTM, one column per protocol (gauged n = 101, random hold-out n = 98, blocked n = 29). The same trained models and the
same gauges and issues are scored with the GEFS forecast rain after the issue (solid) and with the realized AORC rain in its place (dashed; a perfect-rain diagnostic, nothing is
retrained). Top: median NSE over gauges by lead (6-240 h). Bottom: gain from realized rain (NSE with realized rain minus NSE with forecast rain, mean over gauges, 90% gauge-bootstrap
band, 10,000 resamples). Days 1-5 shaded. Reads reports/section5/s5_11_rain_source_curves.csv and s5_11_rain_source_gain.csv (scripts/section5/s5_rain_source.py).
Run: python scripts/section5/fig11_rain_source.py   Writes figures/fig11_rain_source.{pdf,png}.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import BAND, COLOR, DISPLAY, PNAME, PROTOCOLS, read, save

cv = read("s5_11_rain_source_curves.csv")
gn = read("s5_11_rain_source_gain.csv")
FORECAST, REALIZED = "-", (0, (4, 1.6))
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8), sharex=True, sharey="row")
fig.subplots_adjust(left=0.125, right=0.985, bottom=0.225, top=0.93, wspace=0.08, hspace=0.12)
for k, p in enumerate(PROTOCOLS):
    top, bot = axes[0, k], axes[1, k]
    for ax in (top, bot):
        ax.axvspan(1, 5, color=BAND, lw=0, zorder=0)
    d = cv[cv.protocol == p]
    for m in ("LSTM", "M0"):
        for r, ls in (("gefs", FORECAST), ("aorc", REALIZED)):
            s = d[(d.model == m) & (d.rain == r)].sort_values("lead_h")
            top.plot(s.lead_h / 24, s.median_nse, ls=ls, color=COLOR[m], lw=1.7 if m == "M0" else 1.4, zorder=3)
    top.set_title(f"({'abc'[k]}) {PNAME[p]} (n = {int(d.n.iloc[0])})")
    g = gn[gn.protocol == p]
    bot.axhline(0, color="black", lw=0.8, zorder=2)
    for m in ("LSTM", "M0"):
        s = g[g.model == m].sort_values("lead_h")
        bot.fill_between(s.lead_h / 24, s.ci5, s.ci95, color=COLOR[m], alpha=0.2, lw=0, zorder=3)
        bot.plot(s.lead_h / 24, s.mean_gain, color=COLOR[m], lw=1.6, zorder=4)
    bot.set_xlim(0.25, 10)
    bot.set_xticks(range(1, 11))
    bot.set_xlabel("Lead time (days)")
axes[0, 0].set_ylabel("Median NSE")
axes[1, 0].set_ylabel("Gain from realized rain\n(difference in NSE)")
axes[0, 0].set_ylim(-0.5, 1.0)
axes[0, 0].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
for ax in axes[:, 1:].flat:
    ax.tick_params(labelleft=False)
handles = [Line2D([], [], color=COLOR["M0"], lw=1.7), Line2D([], [], color=COLOR["LSTM"], lw=1.4),
           Line2D([], [], color="#444444", lw=1.4, ls=FORECAST), Line2D([], [], color="#444444", lw=1.4, ls=REALIZED)]
fig.legend(handles, [DISPLAY["M0"], DISPLAY["LSTM"], "Forecast rain (GEFS)", "Realized rain (AORC)"], loc="lower center", ncol=4, bbox_to_anchor=(0.54, 0.0),
           columnspacing=1.6, handlelength=2.6)
save(fig, "fig11_rain_source", list(axes.flat))
