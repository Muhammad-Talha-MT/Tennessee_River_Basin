"""Figure S3: the example event (fixed rule, reports/section5_facts.md section 8): observed discharge (black points) and forecasts of the NWM, the LSTM, M0
and M1 valid at each day, issued 6, 12, 18 and 24 h earlier (one value per daily 00Z issue), ten days around the largest observed test-period flow at the
selected gauge. Reads reports/section5/s5_8_example_event.csv. Run: python scripts/section5/figS3_example_event.py   Writes figures/figS3_example_event.{pdf,png}.
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from style import COLOR, DISPLAY, STYLE, WIDTH, read, save

ex = read("s5_8_example_event.csv")
ex["valid_time"] = pd.to_datetime(ex.valid_time)
LEADS = (6, 12, 18, 24)
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True)
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.3, top=0.91, wspace=0.07)
T0, T1 = ex.valid_time.min() - pd.Timedelta(hours=6), ex.valid_time.max() + pd.Timedelta(hours=6)  # one time axis for all panels
for k, (ax, h) in enumerate(zip(axes, LEADS)):
    d = ex[ex.lead_h == h].sort_values("valid_time")
    for m in ["NWM", "LSTM", "M1", "M0"]:
        ax.plot(d.valid_time, d[m], ls=STYLE[m], color=COLOR[m], lw=WIDTH[m], label=DISPLAY[m])
    ax.plot(d.valid_time, d.observed, "o", color="#000000", ms=2.8, label="Observed", zorder=5)
    ax.set_title(f"({'abcd'[k]}) {h} h")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(T0, T1)
    ax.set_xlabel(f"{ex.valid_time.min():%Y}")
axes[0].set_ylabel("Discharge (m$^3$ s$^{-1}$)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=5, bbox_to_anchor=(0.54, 0.0), columnspacing=1.5)
save(fig, "figS3_example_event", list(axes))
