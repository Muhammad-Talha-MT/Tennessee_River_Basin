"""Figure 4: skill against lead time over the full forecast range (6-240 h, 40 leads), GNN (M0) against the LSTM and the NWM reference, one column per protocol
(gauged n = 101, random hold-out n = 98, blocked n = 29). Top: median NSE over gauges. Bottom: paired difference GNN - LSTM (mean over gauges, 90% gauge-bootstrap
band, 10,000 resamples); above zero = GNN higher. Days 1-5 shaded. The ablations (M1, M2, M4) are left out; they have their own figures. Reads
reports/section5/s5_2_lead_curves.csv and s5_4_contrasts_40_leads.csv. Run: python scripts/section5/fig4_skill_lead.py   Writes figures/fig4_skill_lead.{pdf,png}.
"""
import matplotlib.pyplot as plt
from style import BAND, COLOR, PNAME, PROTOCOLS, STYLE, WIDTH, read, save

lc = read("s5_2_lead_curves.csv")
ct = read("s5_4_contrasts_40_leads.csv")
ct = ct[ct.contrast == "M0 - LSTM"]
NAME = {"M0": "GNN", "LSTM": "LSTM", "NWM": "NWM"}
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7), sharex=True, sharey="row")
fig.subplots_adjust(left=0.125, right=0.985, bottom=0.215, top=0.93, wspace=0.08, hspace=0.12)
for k, p in enumerate(PROTOCOLS):
    top, bot = axes[0, k], axes[1, k]
    d = lc[lc.protocol == p]
    for ax in (top, bot):
        ax.axvspan(1, 5, color=BAND, lw=0, zorder=0)
    for m in ("NWM", "LSTM", "M0"):
        s = d[d.model == m]
        top.plot(s.lead_h / 24, s.median_nse, ls=STYLE[m], color=COLOR[m], lw=WIDTH[m], label=NAME[m], zorder=3)
    top.set_title(f"({'abc'[k]}) {PNAME[p]} (n = {int(d.n.iloc[0])})")
    c = ct[ct.protocol == p].sort_values("lead_h")
    bot.axhline(0, color="black", lw=0.8, zorder=2)
    bot.fill_between(c.lead_h / 24, c.ci5, c.ci95, color=COLOR["M0"], alpha=0.22, lw=0, zorder=3)
    bot.plot(c.lead_h / 24, c.mean_diff, color=COLOR["M0"], lw=1.6, zorder=4)
    bot.set_xlim(0.25, 10)
    bot.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bot.set_xlabel("Lead time (days)")
axes[0, 0].set_ylabel("Median NSE")
axes[1, 0].set_ylabel("GNN - LSTM\n(difference in NSE)")
axes[0, 0].set_ylim(-0.5, 1.0)
axes[0, 0].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
axes[1, 0].set_ylim(-0.25, 0.4)
axes[1, 0].set_yticks([-0.2, 0.0, 0.2, 0.4])
for ax in axes[:, 1:].flat:
    ax.tick_params(labelleft=False)
handles, labels = axes[0, 1].get_legend_handles_labels()
order = [labels.index(n) for n in ("GNN", "LSTM", "NWM")]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, bbox_to_anchor=(0.54, 0.0), columnspacing=1.8, handlelength=2.6)
save(fig, "fig4_skill_lead", list(axes.flat))
