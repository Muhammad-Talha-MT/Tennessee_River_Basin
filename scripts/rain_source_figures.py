"""Forecast-rain sources: NWM (GFS-based), GEFS operational and realized AORC rain.

Reads fair_benchmark_gefs (fine-tuned graph model and original-settings LSTM, 3 seeds each)
and scratch_benchmark_gefs_s1 (graph model from scratch), both scored on the same 101 gauges
and 1,917 00Z issues from 2020-09-23, and writes figures/rain_sources. The tuned LSTM was
not run with GEFS rain. Seeds are averaged per lead (median NSE over gauges first).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparison_figures import INK, LEGEND, LSTM, MUTED, NWM, PERSIST, PRETRAINED, SCRATCH, plt  # noqa: E402

M = Path("/data/tva_data/models")
OUT = Path("/data/tva_data/figures/rain_sources")
RAINS = [("nwm", "NWM forecast rain (GFS-based)"), ("gefs", "GEFS operational rain"), ("aorc", "Realized rain (upper bound)")]


def curves():
    fair = pd.read_csv(M / "fair_benchmark_gefs/summary_by_lead.csv")
    scratch = pd.read_csv(M / "scratch_benchmark_gefs_s1/summary_by_lead.csv")

    def mean_seeds(table, prefix, rain):
        names = [f"{prefix}_s{s}+{rain}" for s in (1, 2, 3)]
        part = table[table.method.isin(names)].groupby("lead_hours").median_nse.mean()
        return part

    def one(table, name):
        return table[table.method == name].set_index("lead_hours").median_nse

    result = {}
    for rain, _ in RAINS:
        result[("GNN, from scratch", rain)] = one(scratch, f"scratch_s1+{rain}")
        result[("GNN, fine-tuned (3 seeds)", rain)] = mean_seeds(fair, "graph", rain)
        result[("LSTM, original settings (3 seeds)", rain)] = mean_seeds(fair, "lstm", rain)
    refs = {"NWM streamflow": one(fair, "nwm"), "Persistence": one(fair, "persistence")}
    return result, refs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result, refs = curves()
    models = [("GNN, from scratch", SCRATCH), ("GNN, fine-tuned (3 seeds)", PRETRAINED), ("LSTM, original settings (3 seeds)", LSTM)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (rain, title) in zip(axes, RAINS):
        for name, color in models:
            s = result[(name, rain)]
            ax.plot(s.index, s.values, color=color, label=name)
        ax.plot(refs["NWM streamflow"].index, refs["NWM streamflow"].values, color=NWM, ls="--", label="NWM streamflow")
        ax.plot(refs["Persistence"].index, refs["Persistence"].values, color=PERSIST, ls=":", label="Persistence")
        ax.set_title(title)
        ax.set_xlabel("Lead (hours)")
        ax.set_ylim(-0.6, 1)
    axes[0].set_ylabel("Median NSE, 101 gauges")
    axes[0].legend(loc="lower left", **{**LEGEND, "fontsize": 9})
    fig.tight_layout()
    fig.savefig(OUT / "1_nse_by_lead_and_rain.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (rain, label), style in zip(RAINS, ("-", "--", ":")):
        s = result[("GNN, from scratch", rain)]
        ax.plot(s.index, s.values, color=SCRATCH, ls=style, label=f"GNN, {label}")
    ax.plot(refs["NWM streamflow"].index, refs["NWM streamflow"].values, color=NWM, label="NWM streamflow")
    ax.set_xlabel("Lead (hours)")
    ax.set_ylabel("Median NSE, 101 gauges")
    ax.set_ylim(-0.4, 1)
    ax.legend(**{**LEGEND, "fontsize": 9})
    fig.tight_layout()
    fig.savefig(OUT / "2_rain_effect_gnn_scratch.png", dpi=170)
    plt.close(fig)

    leads = (24, 72, 120)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    width = 0.2
    for ax, lead in zip(axes, leads):
        series = [(n, c, [result[(n, r)][lead] for r, _ in RAINS]) for n, c in models]
        series.append(("NWM streamflow", NWM, [refs["NWM streamflow"][lead]] * 3))
        for k, (n, c, v) in enumerate(series):
            xs = np.arange(3) + (k - 1.5) * width
            ax.bar(xs, v, width, color=c, label=n)
            for x, y in zip(xs, v):
                ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(range(3), ["NWM rain", "GEFS rain", "Realized"])
        ax.set_title(f"{lead} h")
        ax.set_ylim(-0.1, 1)
    axes[0].set_ylabel("Median NSE")
    axes[0].legend(loc="upper left", **{**LEGEND, "fontsize": 8})
    fig.tight_layout()
    fig.savefig(OUT / "3_bars_by_rain.png", dpi=170)
    plt.close(fig)

    rows = [dict(model=n, rain=r, lead_h=int(h), median_nse=float(v)) for (n, r), s in result.items() for h, v in s.items()]
    rows += [dict(model=n, rain="-", lead_h=int(h), median_nse=float(v)) for n, s in refs.items() for h, v in s.items()]
    pd.DataFrame(rows).to_csv(OUT / "nse_by_rain.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
