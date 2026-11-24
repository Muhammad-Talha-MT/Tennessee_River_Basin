"""Phase 1.1 insertion audit, data part: which scored gauges have a visible gauge upstream, how often, and M0 vs M1 below gauges.

Writes reports/v3_tables/insertion_*.csv. Run: python scripts/v3_insertion_audit.py
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from v3_common import *  # noqa: F401,F403

OUT.mkdir(parents=True, exist_ok=True)
ctx = Context()
g = ctx.gauges
INNOVATION = 32  # steps before the issue in which a gauge needs >= 8 values to be shown (Samples.eligible)

steps = np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["step"]
recent_ok = np.stack([np.isfinite(g.observed[s - INNOVATION : s]).sum(0) >= 8 for s in steps])  # [issue, gauge]

# ---- which scored gauges have a visible gauge upstream (static and over the test issues) ---------------------------------
rows = []
for protocol in ("gauged", "heldout", "blocked"):
    up = ctx.upstream_visible(protocol)
    cov = ctx.coverage(protocol)
    for gi, vis in up.items():
        if protocol != "gauged" and ctx.fold[protocol][gi] == 0:
            continue
        elig = recent_ok[:, vis].any(1) if len(vis) else np.zeros(len(steps), bool)
        rows.append(dict(protocol=protocol, gauge=ctx.names[gi], area_km2=ctx.area[gi], fold=int(ctx.fold[protocol][gi]),
                         n_upstream_gauges=len(ctx.upstream_gauges(gi)), n_upstream_visible=len(vis),
                         share_issues_with_visible_upstream=float(elig.mean()), covered_area_share=cov[gi],
                         dam_group=next((n for n, m in ctx.groups if m[gi]), "")))
df = pd.DataFrame(rows)
df.to_csv(OUT / "insertion_upstream_visible_by_gauge.csv", index=False)

summary = []
scored_set = {}
for protocol in PROTOCOLS:
    data_p = load_protocol(protocol, ("M0", "LSTM"))
    ok = np.all([np.isfinite(scores(*data_p[m])["nse"][3]) for m in data_p], axis=0)
    scored_set[protocol] = set(ctx.names[ok])
df = df[[row.gauge in scored_set[row.protocol] for row in df.itertuples()]]
df.to_csv(OUT / "insertion_upstream_visible_by_gauge.csv", index=False)
for protocol in ("gauged", "heldout", "blocked"):
    d = df[df.protocol == protocol]
    summary.append(dict(protocol=protocol, scored_gauges=len(d), with_upstream_gauge=int((d.n_upstream_gauges > 0).sum()),
                        with_upstream_visible_gauge=int((d.n_upstream_visible > 0).sum()),
                        mean_share_issues_with_visible_upstream=float(d.share_issues_with_visible_upstream.mean()),
                        with_visible_upstream_in_over_half_of_issues=int((d.share_issues_with_visible_upstream > 0.5).sum()),
                        median_covered_area_share_when_present=float(d.covered_area_share[d.n_upstream_visible > 0].median())
                        if (d.n_upstream_visible > 0).any() else np.nan))
pd.DataFrame(summary).to_csv(OUT / "insertion_upstream_visible_summary.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))

# ---- M0 vs M1 at gauged-protocol sites below another gauge (6 h and 24 h) --------------------------------------------------
data = load_protocol("gauged", ("M0", "M1"))
sc = {m: scores(*data[m]) for m in data}
scored = np.isfinite(sc["M0"]["nse"][3]) & np.isfinite(sc["M1"]["nse"][3])
gauged = df[df.protocol == "gauged"].set_index("gauge")
below = np.array([n in gauged.index and gauged.loc[n, "n_upstream_gauges"] > 0 for n in ctx.names]) & scored
big = np.array([n in gauged.index and gauged.loc[n, "covered_area_share"] >= 0.5 for n in ctx.names]) & scored
rows = []
for label, mask in (("all gauged sites", scored), ("below at least one gauge", below), ("upstream gauges cover >= 50% of basin area", big),
                    ("no gauge upstream (headwater)", scored & ~below)):
    for h in (6, 24):
        i = h // 6 - 1
        a, b = sc["M0"]["nse"][i][mask], sc["M1"]["nse"][i][mask]
        d = a - b
        rows.append(dict(subset=label, gauges=int(mask.sum()), lead_h=h, median_nse_M0=float(np.median(a)), median_nse_M1=float(np.median(b)),
                         mean_diff_M0_minus_M1=float(d.mean()), median_diff=float(np.median(d)), m0_better=int((d > 0).sum()),
                         wilcoxon_p=wilcoxon_p(d)))
pd.DataFrame(rows).to_csv(OUT / "insertion_M0_vs_M1_gauged_sites_below_gauges.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
