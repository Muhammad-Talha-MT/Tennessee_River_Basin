"""Section 5 facts: every number of reports/section5_facts.md (also PART B of reports/section5_brief.md), computed from the saved forecasts and checkpoints.

Scoring and statistics come from scripts/v3_common.py (scores: NSE, KGE 2009 and parts with the evaluation's scoring rule; boot_mean: gauge-level
bootstrap, 10,000 resamples, seed 0; wilcoxon_p; tost; holm). Models: M0, M1, M2, M4, LSTM (area-scaled L4), NWM (operational forecast, reference).
A model whose forecasts are not saved is MISSING: never filled. No dam or operator data is read (the gauge context is built here, not with
v3_common.Context). Outputs: reports/section5/*.csv and reports/section5_facts.md (read into reports/section5_brief.md by build_brief.py).
Run: python scripts/section5/s5_facts.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd
import torch
from scipy import stats

import v3_common as V  # noqa: E402  (adds src/ to the path)
from tva_flood.hydro import paths as P  # noqa: E402
from tva_flood.hydro import splits  # noqa: E402
from tva_flood.hydro.channel import channel_table, main_path  # noqa: E402
from tva_flood.hydro.data import Arrays, Gauges, Network  # noqa: E402
from tva_flood.hydro.model import build_model  # noqa: E402

REPO, A = V.REPO, V.A
OUT = REPO / "reports" / "section5"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["M0", "M1", "M2", "M4", "LSTM"]
ALL = MODELS + ["NWM"]
PROTOS = list(V.PROTOCOLS)
PNAME = {"gauged": "Gauged", "heldout": "Random hold-out", "blocked": "Blocked basins"}
D15 = V.DAYS15  # lead indices of 24, 48, 72, 96, 120 h
LEADS5 = [24, 48, 72, 96, 120]
D12 = [3, 7]  # days 1-2: leads 24 and 48 h
CONTRASTS = [("M0", "LSTM"), ("M0", "NWM"), ("M0", "M2"), ("M1", "M4"), ("M0", "M1"), ("M1", "LSTM"), ("M2", "LSTM"), ("M4", "LSTM")]
AREA_EDGES, AREA_LABELS = [0, 200, 600, 2000, np.inf], ["< 200", "200-600", "600-2,000", "> 2,000"]
CALIPER = 0.15  # nearest-neighbour matching on log10 area: at most a factor 10^0.15 = 1.41 apart
MIN_SUBSET = 30  # minimum values per lead for an NSE on a subset of issues (item 5b)
md = []


def add(s=""):
    md.append(s)


def table(df, floatfmt=3):
    cols = list(df.columns)
    add("| " + " | ".join(str(c) for c in cols) + " |")
    add("|" + "|".join("---" for _ in cols) + "|")
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else (f"{v:.{floatfmt}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"))
            else:
                cells.append(str(v))
        add("| " + " | ".join(cells) + " |")
    add("")


def save(df, name):
    df.to_csv(OUT / name, index=False)
    return df


# ======================================================================================== context (public data only)
net = Network()
arr = Arrays()
times = arr.times
G = Gauges(net, times)
assert list(G.names) == list(V.STATIONS)
NAMES = np.array(G.names).astype(str)
gt = pd.read_csv(REPO / "figures/section3/gauge_table.csv", dtype={"usgs_id": str}).set_index("gauge")
USGS = np.array([gt.loc[n, "usgs_id"] for n in NAMES])
nodes = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "totdasqkm"]).set_index("comid")
AREA = nodes.totdasqkm.reindex(net.ids[G.reach]).to_numpy(float)  # NHDPlus drainage area at the gauge reach (km2)
FOLD = {"gauged": np.zeros(G.count, int), "heldout": splits.gauge_folds("random", G, net), "blocked": splits.gauge_folds("blocked", G, net)}
SCHEME = {"heldout": "random", "blocked": "blocked"}
_wh = {}


def withheld(proto, k):
    if proto == "gauged" or k == 0:
        return np.zeros(G.count, bool)
    if (proto, k) not in _wh:
        _wh[(proto, k)] = splits.withheld(G, net, k, SCHEME[proto])
    return _wh[(proto, k)]


UP = [np.flatnonzero(np.isin(G.reach, G.basins[g]) & (np.arange(G.count) != g)) for g in range(G.count)]  # catalog gauges upstream of g
NESTED_CAT = np.array([len(u) > 0 for u in UP])  # another of the 128 catalog gauges upstream


def visible_up(proto, g):
    """Upstream catalog gauges of g that the model scoring g may see (not withheld in g's fold)."""
    w = withheld(proto, FOLD[proto][g])
    return UP[g][~w[UP[g]]]


# ======================================================================================== forecasts and scores
NWM_T, NWM_V = V.load_nwm()
DATA = {p: V.load_protocol(p, tuple(MODELS)) for p in PROTOS}
for p in PROTOS:
    DATA[p]["NWM"] = (NWM_T, NWM_V)
SC = {p: {m: V.scores(*DATA[p][m]) for m in DATA[p]} for p in PROTOS}
STEP = np.load(A / "eval_M0_s1" / "forecasts_gauged.npz", allow_pickle=True)["step"].astype(int)  # issue steps (same order in every file)
BASE = {p: np.isfinite(SC[p]["M0"]["nse"][3]) for p in PROTOS}  # the protocol's scored gauges (M0's scoring)
assert {p: int(BASE[p].sum()) for p in PROTOS} == V.N_GAUGES, {p: int(BASE[p].sum()) for p in PROTOS}
EV = BASE["gauged"]
ate = np.zeros(net.n, bool)
ate[G.reach[EV]] = True
NESTED_EV = np.array([ate[b].sum() > 1 for b in G.basins])  # another of the 101 evaluation gauges upstream (sensitivity)


def avail(p):
    return [m for m in ALL if m in SC[p]]


def scored(p, m, lead=3):
    return np.isfinite(SC[p][m]["nse"][lead])


def common(p, models):
    return BASE[p] & np.all([scored(p, m) for m in models], axis=0)


def d15(x):
    return V.days15(x)


def paired(p, a, b, idx=D15, mask=None):
    """Per-gauge mean NSE difference a - b over the leads idx, on the gauges both score (and mask)."""
    c = common(p, [a, b]) if mask is None else common(p, [a, b]) & mask
    with np.errstate(all="ignore"):
        da = np.nanmean(SC[p][a]["nse"][idx], axis=0)
        db = np.nanmean(SC[p][b]["nse"][idx], axis=0)
    return (da - db)[c], c


# ======================================================================================== 1. run inventory
def stage_b(name):
    d = A / name
    if not (d / "DONE").exists():
        return None
    s = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    return int(s["update"])


def lstm_best(name):
    d = A / name
    if not (d / "DONE").exists():
        return None
    v = pd.read_csv(d / "validation_progress.csv")
    return int(v.loc[v["best"], "update"].iloc[-1])


RUNS = {
    "gauged": {"M0": ["M0_s1"], "M1": ["M1_s1"], "M2": ["M2_s1"], "M4": [], "LSTM": ["ladder_L4_s1"]},
    "heldout": {m: [f"{m}_s1_f{k}" for k in range(1, 6)] for m in ("M0", "M1", "M2", "M4")} | {"LSTM": [f"ladder_L4_s1_f{k}" for k in range(1, 6)]},
    "blocked": {m: [f"S1_{m}_s1_b{k}" for k in range(1, 4)] for m in ("M0", "M1", "M2", "M4")} | {"LSTM": [f"ladder_L4_s1_b{k}" for k in range(1, 4)]},
}
PRED = {"gauged": {m: f"{V.HYDRO_FILES['gauged'].get(m, '-')}/forecasts_gauged.npz" for m in ("M0", "M1", "M2", "M4")},
        "heldout": {m: f"{V.HYDRO_FILES['heldout'].get(m, '-')}/forecasts_ungauged.npz" for m in ("M0", "M1", "M2", "M4")},
        "blocked": {m: f"{V.HYDRO_FILES['blocked'].get(m, '-')}/forecasts_ungauged.npz" for m in ("M0", "M1", "M2", "M4")}}
inv = []
for p in PROTOS:
    expected = {"gauged": 1, "heldout": 5, "blocked": 3}[p]
    for m in MODELS:
        names = RUNS[p][m]
        if m == "M4" and p == "gauged":
            inv.append(dict(protocol=PNAME[p], model=m, runs_expected=0, runs_done=0, selected_updates="-", predictions="MISSING (not planned: no all-gauge M4)"))
            continue
        best = [lstm_best(n) if m == "LSTM" else stage_b(n) for n in names]
        done = sum(b is not None for b in best)
        sel = ", ".join("MISSING" if b is None else str(b) for b in best)
        pred = "yes" if m in SC[p] else "MISSING (forecasts not saved)"
        inv.append(dict(protocol=PNAME[p], model=m, runs_expected=expected, runs_done=done, selected_updates=sel, predictions=pred))
    inv.append(dict(protocol=PNAME[p], model="NWM", runs_expected=0, runs_done=0, selected_updates="-", predictions="yes (operational archive)"))
INV = save(pd.DataFrame(inv), "s5_1_inventory.csv")

# ======================================================================================== 2. headline skill and lead curves
head, curves = [], []
for p in PROTOS:
    ms = avail(p)
    c = common(p, ms)
    for m in ms:
        s = SC[p][m]
        row = dict(protocol=PNAME[p], model=m, n=int(c.sum()))
        for k in ("nse", "kge", "r", "alpha", "beta"):
            row[f"{k}_days1_5"] = V.days15_of_median(s[k], c)
        for h, i in zip(LEADS5, D15):
            row[f"nse_{h}h"] = float(np.nanmedian(s["nse"][i][c]))
            row[f"kge_{h}h"] = float(np.nanmedian(s["kge"][i][c]))
        head.append(row)
        for i in range(40):
            curves.append(dict(protocol=p, model=m, n=int(c.sum()), lead_h=6 * (i + 1), median_nse=float(np.nanmedian(s["nse"][i][c]))))
HEAD = save(pd.DataFrame(head), "s5_2_headline.csv")
save(pd.DataFrame(curves), "s5_2_lead_curves.csv")

# ======================================================================================== 3. paired contrasts (days 1-5)
con = []
for p in PROTOS:
    rows = []
    for a, b in CONTRASTS:
        if a not in SC[p] or b not in SC[p]:
            rows.append(dict(protocol=PNAME[p], contrast=f"{a} - {b}", n=0, note="MISSING: " + ", ".join(x for x in (a, b) if x not in SC[p])))
            continue
        d, c = paired(p, a, b)
        mean, lo, hi = V.boot_mean(d, q=(5, 95))
        rows.append(dict(protocol=PNAME[p], contrast=f"{a} - {b}", n=len(d), mean_diff=mean, ci5=lo, ci95=hi, median_diff=float(np.median(d)),
                         share_first_better=float((d > 0).mean()), wilcoxon_p=V.wilcoxon_p(d), tost_p_0p01=V.tost(d, 0.01), tost_p_0p02=V.tost(d, 0.02), note=""))
    t = pd.DataFrame(rows)
    ok = t.n > 0
    for col in ("wilcoxon_p", "tost_p_0p01", "tost_p_0p02"):
        t.loc[ok, col + "_holm"] = V.holm(t.loc[ok, col].to_numpy())  # across the contrasts of this protocol
    t["equivalent_0p01"] = ok & (t.get("tost_p_0p01_holm", np.nan) < 0.05)
    t["equivalent_0p02"] = ok & (t.get("tost_p_0p02_holm", np.nan) < 0.05)
    # status by the writing rule: equivalent within 0.02 (Holm-corrected TOST), different (Holm-corrected Wilcoxon p < 0.05 and the 5-95% interval
    # excludes zero), otherwise undetermined
    diff = ok & (t.get("wilcoxon_p_holm", np.nan) < 0.05) & ((t.get("ci5", np.nan) > 0) | (t.get("ci95", np.nan) < 0))
    t["status"] = np.where(~ok, "MISSING", np.where(t.equivalent_0p02, "equivalent (0.02)", np.where(diff, "different", "undetermined")))
    con.append(t)
CON = save(pd.concat(con, ignore_index=True), "s5_3_contrasts.csv")

# ======================================================================================== 4. contrasts by lead
bylead = []
for p in PROTOS:
    for a, b in CONTRASTS:
        if a not in SC[p] or b not in SC[p]:
            continue
        for h, i in zip(LEADS5, D15):
            d, _ = paired(p, a, b, idx=[i])
            d = d[np.isfinite(d)]
            mean, lo, hi = V.boot_mean(d, q=(5, 95))
            bylead.append(dict(protocol=PNAME[p], contrast=f"{a} - {b}", lead_h=h, n=len(d), mean_diff=mean, ci5=lo, ci95=hi))
BYLEAD = save(pd.DataFrame(bylead), "s5_4_contrasts_by_lead.csv")
flips = []
for (p, cn), grp in BYLEAD.groupby(["protocol", "contrast"], sort=False):
    sg = np.sign(grp.mean_diff.to_numpy())
    ch = [f"{int(grp.lead_h.iloc[k - 1])}->{int(grp.lead_h.iloc[k])} h" for k in range(1, len(sg)) if sg[k] != sg[k - 1]]
    flips.append(dict(protocol=p, contrast=cn, sign_changes=", ".join(ch) if ch else "none"))
FLIPS = save(pd.DataFrame(flips), "s5_4_sign_changes.csv")
# full 40-lead paired curves for the contrasts (figures)
c40 = []
for p in PROTOS:
    for a, b in CONTRASTS:
        if a not in SC[p] or b not in SC[p]:
            continue
        for i in range(40):
            d, _ = paired(p, a, b, idx=[i])
            d = d[np.isfinite(d)]
            mean, lo, hi = V.boot_mean(d, q=(5, 95))
            c40.append(dict(protocol=p, contrast=f"{a} - {b}", lead_h=6 * (i + 1), n=len(d), mean_diff=mean, ci5=lo, ci95=hi))
save(pd.DataFrame(c40), "s5_4_contrasts_40_leads.csv")

# ======================================================================================== 5. upstream observations (RQ3)
# (a) M0 - M1 by position
pos_rows, pos_lead = [], []
for p in PROTOS:
    if not all(m in SC[p] for m in ("M0", "M1")):
        continue
    vis = np.array([len(visible_up(p, g)) > 0 for g in range(G.count)])
    for split, groups in (("visible upstream gauge (protocol)", (("nested", vis), ("headwater", ~vis))),
                          ("128-gauge definition", (("nested", NESTED_CAT), ("headwater", ~NESTED_CAT))),
                          ("101-gauge definition", (("nested", NESTED_EV), ("headwater", ~NESTED_EV)))):
        for gname, mask in groups:
            d, c = paired(p, "M0", "M1", mask=mask)
            if len(d) == 0:
                pos_rows.append(dict(protocol=PNAME[p], definition=split, group=gname, n=0))
                continue
            mean, lo, hi = V.boot_mean(d, q=(5, 95))
            pos_rows.append(dict(protocol=PNAME[p], definition=split, group=gname, n=len(d), mean_diff=mean, ci5=lo, ci95=hi,
                                 median_diff=float(np.median(d)), share_M0_better=float((d > 0).mean()), wilcoxon_p=V.wilcoxon_p(d)))
            if split.startswith("visible"):
                for i in range(40):
                    di, _ = paired(p, "M0", "M1", idx=[i], mask=mask)
                    di = di[np.isfinite(di)]
                    if len(di) == 0:
                        continue
                    mi, li, hi_ = V.boot_mean(di, q=(5, 95))
                    pos_lead.append(dict(protocol=p, group=gname, lead_h=6 * (i + 1), n=len(di), mean_diff=mi, median_diff=float(np.median(di)), ci5=li, ci95=hi_))
POS = save(pd.DataFrame(pos_rows), "s5_5a_position.csv")
save(pd.DataFrame(pos_lead), "s5_5a_position_by_lead.csv")

# (b) within-gauge: issues with a visible upstream gauge against issues without
FIN = np.isfinite(G.observed)
CUM = np.vstack([np.zeros((1, G.count)), np.cumsum(FIN, axis=0)])
RECENT = CUM[STEP] - CUM[STEP - P.INNOVATION]  # [issue, gauge] values in the last 32 steps before each issue
ELIG = RECENT >= 8
within = []
for p in ("gauged", "heldout"):
    if not all(m in SC[p] for m in ("M0", "M1")):
        continue
    (t0, v0), (t1, v1) = DATA[p]["M0"], DATA[p]["M1"]
    for g in np.flatnonzero(BASE[p]):
        ups = visible_up(p, g)
        if len(ups) == 0:
            continue
        seen = ELIG[:, ups].any(axis=1)  # an upstream gauge visible at this issue
        row = dict(protocol=PNAME[p], gauge=NAMES[g], usgs_id=USGS[g], area_km2=AREA[g], issues=len(seen), share_issues_visible=float(seen.mean()))
        for lab, sel in (("visible", seen), ("not_visible", ~seen)):
            def sub(t, v):
                tt, vv = t[:, :, [g]].copy(), v[:, :, [g]].copy()
                tt[~sel] = np.nan
                vv[~sel] = np.nan
                return V.scores(tt, vv, min_count=MIN_SUBSET)["nse"][:, 0]
            n0, n1 = sub(t0, v0), sub(t1, v1)
            with np.errstate(all="ignore"):
                row[f"gain_{lab}"] = float(np.nanmean((n0 - n1)[D15])) if np.isfinite((n0 - n1)[D15]).all() else np.nan
        within.append(row)
WITHIN = save(pd.DataFrame(within), "s5_5b_within_gauge.csv")
wsum = []
for p, grp in WITHIN.groupby("protocol", sort=False):
    both = grp.dropna(subset=["gain_visible", "gain_not_visible"])
    d = (both.gain_visible - both.gain_not_visible).to_numpy()
    row = dict(protocol=p, nested_gauges=len(grp), with_both_subsets=len(both), median_share_issues_visible=float(grp.share_issues_visible.median()))
    if len(d):
        mean, lo, hi = V.boot_mean(d, q=(5, 95))
        row.update(mean_gain_visible=float(both.gain_visible.mean()), mean_gain_not_visible=float(both.gain_not_visible.mean()),
                   mean_diff=mean, ci5=lo, ci95=hi, wilcoxon_p=V.wilcoxon_p(d))
    wsum.append(row)
WSUM = save(pd.DataFrame(wsum), "s5_5b_within_gauge_summary.csv")

# (c) area-matched
abin = np.digitize(AREA, AREA_EDGES[1:-1])
binrows, matchrows = [], []
for p in PROTOS:
    if not all(m in SC[p] for m in ("M0", "M1")):
        continue
    vis = np.array([len(visible_up(p, g)) > 0 for g in range(G.count)])
    gain = np.full(G.count, np.nan)
    d, c = paired(p, "M0", "M1")
    gain[np.flatnonzero(c)] = d
    for i, lab in enumerate(AREA_LABELS):
        row = dict(protocol=PNAME[p], area_km2=lab)
        for gname, mask in (("nested", vis), ("headwater", ~vis)):
            sel = c & mask & (abin == i)
            row[f"n_{gname}"] = int(sel.sum())
            row[f"mean_gain_{gname}"] = float(np.mean(gain[sel])) if sel.any() else np.nan
        binrows.append(row)
    nest = np.flatnonzero(c & vis)
    head_ = list(np.flatnonzero(c & ~vis))
    pairs = []
    for g in nest[np.argsort(AREA[nest])]:
        if not head_:
            break
        dist = np.abs(np.log10(AREA[head_]) - np.log10(AREA[g]))
        j = int(np.argmin(dist))
        if dist[j] <= CALIPER:
            pairs.append((g, head_.pop(j)))
    row = dict(protocol=PNAME[p], nested_gauges=len(nest), matched_pairs=len(pairs), caliper_log10=CALIPER)
    if pairs:
        dd = np.array([gain[a] - gain[b] for a, b in pairs])
        mean, lo, hi = V.boot_mean(dd, q=(5, 95))
        row.update(mean_gain_nested=float(np.mean([gain[a] for a, _ in pairs])), mean_gain_headwater=float(np.mean([gain[b] for _, b in pairs])),
                   mean_diff=mean, ci5=lo, ci95=hi, wilcoxon_p=V.wilcoxon_p(dd),
                   median_area_nested=float(np.median([AREA[a] for a, _ in pairs])), median_area_headwater=float(np.median([AREA[b] for _, b in pairs])))
    matchrows.append(row)
BINS = save(pd.DataFrame(binrows), "s5_5c_area_bins.csv")
MATCH = save(pd.DataFrame(matchrows), "s5_5c_matched.csv")

# (d) gain at days 1-2 against area ratio and travel time from the trained M0 router (and M2, item 6)
DOWN, _, _ = main_path(net.edges, net.n)
CH = channel_table(net)
QREF = torch.as_tensor(net.mean_raw, dtype=torch.float32)  # mean NWM retrospective flow per reach (training years), the router's reference flow
_rt = {}


def reach_times(ckpt):
    """Reach travel times (steps) of a trained Manning router at the mean NWM flow."""
    if ckpt not in _rt:
        s = torch.load(A / ckpt / "best.pt", map_location="cpu", weights_only=False)
        m = build_model(s["config"])
        m.load_state_dict(s["model"])
        m.eval()
        m.router.attach(net, G.reach, torch.device("cpu"), table=CH)
        with torch.no_grad():
            _rt[ckpt] = m.router.reach_time(QREF).numpy().astype(float)
    return _rt[ckpt]


def path_to(j, i):
    path, r = [j], j
    for _ in range(net.n):
        if r == i or DOWN[r] < 0:
            break
        r = DOWN[r]
        path.append(r)
    return path if path[-1] == i else None


def pair(ckpt, j, i):
    """Pair travel time (h) from the middle of reach j to the end of reach i along the main path, and its river length (km)."""
    pth = path_to(j, i)
    if pth is None:
        return np.nan, np.nan
    t = reach_times(ckpt)
    return float((0.5 * t[j] + t[pth[1:]].sum()) * 6.0), float(0.5 * CH["length"][j] + CH["length"][pth[1:]].sum())


gainrows = []
for p in ("gauged", "heldout"):
    if not all(m in SC[p] for m in ("M0", "M1")):
        continue
    d12, c12 = paired(p, "M0", "M1", idx=D12)
    gain12 = np.full(G.count, np.nan)
    gain12[np.flatnonzero(c12)] = d12
    for g in np.flatnonzero(c12):
        ups = visible_up(p, g)
        if len(ups) == 0:
            continue
        u = ups[np.argmax(AREA[ups])]  # nearest visible upstream gauge: the one with the largest drainage area
        ck = "M0_s1" if p == "gauged" else f"M0_s1_f{FOLD['heldout'][g]}"
        tau, length = pair(ck, G.reach[u], G.reach[g])
        gainrows.append(dict(protocol=PNAME[p], gauge=NAMES[g], usgs_id=USGS[g], area_km2=AREA[g], upstream_gauge=NAMES[u], upstream_usgs_id=USGS[u],
                             upstream_area_km2=AREA[u], area_ratio=AREA[u] / AREA[g], river_km=length, travel_time_h_M0=tau, router=ck,
                             gain_days1_2=gain12[g]))
GAIN = save(pd.DataFrame(gainrows), "s5_5d_gain_vs_ratio.csv")
gcor = []
for p, grp in GAIN.groupby("protocol", sort=False):
    for var in ("area_ratio", "travel_time_h_M0"):
        ok = grp[[var, "gain_days1_2"]].dropna()
        rho, pv = stats.spearmanr(ok[var], ok.gain_days1_2)
        gcor.append(dict(protocol=p, variable=var, n=len(ok), spearman_rho=float(rho), p_value=float(pv)))
GCOR = save(pd.DataFrame(gcor), "s5_5d_rank_correlations.csv")

# ---- 5e. noise floor: the blocked protocol, where insertion cannot act ------------------------------------------------------------
nf = []
if all(m in SC["blocked"] for m in ("M0", "M1")):
    d, c = paired("blocked", "M0", "M1")
    fold_of = FOLD["blocked"]
    idx = np.flatnonzero(c)
    nf.append(dict(unit="all blocked gauges", n=len(d), mean_diff=float(d.mean()), median_diff=float(np.median(d))))
    for k in (1, 2, 3):
        sel = fold_of[idx] == k
        nf.append(dict(unit=f"basin B{k}", n=int(sel.sum()), mean_diff=float(d[sel].mean()), median_diff=float(np.median(d[sel]))))
NOISE = save(pd.DataFrame(nf), "s5_5e_noise_floor.csv")

# ---- 5f. robustness of the area-ratio correlation (exploratory) ----------------------------------------------------------------------
rob = []
rng = np.random.default_rng(0)
for prot, grp in GAIN.groupby("protocol", sort=False):
    ok = grp[["area_ratio", "gain_days1_2", "gauge"]].dropna().reset_index(drop=True)
    x, y = ok.area_ratio.to_numpy(), ok.gain_days1_2.to_numpy()
    rho0 = stats.spearmanr(x, y)[0]
    loo = np.array([stats.spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))])
    far = np.argsort(-np.abs(y - np.median(y)))[:3]  # the three gauges with the largest absolute gain
    keep = np.setdiff1d(np.arange(len(x)), far)
    boots = [stats.spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
    rob.append(dict(protocol=prot, n=len(x), rho=float(rho0), loo_min=float(loo.min()), loo_max=float(loo.max()),
                    rho_without_3_largest_gains=float(stats.spearmanr(x[keep], y[keep])[0]), p_without_3=float(stats.spearmanr(x[keep], y[keep])[1]),
                    boot_ci5=float(np.nanpercentile(boots, 5)), boot_ci95=float(np.nanpercentile(boots, 95)),
                    largest_gains=", ".join(f"{ok.gauge[i]} ({y[i]:+.3f})" for i in far)))
ROB = save(pd.DataFrame(rob), "s5_5f_area_ratio_robustness.csv")

# ---- 5g. gauge-based skill by drainage-area stratum (real observations; the smallest gauges) ----------------------------------------------------
sedges = [0, 50, 200, 600, 2000, np.inf]
slab = ["< 50", "50-200", "200-600", "600-2,000", "> 2,000"]
sbin = np.digitize(AREA, sedges[1:-1])
srows = []
for p in PROTOS:
    for i, lab in enumerate(slab):
        row = dict(protocol=PNAME[p], area_km2=lab)
        for m in avail(p):
            c = common(p, avail(p)) & (sbin == i)
            row["n"] = int(c.sum())
            row[f"{m}_median_nse"] = V.days15_of_median(SC[p][m]["nse"], c) if c.any() else np.nan
        srows.append(row)
SSTR = save(pd.DataFrame(srows), "s5_5g_skill_by_area.csv")
sm = common("gauged", avail("gauged")) & (AREA < 200)
SMALL = dict(n_lt50=int(((AREA < 50) & EV).sum()), n_lt200=int(((AREA < 200) & EV).sum()), min_area=float(AREA[EV].min()))

# ======================================================================================== 6. routing and message passing (RQ2)
strata = []
for p in PROTOS:
    for a, b in (("M0", "M2"), ("M1", "M4")):
        if a not in SC[p] or b not in SC[p]:
            strata.append(dict(protocol=PNAME[p], contrast=f"{a} - {b}", area_km2="all", n=0, note="MISSING"))
            continue
        for i, lab in enumerate(AREA_LABELS):
            d, _ = paired(p, a, b, mask=abin == i)
            row = dict(protocol=PNAME[p], contrast=f"{a} - {b}", area_km2=lab, n=len(d), note="")
            if len(d) >= 3:
                mean, lo, hi = V.boot_mean(d, q=(5, 95))
                row.update(mean_diff=mean, ci5=lo, ci95=hi, share_first_better=float((d > 0).mean()))
            strata.append(row)
STRATA = save(pd.DataFrame(strata), "s5_6_area_strata.csv")
tt = []
for g in np.flatnonzero(EV):  # nested evaluation gauges and their nearest upstream catalog gauge (gauged protocol: nothing withheld)
    if len(UP[g]) == 0:
        continue
    u = UP[g][np.argmax(AREA[UP[g]])]
    t0, length = pair("M0_s1", G.reach[u], G.reach[g])
    t2, _ = pair("M2_s1", G.reach[u], G.reach[g])
    tt.append(dict(gauge=NAMES[g], usgs_id=USGS[g], upstream_usgs_id=USGS[u], river_km=length, travel_time_h_M0=t0, travel_time_h_M2=t2,
                   celerity_m_s_M0=length * 1000 / (t0 * 3600) if t0 > 0 else np.nan, celerity_m_s_M2=length * 1000 / (t2 * 3600) if t2 > 0 else np.nan))
TT = save(pd.DataFrame(tt), "s5_6_travel_times.csv")

# high flow (the logic of scripts/v3_kge_highflow.py, here on the section-5 model set)
train = times < pd.Timestamp("2019-01-01")
wy = (times[train] + pd.DateOffset(months=3)).year.to_numpy()
Q2 = np.full(G.count, np.nan)
for g in range(G.count):
    by = pd.Series(G.observed[train, g]).groupby(wy)
    good = by.count() / by.size() >= 0.7
    mx = by.max()[good]
    if len(mx) >= 10:
        Q2[g] = float(np.median(mx))
PEAK_BINS = {"peak <= 24 h": (0, 4), "peak 24-72 h": (4, 12), "peak 72-120 h": (12, 20), "peak 120-240 h": (20, 40)}


def highflow(truth, values, g):
    o, f = truth[:, :, g].astype(float), values[:, :, g].astype(float)
    row = {}
    bias = []
    for lead in D15:
        ok = np.isfinite(o[:, lead]) & np.isfinite(f[:, lead])
        oo, ff = o[ok, lead], f[ok, lead]
        if len(oo) >= 100:
            top = oo >= np.percentile(oo, 95)
            bias.append(ff[top].mean() / oo[top].mean() - 1)
    row["peak_bias_days1_5"] = np.mean(bias) if bias else np.nan
    full = np.isfinite(o).all(1) & np.isfinite(f).all(1)
    if np.isfinite(Q2[g]) and full.any():
        ow, fw = o[full], f[full]
        po, pf = ow.argmax(1), fw.argmax(1)
        flood = ow.max(1) >= Q2[g]
        for lab, (lo, hi) in PEAK_BINS.items():
            sel = flood & (po >= max(lo, 1)) & (po < min(hi, 39))
            row[f"events_{lab}"] = int(sel.sum())
            if sel.sum() >= 3:
                err = (pf[sel] - po[sel]) * 6.0
                row[f"timing_abs_h_{lab}"] = float(np.median(np.abs(err)))
                row[f"magnitude_rel_error_{lab}"] = float(np.median(fw[sel].max(1) / ow[sel].max(1) - 1))
    return row


hf = []
for p in PROTOS:
    ms = avail(p)
    c = common(p, ms)
    for m in ms:
        t_, v_ = DATA[p][m]
        per = pd.DataFrame([highflow(t_, v_, g) for g in np.flatnonzero(c)])
        row = dict(protocol=PNAME[p], model=m, n=int(c.sum()), median_peak_bias_days1_5=float(per.peak_bias_days1_5.median()))
        for lab in PEAK_BINS:
            col = f"timing_abs_h_{lab}"
            row[f"gauges_{lab}"] = int(per[col].notna().sum()) if col in per else 0
            row[f"median_timing_abs_h_{lab}"] = float(per[col].median()) if col in per else np.nan
            row[f"median_magnitude_rel_error_{lab}"] = float(per[f"magnitude_rel_error_{lab}"].median()) if col in per else np.nan
        hf.append(row)
HF = save(pd.DataFrame(hf), "s5_6_highflow.csv")

# ======================================================================================== 7. every reach
rm = np.load(A / "eval_M0_reach" / "reach_metrics.npz", allow_pickle=True)
lm = np.load(A / "ladder_L4_s1_reaches" / "metrics.npz", allow_pickle=True)
assert (rm["comid"] == net.ids).all() and (lm["comid"] == net.ids).all()
RF = rm["reach_fold"]
RAREA = nodes.totdasqkm.reindex(net.ids).to_numpy(float)
RCLASS = [("< 5", 0, 5), ("5-50", 5, 50), ("50-500", 50, 500), ("500-5,000", 500, 5000), ("> 5,000", 5000, np.inf)]
REACH = {"M0": (rm["M0+gefs/nse"], rm["M0+gefs/kge"]), "LSTM": (lm["nse/gefs"], lm["kge/gefs"]), "NWM": (rm["NWM forecast/nse"], rm["NWM forecast/kge"])}
rrows = []
for scope, smask in (("all reaches", np.ones(net.n, bool)), ("inside the basin of a withheld gauge (reach_fold > 0)", RF > 0),
                     ("outside (reach_fold <= 0)", RF <= 0)):
    for lab, lo, hi in RCLASS:
        cls = smask & (RAREA >= lo) & (RAREA < hi)
        row = dict(scope=scope, area_km2=lab, reaches=int(cls.sum()))
        for m in ("M0", "M1", "M2", "M4", "LSTM", "NWM"):
            if m not in REACH:
                row[f"{m}_median_nse"] = "MISSING"
                continue
            nse = d15(REACH[m][0])
            kge = d15(REACH[m][1])
            ok = cls & np.isfinite(nse)
            row[f"{m}_scored"] = int(ok.sum())
            row[f"{m}_median_nse"] = float(np.median(nse[ok])) if ok.any() else np.nan
            row[f"{m}_nse_q25"] = float(np.percentile(nse[ok], 25)) if ok.any() else np.nan
            row[f"{m}_nse_q75"] = float(np.percentile(nse[ok], 75)) if ok.any() else np.nan
            row[f"{m}_median_kge"] = float(np.median(kge[ok])) if ok.any() else np.nan
            row[f"{m}_share_nse_above_0"] = float((nse[ok] > 0).mean()) if ok.any() else np.nan
        rrows.append(row)
REACHT = save(pd.DataFrame(rrows), "s5_7_every_reach.csv")
import xarray as xr  # noqa: E402

with xr.open_dataset(P.PROCESSED / "graph_candidate/static_core.nc") as _d:
    _f = list(_d.static_feature.values.astype(str))
    ELEV = _d.static_values.values[:, _f.index("elevcat")].astype(float)
    ELEV_OK = _d.static_valid.values[:, _f.index("elevcat")].astype(bool)
ELEV = np.where(ELEV_OK, ELEV, np.nan)
HEADWATER = ~np.isin(np.arange(net.n), net.edges[:, 1])  # a reach with no reach upstream of it (first order)
RORDER = pd.read_parquet(P.PROCESSED / "graph_candidate/nodes.parquet", columns=["comid", "streamorde"]).set_index("comid").streamorde.reindex(net.ids).to_numpy()
NWMR = rm["NWM forecast/nse"].astype(float)
PR = pd.DataFrame(dict(comid=net.ids, area_km2=RAREA, reach_fold=RF, is_headwater=HEADWATER, stream_order=RORDER, elev_m=ELEV,
                       nse_M0=d15(REACH["M0"][0]), nse_LSTM=d15(REACH["LSTM"][0]), nse_NWM=d15(NWMR)))
save(PR, "s5_perreach_nse.csv")  # per-reach days 1-5 NSE against the NWM retrospective (M0, LSTM, NWM forecast), headwater flag, elevation: reach maps and headwater plots
PRL = pd.DataFrame(dict(comid=net.ids, area_km2=RAREA))
for _h in (6, 12, 18):  # lead h = 6 * (index + 1): per-reach NSE at one lead against the NWM retrospective
    PRL[f"nse_M0_{_h}h"] = REACH["M0"][0][_h // 6 - 1].astype(float)
    PRL[f"nse_LSTM_{_h}h"] = REACH["LSTM"][0][_h // 6 - 1].astype(float)
save(PRL, "s5_perreach_nse_leads.csv")  # per-reach NSE at leads 6, 12, 18 h (M0, LSTM): reach-map grid
# lead curves of the per-reach median NSE for headwater reaches and other small and large reaches
CLS = [("headwater reaches (no reach upstream)", HEADWATER), ("other reaches < 50 km2", ~HEADWATER & (RAREA < 50)), ("reaches >= 50 km2", RAREA >= 50)]
hl = []
for lab, mask in CLS:
    for m, arr in (("M0", REACH["M0"][0]), ("LSTM", REACH["LSTM"][0]), ("NWM", NWMR)):
        for i in range(40):
            x = arr[i][mask]
            x = x[np.isfinite(x)]
            hl.append(dict(reach_class=lab, model=m, lead_h=6 * (i + 1), reaches=int(mask.sum()), scored=len(x), median_nse=float(np.median(x)) if len(x) else np.nan,
                           q25=float(np.percentile(x, 25)) if len(x) else np.nan, q75=float(np.percentile(x, 75)) if len(x) else np.nan))
HLC = save(pd.DataFrame(hl), "s5_headwater_lead_curves.csv")
ROUTES = pd.DataFrame([dict(route="inside the basin of a random-fold gauge (fold model)", reaches=int((RF > 0).sum())),
                       dict(route="outside every gauge basin (all-gauge model)", reaches=int((RF == -1).sum())),
                       dict(route="inside the basin of a gauge in no fold (all-gauge model)", reaches=int((RF == 0).sum()))])
save(ROUTES, "s5_7_routes.csv")
REACH_ISSUES = int(lm["issues"])
MAN_REACH = json.loads((A / "eval_M0_reach" / "manifest.json").read_text())

# ======================================================================================== 8. example event (fixed rule)
EX = None
if all(m in SC["heldout"] for m in ("M0", "M1", "LSTM")):
    d, c = paired("heldout", "M0", "M1")
    gain = pd.Series(d, index=np.flatnonzero(c))
    cand = [g for g in gain.index if len(visible_up("heldout", g)) > 0]
    ranked = gain[cand].sort_values()
    g = int(ranked.index[(len(ranked) - 1) // 2])  # lower median gain among the nested gauges with a visible upstream gauge
    test = (times >= pd.Timestamp("2020-09-23")) & (times < pd.Timestamp("2026-01-01"))
    tidx = np.flatnonzero(test)
    peak = int(tidx[np.nanargmax(G.observed[test, g])])
    lo_t, hi_t = times[peak] - pd.Timedelta(days=5), times[peak] + pd.Timedelta(days=5)
    rows = []
    series = {"M0": DATA["heldout"]["M0"][1], "M1": DATA["heldout"]["M1"][1], "LSTM": DATA["heldout"]["LSTM"][1], "NWM": NWM_V}
    for h in (6, 12, 18, 24):
        li = h // 6 - 1
        for k, s in enumerate(STEP):
            vt = times[s + li]
            if lo_t <= vt <= hi_t:
                row = dict(lead_h=h, issue=str(times[s]), valid_time=str(vt), observed=float(G.observed[s + li, g]))
                for m, v in series.items():
                    row[m] = float(v[k, li, g])
                rows.append(row)
    EXS = save(pd.DataFrame(rows), "s5_8_example_event.csv")
    ups = visible_up("heldout", g)
    EX = dict(gauge=NAMES[g], usgs_id=USGS[g], area_km2=AREA[g], gain_days1_5=float(gain[g]), rank=f"{(len(ranked) - 1) // 2 + 1} of {len(ranked)}",
              upstream=", ".join(f"{NAMES[u]} ({USGS[u]})" for u in ups), peak_time=str(times[peak]), peak_flow=float(G.observed[peak, g]),
              fold=int(FOLD["heldout"][g]))

# ======================================================================================== per-gauge table (maps)
pg = []
for p in PROTOS:
    for g in np.flatnonzero(BASE[p]):
        row = dict(protocol=p, gauge=NAMES[g], usgs_id=USGS[g], lat=float(gt.loc[NAMES[g], "lat"]), lon=float(gt.loc[NAMES[g], "lon"]), area_km2=AREA[g],
                   nested_visible=bool(len(visible_up(p, g)) > 0), fold=int(FOLD[p][g]))
        for m in avail(p):
            v = d15(SC[p][m]["nse"])[g] if scored(p, m)[g] else np.nan
            row[f"nse_{m}"] = float(v)
        pg.append(row)
save(pd.DataFrame(pg), "s5_pergauge_nse.csv")
# per-gauge NSE at single leads 6, 12, 18 h (lead h = 6 * (index + 1)): gauge-map grids by lead
pgl = []
for p in PROTOS:
    for g in np.flatnonzero(BASE[p]):
        row = dict(protocol=p, gauge=NAMES[g], usgs_id=USGS[g], lat=float(gt.loc[NAMES[g], "lat"]), lon=float(gt.loc[NAMES[g], "lon"]), area_km2=AREA[g])
        for m in avail(p):
            for h in (6, 12, 18):
                row[f"nse_{m}_{h}h"] = float(SC[p][m]["nse"][h // 6 - 1][g]) if scored(p, m)[g] else np.nan
        pgl.append(row)
save(pd.DataFrame(pgl), "s5_pergauge_nse_leads.csv")

# ======================================================================================== 9. caveat numbers
few = []
for p in PROTOS:
    n24 = SC[p]["M0"]["n"][3]
    sel = BASE[p]
    few.append(dict(protocol=PNAME[p], gauges=int(sel.sum()), min_pairs_24h=int(n24[sel].min()), gauges_under_300_pairs=int((n24[sel] < 300).sum()),
                    gauges_under_500_pairs=int((n24[sel] < 500).sum())))
FEW = save(pd.DataFrame(few), "s5_9_test_pairs.csv")
nwmb = []
for p in PROTOS:
    d, c = paired(p, "M0", "NWM")
    nwmb.append(dict(protocol=PNAME[p], n=len(d), gauges_NWM_better=int((d < 0).sum())))
NWMB = save(pd.DataFrame(nwmb), "s5_9_nwm_better.csv")
sign_proto = []
for cn, grp in CON[CON.n > 0].groupby("contrast", sort=False):
    s = np.sign(grp.mean_diff.to_numpy())
    sign_proto.append(dict(contrast=cn, protocols=", ".join(grp.protocol), signs=", ".join("+" if x > 0 else "-" for x in s), flips=bool(len(set(s)) > 1)))
SIGNP = save(pd.DataFrame(sign_proto), "s5_9_sign_by_protocol.csv")

# ======================================================================================== write the facts sheet
add("# Section 5 facts sheet: results")
add()
add("Every number is computed by `scripts/section5/s5_facts.py` (run: `python scripts/section5/s5_facts.py`) from the saved forecasts in "
    "`<data>/models/attr` and the checkpoints, with the scoring and statistics of `scripts/v3_common.py` (`scores`: NSE, KGE (2009) and its "
    "parts, a gauge scored with at least 100 values, positive mean and std >= 1e-3 m3/s; `boot_mean`: 10,000 gauge resamples, seed 0, 5-95%; "
    "`wilcoxon_p`; `tost`; `holm`). Every table is also a CSV in `reports/section5/` (file named in each heading). Days 1-5 = leads 24, 48, 72, 96, "
    "120 h. A median over gauges is taken per lead and then averaged over the five leads; a paired difference is the per-gauge mean over the five "
    "leads. No dam or operator data is used. MISSING = forecasts not saved; nothing is filled in.")
add()
add("## 1. Run inventory (`s5_1_inventory.csv`)")
add()
add("Selected update = the checkpoint `best.pt` (M models: one of 500, 1,000, ..., 3,000 stage-B updates; LSTM: one of 10 evaluations of 68,767 updates). One seed (seed 1) per run.")
add()
table(INV)
add("M4: trained and evaluated for the random hold-out (5 folds) and the blocked basins (3 basins) (`eval_M4_s1`, `eval_S1_M4_s1`, both `--heldout-only`); "
    "no all-gauge M4 was planned, so M4 has no gauged-protocol entries.")
add()
add("## 2. Headline skill (`s5_2_headline.csv`; lead curves over all 40 leads: `s5_2_lead_curves.csv`)")
add()
for p in PROTOS:
    h = HEAD[HEAD.protocol == PNAME[p]]
    add(f"**{PNAME[p]}** (n = {int(h.n.iloc[0])} gauges scored by every model shown)")
    add()
    t = h[["model", "nse_days1_5", "kge_days1_5", "r_days1_5", "alpha_days1_5", "beta_days1_5"] + [f"nse_{x}h" for x in LEADS5] + [f"kge_{x}h" for x in LEADS5]]
    table(t)
add("## 3. Paired contrasts, days 1-5 (`s5_3_contrasts.csv`)")
add()
add("Difference = first minus second model, per gauge, on the gauges both score. Interval = bootstrap 5-95% of the mean. Holm correction across "
    "the contrasts of one protocol (Wilcoxon p, and separately the TOST p at each margin). Equivalent = Holm-corrected TOST p < 0.05. Status: equivalent (0.02) = equivalent at the 0.02 margin; different = Holm-corrected Wilcoxon p < 0.05 and the interval excludes zero; undetermined = neither. The primary quantity is the mean paired difference; the median difference is given beside it (they can disagree).")
add()
for p in PROTOS:
    t = CON[CON.protocol == PNAME[p]].copy()
    add(f"**{PNAME[p]}**")
    add()
    t = t.reindex(columns=["contrast", "n", "mean_diff", "ci5", "ci95", "median_diff", "share_first_better", "wilcoxon_p", "wilcoxon_p_holm",
                           "tost_p_0p01_holm", "tost_p_0p02_holm", "equivalent_0p01", "equivalent_0p02", "status", "note"])
    table(t)
add("## 4. Contrasts by lead (`s5_4_contrasts_by_lead.csv`; all 40 leads: `s5_4_contrasts_40_leads.csv`)")
add()
for p in PROTOS:
    t = BYLEAD[BYLEAD.protocol == PNAME[p]]
    if t.empty:
        continue
    add(f"**{PNAME[p]}** (mean paired difference [5-95%])")
    add()
    wide = t.assign(cell=t.apply(lambda r: f"{r.mean_diff:+.3f} [{r.ci5:+.3f}, {r.ci95:+.3f}]", axis=1)).pivot(index="contrast", columns="lead_h", values="cell")
    wide = wide.reindex([f"{a} - {b}" for a, b in CONTRASTS]).dropna(how="all").reset_index()
    wide.columns = ["contrast"] + [f"{c_} h" for c_ in wide.columns[1:]]
    nn = t.groupby("contrast").n.agg(lambda x: f"{x.min()}-{x.max()}" if x.min() != x.max() else str(x.min()))
    wide.insert(1, "n", wide.contrast.map(nn))
    table(wide)
add("Sign changes across the five leads (`s5_4_sign_changes.csv`):")
add()
table(FLIPS)
add("## 5. Upstream observations (RQ3)")
add()
add("### 5a. M0 - M1 by position (`s5_5a_position.csv`; by lead: `s5_5a_position_by_lead.csv`)")
add()
add("Three definitions of nested: (1) a catalog gauge upstream that the model scoring the gauge may see (not withheld in its fold): the definition the "
    "insertion can use; (2) another of the 128 catalog gauges upstream; (3) another of the 101 evaluation gauges upstream (sensitivity).")
add()
table(POS)
add("### 5b. Within nested gauges: issues with a visible upstream gauge against issues without (`s5_5b_within_gauge.csv`, `s5_5b_within_gauge_summary.csv`)")
add()
add(f"Visible = at least 8 of the last 32 six-hour values at an upstream catalog gauge that is not withheld (the model's rule). NSE per gauge on each "
    f"subset of issues (subset variance; at least {MIN_SUBSET} values per lead), days 1-5 mean of M0 - M1. Difference = gain on visible issues minus gain on "
    "the other issues, per gauge.")
add()
table(WSUM)
for p_, grp in WITHIN.groupby("protocol", sort=False):
    q = grp.share_issues_visible.quantile([0, 0.25, 0.5, 0.75]).to_numpy()
    add(f"{p_}: share of test issues with a visible upstream gauge, per nested gauge: min {q[0]:.3f}, quartiles {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f}; "
        f"{int((grp.share_issues_visible < 0.9).sum())} of {len(grp)} gauges below 0.9. Only the gauges with at least {MIN_SUBSET} values per lead in both "
        "subsets give a within-gauge difference (column with_both_subsets).")
    add()
add("### 5c. Area-matched comparison (`s5_5c_area_bins.csv`, `s5_5c_matched.csv`)")
add()
add("Nested and headwater as in 5a definition (1). Area bins (NHDPlus drainage area):")
add()
table(BINS)
add(f"Nearest-neighbour matching on log10 area, greedy without replacement from the smallest nested gauge, caliper {CALIPER} (a factor "
    f"{10 ** CALIPER:.2f}); difference = gain of the nested gauge minus gain of its matched headwater gauge:")
add()
table(MATCH)
add("### 5d. Gain at days 1-2 against the upstream gauge's area share and travel time (`s5_5d_gain_vs_ratio.csv`, `s5_5d_rank_correlations.csv`)")
add()
add("Gain = per-gauge mean of M0 - M1 NSE at leads 24 and 48 h. Upstream gauge = the visible upstream catalog gauge with the largest drainage area. "
    "Area ratio = its NHDPlus drainage area over the gauge's. Travel time = pair travel time of the trained M0 router that scores the gauge (gauged: "
    "M0_s1; random hold-out: the fold model) at the mean NWM retrospective flow of each reach, from the middle of the upstream gauge reach to the end of "
    "the gauge reach along the main path (`H/channel.py: reach_time, pair_time`).")
add()
table(GCOR)
add("### 5e. Noise floor: M0 - M1 in the blocked protocol (`s5_5e_noise_floor.csv`)")
add()
add("In the blocked protocol no scored gauge has a visible upstream gauge, so insertion cannot act at test time and M0 and M1 can differ only through "
    "training (each was trained once). The difference below is therefore an estimate, from one pair of runs, of the size of training-to-training "
    "differences; it is not a variance estimate. Compare every 'within 0.02' statement with it.")
add()
table(NOISE)
add("### 5f. Area-ratio correlation: robustness, exploratory (`s5_5f_area_ratio_robustness.csv`)")
add()
add("Spearman correlation of the M0 - M1 gain at days 1-2 with the area ratio (5d), with leave-one-out range, the correlation after dropping the three "
    "gauges with the largest absolute gain, and a 5-95% gauge bootstrap (2,000 resamples, seed 0). One seed per model and many looks at the data: exploratory.")
add()
table(ROB)
add("### 5g. Gauge-based skill by drainage-area stratum (`s5_5g_skill_by_area.csv`)")
add()
add(f"Median NSE (days 1-5) of the gauges in each stratum, from observations. The smallest evaluation gauge drains {SMALL['min_area']:.1f} km2; "
    f"{SMALL['n_lt50']} evaluation gauges drain less than 50 km2 and {SMALL['n_lt200']} less than 200 km2. This is the only observation-based evidence below 200 km2 "
    "(the every-reach reference is a model). n is the number of gauges in the stratum scored by every model shown.")
add()
table(SSTR)
add("## 6. Routing and message passing (RQ2)")
add()
add("### 6a. By drainage-area stratum (`s5_6_area_strata.csv`); by lead: section 4")
add()
table(STRATA)
add("### 6b. Learned travel times (`s5_6_travel_times.csv`)")
add()
add("Nested evaluation gauges and their upstream catalog gauge with the largest drainage area; pair travel time at the mean NWM flow from the gauged-protocol "
    "models M0_s1 and M2_s1; river length along the main path from the middle of the upstream reach to the end of the gauge reach.")
add()
tts = TT.dropna(subset=["travel_time_h_M0"])
add(f"Pairs: {len(TT)} ({len(tts)} on the main path). River length median {tts.river_km.median():.1f} km (range {tts.river_km.min():.1f}-{tts.river_km.max():.1f}). "
    f"Travel time M0 median {tts.travel_time_h_M0.median():.1f} h (range {tts.travel_time_h_M0.min():.1f}-{tts.travel_time_h_M0.max():.1f}); M2 median "
    f"{tts.travel_time_h_M2.median():.1f} h (range {tts.travel_time_h_M2.min():.1f}-{tts.travel_time_h_M2.max():.1f}). Implied celerity (length / time) "
    f"median M0 {tts.celerity_m_s_M0.median():.2f} m/s, M2 {tts.celerity_m_s_M2.median():.2f} m/s. Spearman rho of travel time and length: M0 "
    f"{stats.spearmanr(tts.river_km, tts.travel_time_h_M0)[0]:.2f}, M2 {stats.spearmanr(tts.river_km, tts.travel_time_h_M2)[0]:.2f}.")
add()
add("### 6c. High flows (`s5_6_highflow.csv`)")
add()
add("Logic of `scripts/v3_kge_highflow.py`: peak bias = mean over leads 24-120 h of (mean forecast / mean observation - 1) over the top 5% of observed "
    "flows; events = 10-day windows whose observed maximum reaches the 2-year flow (median annual maximum, water years with at least 70% of values, at "
    "least 10 years), timing error = forecast minus observed peak step x 6 h, median over the gauge's events (at least 3), then median over gauges.")
add()
add(("Peak-flow bias (median over gauges, top 5% of observed flows) is negative for every model in every protocol: " if (HF.median_peak_bias_days1_5 < 0).all() else "Peak-flow bias (median over gauges, top 5% of observed flows), not negative for every model: ")
    + "; ".join(f"{r.protocol} {r.model} {r.median_peak_bias_days1_5:+.2f}" for r in HF.itertuples()) + ".")
add()
table(HF[["protocol", "model", "n", "median_peak_bias_days1_5"] + [c_ for c_ in HF.columns if c_.startswith(("gauges_", "median_timing", "median_magnitude"))]])
add("## 7. Every reach (`s5_7_every_reach.csv`, `s5_7_routes.csv`)")
add()
add(f"Against the NWM v3.0 retrospective, {REACH_ISSUES} issues (every second test issue to 2023-01-20; M0 manifest: {MAN_REACH['every_reach_issues']}). Per-reach "
    "NSE and KGE, days 1-5 mean, then median (and quartiles) over the reaches of a class. M0: `eval_M0_reach/reach_metrics.npz`; LSTM (L4): "
    "`ladder_L4_s1_reaches/metrics.npz` (the LSTM can be run at reaches: `scripts/lstm_every_reach.py`, basin-mean inputs of each reach, flow history "
    "hidden everywhere). M1, M2, M4: MISSING (no every-reach evaluation run).")
add()
table(ROUTES)
table(REACHT)
_hw = []
for _lab, _m in (("headwater reaches (no reach upstream)", HEADWATER), ("all other reaches", ~HEADWATER), ("all reaches", np.ones(net.n, bool))):
    _r = dict(reaches=_lab, n=int(_m.sum()), median_area_km2=float(np.median(RAREA[_m])))
    for _k in ("M0", "LSTM", "NWM"):
        _x = PR[f"nse_{_k}"].to_numpy(float)[_m]
        _x = _x[np.isfinite(_x)]
        _r[f"{_k}_scored"], _r[f"{_k}_median_nse"], _r[f"{_k}_share_nse_above_0"] = len(_x), float(np.median(_x)), float((_x > 0).mean())
    _hw.append(_r)
HWT = save(pd.DataFrame(_hw), "s5_7_headwater_summary.csv")
add()
add("Headwater reaches (`s5_7_headwater_summary.csv`; no reach upstream of the reach in the network graph): per-reach days 1-5 NSE against the NWM retrospective; "
    "NWM = the NWM operational forecast scored against the retrospective. Figures S6 a-e; lead curves `s5_headwater_lead_curves.csv`.")
add()
table(HWT)
add()
add("## 8. Example event (`s5_8_example_event.csv`)")
add()
if EX:
    add(f"Rule: random hold-out; among the nested gauges with a visible upstream gauge, the gauge with the (lower) median M0 - M1 days 1-5 gain; within the "
        f"test period (2020-09-23 to 2025-12-31) its largest observed six-hour flow; 5 days either side.")
    add()
    add(f"Gauge {EX['gauge']} (USGS {EX['usgs_id']}), {EX['area_km2']:,.1f} km2, fold {EX['fold']}, gain {EX['gain_days1_5']:+.4f} (rank {EX['rank']}); "
        f"visible upstream gauges: {EX['upstream']}. Peak {EX['peak_flow']:.1f} m3/s at {EX['peak_time']} (six-hour mean). The CSV gives, for leads 6, 12, 18 and "
        "24 h, the issue, the valid time and the observed, NWM, LSTM, M0 and M1 values (NaN where an issue has no forecast or is not scored).")
else:
    add("MISSING")
add()
add("## 9. Numbers for the caveats")
add()
add("Test pairs per scored gauge at 24 h (`s5_9_test_pairs.csv`):")
add()
table(FEW)
add("Gauges where the NWM has the higher days 1-5 NSE than M0 (`s5_9_nwm_better.csv`):")
add()
table(NWMB)
add("Sign of each contrast across protocols (`s5_9_sign_by_protocol.csv`):")
add()
table(SIGNP)
eq = CON[(CON.n > 0) & CON.equivalent_0p02.astype(bool)]
add("Contrasts within the 0.02 NSE equivalence margin (Holm-corrected TOST p < 0.05): " + ("; ".join(f"{r.contrast} ({r.protocol})" for r in eq.itertuples()) or "none") + ".")
eq1 = CON[(CON.n > 0) & CON.equivalent_0p01.astype(bool)]
add()
add("Within the 0.01 margin: " + ("; ".join(f"{r.contrast} ({r.protocol})" for r in eq1.itertuples()) or "none") + ".")
add()

# ---- 9 (text) and 10 ------------------------------------------------------------------------------------------------------------
add("## 9. Caveats for the text")
add()
miss = INV[INV.predictions.str.startswith("MISSING")]
add("- One seed (seed 1) per model and protocol; no seed spread is available, so no difference can be compared with training noise.")
add("- Missing runs: " + "; ".join(f"{r.model} {r.protocol} ({r.predictions})" for r in miss.itertuples()) + ". Every-reach: M1, M2, M4 MISSING.")
add(f"- Blocked protocol: n = {int(BASE['blocked'].sum())} gauges in 3 basins (12, 11, 6 scored); no scored gauge there has a visible upstream gauge, so M0 - M1 "
    "there is not a test of the insertion.")
add(f"- Test pairs: the fewest at 24 h is {int(FEW.min_pairs_24h.min())} (section 9 table); "
    f"{int(FEW.gauges_under_500_pairs.max())} gauge has fewer than 500 in the gauged and random protocols.")
add("- NWM: the operational NWM is nudged toward observations at gauges before each forecast, including gauges withheld from the other models; "
    + "; ".join(f"{r.protocol}: NWM better than M0 at {r.gauges_NWM_better} of {r.n} gauges" for r in NWMB.itertuples()) + ".")
flp = SIGNP[SIGNP.flips]
add("- Sign changes between protocols: " + ("; ".join(f"{r.contrast} ({r.signs} for {r.protocols})" for r in flp.itertuples()) or "none") + ".")
lf = FLIPS[FLIPS.sign_changes != "none"]
add("- Sign changes across leads 24-120 h: " + ("; ".join(f"{r.contrast}, {r.protocol}: {r.sign_changes}" for r in lf.itertuples()) or "none") + ".")
add("- Within the 0.02 NSE margin (Holm-corrected TOST): " + ("; ".join(f"{r.contrast} ({r.protocol})" for r in eq.itertuples()) or "none")
    + ". Not shown equivalent at 0.02 (and not different either) is every other contrast whose interval includes zero.")
add("- Noise floor (5e): the blocked M0 - M1 difference, where insertion cannot act, is "
    + "; ".join(f"{r.unit} {r.mean_diff:+.3f}" for r in NOISE.itertuples()) + " (one pair of runs). Every statement that a difference is within 0.02 NSE has to be read against it.")
add("- The +-0.02 equivalence margin is not yet tied to a seed spread; it should be justified once second seeds exist (section 10).")
add("- 5f: the area-ratio correlation changes sign between the gauged and random protocols and rests on one seed; exploratory only.")
add("- 5b: upstream gauges are visible at nearly every test issue (median share 1.0), so the within-gauge test has 1-2 gauges per protocol; it cannot "
    "separate the insertion from other differences between M0 and M1.")
add("- Every reach: the reference is the NWM retrospective (a model) that M0 (and M1, M2, M4) were pretrained on and the LSTM never saw, and the LSTM at reaches "
    "runs with the flow history hidden everywhere. Contrasts within the M family (M0 - M1, M0 - M2 by area class) would be fair; only M0 has a reach-level run, so "
    "none can be computed (MISSING); the M0 - LSTM comparison at reaches is biased toward M0.")
add("- Holm correction here runs across the contrasts available in each protocol (6 while M4 is missing); earlier reports used other families, so "
    "corrected p values differ from them.")
add()
add("## 10. Extra runs that would make the key contrasts interpretable (not started)")
add()
add("Wall-clock from the logged runs (`<data>/models/attr/*/validation.json` and logs): stage B M0 178 min on 4 GPUs (11.9 GPU-h), M1 167 min (11.1 GPU-h), M2 119 min "
    "(8.0 GPU-h), M4 246 min on 2 GPUs (8.2 GPU-h); one gauge-protocol evaluation about 30 min on 4 GPUs; one every-reach evaluation (M0) about 44 min on 4 GPUs; "
    "L4 LSTM about 17 min on one GPU plus a CPU benchmark. Four L40S GPUs; runs of different sizes can share the machine.")
add()
add("| Priority | Runs | GPU-hours | Wall-clock on 4 GPUs | What it enables |")
add("|---|---|---|---|---|")
add("| 1 | M4 evaluations (random hold-out, blocked): DONE 2026-11-22 | about 4 | about 30 min (parallel) | M1 - M4 (RQ2) in both protocols |")
add("| 2 | L4 LSTM seeds 2-5, gauged protocol (4 runs, 17 min each) + CPU benchmark evaluation | about 1.2 (GPU) | about 1.5 h plus the CPU benchmark | seed spread of the baseline and of its early-checkpoint selection; seed spread for M0 - LSTM |")
add("| 3 | Second stage-B seed, gauged protocol: M0, M1, M2 (reuse stage A) + evaluation | about 31 | about 7.7 h + 1.5 h | seed spread for the contrasts that carry the equivalence claims (M0 - M1, M0 - M2); a data-based justification of the +-0.02 margin |")
add("| 4 | Every-reach evaluation of M1, M2 and M4 (existing models, no training) | about 12 | about 2.2 h (3 x 44 min) | within-family contrasts by area class that are not biased by the shared NWM reference (Figure 7) |")
add("| 5 | Seed 2, blocked basins: M0, M1, M2, M4 x 3 basins + evaluations | about 120 | about 31 h | seed spread for the blocked contrasts (n = 29), after items 2-3 |")
add("| 6 | Seed 2, random hold-out: M0, M1, M2, M4 x 5 folds + evaluation | about 197 | about 50 h | seed spread for the 98-gauge contrasts, if still needed |")
add()
add("Retraining stage A for a second seed would add 15.9 (M0/M1), 9.1 (M2) and 12.0 (M4) GPU-hours; items 3, 5 and 6 above reuse the existing stage A, so they measure stage-B noise only.")
add()
(REPO / "reports" / "section5_facts.md").write_text("\n".join(md) + "\n")  # kept; scripts/section5/build_brief.py reads it into reports/section5_brief.md
print("wrote reports/section5_facts.md and", len(list(OUT.glob("*.csv"))), "CSVs")
