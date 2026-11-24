"""Compare the repository's dam list (raw/tva_dams/selected_dams.csv, processed/hydro/dams.csv) with the public USACE National Inventory of Dams (NID).
NID records inside the basin bounding box were downloaded from the public NID map service (layer 0, 2026-11-21) into a CSV given as argv[1].
For each repository dam: nearest NID dam by great-circle distance (km), NID name, NID drainage area (square miles, converted to km2) against the repository area,
and the distance from the graph reach where the dam was placed (flowline outlet) to the NID dam. Run: python scripts/section3/dams_vs_nid.py <nid_bbox.csv>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geopandas as gpd
import numpy as np
import pandas as pd
from common import INPUTS, REPO

nid = pd.read_csv(sys.argv[1])
sel = pd.read_csv("/data/tva_data/raw/tva_dams/selected_dams.csv")
dams = pd.read_csv("/data/tva_data/processed/hydro/dams.csv")


def km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


# graph placement: outlet of the placed reach, in WGS84
lines = gpd.read_file(INPUTS["flowlines"], columns=["COMID"])
lines = lines[lines.COMID.isin(dams.comid)].set_index("COMID").to_crs(4326)


def outlet(comid):
    g = lines.geometry[comid]
    g = g.geoms[-1] if g.geom_type == "MultiLineString" else g
    return g.coords[-1][1], g.coords[-1][0]


import json
import re
import urllib.parse
import urllib.request

WIKI = {"NRST1": "Norris Dam", "CRKT1": "Cherokee Dam", "DUGT1": "Douglas Dam", "WBOT1": "Watts Bar Dam", "FLDT1": "Fort Loudoun Dam", "TDTT1": "Tellico Dam",
        "SHDT1": "South Holston Dam", "NOTG1": "Nottely Dam"}  # main dams that have only saddle dams or dikes in the NID extract


def wikipedia_coordinates(titles):
    q = dict(action="query", prop="coordinates", titles="|".join(titles), format="json", redirects="1")
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q), headers={"User-Agent": "dam-check/0.1 (research script)"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    redir = {r["from"]: r["to"] for r in d["query"].get("redirects", [])}
    pages = {p["title"]: p for p in d["query"]["pages"].values()}
    return {t: (pages[redir.get(t, t)]["coordinates"][0]["lat"], pages[redir.get(t, t)]["coordinates"][0]["lon"]) for t in titles if "coordinates" in pages.get(redir.get(t, t), {})}


MANUAL_NID = {"OCCT1": "Ocoee No. 3 Dam", "NANN7": "Nantahala - Gated Spillway and 2 Fuseplugs", "THPN7": "Thorpe - Gated Spillway and 6 Fuseplugs"}  # raw coordinate is 6-9 km from the NID record (inflow-node point); matched by exact NID name
wiki = wikipedia_coordinates(list(WIKI.values()))
rows = []
dd = dams.set_index("dam")
for _, r in sel.iterrows():
    token = re.sub(r"\s*(Dam|Lock and Dam)\s*$", "", r.Descript.split("@")[-1].strip()).strip()
    token = {"Ft Patrick Henry": "Fort Patrick Henry", "Nolinchucky": "Nolichucky", "Ft Loudoun": "Fort Loudoun"}.get(token, token)
    token = token.replace("Dam #1", "No. 1").replace("Dam #3", "No. 3").replace("Occoee", "Ocoee").replace("Dam", "").strip()
    d = km(r.latitude, r.longitude, nid.LATITUDE.to_numpy(), nid.LONGITUDE.to_numpy())
    near = nid[(d <= 8) & nid.NAME.str.contains(token.split()[0], case=False, na=False) & ~nid.NAME.str.contains("Saddle|Dike|Backwater|Pond|Nuclear|Fossil|Reregulation|Forebay", case=False, na=False)]
    if r.NWSID in MANUAL_NID:
        near = nid[nid.NAME == MANUAL_NID[r.NWSID]]
    if len(near):
        j = near.index[np.argmin(d[near.index])]
        ref_name, ref_lat, ref_lon, src = nid.NAME[j], nid.LATITUDE[j], nid.LONGITUDE[j], "NID"
        area = float(nid.DRAINAGE_AREA[j]) * 2.58999 if pd.notna(nid.DRAINAGE_AREA[j]) else np.nan
        owner, storage = nid.PRIMARY_OWNER_TYPE[j], nid.NID_STORAGE[j]
    elif r.NWSID in WIKI and WIKI[r.NWSID] in wiki:
        (ref_lat, ref_lon), ref_name, src, area, owner, storage = wiki[WIKI[r.NWSID]], WIKI[r.NWSID], "Wikipedia (no NID record for the main dam)", np.nan, "", np.nan
    else:
        ref_name, ref_lat, ref_lon, src, area, owner, storage = "", np.nan, np.nan, "none", np.nan, "", np.nan
    row = dict(dam=r.NWSID, repo_name=r.Descript, repo_lat=round(r.latitude, 4), repo_lon=round(r.longitude, 4), ref_source=src, ref_name=ref_name,
               raw_coord_to_ref_km=round(float(km(r.latitude, r.longitude, ref_lat, ref_lon)), 2) if src != "none" else np.nan, nid_drainage_km2=round(area) if pd.notna(area) else np.nan,
               nid_owner=owner, nid_storage_acft=storage)
    if r.NWSID in dd.index:
        c = dd.loc[r.NWSID]
        la, lo = outlet(int(c.comid))
        row.update(placed=True, graph_area_km2=c.area_km2, how=c.how, placed_to_ref_km=round(float(km(la, lo, ref_lat, ref_lon)), 2) if src != "none" else np.nan)
    else:
        row.update(placed=False)
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(REPO / "reports/dams_vs_nid.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 30)
print(out[["dam", "ref_source", "ref_name", "raw_coord_to_ref_km", "nid_drainage_km2", "graph_area_km2", "placed_to_ref_km"]].to_string(index=False))
print("\nreference found:", int((out.ref_source != "none").sum()), "of", len(out), "| NID:", int((out.ref_source == "NID").sum()), "| Wikipedia:", int(out.ref_source.str.startswith("Wiki").sum()))
print("raw coordinate within 1 km of the reference:", int((out.raw_coord_to_ref_km <= 1).sum()), "| over 1 km:", out[out.raw_coord_to_ref_km > 1][["dam", "raw_coord_to_ref_km"]].values.tolist())
pl = out[out.placed]
print("placed reach outlet within 1 km of the reference:", int((pl.placed_to_ref_km <= 1).sum()), "of", len(pl), "| within 3 km:", int((pl.placed_to_ref_km <= 3).sum()), "| over 3 km:", pl[pl.placed_to_ref_km > 3][["dam", "placed_to_ref_km"]].values.tolist())
ok = pl[pl.nid_drainage_km2.notna() & (pl.nid_drainage_km2 > 100)]
ratio = ok.graph_area_km2 / ok.nid_drainage_km2
print("graph/NID drainage-area ratio (NID area > 100 km2):", len(ok), "median", round(float(ratio.median()), 2), "| within 0.8-1.25:", int(((ratio >= 0.8) & (ratio <= 1.25)).sum()), "| outside:", ok[(ratio < 0.8) | (ratio > 1.25)][["dam", "graph_area_km2", "nid_drainage_km2"]].values.tolist())
