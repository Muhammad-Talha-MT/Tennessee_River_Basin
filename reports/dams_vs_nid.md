# Repository dams against public data (internal check, not for the paper)

Script: `scripts/section3/dams_vs_nid.py`; table: `reports/dams_vs_nid.csv`. Public sources: USACE National Inventory of Dams (public map service, layer 0, 7,440 unique records in the basin bounding box, downloaded 2026-11-21) and, for eight main dams without a main-dam NID record, Wikipedia page coordinates (public API). Nothing in the repository was changed.

| Check | Result |
|---|---|
| Dams in `raw/tva_dams/selected_dams.csv` | 44 (42 placed on the graph; Barkley and Great Falls are on the Cumberland) |
| Public reference found | 44 of 44: 36 NID records, 8 Wikipedia (Norris, Cherokee, Douglas, Watts Bar, Fort Loudoun, Tellico, South Holston, Nottely: NID has only saddle dams or dikes for them) |
| Raw coordinate within 1 km of the public location | 36 of 44 |
| Raw coordinate over 1 km away | BRDG1 2.2, GGRT1 1.25, OCAT1 1.8, WHLA1 1.1, THPN7 6.1, NANN7 8.6, OCCT1 8.8 km; NOTG1 140.8 km (the row repeats Norris Dam's coordinates) |
| Placed reach outlet within 1 km of the public location | 35 of 42; within 3 km: 39 |
| Placed over 3 km away | NANN7 8.8, OCCT1 9.1, THPN7 7.1 km (reach on the right river with a plausible area, but 7-9 km from the dam) |
| NOTG1 (Nottely) placement | correct: the reach was fixed by hand and is 0.2 km from the real dam; only the raw coordinate row is wrong |
| Drainage area, graph against NID (30 placed dams with NID area above 100 km2) | median ratio 1.00; 29 of 30 within 0.8-1.25; outlier Wheeler (NID lists 601 km2 against 75,998 in the graph, NID's field is not the basin area there) |
| Not verifiable from public data | `mean_release` (dam release volumes) and `release_over_nwm`, which the placement rule uses |
