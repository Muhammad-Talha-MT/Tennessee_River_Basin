# Section 3 figures

Scripts in `scripts/section3/`; shared loaders in `common.py` (reuses `hydro.data.Network/Gauges/Arrays/Rain`, `hydro.splits`, `hydro.evaluate.regulation`, the geometry approach of `scripts/headwater_map.py`).
Run from the repo root: `python scripts/section3/<script>.py` (about 30 s each). Each writes a PDF (vector) and a 300 dpi PNG here.
Width: full width, 17.4 cm. Font 7-8 pt at final size. Colors: Okabe-Ito / Paul Tol palettes; the evaluation-gauge blue (#2B7BE0 in Fig 1, #0072B2 in Fig 3) is used for evaluation gauges only.
Facts and the per-gauge table: `python scripts/section3/facts.py` -> `reports/section3_facts.md`, `figures/section3/gauge_table.csv`.

Inputs (all under `/data/tva_data`; full list in `scripts/section3/common.py: INPUTS`):
`raw/geography/nhdplus_v21/.../NHDFlowline.shp`, `raw/geography/tennessee_huc2.geojson`, `raw/geography/us_states/cb_2022_us_state_5m.shp`,
`processed/graph_candidate/{nodes.parquet,catchments_clean.parquet,static_core.nc}`, `processed/gauges_public/{gauges.nc,gauges.csv,gauges.json}`,
`processed/hydro/routing.npz` (Figs 1-4 do not use dam data), `processed/rain_forecasts/{gefs_reforecast_240,gefs_operational_240}`,
`models/attr/eval_M0_s1/forecasts_gauged.npz`, `eval_M0_s1/forecasts_ungauged.npz`, `eval_S1_M0_s1/forecasts_ungauged.npz` (which gauges are scored; test issue dates).

## Fig 1: `fig1_map.py` -> `fig1_study_area`
Encodings: (a) river network (reaches of at least 100 km2 only, drawn thin and pale for 100-1,000 km2 and stronger above; the 58,834 reaches in total), state boundaries and names, the basin in pale blue with a bold navy outline, the 101 evaluation gauges as one blue circle, north arrow and scale bar, CONUS inset with state boundaries.
(b) the same base map; blocked basins B1-B3 shaded in orange, teal and purple (labelled on the map); gauges scored in the blocked protocol in their basin's hue, all other evaluation gauges small gray circles. One marker shape for gauges; color encodes at most one variable. No counts appear in the legends.
Dam locations are not public and are not drawn.
Counts drawn (printed by the script, matching `reports/section3_facts.md`): 101 evaluation gauges (a); in (b) 72 gray + 12 (B1) + 11 (B2) + 6 (B3) = 101.
Counts for the caption: 128 catalog gauges (125 CAMELS-H, 3 USGS-only); 101 used in the gauged protocol; 98 in the random-fold protocol (5 folds of 21, 22, 18, 19 and 18 scored gauges; 3 of the 101 are in no fold because they have no observations before 2019); 29 in the blocked protocol (B1 12, B2 11, B3 6 scored gauges; 15, 13 and 12 catalog gauges are withheld in B1, B2 and B3; basin areas 4,801, 9,639 and 5,971 km2; 2,847, 5,235 and 3,228 reaches).
Gauge groups: 101 evaluation gauges, 44 nested and 57 headwater (another of the 128 catalog gauges upstream or not); blocked protocol 29 gauges (12 + 11 + 6).
Draft caption: (a) The Tennessee River Basin with its 58,834 NHDPlus reaches, state boundaries and the 101 gauges used for evaluation; inset: location in the contiguous United States. (b) The three blocked basins B1-B3 (shaded) and the gauges they hold (same hue); all other evaluation gauges are gray.
Caveats: gauges are drawn at the downstream end of their reach, so gauges on one reach overlap; the blocked-basin shapes are the union of catchment polygons simplified by 200 m; the 128 - 101 = 27 catalog gauges that are not scored are not drawn; state names are placed outside the basin (Tennessee inside) with a white halo; reaches below 100 km2 are not drawn so that the main stems stay visible; the map extent is the basin plus a 4% (east-west) and 6% (north-south) margin, so state labels are placed inside that frame; B1-B3 labels sit outside their basins with a short leader to the basin's gauge centroid.

## Fig 2: `fig2_timeline.py` -> `fig2_timeline`
A single panel (full width 17.4 cm, about 5 cm high, font 8 pt): one forecast issue. Two colors only: blue for inputs known at the issue time (rows "Observed forcing (AORC), all reaches" and "Observed discharge, visible gauges", ending at day 0) and vermilion for the forecast (row "Forecast rain (GEFSv12 control), all reaches", day 0 to +10). A vertical line marks the issue time (00Z), days 1-5 (24-120 h) are shaded with the label below the axis, and a bracket marks the 240 h horizon (40 six-hourly steps). No dates and no legend. The data periods are not drawn; they are in `reports/section3_facts.md` section 13.
Values read from the code, not typed: look-back = `P.HISTORY` (144 steps = 36 days, hydro/paths.py; the model reads this many steps of weather and of observed discharge, `Device.assemble`), horizon = `P.HORIZON` (40 steps = 240 h).
Draft caption: One forecast issue. At the 00Z issue time the model has the observed forcing (AORC) at every reach and the observed discharge at the gauges visible to it for the previous 36 days; after the issue it has only the GEFSv12 control-member rain forecast, to 240 h (40 six-hourly steps). Skill is reported for days 1-5 (leads 24-120 h, shaded).
Caveats: the gauge head reads the last 30 days of discharge and insertion the last 20 days of it (not drawn); a gauge is visible only with at least 8 values in the last 8 days. Observed discharge is assumed available up to 00Z: the code models no reporting latency, which a real-time application would face.

## Fig 3: `fig3_representativeness.py` -> `fig3_representativeness` (also copied to `figures/`), and Fig S1: `figS1_record_length.py` -> `figS1_record_length`
Inputs: `common.load()` (NHDPlus `totdasqkm` at reaches and gauge reaches, saved gauged-protocol scoring, observed discharge array). Fig 3: 12.5 x 13.2 cm, constrained layout, light grid, (a) over (b) with a shared log x-axis 10^-1 to 10^4.5 km2; colors (Okabe-Ito): vermilion #D55E00 reaches, blue #0072B2 all evaluation gauges in (a); in (b) bluish green #009E73 headwater gauges and reddish purple #CC79A7 nested gauges (dot layout computed in points so no two dots touch), gray #999999 dashed catalog gauges. Fig S1: one panel, 6.5 in wide, test period blue solid, training period #333333 dashed. Both scripts fail on overlapping text, lines or dots (last run: Fig 3 none; see the script output for S1).
Draft caption Fig 3: (a) Cumulative distribution of drainage area for all reaches, the evaluation gauges and, dashed gray, the 128 catalog gauges; dots mark the medians. (b) Drainage area of the evaluation gauges, headwater gauges (no other catalog gauge upstream) against nested gauges (another catalog gauge upstream), one dot per gauge (vertical position only separates dots), bars mark the medians.
Draft caption Fig S1: Fraction of six-hourly values observed at each evaluation gauge in the training period (2000-2018) and the test period (2020-09-23 to 2025-12-31); each curve is ranked on its own, so a rank is not the same gauge in the two curves.
Numbers kept out of the figures (script output and `reports/section3_facts.md` sections 11-12 and 15):
- n: 58,834 reaches, 128 catalog gauges, 101 evaluation gauges; headwater gauges 57, nested gauges 44.
- Medians (km2): reaches 5.8; catalog gauges 375.8; evaluation gauges 382.3; headwater gauges 275.2, nested gauges 1,237.9. 77.7% of reaches but 4.0% of evaluation gauges are under 50 km2.
- Outside the x-range: 358 reaches < 0.1 km2 and 640 reaches > 10^4.5 km2; no gauge is outside it.
- Availability (Fig S1): training availability < 50% at 19 gauges, < 10% at 3 gauges; the 3 have exactly 0%: 03566420 (CAMELS-H + NWIS, first observation 2019-09-18), 03591800 (NWIS only, 2022-10-20), 03572110 (NWIS only, 2021-11-10); they are in no random fold.
- Test period: the last test forecast is issued 2025-12-22 and its horizon ends at the 2025-12-31 18:00 step, the last step of the data arrays; availability is computed over 2020-09-23 to 2025-12-31.
Caveats: all areas are NHDPlus `totdasqkm`; three gauges have a graph area that differs from it by more than 10% (facts sheet). The reason for low availability is not shown (record start dates are in `gauge_table.csv`).

## Fig 4 (optional): `fig4_elevation.py` -> `fig4_elevation`
Inputs: `processed/graph_candidate/static_core.nc` (StreamCat `elevcat`, one value per NHDPlus catchment), `catchments_clean.parquet`, `tennessee_huc2.geojson`, Census state boundaries (cb_2022). Frame, state boundaries, scale bar, north arrow, CONUS inset and blue ramp follow Fig 1.
Numbers: 57,642 catchments, all with a value; range 92 to 1,748 m. Water-body holes inside the catchment polygons: 180, 0.92 km2 in total (largest 0.39 km2); they are filled with the catchment value. Polygons are simplified by 60 m for file size.
Draft caption: Mean elevation of each NHDPlus catchment of the basin (m), from the StreamCat Elevation attribute (one value per catchment, so the surface is catchment-level and not a DEM). Water bodies (180 polygons, 0.9 km2 in total) inside the catchments are filled with the catchment value. The outline is the HUC2 region.
Caveat: only elevation was drawn; precipitation and land-cover maps would need data not prepared for this purpose, and nothing was downloaded.

## Checks
- Counts agree across figures and facts sheet: 101 gauged, 98 random-fold (the 3 missing gauges 03566420, 03591800 and 03572110 are in no fold: no observations before 2019), 29 blocked (12 + 11 + 6); 44 nested and 57 headwater evaluation gauges.
- All scripts were re-run from a clean environment (`env -i`) and reproduce the files.
- Figures were viewed at full size; Fig 1 was also checked in grayscale (the three basin fills differ in luminance and are labelled B1-B3 on the map).

## Fig 4 (replacement): `fig4_basin_grid.py` -> `fig4_basin_grid` (data: `basin_attributes.py` -> `basin_attributes.csv`)
3 x 2 maps, full width 17.4 cm, one value per NHDPlus catchment (57,642 with values; no DEM, no interpolation), fixed class breaks on each colorbar:
(a) mean catchment elevation (StreamCat elevcat; hypsometric tints, rivers of 5,000 km2 or more in blue; north arrow and scale bar in (a) only; framed panels with state names, no CONUS inset), (b) mean annual AORC precipitation 2000-2019,
(c) river network by Strahler order (orders 1-8; line width and color), (d) reach slope (log classes), (e) forest cover and (f) developed cover of the local catchment (StreamCat NLCD 2011).
Shares of catchments per class are printed by the script. Map layers are raster inside the PDF (4.7 MB), text and colorbars are vector. The earlier single-panel `fig4_elevation` is kept.
Caveats: slope is missing for 18 reaches (gray); precipitation and elevation are drawn only for the 57,642 reaches with a catchment polygon; the classes differ between panels by design.
