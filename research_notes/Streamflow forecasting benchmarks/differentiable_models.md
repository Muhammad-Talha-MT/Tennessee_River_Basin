# Differentiable / Physics-Informed (Hybrid) Hydrologic Models as Baselines for a Tennessee River Basin GNN Streamflow Forecaster (6-hourly, 1–10 day lead, NHDPlus v2, USGS gauges 2020–2025)

Research date: 2026-10-12. Peer-review status is flagged for each item. Most numbers below are **simulation** skill (driven by observed/analysis forcings), not forecast skill. That matters when comparing them with a 1–10 day forecasting GNN.

---

## 1. The dPL / δHBV lineage (Tsai 2021 → Feng 2022 → δHBV1.1p): what each is, and its CAMELS numbers against LSTM

### Takeaway
dPL (Tsai et al. 2021) introduced the idea of an NN learning parameters for a process model end to end. δHBV (Feng et al. 2022, WRR) brought lumped daily HBV on CAMELS to within about 0.01–0.02 median NSE of LSTM: 0.732 vs 0.748 with Daymet forcing, and 0.715 vs 0.722 with a second forcing set. δHBV1.1p (Song et al. 2026, WRR) beats LSTM on rare extremes, with median NSE about 0.06 higher for floods of 50-year return period or rarer. All of these are daily, lumped, CAMELS-basin models.

### Cited Findings
- **dPL (Tsai et al. 2021, Nature Communications, peer-reviewed)**, "From calibration to parameter learning: Harnessing the scaling effects of big data in geoscientific modeling," doi:10.1038/s41467-021-26107-z. It proposes a differentiable parameter learning (dPL) framework that "efficiently learns a global mapping between inputs (and optionally responses) and parameters." In its tests, dPL "drastically outperformed existing evolutionary and regionalization methods, or required only ~12.5% of the training data to achieve similar performance" — [arXiv 2007.15751](https://arxiv.org/abs/2007.15751); [DOAJ record](https://doaj.org/article/79ba5184e4a24bf6ac1ca15f35085535)
- The streamflow experiment in Tsai et al. used VIC as the process model on 531 CAMELS basins (daily). Its baseline was Multiscale Parameter Regionalization, and an NN surrogate stood in for VIC — [eesm.science.energy.gov summary](https://eesm.science.energy.gov/publications/calibration-parameter-learning-harnessing-scaling-effects-big-data-geoscientific). The authors' cost comparison, as quoted by a later paper: CONUS calibration with an evolutionary algorithm would take "a 100-processor cluster ... 2-3 days" versus "1 single GPU for 1 hour" for dPL — via the same search results (secondary quote; I did not see it in the primary text).
- **δHBV (Feng, Liu, Lawson & Shen 2022, WRR 58(10), doi:10.1029/2022WR032404, peer-reviewed).** HBV is the backbone, with embedded NNs that "parameterize, enhance, or replace" its modules. Results: "δ models can obtain a median Nash Sutcliffe efficiency of 0.732 for 671 basins across the USA for the Daymet forcing dataset, compared to 0.748 from a state-of-the-art LSTM model with the same setup. For another forcing dataset, the difference is even smaller: 0.715 vs. 0.722." The model also outputs untrained variables (ET, baseflow, snowpack, soil and groundwater storage) — [arXiv 2203.14827 (journal ref WRR 58, 2022)](https://arxiv.org/abs/2203.14827)
- A 2026 HESS article cites Feng et al. 2022 for the claim that dynamic-parameter gains are small: "median NSE of 0.732 versus 0.714 in CAMELS-US." That is a different pairing from the abstract's LSTM 0.748, and probably compares δHBV with dynamic parameters against δHBV with static ones. Cite the abstract's numbers instead — [HESS 30/5173/2026 (via search)](https://hess.copernicus.org/articles/30/5173/2026/); conflicts with the reading of [arXiv 2203.14827](https://arxiv.org/abs/2203.14827)
- **δHBV1.1p (Song, Sawadekar, Frame, Pan, Clark, Knoben, Wood, Lawson, Patel & Shen 2026, WRR 62(2), e2025WR040414, doi:10.1029/2025WR040414, peer-reviewed).** Tested on CAMELS against LSTM. "For events with a 50-year return period or rarer, δHBV1.1p had a median NSE 0.06 higher than LSTM" and lower peak errors for 80% of those events. Its advantage over LSTM starts at events with return period ≥5 years and grows with return period. The loss function affected δHBV1.1p more than LSTM, and more dynamic parameters improved extreme-event metrics — [Penn State Pure](https://pure.psu.edu/en/publications/physics-informed-differentiable-hydrologic-models-for-capturing-u/); [UCAR record](https://impacts.ucar.edu/en/publications/physics-informed-differentiable-hydrologic-models-for-capturing-u/)
- δMG (the generic_deltamodel README) lists δHBV1.0 ("lumped rainfall-runoff"), δHBV1.1p ("improved lumped model, unseen extreme events test") and δHBV2.0 ("national and global distributed modeling") as its supported models — [github.com/mhpi/generic_deltamodel](https://github.com/mhpi/generic_deltamodel)
- An older companion paper: Feng et al. (2023) on the suitability of δ models for ungauged regions and climate-change assessment. I found it only as a title reference and did not verify its numbers (see Gaps).
- Review and perspective: Shen et al. (2023), *Nature Reviews Earth & Environment*, "Differentiable modeling to unify machine learning and physical models and advance Geosciences," is the main framework citation for δMG — [arXiv 2301.04027](https://arxiv.org/abs/2301.04027); [generic_deltamodel README](https://github.com/mhpi/generic_deltamodel)

### Inferences
- On CAMELS daily NSE, a well-tuned LSTM still matches or slightly beats δHBV on median NSE (by about 0.01–0.02). δ models win on extremes and peak bias, and they output physical states. For a flood-focused Tennessee paper, the extreme-event comparison (δHBV1.1p) is the more relevant one.
- The lumped δHBV1.0 and 1.1p models predict only at basin outlets and have no network routing, so they compare only at gauge points. They cannot give a reach-level (NHDPlus) comparison.

### Gaps
- I could not open the full texts of Tsai 2021 or Feng 2022 (Wiley returned 403), so there are no KGE values or regional (Southeast) breakdowns. The CAMELS set includes some Tennessee Valley basins, but I found no per-region numbers for them.
- I could not find δHBV1.1p's absolute median NSE or KGE in the abstract-level sources.
- Feng et al. 2023 (HESS, ungauged regions) numbers are unverified.

---

## 2. The high-resolution national-scale δ models (δHBV2.0 with UH or δMC routing; Song et al. 2025) and δMC routing (Bindas 2024; ddr)

### Takeaway
Song, Bindas et al. (2025, WRR, peer-reviewed) trained δHBV2.0 on MERIT unit catchments of about 37 km² over CONUS, in two versions: implicit UH routing, and explicit Muskingum–Cunge network routing (δMC). The models were trained on 2,807 basins and evaluated on 4,997, and raised median daily NSE to about 0.68 from 0.48 for NWM3.0. Bindas et al. (2024, WRR) introduced the δMC routing model (learned Manning's n) on the Juniata/Susquehanna. `deepgroundwater/ddr` is the Apache-2.0 continuation of that routing code for the NextGen hydrofabric and MERIT. The 0.68 vs 0.48 numbers come from Song et al. 2025; ddr only cites them.

### Cited Findings
- **Song, Bindas, Shen, Ji, Knoben, Lonzarich, Clark, et al. (2025), "High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning," WRR 61(4), e2024WR038928, doi:10.1029/2024WR038928 (peer-reviewed).** "two novel versions of high-resolution (∼37 km²) differentiable models ... one with implicit, unit-hydrograph-style routing and another with explicit Muskingum-Cunge routing in the river network ... trained simultaneously ('end-to-end') on 2,807 basins across the CONUS and evaluated on 4,997 basins ... the median daily Nash-Sutcliffe efficiency of all 4,997 basins is improved to around 0.68 from 0.48 of NWM3.0." Both versions "greatly improved simulations in the western CONUS and also in the Prairie Pothole Region." "The Muskingum-Cunge version further improved performance for basins >10,000 km²." "The modeling system supports the Basic Model Interface (BMI)" — [OSTI/DOE PAGES](https://www.osti.gov/pages/biblio/2555891-highresolution-nationalscale-water-modeling-enhanced-multiscale-differentiable-physicsinformed-machine-learning); abstract also reproduced in the [ddr README](https://github.com/deepgroundwater/ddr); [Penn State Pure](https://pure.psu.edu/en/publications/high-resolution-national-scale-water-modeling-is-enhanced-by-mult/); [RTI](https://www.rti.org/publication/highresolution-nationalscale-water-modeling-is-enhanced-by-multis)
  - **Conflict:** the ESSOAr preprint (doi:10.22541/essoar.172736277.74497104/v1) reportedly gives the NWM3.0 baseline as **0.49**, while the published abstract says **0.48**. Use 0.48, from the published WRR version — [ESSOAr preprint (via search snippet)](https://essopenarchive.org/doi/full/10.22541/essoar.172736277.74497104/v1)
  - δHBV2.0 provides "40 years of daily runoff for ~180,000 MERIT unit catchments" — [ddr README](https://github.com/deepgroundwater/ddr). δMG describes it as "~180,000 CONUS river reaches" — [generic_deltamodel](https://github.com/mhpi/generic_deltamodel)
  - Architecture (from the dhbv2 README): an LSTM supplies dynamic parameters and an MLP supplies static parameters, from forcings (P, T, PET) and static attributes per unit basin. HBV then computes fluxes and states. Daily timestep; requires the "prior 365 days" of forcing for warm-up — [github.com/mhpi/dhbv2](https://github.com/mhpi/dhbv2)
- **Bindas, Tsai, Liu, Rahmani, Feng, Bian, Lawson & Shen (2024), "Improving River Routing Using a Differentiable Muskingum-Cunge Model and Physics-Informed Machine Learning," WRR, doi:10.1029/2023WR035337 (peer-reviewed).** δMC-Juniata-hydroDL2 "mimics the classical Muskingum-Cunge routing model over a river network but embeds an NN to infer parameterizations for Manning's roughness (n) and channel geometries from raw reach-scale attributes like catchment areas and sinuosity," trained only on downstream hydrographs. In synthetic tests, n "can be identified with moderate precision" while channel geometry was unidentifiable. For subbasins >2,000 km² it routed more accurately over the long term than a homogeneous ML model or summed subbasin runoff — [eesm.science.energy.gov](https://eesm.science.energy.gov/publications/improving-river-routing-using-differentiable-muskingum-cunge-model-and-physics); [NOAA repository](https://repository.library.noaa.gov/view/noaa/63586)
  - Code: [github.com/mhpi/dMC-Juniata-hydroDL2](https://github.com/mhpi/dMC-Juniata-hydroDL2), CC-BY-4.0, Zenodo DOI 10.5281/zenodo.10183448. River graph data for the Susquehanna is on Zenodo (records/10183429). The data loader is specific to the Susquehanna basin. Last pushed April 2024 (research release; not maintained).
- **deepgroundwater/ddr (Distributed Differentiable Routing):** "An implementation of differentiable river routing methods for the NextGen Framework and Hydrofabric." Its solver "was inspired by" RAPID (a matrix-based Muskingum method). A geometry predictor uses KAN-predicted parameters (n, p, q). Supported spatial units are NextGen hydrofabric v2.2 (`conus_nextgen.gpkg`), MERIT, and a 0.5° ISIMIP DDM30 grid. **NHDPlus v2 is not supported** (it is mentioned only through the RAPID citation). Lateral inflow comes from the δHBV2.0 daily MERIT runoff dataset. **License: Apache-2.0.** CLI: `ddr train|test|route|train-and-test|summed-q-prime`, installed with `uv sync`, with CUDA 12/13 groups. The README gives no DDR-specific skill scores and no forecasting mode. Pretrained references sit under `DeepGroundwater/references/mhpi/dHBV2.0UH` — [github.com/deepgroundwater/ddr](https://github.com/deepgroundwater/ddr)
- **δHBV2.0 global (Ji, Song et al. 2025, arXiv 2504.10707, preprint):** a high-resolution global physics-embedded model with "more accurate simulations at monthly and daily scales than current operational systems." It uses δMC as explicit routing on MERIT flowlines — [arXiv 2504.10707](https://arxiv.org/abs/2504.10707); [generic_deltamodel README](https://github.com/mhpi/generic_deltamodel)

### Inferences
- "ddr with median NSE ~0.68 vs NWM3.0 0.48" is a slight misattribution. The 0.68/0.48 figures are **daily** NSE from Song et al. 2025 (δHBV2.0 with UH or MC routing on MERIT). ddr is the routing code that cites that paper.
- Song 2025 runs on MERIT-Basins (~37 km²), not NHDPlus v2. Comparing it against an NHDPlus v2 GNN means matching at USGS gauges, not reach by reach.

### Gaps
- I could not access the full Song 2025 text (Wiley 403; the ESSOAr page returned 403). Missing: the forcing dataset for the daily model (likely multiple products; unverified), train/test periods, KGE, and Southeast/Tennessee regional numbers. The abstract highlights gains in the West and the Prairie Pothole Region, which suggests smaller margins over NWM in the humid Southeast, but this is unconfirmed.
- I found no published DDR-only skill numbers.

---

## 3. Hourly δ models: δHBV2.0MTS (hourly, NextGen hydrofabric, AORC)

### Takeaway
The hourly δ model is δHBV2.0MTS-MC (Yang, Lonzarich, Song, Ji, Pan, Lawson & Shen, arXiv 2609.06794, Sep 2026, **preprint/under review**). It runs on the NextGen Hydrofabric (>800,000 unit basins, median 7.2 km²) with AORC forcing and end-to-end Muskingum–Cunge routing. Its CONUS median hourly NSE is 0.683, against 0.461 for NWM3.0 and 0.359 for LSTM-TRoute, over 2,831 gauges in a 2009–2018 test period. It is the closest-matched physics-ML baseline for a sub-daily (6-hourly) Tennessee forecasting study.

### Cited Findings
(All from [arXiv 2609.06794](https://arxiv.org/abs/2609.06794), "Hourly U.S.-wide flood simulation beyond the limits of traditional and data-driven models", v1 6 Sep 2026, CC BY 4.0, preprint)
- Across 2,831 CONUS gauges, δHBV2.0MTS-MC scores median hourly NSE **0.683** (NWM3.0: 0.461), KGE **0.719** (0.591), and |PBIAS| 10.24% (15.28%). LSTM-TRoute, an "operational" NextGen LSTM coupled with t-route, scores median hourly NSE **0.359**. |FHV| and |FLV| show a 20–21% relative reduction compared with NWM3.0.
- On CAMELS (Table 1; the δ models were trained only on CAMELS, 1991–2003):
  - δHBV2.0MTS-MC: NSE 0.710, KGE 0.736
  - δHBV2.0MTS-UH: NSE 0.716, KGE 0.758
  - NWM3.0: NSE 0.545, KGE 0.684
  - LSTM-TRoute: NSE 0.457, KGE 0.410
  - **MTS-LSTM** (Gauch/Kratzert): NSE 0.726, KGE 0.723
  - **MF-LSTM** (Acuña Espinoza): NSE 0.716, KGE 0.732
  - The pure-LSTM hourly benchmarks use NLDAS forcing. All δ variants, NWM3.0 and LSTM-TRoute use AORC. Overall skill is therefore comparable to state-of-the-art hourly LSTMs, but δ has lower high-flow bias (|FHV| 13.4% vs 16.8% for MTS-LSTM).
- For basins >10,000 km², MC routing scores median NSE 0.668 against 0.546 for UH. MC "produces hourly streamflow at every reach," while UH is designed for gauged outlets.
- On flood peaks, timing error falls from 7–8 h (NWM3.0) to 4.5–6 h. δHBV2.0MTS-MC captures 33% more ≥50-year floods than NWM3.0 and 159% more than LSTM-TRoute, and reduces peak-magnitude error for >100-year floods by 34% relative to MTS-LSTM.
- Skill varies with climate: median hourly NSE is 0.729 in humid basins (aridity 0.2–0.8) and 0.343 in the most arid. "The bias reduction relative to NWM3.0 is concentrated in the arid West and the Great Plains."
- Data and split: AORC forcing (1-km, aggregated to hydrofabric unit basins; PET computed from AORC), NOAA-OWP Hydrofabric, USGS hourly IV. Train 1991–2003, validation 2004–2008, test 2009–2018. Training excluded gauges with drainage area >50,000 km² because of compute.
- Compute: one CONUS training run takes "about five days on one A100 GPU for the unit-hydrograph configuration." A "10-year hourly simulation of the full network of more than 800,000 reaches with exact Muskingum-Cunge routing takes about four hours on a single GPU."
- Limitations stated by the authors: performance on the largest rivers is untested. Extending to large basins "will also require attention to reservoir operations, diversions, water use, floodplain storage, and long routing memory."
- The δHBV2.0MTS BMI caches about 351 days of daily inputs plus about 7 days of hourly inputs, re-warming every 7 days. NextGen runs need the "prior 358 days" of forcing — [github.com/mhpi/dhbv2](https://github.com/mhpi/dhbv2)
- Related hourly work from the same group: **h-Diffusion** (arXiv 2510.08488, preprint) is a diffusion model that downscales daily discharge (observed or simulated by a hydrologic model) into hourly hydrographs. It outperforms MTS-LSTM and MF-LSTM on CAMELS-hourly and uses inpainting with recent observations for DA — [arXiv 2510.08488](https://arxiv.org/abs/2510.08488)
- Note: the dhbv2 README's citation [2] mixes titles ("Diffusion-Based Probabilistic Modeling…" in the text, "Hourly U.S.-wide Flood Simulation…" in the BibTeX), both pointing to arXiv 2609.06794. The hourly-flood paper is the correct one for that ID — [dhbv2 README](https://github.com/mhpi/dhbv2); [arXiv 2609.06794](https://arxiv.org/abs/2609.06794)

### Inferences
- For a 6-hourly Tennessee study, δHBV2.0MTS-MC is the natural δ baseline. It is hourly (aggregate to 6 h), uses AORC (the same forcing as NWM3.0 retrospective), and runs on the NextGen hydrofabric, which also serves as NWM's native network and as a bridge to NWM3.0 comparisons.
- **Tennessee-specific red flag:** the mainstem Tennessee River gauges (e.g., Chattanooga, roughly 55,000 km² drainage area — my estimate, verify) and the lower river exceed the 50,000 km² training cutoff. The model also has no reservoir operations, so on the TVA-regulated mainstem and large tributaries (e.g., below Norris, Fontana, Watauga and Cherokee dams) it would likely do poorly unless restricted to unregulated or headwater gauges or given observed reservoir releases. The authors list this as a limitation; the Tennessee-specific consequence is my inference.
- Humid-basin median NSE (0.729) suggests the Tennessee basin falls in the model's stronger climate regime. The gains over NWM3.0 there are probably smaller than CONUS-wide, since bias gains are concentrated in the West.

### Gaps
- The paper reports no Southeast- or Tennessee-specific numbers (only CONUS maps in supplementary figures, which I did not access).
- No pretrained-weights download location appears in the paper. The dhbv2 repo implies weights for BMI deployment, but I did not verify they are publicly downloadable.

---

## 4. Use within NOAA NextGen / CIROH

### Takeaway
δHBV2.0 (daily) and δHBV2.0MTS (hourly) ship as BMI Python formulations for NextGen (`dhbv2` on PyPI). MHPI maintains a NextGen fork with dhbv2 integration, and CIROH funds the work (NOAA cooperative agreement NA22NWS4320003). The operational ML comparator inside NextGen is LSTM-TRoute (NOAA-OWP `lstm` BMI + t-route), run through NextGen In A Box (NGIAB).

### Cited Findings
- `mhpi/dhbv2` is described as "an operations-level module for NOAA-OWP's Next Generation National Water Modeling Framework (NextGen)" with BMI adapters `dhbv2.bmi.DeltaModelBmi` (daily δHBV 2.0) and `dhbv2.mts_bmi.MtsDeltaModelBmi` (hourly MTS). It installs with `uv pip install dhbv2`, is built on δMG, carries a "Non-Commercial (PSU)" license badge, and is funded by NOAA via CIROH (NA22NWS4320003). The daily model's prediction "is then distributed across the proceeding 24 hours" in NextGen. Example realization configs use `"time_step": 3600` — [github.com/mhpi/dhbv2](https://github.com/mhpi/dhbv2)
- MHPI GitHub org repos: `ngen` ("Next Generation Water Modeling Engine and Framework Prototype w/ dhbv2 integration", Apache-2.0, pushed 2026-10-07); `t-route` fork; `CIROH_DL_NextGen` (MIT), a CIROH project to "integrate and test deep learning (DL) and physics-informed deep learning (PIML) streamflow models and data assimilation into NOAA's Next Generation Water Modeling Engine," pointing to AORC 4 km forcing, NOAA-OWP hydrofabric/Lynker-spatial, and NOAA-OWP/lstm — [GitHub API listing of github.com/mhpi](https://github.com/mhpi); [CIROH_DL_NextGen README](https://github.com/mhpi/CIROH_DL_NextGen)
- δMG is "CSDMS BMI compliant" for NOAA-OWP's NextGen and AWI's NGIAB — [generic_deltamodel](https://github.com/mhpi/generic_deltamodel)
- Song et al. 2025: "The modeling system supports the Basic Model Interface (BMI), which allows seamless integration with the next-generation NWM" — [OSTI](https://www.osti.gov/pages/biblio/2555891-highresolution-nationalscale-water-modeling-enhanced-multiscale-differentiable-physicsinformed-machine-learning)
- The hourly paper calls LSTM-TRoute "an operational baseline ... within the current Water Resources Modeling Framework" (Patel et al.). Its simulations were generated with NOAA-OWP's LSTM BMI and NGIAB, and the paper stresses that δHBV2.0MTS-MC is "natively deployable in the same operational infrastructure" because it operates on the NextGen Hydrofabric — [arXiv 2609.06794](https://arxiv.org/abs/2609.06794)
- A CIROH project led by Shen develops and benchmarks DA methods for ML and differentiable models on a standardized testbed — [CIROH project page](https://ciroh.ua.edu/research-project/developing-and-benchmarking-data-assimilation-methods-on-a-standardized-testbed/)

### Inferences
- Running δHBV2.0/MTS through NGIAB or the MHPI ngen fork on a Tennessee hydrofabric subset (VPU 06) is the "operational-style" route. Running δMG/hydrodl2 directly in PyTorch is simpler for research batch runs.
- The NextGen hydrofabric differs from NHDPlus v2 but is derived from NHDPlus-family reference fabrics (my understanding, not verified here). A crosswalk at USGS gauge locations is the practical comparison point.

### Gaps
- I found no documentation of δHBV running in any *operational* NWS/NWC NextGen deployment. Status appears to be research-to-operations (CIROH), not operational.
- I could not confirm whether CIROH or NGIAB publishes a ready-made dHBV2 realization for VPU 06 (Tennessee).

---

## 5. Forecast-mode and data-assimilation differentiable models

### Takeaway
No δ model I found has been evaluated with NWP forecast forcings at 1–10 day lead. The forecast-like evidence is: (i) variational DA for δHBV on CAMELS (Jamaat et al. 2025), which raised one-day-lead median NSE from 0.75 to 0.82, on par with LSTM+DA; (ii) diffusion-based probabilistic models (DRUM, h-Diffusion) from the same group. A Tennessee 1–10 day comparison would need new experiments.

### Cited Findings
- **Jamaat, Song, Rahmani, Liu, Lawson & Shen, "Update hydrological states or meteorological forcings? Comparing data assimilation methods for differentiable hydrologic models"** (arXiv 2502.16444. Penn State Pure lists it as a December 2025 Elsevier journal article; journal name not confirmed here). It builds variational DA adjusting precipitation (δHBV-DA-P), states (δHBV-DA-S), or both (δHBV-DA-PS). "Differentiable streamflow models using the CAMELS dataset can benefit strongly and equivalently from variational DA as LSTM, with one-day lead time median NSE elevated from 0.75 to 0.82." It "matched or outperformed LSTM with DA in the eastern, northwestern, and central Great Plains regions." "Both precipitation and state adjusters were needed ... the latter being substantially more effective on its own." The framework "does not need systematic training data and could serve as a practical DA scheme for whole river networks" — [arXiv 2502.16444](https://arxiv.org/abs/2502.16444); [Penn State Pure](https://pure.psu.edu/en/publications/update-hydrological-states-or-meteorological-forcings-comparing-d/)
- **DRUM (arXiv 2412.11942, preprint):** a diffusion-based probabilistic runoff model for extreme-flood forecasting on CONUS basins. It "extends reliable lead times by nearly a full day for 20- and 50-year floods" under operational scenarios, and gains 3–7 days of lead time in eastern and northwestern U.S. precipitation-driven flood zones when driven by measured precipitation. It is pure DL, not a δ model, but comes from the same group — [arXiv 2412.11942](https://arxiv.org/abs/2412.11942)
- **h-Diffusion (arXiv 2510.08488, preprint):** hourly diffusion downscaling with inpainting-based DA "that largely improves flood forecasting performance" — [arXiv 2510.08488](https://arxiv.org/abs/2510.08488)
- Earlier LSTM "data integration" forecasting (Feng, Fang & Shen 2020, arXiv 1912.08949) is the LSTM-with-lagged-observations analogue that the DA paper compares against — [arXiv 1912.08949](https://arxiv.org/abs/1912.08949)
- The hourly δHBV2.0MTS paper says the model "sets up the stage for smart ensembles, probabilistic forecasting, and data assimilation" (future work, not demonstrated) — [arXiv 2609.06794](https://arxiv.org/abs/2609.06794)

### Inferences
- To build a δ forecast baseline for Tennessee 2020–2025: warm up δHBV2.0MTS with AORC through the issue time (358-day cache), then drive it with forecast forcings out to 10 days. Candidates are the NWM short/medium-range forcing (HRRR/GFS-based), GEFS, or an AORC "perfect-forecast" upper bound. Optionally apply δHBV-DA-S-style state adjustment from USGS observations at t0. This would be a new experiment, not a published configuration.
- Without DA, a δ model in forecast mode does not use recent observed streamflow. A GNN that ingests lagged gauge observations would hold an unfair advantage at 1–3 day leads. Report both "simulation-mode δ" and "δ+DA" if possible, and compare against persistence.

### Gaps
- I found no published δ-model results with GFS, GEFS or HRRR forecast forcings, or multi-day lead verification, for any region.
- The DA paper's per-lead (>1 day) skill and its Southeast results beyond "eastern" were not available at abstract level.

---

## 6. Other hybrid / physics-ML baselines (NWM post-processing, NWM emulators, physics-guided graph models)

### Takeaway
The standard NWM+ML baseline is Frame et al. (2021, JAWRA), an LSTM post-processor of NWM output. On 531 CAMELS basins it reached median NSE 0.73–0.75 against NWM's 0.62 (daily, NWM v2 retrospective). GNN emulators of the NWM (Sun et al., HESS) and physics-guided recurrent graph models are the closest methodological relatives of a GNN forecaster.

### Cited Findings
- **Frame, Kratzert, Raney, Rahman, Salas & Nearing (2021), "Post-Processing the National Water Model with Long Short-Term Memory Networks for Streamflow Predictions and Model Diagnostics," JAWRA 57(6), 885–905 (peer-reviewed).** Three LSTMs on 531 CAMELS basins:
  - LSTM_PP (NWM outputs): median NSE 0.73
  - LSTM_PPA (NWM outputs + atmospheric forcings): median NSE 0.75
  - LSTM_A (forcings only): median NSE 0.74
  - NWM reanalysis: median NSE 0.62
  - Trained 2004–2014 and evaluated 1994–2002. LSTM_A did best in ungauged basins. Diagnostics pointed to the land-surface component for mass bias and to the channel router for timing error. The authors recommend NWM routing as "a priority for NWM improvement."
  - Sources: [Google Research pub page](https://research.google/pubs/post-processing-the-national-water-model-with-long-short-term-memory-networks-for-streamflow-predictions-and-model-diagnostics/); [NOAA repository accepted manuscript](https://repository.library.noaa.gov/view/noaa/53162); [CIROH_DL_NextGen reference list](https://github.com/mhpi/CIROH_DL_NextGen)
- **GNN emulation of NWM (Sun et al., HESS preprint hess-2022-111, as described in reviewer and author comments):** "the GNN can almost fully emulate the NWM," with KGE about 0.9 against NWM output, then fine-tuned on observed streamflow because NWM is poor for high flows — [HESS discussion supplement RC2](https://hess.copernicus.org/preprints/hess-2022-111/hess-2022-111-RC2-supplement.pdf); [AC2](https://hess.copernicus.org/preprints/hess-2022-111/hess-2022-111-AC2-supplement.pdf). Final publication status not verified.
- **Physics-guided recurrent graph model** (NSF PAR 10287145) predicts flow and temperature in river networks, capturing interactions among segments and transferring knowledge from physics-based models. Streamflow was tested using NWM-simulated data for a Houston river network — [NSF PAR](https://par.nsf.gov/biblio/10287145-physics-guided-recurrent-graph-model-predicting-flow-temperature-river-networks)
- **LSTM-TRoute** (NOAA-OWP LSTM BMI + t-route, Patel et al.) is the NextGen operational-style ML+physical-routing hybrid. It scored median hourly NSE 0.359 on 2,831 CONUS gauges and 0.457 on CAMELS in the Yang et al. comparison — [arXiv 2609.06794](https://arxiv.org/abs/2609.06794)
- The hourly paper lists other LSTM+routing hybrids: LSTM-RAPID, spatial recursive modeling, DROP, LSTM-LTI, LSTM-TRoute. These pair pretrained LSTM runoff with network routing — [arXiv 2609.06794](https://arxiv.org/abs/2609.06794) (I did not verify the individual papers).
- CIROH abstract: "Improving NWM retrospective data using LSTM and high-resolution SWE estimates" (another NWM-correction line; details not verified) — [CIROH abstract](https://ciroh.ua.edu/abstracts/improving-national-water-model-retrospective-data-using-long-short-term-memory-and-high-resolution-swe-estimates/)
- An independent evaluation of NWM v3.0 retrospective at more than 4,300 USGS gauges found NWMv3.0 generally beats v2.1 on daily and peak-flow statistics — [U. Michigan CSS summary (via search)](https://css.umich.edu/publications/research-publications/comprehensive-assessment-and-comparison-streamflow-simulations)

### Inferences
- For the Tennessee paper, an "NWM + LSTM post-processor" (Frame-style) at gauges is cheap to build from the NWM3.0 retrospective (AWS) plus NWM operational short/medium-range forecast archives. It is the most direct NWM-hybrid baseline for the forecast setting.
- LSTM-TRoute's low published skill makes it a weak baseline. Including it would show the gap between operational NextGen ML and research models.

### Gaps
- I found no Tennessee- or Southeast-specific NWM post-processing study.
- The Sun et al. GNN-NWM emulator's final journal version and test region were not verified.

---

## 7. Practical: code, licence, data needs, runtime, and how to run for the Tennessee basin 2020–2025

### Takeaway
The code is public and fairly mature: δMG/`dmg` and `hydrodl2` (PyPI), `dhbv2` (PyPI, BMI), and `ddr` (Apache-2.0). Most MHPI code, however, carries a **Non-Commercial (PSU)** license. Hourly δHBV2.0MTS needs AORC forcing (available to the present on AWS), NextGen hydrofabric v2.2 attributes, and about 358 days of warm-up. CONUS training costs about 5 A100-days; inference is cheap (10 years hourly for all of CONUS in about 4 GPU-hours).

### Cited Findings
- **δMG (`generic_deltamodel`)**: PyTorch framework, "successor to HydroDL." Install with `uv pip install dmg`, extras `dmg[hydrodl2]`, `dmg[logging]`, `dmg[tune]`. Configs ship for CAMELS 531/671; custom regions need new observation configs (`data_path`, `gage_info`, `subset_path`). License badge "Non-Commercial (PSU)" — [github.com/mhpi/generic_deltamodel](https://github.com/mhpi/generic_deltamodel)
- MHPI repo licenses are reported by the GitHub API as NOASSERTION (custom) for generic_deltamodel, dhbv2, hydrodl2, hydroDL, dPLHBVrelease and HydroDLAdj. ngen is Apache-2.0, CIROH_DL_NextGen MIT, dMC-Juniata CC-BY-4.0. Recent activity: generic_deltamodel, dhbv2, hydrodl2 and ngen were all pushed in October 2026. The org also has `dHBV-camels-de` and `DataAssimilation` repos — [github.com/mhpi](https://github.com/mhpi)
- `ddr`: Apache-2.0, `uv`-based, CUDA 12/13, hydrofabric v2.2 adjacency build script (`build_hydrofabric_v2.2_matrices.py <conus_nextgen.gpkg> --gages ...`) — [github.com/deepgroundwater/ddr](https://github.com/deepgroundwater/ddr)
- Forcing and data for the hourly model: AORC (AWS Open Data `noaa-nws-aorc`), NOAA-OWP Hydrofabric, USGS hourly via REST API. NWM3.0 retrospective comes from AWS `nwm-archive` — [arXiv 2609.06794](https://arxiv.org/abs/2609.06794). The CIROH_DL_NextGen page points to AORC at 4 km and ngen-datastream forcingprocessor — [CIROH_DL_NextGen](https://github.com/mhpi/CIROH_DL_NextGen)
- Feng et al. 2022 evaluated δHBV with Daymet and a second CAMELS forcing set (lumped daily) — [arXiv 2203.14827](https://arxiv.org/abs/2203.14827)
- Runtime: about 5 days on one A100 for CONUS training (UH config). A 10-year hourly simulation of >800k reaches takes about 4 h on one GPU — [arXiv 2609.06794](https://arxiv.org/abs/2609.06794). dPL CONUS calibration: "1 single GPU for 1 hour" versus 2–3 days on a 100-CPU cluster for evolutionary calibration (secondary quote) — [eesm.science.energy.gov](https://eesm.science.energy.gov/publications/calibration-parameter-learning-harnessing-scaling-effects-big-data-geoscientific)
- Warm-up: 365 days of forcing (daily δHBV2.0) or 358 days (MTS) before the simulation start — [dhbv2 README](https://github.com/mhpi/dhbv2)

### Inferences (recipe for the Tennessee basin 2020–2025)
1. **Simulation baseline (cheapest, strongest evidence base):** run pretrained CONUS δHBV2.0MTS-MC, if weights are obtainable from MHPI/dhbv2, on the hydrofabric v2.2 VPU 06 subset with AORC from 2019 (for warm-up) through 2025. Aggregate the hourly output to 6-hourly and evaluate at the USGS gauges matched to the GNN's NHDPlus v2 reaches. The 2020–2025 window lies outside the published 2009–2018 test period, which makes it a clean out-of-sample test.
2. **Daily alternative:** use δHBV2.0 with UH routing (Song 2025, MERIT) or the published 40-year MERIT daily runoff routed with `ddr`. Daily output cannot resolve 6-hourly peaks; its daily-to-hourly BMI disaggregation is a flat distribution.
3. **Retraining option:** fine-tune or retrain on Tennessee and Southeast gauges with δMG and hydrodl2 (GPU-days). Retraining is needed if the dataset period or gauges differ, or if regulated basins are included.
4. **Forecast baseline:** warm-start at each issue time, then drive with archived NWM medium-range or GEFS forcings out to 10 days (new work), optionally with δHBV variational DA at t0.
5. **Regulation caveat:** TVA reservoir operations are not represented, and gauges >50,000 km² were excluded from training. Stratify evaluation into unregulated and headwater gauges versus regulated mainstem gauges, or use observed dam releases as upstream boundary conditions (would need custom code).
6. **Licence:** the PSU non-commercial license is fine for academic baselines. Check before any TVA (commercial or operational) use. `ddr` (Apache-2.0) and the dMC-Juniata release (CC-BY) are permissive.

### Gaps
- I could not confirm whether pretrained δHBV2.0MTS-MC weights and the hourly CONUS simulation dataset are publicly released (the paper's data statement lists inputs, not outputs).
- AORC's latency and availability through 2025 on AWS were not verified here. For real-time-like forecasts, AORC would be replaced by an analysis such as the NWM analysis forcing.
- No source gives δ-model runtime for a basin-sized (about 105,000 km²) Tennessee subset. Scaling the CONUS inference cost down suggests minutes on one GPU (inference, unverified).
- I found no published δ-model results specific to the Tennessee River basin.
