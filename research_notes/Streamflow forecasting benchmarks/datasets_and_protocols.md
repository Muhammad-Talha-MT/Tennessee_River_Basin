# Streamflow Forecasting Benchmarks: Datasets and Evaluation Protocols (for a 6-hourly, 1-10 day GNN forecast in the Tennessee River basin)

Research date: 2026-10-12. Scope: (a) datasets relevant to a CONUS/Tennessee sub-daily graph forecast; (b) evaluation conventions, metrics, baselines and significance testing in leading hydrologic ML forecasting papers.

Note on sourcing: items marked "[not re-verified this session]" are well-known citations I'm giving from prior knowledge, with a DOI link. I did not fetch them during this research.

---

## 1. Datasets: what exists, coverage, and which include river-network topology

### Takeaway
The "CAMELS-H" the brief refers to is almost certainly **CAMELSH** (Tran, Xu, Nguyen, Kim & Ivanov, *Scientific Data* 2025). It is the strongest base dataset for this project: hourly USGS streamflow plus NLDAS-2 and ERA5-Land forcing for 9,008 GAGES-II basins (3,166 with observed hourly flow), covering 1980-2024. Later Zenodo versions add NWM retrospective v2.1/v3.0 benchmarks and 2025 NWM short-range forecasts. None of the CAMELS-style US datasets ship river-network topology. LamaH-CE is the standard large-sample dataset that does, and it is the de facto GNN benchmark.

### Cited Findings

**CAMELSH (the "CAMELS-H" in the brief)**
- Tran, Xu, Nguyen, Kim, Ivanov (2025), "CAMELSH: A Large-Sample Hourly Hydrometeorological Dataset and Attributes at Watershed-Scale for CONUS," *Scientific Data* 12:1307, DOI 10.1038/s41597-025-05612-6. It covers 9,008 CONUS basins from GAGES-II, 3,166 with observed hourly streamflow, from 1980 to 2024. Basin areas run from about 3.8 to 5×10⁴ km². — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12304116/); [Nature](https://www.nature.com/articles/s41597-025-05612-6)
- Streamflow comes from the USGS NWIS Instantaneous Values service: native 15-60 min data aggregated to hourly, converted to m³/s, and set to UTC+0. USGS qualifier flags (A, e, R, P) are retained, but no further QC is applied. Missing values are NaN, and coastal gauges can be negative because of tides. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12304116/)
- Forcing is NLDAS-2 (11 hourly variables at 0.125°, basin-averaged), with ERA5-Land as a second forcing set. Attributes are 439 from GAGES-II, 195 from HydroATLAS and 9 CAMELS-style climate indices. The paper does not describe river-network topology, flow connectivity or nesting. It reports no ML benchmark results. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12304116/)
- Zenodo record v8 (10 July 2026) adds basin-averaged hourly NWM forecast forcing (18-h lead, 1 Jan 2025 to 9 Mar 2026) for all 9,008 basins and NWM short-range (18-h) streamflow forecasts for 8,723 locations in 2025. The update history lists:
  - hourly NWM v2.1 (1980-2020) and v3.0 (1980-2023) retrospective simulations at 8,664 USGS-matched locations;
  - hourly observed streamflow and water level for 5,767 USGS gauges, 1980-01-01 to 2024-12-31 (DOI 10.5281/zenodo.16729675);
  - benchmark NWM data under DOI 10.5281/zenodo.16763144.
  
  The record says a newer version exists. — [Zenodo 21302482](https://zenodo.org/records/21302482)
- Code is at github.com/vinhngoctran/CAMELSH. NLDAS forcing DOIs are 10.5281/zenodo.15066778 and 10.5281/zenodo.15070091, and the ERA5-Land DOI is 10.5281/zenodo.15264813. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12304116/); [Zenodo](https://zenodo.org/records/21302482)

**Hourly CAMELS (Gauch et al. 2021 data)**
- Gauch et al. (2021), HESS 25:2045, uses 516 of the 531 CAMELS benchmark basins that have hourly USGS IV data, with hourly NLDAS-2 forcing. The data DOI is 10.5281/zenodo.4072701, and models and predictions are at 10.5281/zenodo.4071886. — [arXiv/ar5iv](https://ar5iv.arxiv.org/html/2010.07921); [HESS](https://hess.copernicus.org/articles/25/2045/2021/)

**CAMELS-US (original)**
- [not re-verified this session] Newman et al. (2015), HESS 19:209, "Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA": 671 basins with daily Daymet, Maurer and NLDAS forcing — https://doi.org/10.5194/hess-19-209-2015. Addor et al. (2017), HESS 21:5293, "The CAMELS data set: catchment attributes and meteorology for large-sample studies" — https://doi.org/10.5194/hess-21-5293-2017. Daily only, 1980-2014, and no topology.

**Caravan and extensions**
- [not re-verified this session] Kratzert et al. (2023), "Caravan - A global community dataset for large-sample hydrology," *Scientific Data* 10:61 — https://doi.org/10.1038/s41597-023-01975-w. Daily ERA5-Land forcing and HydroATLAS attributes in UTC.
- **GRDC-Caravan** (Färber et al., ESSD 17:4613, 19 Sept 2025): 5,356 catchments in 25 countries, 1950-2023, ERA5-Land plus HydroATLAS, CC-BY-4.0, Zenodo 10.5281/zenodo.15349031 (v0.6, May 2025). I found no evidence of an hourly version. — [ESSD](https://essd.copernicus.org/articles/17/4613/2025/essd-17-4613-2025.html); [GRDC news](https://grdc.bafg.de/news/2025/new_paper_grdc_caravan/); [Zenodo](https://zenodo.org/records/15349031)
- **Caravan MultiMet** (Shalev & Kratzert, Google, arXiv 2411.09459, Nov 2024):
  - adds precipitation nowcasts (CPC, IMERG v07 Early, CHIRPS) and weather forecasts (ECMWF IFS HRES, GraphCast, CHIRPS-GEFS) to Caravan;
  - IFS and GraphCast run to 10-day lead and CHIRPS-GEFS to 16 days;
  - data are daily, basin-averaged and in UTC;
  - billed as the first open large-sample dataset with weather-forecast inputs, intended for benchmarking in "real-time forecasting scenarios."
  
  — [arXiv](https://arxiv.org/abs/2411.09459v1); [Zenodo Part 2](https://zenodo.org/records/14161281)

**LamaH-CE (topology-aware; the de facto GNN benchmark)**
- Klingler, Schulz & Herrnegger (2021), ESSD 13:4529, covers 859 gauged basins in Central Europe (Danube). Data are daily and hourly with quality flags, and most series exceed 35 years. It adds intermediate-catchment delineations and attributes "allow[ing] the hydrological network and river topology to be considered." Hourly plus daily is about 70 GB. Zenodo 10.5281/zenodo.5153305, CC BY 4.0. — [ESSD discussion](https://essd.copernicus.org/articles/13/4529/2021/essd-13-4529-2021-discussion.html); [Zenodo](https://zenodo.org/record/4525244)

**NWM retrospective and AORC**
- NWM v3.0 retrospective is a 44-year CONUS run (Feb 1979 to Jan 2023) forced by AORC, with hourly CHRTOUT streamflow and 3-hourly land-surface output. **No streamflow data assimilation** is done in any retrospective run. v2.1 also used AORC, while v2.0 and v1.2 used NLDAS. NCAR's rechunked table gives the end as Feb 2023, which differs slightly from the AWS registry. — [AWS Registry](https://registry.opendata.aws/nwm-archive/); [NCAR rechunk](https://github.com/NCAR/rechunk_nwm_retro)
- A University of Michigan assessment reports that NWM v3.0 retrospective generally outperformed v2.1 for daily and peak-flow statistics at more than 4,300 USGS gauges, 1979-2020. — [UMich CSS](https://css.umich.edu/publications/research-publications/comprehensive-assessment-and-comparison-streamflow-simulations)

### Inferences
- **Tennessee coverage.** CAMELSH (GAGES-II-based, 3,166 gauges with hourly data) almost certainly has many more Tennessee-basin gauges than CAMELS-US. Its record runs through 2024, and the observed-streamflow extension runs to 31 Dec 2024. Test years 2020-2024 can therefore come straight from CAMELSH. **2025 observations would have to be pulled from USGS NWIS IV** with the same aggregation: hourly mean, UTC, then 6-hourly.
- **NWM benchmarks for the test window.** The CAMELSH NWM benchmarks cover the 2020-2023 part of the test window with v3.0 retrospective (a simulation, not a forecast). For 2025 they offer only short-range (18-h) forecasts, so no 1-10 day operational benchmark is bundled. NWM medium-range archives would have to be obtained separately; see Gaps.
- **Topology must be built by the authors.** None of CAMELS-US, CAMELSH or Caravan provide graph edges. The usual options are NHDPlus HR/V2 flowlines, GAGES-II nesting, or USGS NLDI upstream navigation. LamaH-CE is the community's topology-aware dataset, so running the GNN on LamaH-CE as a secondary sanity benchmark would strengthen a paper's generality claim.
- **Regulation.** GAGES-II attributes in CAMELSH include regulation indicators. The Tennessee basin is heavily regulated by TVA reservoirs, so stratifying results by regulated versus reference gauges is advisable.

### Gaps
- I did not verify the exact number of CAMELSH or CAMELS-US gauges inside HUC-06 (Tennessee region).
- I found no "Caravan-hourly" dataset. GRDC-Caravan and MultiMet are both daily.
- I did not check availability or archive locations of NWM operational medium-range (10-day) forecasts for 2020-2025. NOAA's AWS NWM archive is a likely source, but this is unverified.
- AORC dataset papers were not fetched. [not re-verified] Fall et al. (2023), JAWRA, describes AORC as a 1979-present hourly ~1 km CONUS forcing.

---

## 2. Evaluation conventions in key forecasting papers (splits, aggregation, lead time)

### Takeaway
Leading papers in this line (Google/JKU: Kratzert, Gauch, Klotz, Nearing) share a common protocol:
- multi-basin training (never single-basin);
- strictly temporal out-of-sample test periods, and often spatial k-fold for "ungauged" claims;
- **median** per-basin metrics, with mean sometimes added;
- several random-seed ensembles;
- paired **Wilcoxon signed-rank** tests with effect sizes (Cohen's d);
- metrics reported **per lead time** in forecast studies.

Nearing et al. (2024, *Nature*) is the template for event-based "reliability": precision, recall and F1 at return-period thresholds, with a ±2-day matching window.

### Cited Findings

**Nearing et al. (2024), "Global prediction of extreme floods in ungauged watersheds," *Nature* 627:559-563, DOI 10.1038/s41586-024-07145-1**
- Cross-validation combines temporal and spatial splits so every record is out-of-sample in both. Temporal folds leave no training data from any gauge within 1 year (the encoder sequence length) of any test data. Spatial folds are random k-fold with k = 10. Robustness splits use continents (k = 6), climate zones (k = 13) and hydrologically separated terminal-basin groups (k = 8). — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- Reanalysis covers 1984-2021 and reforecasts 2014-2021. Hydrograph metrics are computed over 2014-2021. Lead times run 0-7 days. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- Architecture: the encoder LSTM ingests 365 days of historical meteorology (ERA5-Land, CPC, IMERG-early). The decoder LSTM runs across the 0-7 day horizon on ECMWF IFS HRES forecast inputs. This is true **forecast mode** with archived NWP, not perfect-forcing hindcasts. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- GloFAS v4 was benchmarked only at 0-day lead, because its reforecast archive does not span the full year. The authors note the comparison favours GloFAS, which was calibrated on some evaluation gauges and periods. GloFAS was evaluated on 4,089 gauges, 1,144 of them used in its calibration. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- Statistics: two-sided paired Wilcoxon signed-rank tests with Cohen's d effect sizes. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- Headline result: AI forecasts reach reliability at up to 5-day lead in ungauged basins that is similar to or better than GloFAS nowcasts. Accuracy on 5-year return-period events is similar to or better than current accuracy on 1-year events. — [MURAL abstract](https://mural.maynoothuniversity.ie/id/eprint/21627/); [Google Research](https://research.google/pubs/ai-increases-global-access-to-reliable-flood-forecasts/)
- Gauge-count inconsistency: the text says 5,680 gauges, while the Extended Data Fig. 6 caption says 5,860. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)

**Gauch et al. (2021), MTS-LSTM, HESS 25:2045**
- Uses 516 hourly CAMELS basins. Train Oct 1990 to Sep 2003, validation Oct 2003 to Sep 2008, test Oct 2008 to Sep 2018. Results are reported as the **median across basins** (plus a mean-NSE row) and averaged over **10 independently trained models**. — [ar5iv](https://ar5iv.arxiv.org/html/2010.07921)
- Statistics: Wilcoxon signed-rank test at α = 0.001 with Cohen's d. No multiple-comparison correction is mentioned. — [ar5iv](https://ar5iv.arxiv.org/html/2010.07921)
- The NWM v2 reanalysis (hourly, 1993-2018) is the process-model benchmark. For the 3- and 6-hourly comparisons, NWM is aggregated from hourly output. The MTS-LSTM had significantly higher NSE than NWM. — [ar5iv](https://ar5iv.arxiv.org/html/2010.07921); [HESS](https://hess.copernicus.org/articles/25/2045/2021/)

**Nearing et al. (2022), DA vs autoregression, HESS 26:5493**
- Compares autoregression (feeding lagged observed streamflow as an input) with variational data assimilation that updates LSTM states by backpropagation. **Autoregression was more accurate and cheaper.** It is sensitive to missing observations, which a simple training strategy (randomly masking lagged-flow inputs) largely fixes. A reviewer felt the gap between methods was overstated. — [HESS](https://hess.copernicus.org/articles/26/5493/2022/); [discussion](https://hess.copernicus.org/articles/26/5493/2022/hess-26-5493-2022-discussion.html)

**Kratzert, Gauch, Klotz & Nearing (2024), "HESS Opinions: Never train an LSTM on a single basin," HESS 28:4187**
- Argues LSTM rainfall-runoff models are best trained on many basins, and that training on single or small homogeneous sets is a common mistake. The preprint survey found 122 of 150 papers trained on individual catchments. That figure is from the preprint and may differ in the final version. — [HESS](https://hess.copernicus.org/articles/28/4187/2024/); [Google Research](https://research.google/pubs/hess-opinions-never-train-a-long-short-term-memory-lstm-network-on-a-single-basin/)

**Frame et al. (2022), "Deep learning rainfall-runoff predictions of extreme events," HESS 26:3377**
- LSTM and MC-LSTM stayed relatively accurate on high-return-period events against the Sacramento model and the NWM, **even when extremes were excluded from training**. The test splits years by the annual peak flow so that extremes are unseen. Adding mass-balance constraints reduced skill on extremes. A 2023 corrigendum fixed Figs. 2 and C2: models trained only on years with return period ≤5 years show performance degrading as event return period increases. — [HESS](https://hess.copernicus.org/articles/26/3377/2022/); [corrigendum](https://hess.copernicus.org/articles/26/3377/2022/hess-26-3377-2022-corrigendum.pdf)

**Klotz et al. (2022), "Uncertainty estimation with deep learning for rainfall-runoff modeling," HESS 26:1673**
- Sets up an uncertainty-estimation benchmarking procedure with four DL baselines: three mixture-density-network variants (Gaussian mixtures, countable mixtures of asymmetric Laplacians, uncountable mixtures of asymmetric Laplacians) and MC dropout. The mixture density networks are strong baselines. Implemented in NeuralHydrology. — [HESS](https://hess.copernicus.org/articles/26/1673/2022/)

**GNN-specific precedents (relevant for reviewer expectations)**
- Kirschstein & Sun (2024), "The Merit of River Network Topology for Neural Flood Forecasting," arXiv 2405.19836 (ICML 2024 per keywords; acceptance not confirmed in the text).
  - Data: LamaH-CE hourly, 358-gauge connected Danube component, 2000-2017.
  - Splits: test years 2016-2017; three 8-year training splits; random 1/5 holdout for validation.
  - Task and inputs: 6-h lead time from a 24-h window of past discharge, so the model is autoregressive.
  - Baselines and metrics: MLP baselines, no persistence baseline; NSE averaged across gauges as mean ± SD over folds; no significance tests.
  - Finding: **the GNN did not benefit from topology**. — [arXiv html](https://arxiv.org/html/2405.19836v1)
- Wang et al. (arXiv 2410.05536) attribute the weak use of topology to over-squashing. They propose a reachability-based graph transformation that beats EA-LSTM on extreme floods. — [arXiv](https://arxiv.org/html/2410.05536v3)
- A 2026 HESS paper (vol. 30, p. 2079) on the LamaH-CE Upper Danube (530 subbasins, 1987-2017) reports that all LSTM-GNN variants beat a baseline LSTM. LSTM-GAT was best, with mean NSE 0.61 and KGE 0.65. I only saw this in a search snippet and did not read the full paper. — [HESS PDF](https://hess.copernicus.org/articles/30/2079/2026/hess-30-2079-2026.pdf)

### Inferences
- **Recommended protocol for this project.**
  - Temporal split: train 1979-2018, validate 2019, test 2020-2025. This matches the convention.
  - Gap between train and test: add a buffer of at least one input-sequence length, following Nearing 2024's 1-year rule. 2019 already serves as that buffer.
  - Ungauged claim: add a spatial k-fold (e.g., k = 5-10 over gauges, or hydrologically separated sub-basins, which is especially relevant for a GNN) to test ungauged or masked-node prediction.
  - Aggregation: report median and mean across gauges, per lead time (6-h steps up to 240 h, or at least days 1-10).
  - Seeds: use a seed ensemble (Gauch used 10).
  - Significance: paired Wilcoxon with Cohen's d. With many lead times and baselines, add a Holm or Benjamini-Hochberg correction, which Gauch did not use.
- **Topology control.** Given Kirschstein & Sun's negative result, reviewers will expect an "isolated-node" (no-edge) version of the same architecture as a control. Without it, the GNN's gain cannot be attributed to topology.
- **Autoregressive inputs.** If the GNN uses lagged observed flow (as Kirschstein does, and as Nearing 2022 recommends), the persistence baseline is crucial. State it explicitly and test robustness to missing lagged inputs.
- **Forecast mode versus perfect forcing.** Nearing 2024 used archived IFS HRES forecasts as decoder inputs. Feeding observed future forcing (AORC or NLDAS during the lead window) is a "perfect-forcing hindcast" that overstates forecast skill. Report both and label them clearly. Possible archived NWP sources for 2020-2025 over CONUS include GEFSv12 reforecasts (2000-2019) plus the operational GEFS archive, ECMWF IFS (via Caravan MultiMet, but daily and only for Caravan basins), and NWM medium-range forcing. These are not verified here.

### Gaps
- I did not retrieve per-lead-time reporting conventions from a CONUS-specific multi-day LSTM forecasting paper. Nearing 2024 reports per lead time 0-7 days; other examples were not found within budget.
- I found no dedicated peer-reviewed "best-practices for fair forecast comparison" paper (e.g., matched issue times, forecast vs simulation) beyond Pappenberger 2015 (Section 4) and the implicit practice in Nearing 2024. A dedicated search may still surface one.

---

## 3. Metrics (continuous, hydrological signatures, event-based, probabilistic)

### Takeaway
The standard metric set is NeuralHydrology's: NSE, KGE plus components, α-NSE, β-NSE, Pearson r, FHV/FMS/FLV (Yilmaz 2008), peak timing, missed peaks, peak MAPE, and CRPS/PICP/MPIW for probabilistic output. For flood forecasting, add Nearing-2024-style event precision, recall and F1 at 1-, 2-, 5- and 10-year return periods with a ±2-day tolerance. CSI is a natural and equivalent addition: CSI = hits / (hits + misses + false alarms).

### Cited Findings
- **NeuralHydrology metrics module.** The metrics and defaults are listed below. — [NeuralHydrology docs](https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html)

  | Metric | Notes and defaults |
  |---|---|
  | NSE | — |
  | MSE, RMSE | — |
  | KGE | weights [1,1,1] |
  | Beta-KGE, Alpha-NSE, Beta-NSE | Gupta et al. 2009 |
  | Pearson r | — |
  | FHV | h = 0.02, i.e., top 2% of the flow duration curve |
  | FMS | 0.2-0.7 |
  | FLV | l = 0.3 |
  | Peak-Timing | window 3 for daily, 12 for hourly; Kratzert et al. 2020 |
  | Missed-Peaks | percentile 80; window 1 daily, 12 hourly |
  | Peak-MAPE | — |
  | CRPS | ensemble/samples |
  | MPIW, PICP | α = 0.1 |

- **Gauch 2021 peak-timing definition.**
  - Observed peaks are found by topographic prominence of at least the observed SD.
  - Smaller peaks are removed until peaks are at least 100 steps apart.
  - For each observed peak, the largest prediction is taken within ±1 day (hourly) or ±3 days (daily).
  - The error is the mean absolute time offset.
  
  Gauch 2021 also reports NSE, KGE, r, α-NSE, β-NSE, FHV, FMS and FLV. — [ar5iv](https://ar5iv.arxiv.org/html/2010.07921)
- **Nearing 2024 event metrics.**
  - Return periods are computed separately per gauge for observed and modelled series, "using the methodology described by the US Geological Survey Bulletin 17b."
  - An event is a hit when both hydrographs cross their return-period thresholds "within two days of each other."
  - Precision, recall and F1 (harmonic mean) are computed per gauge. They can be undefined when there are no observed or predicted events.
  - "Reliability" is operationally these event scores.
  - Hydrograph metrics reported: bias, NSE, KGE and α-NSE. Peak timing and FHV/FLV are not used.
  
  — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- **Return periods evaluated.** The AI model achieved higher precision and recall than GloFAS for return periods of 1 to 10 years. — [Nature listing / search summary](https://www.nature.com/articles/s41586-024-07145-1)
- **KGE baseline.** A mean-flow prediction scores KGE = 1 − √2 ≈ −0.41, not 0 as with NSE. Negative KGE can therefore still beat the mean-flow benchmark, and "KGE > 0 = good" is unreliable. The paper recommends purpose-specific metrics and benchmarks. — Knoben, Freer & Woods (2019), HESS 23:4323, [HESS](https://hess.copernicus.org/articles/23/4323/2019/)
- [not re-verified this session] Yilmaz, Gupta & Wagener (2008), WRR 44, W09417, "A process-based diagnostic approach to model evaluation" (FHV/FMS/FLV) — https://doi.org/10.1029/2007WR006716. Gupta et al. (2009), J. Hydrol. 377:80-91 (KGE decomposition) — https://doi.org/10.1016/j.jhydrol.2009.08.003. Kling et al. (2012), J. Hydrol. 424-425:264 (KGE′ using the coefficient of variation) — https://doi.org/10.1016/j.jhydrol.2012.01.011.

### Inferences
- **Suggested primary metric set, per lead time (6 h to 240 h), median across test gauges:**
  - NSE and KGE, with r, α and β components;
  - log-NSE or FLV for low flows, if relevant;
  - FHV (2%) and peak-timing error with NeuralHydrology windows, adjusted for 6-hourly data (e.g., ±2 steps = ±12 h);
  - peak magnitude error (Peak-MAPE);
  - event POD (= recall), FAR (= 1 − precision), CSI and F1 at 1-, 2- and 5-year (optionally 10-year) return periods with a ±2-day matching window, as in Nearing 2024.
- **Event metrics per gauge.** Compute them per gauge and summarise as medians, since precision and recall are undefined where no events occur. Also report pooled (micro-averaged) CSI across gauges, because few 5-year events will occur in a 6-year test period.
- **Bulletin version.** Nearing 2024 cites Bulletin 17B. Bulletin 17C (2018) is the current US standard and would be the defensible choice for US gauges.

### Gaps
- I did not verify a canonical CSI or FAR definition paper for hydrology, though these are standard contingency-table scores. A citation such as Jolliffe & Stephenson (2012), *Forecast Verification*, would be standard but was not verified here.
- The log-NSE convention (e.g., ε offset for zero flows) was not researched.

---

## 4. Reference/benchmark forecasts, skill scores and significance

### Takeaway
Pappenberger et al. (2015) is the canonical citation for benchmark choice. **The apparent skill depends heavily on the benchmark.** Long-term-average (climatology) benchmarks are trivially beaten and create "naïve skill," while **meteorological persistence** was the most useful and hardest benchmark for EFAS. For a 1-10 day ML forecast, report skill against:
- streamflow persistence (last observed value), which is very hard to beat at short leads;
- day-of-year climatology;
- the NWM (v3.0 retrospective as a simulation reference, and medium-range forecasts if available);
- a strong ML baseline (e.g., an LSTM without a graph).

### Cited Findings
- **Pappenberger et al. (2015), J. Hydrol. 522:697-713, DOI 10.1016/j.jhydrol.2015.01.024.**
  - Compared 23 benchmark-derivation methods with CRPS in EFAS.
  - Meteorological persistence (driving the hydrological model with the latest observed precipitation and temperature) was the most useful benchmark and avoided the most naïve skill.
  - Hydrological long-term-average benchmarks were "very easily beaten" and produced much naïve skill.
  - The 2014 poster version recommends different benchmarks for different flow regimes (rising and falling limb, 20th and 80th percentile flows) and for short range (<3 days) versus medium range (3-30 days). These recommendations come from the poster and may differ in the paper.
  
  — [CentAUR open access](https://centaur.reading.ac.uk/39072/); [HAL poster](https://hal.inrae.fr/hal-02599834v1)
- Persistence forecasts are a low-cost benchmark often considered "hard-to-beat" at medium-range timescales. — [arXiv 2106.09547 (ML post-processing of ensemble streamflow forecasts)](https://arxiv.org/pdf/2106.09547)
- Persistence-based skill depends strongly on lead time, flow threshold, basin size and flow regime (65 Himalayan gauges, 1-12 day leads, comparing persistence, climatology and anomaly persistence). — [MDPI Forecasting 2(3):13](https://doi.org/10.3390/forecast2030013)
- Example of CRPSS decay with lead time: a mean CRPSS of 0.8 at 1 day falls to 0.32 at 2 weeks, relative to climatology. — [HESS 25:4159 (2021)](https://hess.copernicus.org/articles/25/4159/2021/)
- KGE-based skill must be judged against the −0.41 mean-flow baseline. — [Knoben et al. 2019](https://hess.copernicus.org/articles/23/4323/2019/)
- Significance-testing precedent: paired two-sided Wilcoxon signed-rank tests across gauges with Cohen's d (Nearing 2024), and Wilcoxon at α = 0.001 with Cohen's d (Gauch 2021). — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/); [ar5iv](https://ar5iv.arxiv.org/html/2010.07921)
- The NWM retrospective has **no data assimilation**, so it is a simulation benchmark rather than a forecast. Comparing a model that assimilates observed flow (autoregressive) against it is unfair at short leads. — [AWS Registry](https://registry.opendata.aws/nwm-archive/)
- Nearing 2024 explicitly flags benchmark asymmetry: GloFAS was scored at 0-day lead and partly in calibration. This is a model for transparently disclosing an unequal comparison. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)

### Inferences
- **Skill score form.** A generic skill score is SS = (S_model − S_ref)/(S_perfect − S_ref). Two standard instances:
  - The NSE computed with persistence as the reference instead of the mean is the "persistence index" (PI). I didn't verify the citation for PI: Kitanidis & Bras 1980, WRR. It should be reported per lead time, because persistence decays fast and PI separates real forecasting skill from autocorrelation.
  - With ensembles, use CRPSS against climatology and persistence ensembles.
- **Matched-sample comparison.** Compare all models on identical issue times (e.g., 00/06/12/18 UTC), identical gauges and identical valid times, dropping any timestep where any model or the observation is missing. This is standard verification practice that I inferred; I found no single source stating it for ML hydrology.
- **Bootstrap confidence intervals.** Report CIs on median metrics by bootstrapping over gauges, and over years or blocks for temporal dependence, alongside Wilcoxon p-values. This is common practice; I found no specific hydrology-ML citation within budget.

### Gaps
- I did not verify a source for the "persistence index" (Kitanidis & Bras 1980) or the KGE skill score formula (Knoben 2019 proposes KGE benchmark-relative interpretation, but I did not confirm whether it defines a formal KGESS).
- I found no peer-reviewed reference applying a block-bootstrap protocol specifically to LSTM or GNN streamflow forecasts.

---

## 5. Return-period flood thresholds and NWS flood stages

### Takeaway
For US gauges, compute 1-, 2-, 5- and 10-year thresholds from training-period annual maxima using **Bulletin 17C** (log-Pearson Type III fitted with the Expected Moments Algorithm (EMA) and Multiple Grubbs-Beck low-outlier test, implemented in USGS PeakFQ). Do this separately for observed and modelled series, as Nearing 2024 did. NWS flood categories (action/minor/moderate/major) exist only at NWS forecast points, are stage-based, and change over time. They are useful as a supplementary impact-based threshold set via a rating-curve conversion, but not as the primary threshold.

### Cited Findings
- **Bulletin 17C** (England et al., USGS Techniques and Methods 4-B5, 2018; v1.1 May 2019) is the US federal guideline. It fits log-Pearson Type III with the Expected Moments Algorithm, which handles interval and censored data, and uses a generalized low-outlier test (Multiple Grubbs-Beck). USGS PeakFQ implements it. — [USGS pubs](https://pubs.usgs.gov/publication/tm4B5); [PeakFQ](https://www.usgs.gov/tools/peakfq)
- Nearing 2024 computed per-gauge return periods for observed and simulated series following Bulletin 17B. It notes that some gauges lack enough samples for some return periods. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- **NWS flood categories** (minor, moderate, major, plus action stage) are defined per forecast point. Not all categories exist at every point. — [NWS directive 10-950](https://www.weather.gov/directives/sym/pd01009050curr.pdf)
- NWS flood stages are periodically revised (e.g., a 2025 change at one gauge), so cached thresholds can go stale. — [NWS LWX example](https://www.weather.gov/lwx/GeorgesCreek2025)
- A bulk CSV of NWPS gauge thresholds (action, minor, moderate, major) is reportedly at water.noaa.gov/resources/downloads/reports/nwps_all_gauges_report.csv. This is secondhand and unverified. NWS also exposes a riv_gauges ArcGIS MapServer. — [secondhand listing](https://mcp.directory/skills/nws-flood-thresholds); [NWS MapServer](https://mapservices.weather.noaa.gov/eventdriven/rest/services/water/riv_gauges/MapServer)

### Inferences
- **Recommended threshold procedure.**
  - Build annual maxima (water years) of instantaneous or 6-hourly flow from the 1979-2018 training period only, so test events are not used to set thresholds.
  - Fit LP3 via EMA (PeakFQ or a Python port).
  - Derive Q1.01 or Q1.5 (≈ "1-year"), Q2, Q5 and Q10.
  - Use USGS annual peak-flow files (NWIS peak) as an alternative input for long records.
- **Defining the "1-year" event.** A "1-year" return period is ill-defined for annual-maximum series, since its exceedance probability is 1. Papers using it (Nearing 2024, Google Flood Hub) use small return periods computed from the fitted distribution. State explicitly how the "1-year" event is defined (e.g., 1.01-yr or a partial-duration series).
- **Regulated gauges.** TVA regulation makes stationarity and frequency assumptions weaker for regulated gauges. Stratify, or use the regulation attribute to flag gauges.
- **NWS-to-USGS mapping.** Many NWS forecast points in the Tennessee basin are co-located with USGS gauges, but mapping NWS LIDs to USGS site IDs and converting stage to flow via USGS ratings would need verification.

### Gaps
- I did not confirm the official NWPS API endpoint or the authoritative bulk-download URL for flood categories, nor a crosswalk from NWS LID to USGS site ID.
- Minimum record length for stable Q5 and Q10 estimates was not researched. With about 40 training years this is adequate for Q2-Q10, but extrapolation for Q25 and above is uncertain.
