# LSTM Rainfall-Runoff and Streamflow Forecasting Baselines (for a GNN 6-hourly, 1-10 day Tennessee River basin forecast paper)

Research date: 2026-10-12. Scope: reproducible/public LSTM baselines. Where a statement is from my prior knowledge and was not re-verified in this session, it is in "Inferences" or "Gaps", not "Cited Findings".

---

## 1. Kratzert et al. LSTM / EA-LSTM / regional LSTM and the NeuralHydrology library (version, model classes, hourly support, forecast models, autoregression/DA, hindcast vs forecast forcings)

### Takeaway
NeuralHydrology (NH) is the de facto reproducible LSTM baseline library. The latest release is v1.13.0 (14 Jan; year not shown on the page, probably 2026). It has CudaLSTM, EA-LSTM, MTS-LSTM, ARLSTM (autoregressive), GRU, Transformer, Mamba, xLSTM, MC-LSTM, hybrid conceptual models, and four encoder-decoder "forecast" LSTMs (handoff, multihead, sequential, stacked), added in v1.8.0. They take separate `hindcast_inputs`/`forecast_inputs` and `forecast_seq_length`. v1.13.0 adds AORC hourly forcing support. The core library documents no DA. DA lives in a Nearing et al. (2022) fork branch and in Google's fork.

### Cited Findings
**Original Kratzert papers (daily CAMELS benchmarks):**
- Kratzert et al. 2018, HESS 22, 6005–6022 ("Rainfall–runoff modelling using LSTM networks"):
  - Data: 241 CAMELS basins in 4 hydrologic units, daily Daymet forcing, 365-day input sequences, 2-layer LSTM with 20 units.
  - Single-basin LSTM: mean validation NSE 0.63 vs 0.58 for SAC-SMA+Snow-17.
  - Regional model fine-tuned per basin: median NSE 0.72 vs 0.71 and mean 0.68 vs 0.58 for the benchmark. — [HESS 2018](https://hess.copernicus.org/articles/22/6005/2018/)
- Kratzert et al. 2019, HESS 23, 5089–5110 ("Towards learning universal, regional, and local hydrological behaviors…"):
  - Setup: one regional LSTM/EA-LSTM over 531 CAMELS basins, daily Maurer forcing plus 27 static attributes. Train 1 Oct 1999–30 Sep 2008; validation 1 Oct 1989–30 Sep 1999. Loss is a basin-normalized NSE*.
  - EA-LSTM median NSE: 0.71 for single models, 0.74 for an 8-member ensemble.
  - Benchmarks: regionally calibrated VIC median 0.31 and mHM 0.53. The EA-LSTM significantly outperformed basin-calibrated HBV-ensemble and mHM.
  - An LSTM with static inputs was slightly better than EA-LSTM (small effect size). — [HESS 2019](https://hess.copernicus.org/articles/23/5089/2019/)

**NeuralHydrology releases:**
- v1.13.0 (14 Jan): moved from conda to uv. Added an xLSTM backbone, early stopping, LR scheduling, embedding networks inside MTS-LSTM, CAMELS-IND and CAMELS-DE, "AORC hourly forcing support" (PR #262), and a dataset registry for external datasets. — [NH release v1.13.0](https://github.com/neuralhydrology/neuralhydrology/releases/tag/v1.13.0)
- v1.12.0 (15 Mar): NaN input handling; dynamic features returned as a dict; hourly frequency identifier changed from `H` to `h`; fixed forecast config docs and the forecast-overlap regularization. — [NH releases](https://github.com/neuralhydrology/neuralhydrology/releases)
- v1.11.0: Mamba state-space model and AdamW. v1.10.0: HybridModel/SHM conceptual model. — [NH releases](https://github.com/neuralhydrology/neuralhydrology/releases)
- v1.8.0 (07 Jul): added HandoffForecastLSTM, MultiheadForecastLSTM, SequentialForecastLSTM, StackedForecastLSTM, the `forecast_inputs`/`hindcast_inputs` config options, and a timestep counter. — [NH releases](https://github.com/neuralhydrology/neuralhydrology/releases)

**NeuralHydrology model zoo** ([NH models docs](https://neuralhydrology.readthedocs.io/en/latest/usage/models.html)):
- Documented models: ARLSTM, CudaLSTM, CustomLSTM, EA-LSTM, EmbCudaLSTM (deprecated), GRU, Hybrid-Model, Mamba, MC-LSTM, MTS-LSTM, ODE-LSTM, Transformer (encoder-only), XLSTM, plus the four forecast LSTMs.
- **ARLSTM:** a time-lagged copy of the target is the last `x_d` input. Missing AR values are imputed with lagged model output. It runs one step at a time, so it is "much slower than CudaLSTM".
- **Handoff-Forecast-LSTM:** a hindcast LSTM runs up to issue time. A `state_handoff_network` maps its (h, c) states to a separate forecast LSTM with its own weights, head, and embeddings. `hindcast_hidden_size` and `forecast_hidden_size` set the sizes. `forecast_overlap` lets the two run together for N steps, and `ForecastOverlapMSERegularization` is recommended.
- **Multihead-Forecast-LSTM:** a hindcast LSTM, then an FC layer predicts all lead steps at once (no rollout).
- **Sequential-Forecast-LSTM:** one LSTM rolls through hindcast then forecast, with a separate embedding for each period.
- **Stacked-Forecast-LSTM:** two stacked LSTMs. Example: 365-day spin-up plus 7-day forecast gives `seq_length` 372, `forecast_seq_length` 365, `forecast_overlap` 358. `bidirectional_stacked_forecast_lstm` fixes a lag issue.
- The model page documents no data assimilation.

**NeuralHydrology config arguments** ([NH config docs](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html)):
- **Forecast settings:**
  - `forecast_inputs` and `hindcast_inputs` must also appear in `dynamic_inputs`. The docs say "this does not currently support a forecast rollout", and it is not supported for multi-timescale models.
  - `forecast_seq_length` is part of `seq_length`, not added to it.
  - Other forecast options: `forecast_overlap`, `state_handoff_network`, `timestep_counter` (lead-time counter input), `forecast_hidden_size`/`hindcast_hidden_size`, `forecast_network`.
- **Autoregression and lags:**
  - `autoregressive_inputs` allows only one feature, which must also be in `lagged_features`. Missing data is handled with binary flags, and it works only with models that support AR.
  - `lagged_features` is a dict of feature to shift(s), creating `_shiftN` copies.
- **Multi-frequency:** `use_frequencies` takes pandas strings like `'1h'`/`'1D'`. `seq_length` and `predict_last_n` become per-frequency dicts. Also `no_loss_frequencies`.
- **NaN handling:** `nan_handling_method` ∈ {`masked_mean`, `input_replacing`, `attention`}, with `nan_sequence_probability` and `nan_step_probability` for training-time input dropout.
- **Datasets:** `camels_us`, `hourly_camels_us` (516 basins, `nldas_hourly`), `camels_gb/cl/br/aus`, `lamah_*`, `generic`.
- **Losses and heads:** losses MSE/NSE/RMSE/GMM/CMAL/UMAL; heads regression/gmm/cmal/umal. Probabilistic metrics CRPS/PICP/MPIW.

**Nearing et al. 2022, HESS 26, 5493–5513 (DA and autoregression)** ([HESS 2022](https://hess.copernicus.org/articles/26/5493/2022/)):
- Data: daily, 531 CAMELS basins, three forcings (Daymet, NLDAS, Maurer). Train 1999–2008; test 1989–1999.
- Methods:
  - Autoregression (AR): lagged observed Q as input; missing values replaced by model predictions, with an observed/simulated flag. Trained with 0–100% of lagged values withheld.
  - Variational DA: back-propagates to update cell states over a 5-step window.
  - AR lags tested: 1, 2, 4, 8, 10 days.
- Results:
  - Baseline simulation LSTM median NSE 0.796.
  - AR raised it to 0.879 (about 10%); DA gave about +8%.
  - AR beat DA on most metrics at a 1-day lag with no missing data.
  - AR can be worse than simulation when no observations are available at inference. DA falls back to simulation.
  - Flashy basins benefit less from lagged streamflow.
- Code: a NeuralHydrology fork with an "assimilation" branch, archived on Zenodo.

### Inferences
- For a 6-hourly baseline, the closest NH architecture to Google's v1 operational model is `handoff_forecast_lstm`. Simpler options are `sequential_forecast_lstm` or `multihead_forecast_lstm`.
- Observed flow up to issue time can go in as a hindcast-only dynamic input via `lagged_features`/`hindcast_inputs`. This works even without ARLSTM, because forecast models are not documented as supporting `autoregressive_inputs`. This needs testing.
- The "no forecast rollout" note probably means core NH indexes forecast inputs by valid time only, as one time series per variable. It cannot natively represent different NWP forecasts valid at the same time from different issue times. That matters for using real (not perfect) forecast precipitation. Google's fork was explicitly "extended to support forecast sequences" (see §3).
- The 2019 HESS repo for EA-LSTM (github.com/kratzert/ealstm_regional_modeling) is from my prior knowledge and was not re-verified here. NH supersedes it.

### Gaps
- The v1.13.0 release year is not printed on GitHub. The usage and dataset class for "AORC hourly forcing support" (PR #262) are not documented in the release notes. Check PR #262 and the dataset docs.
- I could not confirm whether NH's forecast models accept `autoregressive_inputs` directly. The docs say AR only works for models that support it, and only ARLSTM is documented as AR.
- The exact DA NSE value from Nearing et al. 2022 (Table 2) was not extracted.

---

## 2. Gauch et al. 2021 MTS-LSTM / sMTS-LSTM (hourly CAMELS)

### Takeaway
MTS-LSTM and sMTS-LSTM predict daily and hourly flow jointly. On 516 CAMELS basins with hourly NLDAS-2, they reach median hourly NSE of about 0.75 and daily NSE of about 0.76, beating NWM by about 0.19 (hourly). Trained models, predictions, and hourly forcing/streamflow data are all on Zenodo, which makes this the standard hourly LSTM reference point.

### Cited Findings
- Gauch et al. 2021, HESS 25, 2045–2062. Setup:
  - Basins: 516 CAMELS basins.
  - Forcings: hourly NLDAS-2 (11 variables) plus 27 static attributes.
  - Target: USGS instantaneous values aggregated to hourly and daily.
  - Periods: train 1 Oct 1990–30 Sep 2003; validation 2003–2008; test 1 Oct 2008–30 Sep 2018.
  - Sequences: the naive hourly LSTM uses 4320 h (180 d). sMTS-LSTM uses 365 daily steps then 336 hourly steps.
  - Training adds a daily–hourly consistency regularization. — [HESS 2021](https://hess.copernicus.org/articles/25/2045/2021/)
- Results:
  - sMTS-LSTM was best at both timescales. Its regularized daily median NSE was 0.762, and the MTS-LSTM hourly median NSE was 0.748.
  - NWM was worse by about 0.11–0.16 (daily) and about 0.19 (hourly).
  - Median peak-timing error was about 3.5 h for sMTS-LSTM vs over 6 h for NWM (hourly), and 0.3 d vs 0.5 d (daily).
  - Adding daily Daymet and Maurer forcings raised validation NSE from 0.766 to 0.811 (daily) and from 0.776 to 0.812 (hourly).
  - Compute: the multi-timescale models run inference about 40× faster than the naive hourly LSTM (13 min vs about 9 h for 10 years × 516 basins). — [HESS 2021](https://hess.copernicus.org/articles/25/2045/2021/)
- Data and code: NH (doi:10.5281/zenodo.4688003), analysis code (doi:10.5281/zenodo.4687991), trained models and predictions (doi:10.5281/zenodo.4071885), hourly NLDAS forcings and streamflow (doi:10.5281/zenodo.4072700). — [HESS 2021](https://hess.copernicus.org/articles/25/2045/2021/)
- Independent re-evaluation by Yang et al. (arXiv 2609.06794, Sept 2026), on CAMELS 2009–2018 using the publicly released simulations, with mean of individual realizations:

  | Model | Median hourly NSE | KGE | abs(FHV) % |
  |---|---|---|---|
  | MTS-LSTM | 0.726 | 0.723 | 16.8 |
  | MF-LSTM (Acuña Espinoza et al.) | 0.716 | 0.732 | 15.4 |
  | NWM3.0 | 0.545 | 0.684 | — |
  | LSTM-TRoute | 0.457 | 0.410 | — |

  — [Yang et al. 2026 arXiv](https://arxiv.org/abs/2609.06794)
- Yang et al. also report a lumped MTS-LSTM median peak APE of 59.4% and timing error of 5.5 h for flood peaks on CAMELS. MTS-LSTM simulations are on Zenodo record 18953458 and MF-LSTM on Zenodo record 14780059. — [Yang et al. 2026 arXiv](https://arxiv.org/pdf/2609.06794)

### Inferences
- MTS-LSTM is a simulation (no lagged Q, perfect forcing) model, so it is an upper-bound-forcing / lower-bound-information reference rather than a forecast baseline. It is still the most-cited hourly CAMELS number. CAMELS-H gauges in the Tennessee basin overlap the CAMELS set, so its released predictions could be pulled for overlapping basins, but only for 2008–2018, not 2020–2025.
- NH notes that `forecast_inputs` are not supported for multi-timescale models, so MTS-LSTM cannot be used directly as a forecast model with NWP inputs.

### Gaps
- Full Table 4 values for all model variants in Gauch et al. 2021 were not extracted.

---

## 3. Google's operational flood forecasting model (Nearing et al. 2024 Nature; v2 ME-LSTM; 2026 open-source framework; Flood Hub)

### Takeaway
- **v1 (Nature 2024):** a daily encoder-decoder (hindcast/forecast handoff) LSTM. It uses 365-day hindcast forcing and a 7-day HRES forecast, has no streamflow input, and has an asymmetric-Laplacian output.
- **v2 (operational, Nov 2024 onward):** a masked "mean-embedding" LSTM (ME-LSTM) trained on about 16,000 gauges, adding GraphCast forcing. Its 7-day accuracy is reported as similar to v1's 5-day.
- **Open release (June 3, 2026):** google-research/flood-forecasting ("OpenHydroNet", Apache-2.0). It is a NeuralHydrology fork with both models, gradient-based DA, and a `pretrained-models` folder.
- **Public data:** Flood Hub reforecasts are available only as the GRRR dataset (reforecasts 2016–2023, reanalysis 1980–2023) and the Nature 2024 Zenodo archive. The Flood Hub API gives real-time forecasts only. I found no route to archived Flood Hub forecasts for US gauges over 2020–2025, and the model is daily, not 6-hourly.

### Cited Findings
**v1: Nearing et al. 2024, Nature** ([PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)):
- Architecture: encoder (hindcast) LSTM over 365 days, decoder LSTM over a 7-day forecast, both with hidden size 256. A linear network transfers the cell state and a nonlinear FC network transfers the hidden state.
- Output: an asymmetric Laplacian per lead; the median is reported. Three models are averaged.
- Training: 50,000 minibatches of size 256, about 10 h on one V100.
- Inputs: daily, no streamflow. ECMWF IFS HRES and ERA5-Land (6 variables each), NOAA CPC precipitation, NASA IMERG Early, and HydroATLAS attributes.
- Data: about 5,680 GRDC gauges (the text also says 5,860).
- Evaluation: 4,089 gauges; reanalysis 1984–2021, reforecasts 2014–2021. Random 10-fold spatial CV plus temporal splits.
- Metrics: precision, recall, and F1 for 1/2/5/10-year return-period events (±2 days).
- Results: equal or better than GloFAS v4 at 0-day lead at about 60–70% of gauges. 5-day AI forecasts are equal to or better than GloFAS nowcasts for 1-, 2-, and 5-year events. No US-specific results are reported.
- Data and code:
  - Reanalysis, reforecasts, GloFAS benchmark, and trained models are on Zenodo, doi:10.5281/zenodo.10397664.
  - Input data is not redistributed.
  - Model code is in NeuralHydrology; analysis code is at google-research-datasets/global_streamflow_model_paper.

**v2 (Google Research blog, Nov 11, 2024)** ([Google blog 2024](https://research.google/blog/a-flood-forecasting-ai-model-trained-and-evaluated-globally/)):
- Adds GraphCast precipitation and temperature.
- Each weather product gets its own embedding network, and embeddings are combined according to availability.
- CMAL probabilistic head.
- Training gauges rose from 5,680 to 15,980, mostly from Caravan.
- 7-day lead accuracy is similar to v1's 5-day.
- NSE was evaluated over 3,079 basins (CDFs at 0, 4, and 7 days; no summary numbers given).
- GRRR covers reforecasts 2016–2023 and reanalysis 1980–2023.
- The API (real-time, 7-day lead) is waitlisted for expert users.

**EGU26 abstract (Loike, Nearing et al.)** ([EGU26-6973](https://meetingorganizer.copernicus.org/EGU26/EGU26-6973.html)):
- Describes a masked mean-embedding LSTM trained on about 16,000 gauges, with AI weather forecasts as forcing.
- The design is meant to avoid encoder-decoder state hand-off inconsistencies and to keep running through weather-data outages.
- Same 7-day vs 5-day claim.
- The peer-reviewed v2 paper was not located (the blog cites an EGUsphere preprint).

**Open-source framework blog (June 3, 2026, Nearing & Cohen)** ([Google blog 2026](https://research.google/blog/the-next-chapter-in-flood-resilience-open-sourcing-googles-hydrology-framework/)):
- Released a PyTorch package with v1 (2024 study) and v2 (ME-LSTM, powering Flood Hub), a training pipeline, and docs, under Apache-2.0.
- Daily predictions.
- v2 "extends reliable predictive horizon by six days in gauged basins and one day in ungauged basins" vs v1.
- Weights are not mentioned in the blog.

**Repo README** ([google-research/flood-forecasting](https://github.com/google-research/flood-forecasting/)):
- A "heavily modified fork of NeuralHydrology" with Mean-Embedding-Forecast-LSTM (production as of Dec 2025) and Handoff-Forecast-LSTM (former production).
- DA is optional at evaluate/infer time. It adjusts static and dynamic embeddings by gradient descent to match recent observed streamflow, without changing weights, and works with the Mean-Embedding model only (`--assimilate`, `assimilation_config`; window constraint `1 <= assimilation_window <= seq_length - lead_time`).
- Data: Caravan (Zarr) plus MultiMet forcings from `gs://caravan-multimet/v1.1`. Example datasets include CAMELS-US (531).
- Tools: delineation and static-extraction tools, plus a `pretrained-models` folder and a Colab fine-tuning tutorial.
- Docs: openhydronet.readthedocs.io. "Not an officially supported Google product."

**Missing-input methods behind ME-LSTM: Gauch et al. 2025, HESS 29, 6221–6235 ("How to deal with missing input data")** ([HESS 2025](https://hess.copernicus.org/articles/29/6221/2025/)):
- Compares input replacing, masked mean (embed each forcing product, then average the available ones), and attention.
- Masked mean is a special case of attention and is "practically equally accurate".

**Caravan MultiMet (Shalev & Kratzert, arXiv 2411.09459, Nov 2024)** ([arXiv](https://arxiv.org/abs/2411.09459)):
- Adds ECMWF IFS HRES, GraphCast, and CHIRPS-GEFS forecasts, plus CPC, IMERG v07 Early, and CHIRPS nowcasts, to Caravan (ERA5-Land).
- Hosted on Zenodo and GCP under CC-BY-4.0.

**Flood Hub API and GRRR** ([Google flood resources](https://sites.research.google/gr/floodforecasting/resources/)):
- API: real-time forecasts, mainly for non-commercial use, Google account required, CC BY 4.0.
- GRRR: 1980–2023 discharge, accessed through a Colab.
- Flood Hub riverine forecasts run up to 7 days ahead.

### Inferences
- The Google model is the strongest "operational" LSTM comparator. It is daily and runs to 7 days, so comparing it with a 6-hourly, 10-day GNN would need aggregation to daily and truncation to 7 days.
- GRRR reforecasts end in 2023 and the Nature Zenodo reforecasts end in 2021, so neither fully covers 2020–2025. A partial-overlap comparison for 2020–2023 at Tennessee gauges near Caravan / Flood Hub points may be possible.
- The most defensible use is to retrain OpenHydroNet's Handoff or ME-LSTM on the 128 Tennessee gauges with 6-hourly data. This needs the fork to accept sub-daily frequency (unverified). Alternatively, fine-tune the pretrained daily model. Its native support for issue-time-dependent forecast sequences and DA matches the paper's forecast mode better than core NeuralHydrology.

### Gaps
- Whether OpenHydroNet supports sub-daily (6-hourly) training: the README does not state the temporal resolution.
- What exactly is in `pretrained-models`: the weights' license and resolution are unknown.
- GRRR resolution, location count, and whether it includes US/Tennessee points: not stated on the resources page.
- No public archive of operational Flood Hub forecasts for US gauges over 2020–2025 was found.
- The v2 preprint's quantitative NSE/KGE numbers were not retrieved.

---

## 4. Other forecast / operational LSTMs (NWM post-processing, Hunt et al., NextGen LSTM, Feng et al. DI, MF-LSTM, Yang et al. 2026)

### Takeaway
- **Hunt et al. 2022** is the closest published analogue to the paper's setup: a 6-hourly, 10-day (40-step) LSTM forecast driven by ECMWF IFS, but only at 10 western-US gauges and single-gauge models.
- **NOAA-OWP / CIROH NextGen LSTM** is an hourly CudaLSTM BMI module with CAMELS-trained weights. It is the US operational-framework LSTM, but it is a simulation model with no observed-flow input, and it performs poorly when coupled with t-route (median hourly NSE 0.36–0.46).
- **Feng et al. 2020** "data integration" (lagged-Q) LSTM reported median daily NSE 0.86 on CAMELS.
- **Frame et al. 2021** NWM post-processing LSTMs reach median daily NSE of about 0.73–0.75 vs NWM 0.62.

### Cited Findings
**Hunt et al. 2022, HESS 26, 5449–5472** ([HESS 2022 Hunt](https://hess.copernicus.org/articles/26/5449/2022/)):
- Data: 10 western-US gauges. Target is observed 3-hourly streamflow, Jan 1990–Jun 2019. Inputs are 23 catchment-mean ERA5 variables plus GloFAS-ERA5 output.
- Forecast mode: daily forecasts using the ECMWF IFS **control** member (6-hourly, 0.1°) out to 10 days at 6-hourly steps (40 steps).
- Architecture: input is a 7-day sequence (28 six-hourly steps). Three stacked LSTM layers of 50 units; 100 random seeds per gauge, with the top 5 averaged.
- Testing (ERA5-forced): LSTM skilful at 9/10 gauges and highly skilful at 6, vs 7/4 for bias-corrected GloFAS.
- Operational period (IFS-forced): at 5-day lead, the LSTM beat bias-corrected GloFAS at all 10 stations. It had KGE comparable to persistence but higher NSE at most stations.
- Per-lead skill appears only in figures. Whether lagged observed Q is an input was unclear in the extracted text.
- The article text has an inconsistency on the operational dates (Sep 2020–Oct 2021 vs Sep 2019–Oct 2020 in a caption).

**Frame et al. 2021, JAWRA 57(6), 885–905 ("Post-processing the National Water Model with LSTM networks…")** ([NOAA repository](https://repository.library.noaa.gov/view/noaa/53162); [Google Research pub page](https://research.google/pubs/post-processing-the-national-water-model-with-long-short-term-memory-networks-for-streamflow-predictions-and-model-diagnostics/)):
- 531 CAMELS basins, daily.
- Median NSE 0.73 (LSTM_PP, post-processor), 0.75 (LSTM_PPA, post-processor plus atmospheric inputs), 0.74 (LSTM_A, atmospheric only), vs 0.62 for NWM.
- A model trained without NWM outputs performed comparably.
- These figures come from the abstract and metadata only, not the full text.

**NOAA-OWP/lstm and CIROH-UA/lstm (NextGen BMI LSTM)** ([CIROH-UA/lstm](https://github.com/CIROH-UA/lstm)):
- A BMI wrapper whose forward pass is "heavily based on" NeuralHydrology CudaLSTM. CIROH-UA/lstm is a fork of NOAA-OWP/lstm (`ngiab` branch).
- Hourly. The trained `.pt` weights come from "500+ CAMELS catchments". The recommended model uses 8 forcings and 26 static attributes, with sample NLDAS data.
- Observed streamflow "isn't necessary to run the model" (no DA/AR).
- The README reports no performance numbers.

**Yang, Lonzarich, Song, Ji, Pan, Lawson, Shen (arXiv 2609.06794, Sept 6, 2026)** ("Hourly U.S.-wide flood simulation…", [arXiv](https://arxiv.org/abs/2609.06794); [PDF](https://arxiv.org/pdf/2609.06794)):
- Setup: 2,831 CONUS gauges, AORC forcing on the NOAA Hydrofabric, test 2009–2018.
- **LSTM-TRoute** (NOAA-OWP/lstm BMI LSTMs run in NGIAB and coupled with t-route) is called the "operational LSTM baseline".
  - Median hourly NSE: 0.359 across CONUS gauges; 0.457 on CAMELS (KGE 0.410, abs(FHV) 37.9%).
  - The differentiable δHBV2.0MTS-MC captured 159% more ≥50-year floods than LSTM-TRoute.
- NWM3.0 median hourly NSE: 0.461 across CONUS (KGE 0.591).
- δHBV2.0MTS-MC: median hourly NSE 0.683 and KGE 0.719 across CONUS; NSE 0.710 on CAMELS. Median hourly NSE is 0.729 in humid basins.
- LSTM-TRoute simulations were generated with github.com/NOAA-OWP/lstm and NGIAB.
- This is a preprint, not yet peer-reviewed.

**Feng, Fang & Shen 2020, WRR ("Enhancing streamflow forecast and extracting insights using LSTM networks with data integration at continental scales")** ([arXiv 1912.08949](https://arxiv.org/pdf/1912.08949); [NSF PAR record](https://par.nsf.gov/biblio/10166420)):
- "Data integration" (DI) feeds lagged streamflow observations directly or through a CNN unit.
- Reached median daily NSE 0.86 on CAMELS.
- Nearing et al. 2022 note Feng's gain was about 19% from a 0.714 baseline. — [HESS 2022](https://hess.copernicus.org/articles/26/5493/2022/)

### Inferences
- **NextGen LSTM:** the most "operationally US" LSTM baseline, and it already runs on AORC at hourly resolution. It is a perfect-forcing simulation model with no observed-flow assimilation, so it is a weak but relevant floor. Its weights are CAMELS-trained, and running it over the Tennessee NHDPlus/Hydrofabric with t-route is possible via NGIAB.
- **Feng/Shen DI LSTM:** code is in the MHPI hydroDL GitHub (prior knowledge; not re-verified this session).
- **Hunt et al.:** its 6-hourly, 40-lead IFS-forced design is a near-exact template for a "forecast-mode LSTM" baseline. Its single-gauge models and its 10-gauge, one-year evaluation limit direct number comparison.

### Gaps
- Hunt et al. code repository URL was not extracted.
- NeuralHydrology's own forecast-tutorial skill numbers were not located.
- Lees et al. LSTM forecasting work was not covered. Lees et al. 2021 (CAMELS-GB) and 2022 (concept formation) are simulation papers, and I did not verify a Lees forecast paper.
- "Liu et al. DA LSTM" could not be identified unambiguously and was not verified.
- Frame et al. 2021 full text (periods, code) was not retrieved. The abstract-derived train 2004–2014 / eval 1994–2002 ordering needs checking.

---

## 5. Reported skill numbers for hourly/daily forecasts at 1-10 day leads (CONUS/CAMELS; Southeast/Tennessee)

### Takeaway
Published LSTM numbers on CONUS/CAMELS fall into three groups:
- **Simulation (perfect forcing, no observed Q):** median NSE about 0.72–0.80 daily and about 0.72–0.75 hourly.
- **With lagged observed Q (AR/DI):** median daily NSE about 0.86–0.88.
- **Operational hourly NextGen LSTM-TRoute:** about 0.36–0.46.

Lead-time-resolved NSE/KGE for real-NWP-forced LSTM forecasts at 1–10 days is scarce. Google reports only CDFs or event F1, and Hunt et al. report figures at 10 western gauges. I found no published LSTM forecast skill specific to the Tennessee River basin or the Southeast US.

### Cited Findings
| Model / study | Setting | Median NSE (other) | Source |
|---|---|---|---|
| Regional fine-tuned LSTM (Kratzert 2018) | daily, 241 CAMELS basins, Daymet | 0.72 (bench 0.71) | [HESS 2018](https://hess.copernicus.org/articles/22/6005/2018/) |
| EA-LSTM ensemble (Kratzert 2019) | daily, 531 CAMELS, Maurer | 0.74 (single 0.71) | [HESS 2019](https://hess.copernicus.org/articles/23/5089/2019/) |
| Simulation LSTM, 3 forcings (Nearing 2022) | daily, 531 CAMELS | 0.796 | [HESS 2022](https://hess.copernicus.org/articles/26/5493/2022/) |
| AR LSTM, 1-day lag Q (Nearing 2022) | daily, 531 CAMELS | 0.879 | [HESS 2022](https://hess.copernicus.org/articles/26/5493/2022/) |
| DI LSTM (Feng 2020) | daily, CAMELS | 0.86 | [arXiv 1912.08949](https://arxiv.org/pdf/1912.08949) |
| NWM post-proc LSTMs (Frame 2021) | daily, 531 CAMELS | 0.73–0.75 (NWM 0.62) | [NOAA repo](https://repository.library.noaa.gov/view/noaa/53162) |
| sMTS / MTS-LSTM (Gauch 2021) | hourly NLDAS, 516 CAMELS, 2008–2018 | daily 0.762; hourly 0.748; peak timing about 3.5 h | [HESS 2021](https://hess.copernicus.org/articles/25/2045/2021/) |
| MTS-LSTM / MF-LSTM re-evaluated (Yang 2026) | hourly, CAMELS 2009–2018 | 0.726 / 0.716 (KGE 0.723 / 0.732) | [arXiv 2609.06794](https://arxiv.org/pdf/2609.06794) |
| LSTM-TRoute (NextGen BMI LSTM + t-route) | hourly AORC, 2,831 CONUS / CAMELS | 0.359 / 0.457 | [arXiv 2609.06794](https://arxiv.org/pdf/2609.06794) |
| NWM3.0 | hourly, CONUS / CAMELS | 0.461 / 0.545 | [arXiv 2609.06794](https://arxiv.org/pdf/2609.06794) |
| Google v1 encoder-decoder LSTM | daily, 0–7 d leads, global | F1 at 5-day lead ≥ GloFAS 0-day (1/2/5-yr events) | [Nature 2024 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/) |
| Google v2 ME-LSTM | daily, 0–7 d, 3,079 basins | 7-day ≈ v1 5-day (CDF only) | [Google blog 2024](https://research.google/blog/a-flood-forecasting-ai-model-trained-and-evaluated-globally/) |
| Hunt et al. LSTM | 6-hourly, 1–10 d, IFS control, 10 western gauges | beat bias-corrected GloFAS at 10/10 at 5-day lead | [HESS 2022 Hunt](https://hess.copernicus.org/articles/26/5449/2022/) |

- Yang et al. 2026 report that median hourly NSE varies with aridity: 0.729 in humid basins (aridity index 0.2–0.8) vs 0.343 in the most arid. This is for their δHBV model, not an LSTM. — [arXiv 2609.06794](https://arxiv.org/pdf/2609.06794)
- Kratzert et al. 2018's 241 basins include the South Atlantic-Gulf hydrologic unit, which is adjacent to but not the Tennessee region (HUC 06). — [HESS 2018](https://hess.copernicus.org/articles/22/6005/2018/)

### Inferences
- The Tennessee basin is humid and rain-dominated, so CAMELS-trained LSTMs probably score above the CONUS median there. Many TVA-gauge flows are reservoir-regulated, though, and CAMELS LSTMs were trained on minimally disturbed basins. Regulated TVA gauges are therefore likely to score lower without operations data, and this is the key fairness caveat.
- Comparisons in the paper should state which information regime each baseline uses: perfect forcing vs NWP forcing, and with vs without observed Q. The NSE gap between those regimes (about 0.08–0.1 median daily) is as large as typical architecture differences.

### Gaps
- No peer-reviewed LSTM forecast skill numbers specific to the Tennessee River basin, TVA, or the Southeast US were found (one broad search).
- No study reporting NSE/KGE vs lead time (1–10 d) for NWP-forced LSTMs over CAMELS-US at hourly or 6-hourly resolution was found.

---

## 6. Practical recipe: a fair NeuralHydrology (or OpenHydroNet) LSTM baseline on the 128 Tennessee gauges (34 TVA + 94 CAMELS-H), AORC, 6-hourly, with observed-Q history and forecast precipitation

### Takeaway
Use NH (≥ v1.12/1.13) with the `generic` dataset or the new AORC support. Train a `handoff_forecast_lstm` (Google v1-style) and/or `sequential_forecast_lstm`:
- **Hindcast inputs:** 6-hourly AORC history plus lagged observed Q, with missing-value handling.
- **Forecast inputs:** precipitation from the same NWP product the GNN uses, plus `timestep_counter`, with `forecast_seq_length` = 40.
- **Loss and splits:** NSE loss (or CMAL); train 1979–2018, validation 2019, test 2020–2025; seed ensemble.

The main technical hurdle is that core NH indexes forecast inputs by valid time ("no forecast rollout"). Real issue-time-dependent NWP forecasts are better handled with Google's OpenHydroNet fork, or by a lead-specific data layout.

### Cited Findings (building blocks)
- Forecast models and their config keys (`hindcast_inputs`, `forecast_inputs`, `forecast_seq_length`, `forecast_overlap`, `state_handoff_network`, `timestep_counter`, `hindcast_hidden_size`/`forecast_hidden_size`) are described in §1. — [NH models](https://neuralhydrology.readthedocs.io/en/latest/usage/models.html); [NH config](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html)
- Lagged observed Q: `lagged_features` creates `_shiftN` copies. `autoregressive_inputs` (one feature, with missing-value flags) works only for AR-capable models such as ARLSTM. — [NH config](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html); [NH models](https://neuralhydrology.readthedocs.io/en/latest/usage/models.html)
- Missing inputs and gauge gaps: `nan_handling_method: masked_mean` with `nan_step_probability`/`nan_sequence_probability`. — [NH config](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html); method paper [Gauch et al. 2025 HESS](https://hess.copernicus.org/articles/29/6221/2025/)
- Training with randomly withheld lagged observations makes AR models robust to missing data. — [Nearing et al. 2022](https://hess.copernicus.org/articles/26/5493/2022/)
- AORC hourly forcing support exists in NH v1.13.0 (PR #262). — [NH v1.13.0](https://github.com/neuralhydrology/neuralhydrology/releases/tag/v1.13.0)
- AORC is on the AWS Open Data Registry (registry.opendata.aws/noaa-nws-aorc). — [Yang et al. 2026](https://arxiv.org/pdf/2609.06794)
- Losses: NSE (basin-normalized), CMAL head for probabilistic output (Google v2 uses CMAL). Metrics include CRPS. — [NH config](https://neuralhydrology.readthedocs.io/en/latest/usage/config.html); [Google blog 2024](https://research.google/blog/a-flood-forecasting-ai-model-trained-and-evaluated-globally/)
- Google v1 hyperparameters for reference: 365-day hindcast, hidden 256, batch 256, 50k updates, 3-model ensemble. — [Nature 2024 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10954541/)
- Hunt et al. used 7-day (28 × 6-h) input windows and 40 × 6-h lead steps with IFS forecast forcing. — [Hunt et al. 2022](https://hess.copernicus.org/articles/26/5449/2022/)
- Example stacked-forecast arithmetic: `seq_length` = hindcast + forecast; `forecast_seq_length` = overlap + forecast. — [NH models](https://neuralhydrology.readthedocs.io/en/latest/usage/models.html)
- OpenHydroNet: train/evaluate/infer CLI, `--assimilate` DA, YAML configs, and an extension "to support forecast sequences". — [google-research/flood-forecasting](https://github.com/google-research/flood-forecasting/)

### Inferences (proposed recipe; untested)
1. **Data.** For each of the 128 gauges:
   - Delineate the upstream area using NHDPlus v2 catchments (or OpenHydroNet's `delineate-catchment`).
   - Area-average hourly AORC (8 variables), then aggregate to 6-hourly means (sum for precipitation).
   - Aggregate USGS/TVA observed flow to 6-hourly means, the same target the GNN uses.
   - Attach static attributes: CAMELS-style for CAMELS-H basins; for TVA basins, derive the same set (e.g., HydroATLAS via `extract-caravan-static`).
   - Write NH `generic` netCDF files per basin with a 6-hourly DatetimeIndex.
2. **Observed-Q history.** Add `qobs_lag = qobs.shift(k)`, plus a 0/1 availability flag, as hindcast-only dynamic inputs. With k = 1 step, the last value is at issue time. At inference, set lagged Q to NaN in the forecast window so that no future flow leaks in. Train with random withholding, as in Nearing 2022.
3. **Forecast precipitation.** Two variants:
   - (a) A perfect-forcing upper bound: AORC as `forecast_inputs`.
   - (b) True forecast mode: the same NWP precipitation as the GNN (e.g., GEFS/GFS or HRRR, regridded to basins).

   Core NH cannot hold several issue times per valid time. Workarounds:
   - Train on AORC as forecast input and substitute NWP at test time (common, but it creates a train/test forcing mismatch).
   - Build lead-specific columns per sample.
   - Use OpenHydroNet's forecast-sequence data layout (preferred if 6-hourly works).
4. **Model.**
   - Main: `model: handoff_forecast_lstm`, `seq_length` ≈ 1460 + 40 (365 d hindcast plus 10 d forecast at 6 h), `forecast_seq_length: 40`, `forecast_overlap: 0` (or a small overlap with `ForecastOverlapMSERegularization`), `timestep_counter: true`, `hidden_size` 128–256, `loss: NSE` (or `head: cmal` with `loss: CMALLoss`), `predict_last_n` = 40.
   - Ablations: `sequential_forecast_lstm` and a plain `cudalstm` simulation model (no Q, perfect forcing) as a floor.
5. **Splits.** Train 1979-10 to 2018 (AORC starts 1979), validation 2019, test 2020–2025. Train one regional model on all 128 gauges and ensemble 3–8 seeds, as Kratzert 2019 and Nearing 2024 did.
6. **Evaluation.** Report metrics per lead (1–40 × 6 h): NSE, KGE, peak timing, FHV, and event-based F1. Compute them at the same gauge-issue times as the GNN, and give persistence and NWM as reference points. Stratify regulated TVA gauges vs CAMELS-H gauges.
7. **Other reproducible references** to include alongside:
   - NextGen BMI LSTM (NOAA-OWP/lstm) driven by AORC: an operational simulation floor.
   - Google OpenHydroNet ME-LSTM with `--assimilate` (DA), retrained or fine-tuned on the 128 gauges.
   - If feasible, a daily-aggregated comparison against GRRR reforecasts for 2020–2023.

### Gaps
- Not verified:
  - That NH's `generic` dataset and forecast models run cleanly at a 6-hourly frequency string (`'6h'`).
  - How NH v1.13's AORC hourly loader is configured.
  - Whether OpenHydroNet supports sub-daily data and issue-time-indexed forecast inputs. Its README does not document resolution.
- No published example of an NH forecast-LSTM trained with lagged observed Q plus real NWP forcing at sub-daily resolution was found. This baseline would be novel in configuration and should be described in full in the paper.
