# Public operational streamflow forecast systems and archives usable as benchmarks (Tennessee River basin, 2020-2025)

Notes compiled 2026-10-12. Some facts below come from listing the public buckets directly with anonymous HTTP requests on 2026-10-12. These are marked "direct listing" and cite the bucket endpoint. Everything else cites a web source.

## 1. NOAA National Water Model (NWM) operational forecasts: versions, configurations, archives, ID mapping

### Takeaway
The NWM is the only public operational forecast archive that is native to NHDPlus v2 COMIDs (feature_id = COMID in CONUS). The archive is effectively complete for 2020-2025 in the Google Cloud bucket `gs://national-water-model` (NetCDF, one file per cycle and lead hour, CONUS-wide). There are three versions in the test period: v2.0 until 2021-04-20, v2.1 until 2023-09-20, and v3.0 after that. These change the time step, the ensemble size and the forcing, so results should be stratified by version.

### Cited Findings
**Versions in operation 2020-2025**
- NWM v2.0 (2019) added "seven-member medium-range CONUS ensemble, use of RFC MPE precipitation, improved calibration, increased reservoir representation" — [Cosgrove et al. 2024, JAWRA, Table 5](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf)
- NWM v2.1 took effect "on or about April 20, 2021, beginning with the 1200 UTC run". It added Puerto Rico/USVI, the Great Lakes basins and "an enhanced treatment of reservoirs" — [NWS SCN 20-119 (v2.1)](https://www.weather.gov/media/notification/pdf2/scn20-119nwm_v2.1_aad.pdf)
- v2.1 is described as the "fifth major upgrade" and "more than doubles simulation skill with respect to hourly streamflow correlation, Nash Sutcliffe Efficiency, and bias reduction" relative to 2016 — [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536)
- NWM v3.0 took effect "on or about September 20, 2023, beginning with the 1200 UTC run". It added the "National Blend of Models (NBM) as a forcing source for NWM CONUS medium-range forecasts", "Ingest of RFC-supplied reservoir outflow forecasts at 77 additional locations, bringing the total ... to 392", a "new surface runoff scheme (Xinanjiang)", and hydrofabric fixes (CONUS: "Removal of four stream segments (Great Lakes region)" and "12 Stream segment topology and attribute fixes") — [NWS SCN 23-76 (v3.0)](https://www.weather.gov/media/notification/pdf_2023_24/scn23-76_national_water_model_v3.0_aab.pdf)
- v3.0 changed `feature_id` to int64, and "channel_rt" post-processed files now use NETCDF4 — [SCN 23-76](https://www.weather.gov/media/notification/pdf_2023_24/scn23-76_national_water_model_v3.0_aab.pdf)
- No separate operational "v2.2" upgrade notice was found. However, HydroShare RouteLink resources list RouteLink versions "2.1.6, 2.2.3, and 3.0.7", so a v2.2 parameter/hydrofabric build existed, likely for retrospective or testing use — [Regina, NWM RouteLinks CSV (HydroShare)](https://beta.hydroshare.org/resource/e9fe66730d184bdfbaea19639bd7cb55)

**Forecast configurations (CONUS)**
- Configuration table for v2.1: Short Range (SR) "18 h forecast, 24x day, HRRR, RAP". Medium Range (MR) "240 h forecast, 4x day, GFS, 7 ensemble members". Long Range (LR) "30 day forecast, 4x day, CFS, 4 ensemble members". Analysis and Assimilation (AnA) has a 3 h lookback, runs 24x/day with HRRR/RAP/MRMS/MPE and "Features data assimilation". Extended AnA has a 28 h lookback and runs once a day — [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf)
- MR ensemble design: "Member 1 ... forced with model output from the most recent GFS ... extends out to 10 days ... members 2–7 ingest time-lagged forcing data from successively older forecast cycles of the GFS. These six members extend to 8.5 days, creating an overall seven member equal-weighted medium range ensemble forecast with an 8.5 day forecast horizon." LR: "four members which are each forced with a different Climate Forecast System (CFS) forecast member" — [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf)
- The v3.0 medium-range blend (NBM-forced, deterministic) files are `nwm.tCCz.medium_range_blend.channel_rt.f###.conus.nc`, "Where CC is cycle (00, 06, 12, 18) and ### is 001-240". Blend forcing is in `forcing_medium_range_blend/` — [SCN 23-76](https://www.weather.gov/media/notification/pdf_2023_24/scn23-76_national_water_model_v3.0_aab.pdf)
- Markert et al. (2024) describe the medium-range ensemble as "10 days (member 1) and 8.5 days (members 2–6) with a 3-h time step" and the long range as a "30-day four-member ensemble forecast on a 6-h time step" — [Markert et al. 2024, Env. Modelling & Software 179:106123](https://repository.library.noaa.gov/view/noaa/66525/noaa_66525_DS1.pdf). The direct listing below shows the channel time step is actually hourly from v2.1 onward. The 3-h step applied to v2.0 channel output and to land output.
- Direct listing of `gs://national-water-model`, CONUS `channel_rt` files — [GCS JSON API listing](https://storage.googleapis.com/storage/v1/b/national-water-model/o?delimiter=/&prefix=nwm.20240601/&fields=prefixes):
  - **2020-06-01 (v2.0):** `medium_range_mem1` has 320 files/day = 4 cycles x 80 lead times, f003...f240, i.e. a **3-hourly** step. mem2 runs to f204 (8.5 d). Seven member folders (mem1-mem7). About 5.0 MB per file.
  - **2021-05-01 onward (v2.1):** mem1 has 960 files/day = 4 cycles x 240, f001-f240, i.e. **hourly**. mem2-7 run f001-f204. About 14.3 MB per file. Seven members. A `medium_range_no_da` folder is present in 2022.
  - **2023-10-01 onward (v3.0):** **six** GFS members (mem1-mem6; mem7 no longer present), plus `medium_range_blend` (960 files/day, f001-f240, hourly, ~14.4 MB/file) and `medium_range_no_da`.
  - **Long range (2024-06-01):** `long_range_mem1` has 480 files/day = 4 cycles x f006-f720 (6-hourly), ~4.8 MB/file.
- File content (v3.0 blend `channel_rt` file, direct inspection): dimension `feature_id = 2776734`. The attribute reads `feature_id:comment = "NHDPlusv2 ComIDs within CONUS, arbitrary Reach IDs outside of CONUS"`. Variables are `streamflow` (m3 s-1, int with scale_factor 0.01), `nudge`, `velocity`, `qSfcLatRunoff`, `qBucket`, plus `time` and `reference_time` (minutes since 1970) — [sample file](https://storage.googleapis.com/national-water-model/nwm.20240601/medium_range_blend/nwm.t00z.medium_range_blend.channel_rt.f024.conus.nc)

**Where the 2020-2025 archive lives**
- **Google Cloud (most complete).** The bucket holds operational forecasts "since 2018 ... continuously updated daily" and "archives of both operational forcing files and forecast files dating back to 2018". Markert et al. call it "perhaps the most comprehensive and consistently updated data repository" — [Markert et al. 2024](https://repository.library.noaa.gov/view/noaa/66525/noaa_66525_DS1.pdf); [GCP Marketplace listing](https://console.cloud.google.com/marketplace/details/noaa-public/national-water-model)
- Direct listing: `gs://national-water-model` has daily folders `nwm.YYYYMMDD/` from **2018-09-17 to 2026-10-12**: 2,948 folders and **zero missing days** — [GCS listing](https://storage.googleapis.com/storage/v1/b/national-water-model/o?delimiter=/&prefix=nwm.2020&fields=prefixes)
- File-level gaps do exist. In a sample of the 1st of each month for 2020-2025, every date had the full 320 (v2.0) or 960 (v2.1+) mem1 `channel_rt` files, except **2020-04-01, which had only 246 of 320**. Blend files were complete (960) on every sampled date from 2023-10-01. The sample covered one day per month, not an exhaustive audit — direct listing ([GCS JSON API](https://storage.googleapis.com/storage/v1/b/national-water-model/o?prefix=nwm.20200401/medium_range_mem1/))
- **BigQuery and Parquet derivatives.** The public dataset "BigQuery-public-data.national_water_model" has tables named by configuration and output, e.g. "short_range_channel_rt", in long format "on the order of trillions of rows", with ensemble members in the same table. Parquet copies are at `gs://national-water-model-parq`, and there is a REST API at https://nwm-api.ciroh.org/docs/ — [Markert et al. 2024](https://repository.library.noaa.gov/view/noaa/66525/noaa_66525_DS1.pdf). The paper does not give the date coverage of the BigQuery tables (see Gaps).
- **NOMADS.** It "only provides forecast data for the most recent two days" — [Markert et al. 2024](https://repository.library.noaa.gov/view/noaa/66525/noaa_66525_DS1.pdf). A Scientific Data paper says outputs are on NOMADS for "a 48-hour rolling window" and have been archived on GCP since v2.0 — [Sci. Data 2023 (NWM retrospective restructuring)](https://www.nature.com/articles/s41597-023-02316-7)
- **AWS `noaa-nwm-pds`.** The registry describes "A rolling four week archive of NWM data" — [AWS Registry](https://registry.opendata.aws/noaa-nwm-pds/). **This conflicts with the direct listing on 2026-10-12**, which shows 650 daily folders from `nwm.20250101/` to `nwm.20261012/` — [S3 listing](https://noaa-nwm-pds.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=1000). Either way, AWS does not cover 2020-2024. The bucket also has kerchunk references in `noaa-nodd-kerchunk-pds` ("Cloud-optimized zarr reference files") — [AWS Registry](https://registry.opendata.aws/noaa-nwm-pds/)
- **Microsoft Azure / Planetary Computer.** Stores "the operational products starting in 2018 ... in the raw NetCDF file format" — [Markert et al. 2024](https://repository.library.noaa.gov/view/noaa/66525/noaa_66525_DS1.pdf); [Planetary Computer noaa-nwm](https://planetarycomputer.microsoft.com/dataset/storage/noaa-nwm)
- **Older archives.** A USGS data release covers v1.2 operational runs 2017-05-01 to 2018-05-01 at selected gauges, irrelevant for 2020-2025 — [USGS data release](https://www.usgs.gov/data/national-water-model-v12-retrospective-and-operational-model-run-archive-selected-nwis-gage)
- Each cloud NetCDF "represents one time step of one forecast across all 2.7 million streams", so single-reach extraction requires reading whole files. "Each cloud provider stores the data in differing directory structures" — [Markert et al. 2024](https://repository.library.noaa.gov/view/noaa/66525/noaa_66525_DS1.pdf)

**ID mapping (feature_id to COMID to USGS gauge)**
- In CONUS, `feature_id` is the NHDPlusV2 COMID (file attribute, direct inspection above). The NWM uses "the 1:100,000 scale flowlines, catchments, waterbodies, gauge associations ... of the Medium Resolution NHDPlus" and NHDPlus v2.1 "underpins the vast majority of the features" — [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf). NOAA map services crosswalk "NWM feature_id with the NHDPlus v2.1 ComID" — [search summary of NOAA NWC services](https://www.weather.gov/media/owp/operations/Public_Handbook_NWC_Visualization_Services.pdf)
- The RouteLink file (NetCDF) carries the gage association per link. CSV versions for NWM 2.1.6, 2.2.3, 3.0.7 (and 3.0.17) are on HydroShare, with build scripts at github.com/jarq6c/NWM_RouteLinks — [HydroShare RouteLinks](https://beta.hydroshare.org/resource/e9fe66730d184bdfbaea19639bd7cb55); [UCAR streamflowEval](https://ral.ucar.edu/sites/default/files/public/streamflowEval.html)
- The USGS GageLoc dataset indexes ~26,000 gauges to NHDPlus V2.1 COMIDs. It warns that network locations "differ from stations' latitude/longitude provided in NWIS" — [ScienceBase GageLoc](https://sciencebase.gov/catalog/item/5c86a747e4b09388244b3da1)
- Known crosswalk error example: gauge 07010040 is associated with the wrong NWM reach (hydrotools issue #206) — [GitHub NOAA-OWP/hydrotools #206](https://github.com/NOAA-OWP/hydrotools/issues/206)
- Data assimilation: AnA "features data assimilation" of USGS streamflow, which nudges forecast initial conditions — [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf). The bucket also holds `usgs_timeslices/` and `analysis_assim_no_da`/`medium_range_no_da` (v2.1+), per the direct listing ([GCS](https://storage.googleapis.com/storage/v1/b/national-water-model/o?delimiter=/&prefix=nwm.20240601/&fields=prefixes)).

**Tools**
- The BYU/CIROH NWM API, the BigQuery tables, and benchmark notebooks at github.com/BYU-Hydroinformatics/api-nwm-gcp — [Markert et al. 2024](https://repository.library.noaa.gov/view/noaa/66525/noaa_66525_DS1.pdf)
- TEEHR (Tools for Exploratory Evaluation in Hydrologic Research) is CIROH's open-source evaluation framework for "joining timeseries, built-in metrics, event detection ..." aimed at NWM evaluation — [CIROH](https://ciroh.ua.edu/?p=1429); [CIROH evaluation system project](https://ciroh.ua.edu/research-project/expansion-of-the-ciroh-integrated-evaluation-system-to-accelerate-research-to-operations/)
- `fcast` (Dewberry) reads short-range forecasts from GCS via `gcsfs.GCSFileSystem(project='national-water-model')` — [fcast docs](https://dewberry.github.io/fcast/ShortRange/)

### Inferences
- **Recommended NWM benchmarks for 1-10 day leads:**
  - Medium-range **mem1** (deterministic, GFS; 10 d) for the whole of 2020-2025.
  - **Ensemble mean** of mem1-7 (v2.x) or mem1-6 (v3.0), which only reaches 8.5 d (204 h).
  - From 2023-09-20, the **medium_range_blend** (NBM-forced deterministic, 10 d).
  - Hourly (v2.1+) or 3-hourly (v2.0) streamflow can be averaged to 6-hourly means. For v2.0 (Jan 2020 to Apr 2021), 6-h means come from only two 3-h instantaneous values.
- **Version breaks:** v2.0 to v2.1 on 2021-04-20 and v2.1 to v3.0 on 2023-09-20. Ensemble size changes 7 to 6, and the time step changes 3 h to 1 h. Scores should be reported per version period or with the breaks flagged.
- **Volume:** one mem1 channel_rt file is about 14.5 MB (v2.1+). All cycles and leads come to about 14 GB/day per member, or roughly 30 TB for 2020-2025 per member. Taking only the 00Z cycle at 6-hourly leads (40 files) is about 0.6 GB/day, roughly 1.3 TB for 6 years. Byte-range reads via kerchunk/HTTP or the BigQuery/Parquet route avoid full downloads.
- Because the model's targets are NHDPlus v2 COMIDs, the NWM is a like-for-like benchmark at the exact same reaches. Gauge-to-COMID links should still be spot-checked (GageLoc vs RouteLink vs CAMELS-H metadata).
- **Regulation caveat:** NWM forecasts at Tennessee mainstem and tributary reaches below TVA dams depend on how the NWM handles reservoirs (level-pool or persistence vs RFC-supplied outflow forecasts). Fair comparisons should separate unregulated headwater gauges from regulated reaches. Whether any TVA reservoirs are among the 392 RFC-forecast reservoirs in v3.0 was not verified.

### Gaps
- No complete file-level audit of 2020-2025 completeness was done (only 1st-of-month sampling). One partial day (2020-04-01) was found.
- The date coverage of the BigQuery `national_water_model` tables (whether medium_range back to 2020 is loaded) was not confirmed.
- No official OWP statement explains why the AWS `noaa-nwm-pds` retention changed from "four-week rolling" to multi-year (listing shows from 2025-01-01).
- No source confirmed whether v3.0 reduced the GFS MR ensemble to 6 members by design. The 6-member count is from the bucket listing, and Markert et al. also say "members 2–6".
- No primary documentation for a "v2.2" operational upgrade was found.

## 2. NWM retrospective v2.1 and v3.0 (simulation baselines, not forecasts)

### Takeaway
The v2.1 retrospective (Feb 1979 to Dec 2020) and v3.0 retrospective (Feb 1979 to Jan 2023) are AORC-forced open-loop simulations on AWS (NetCDF and Zarr). They cover the test period only partially: v2.1 covers 2020 and v3.0 covers 2020 to Jan 2023. They act as a "perfect-forcing" upper-bound baseline, not a forecast benchmark.

### Cited Findings
- v3.0: "Feb 1979 to Jan 2023", bucket `noaa-nwm-retrospective-3-0-pds`, NetCDF plus Zarr. v2.1: Feb 1979 to Dec 2020, `noaa-nwm-retrospective-2-1-pds` (NetCDF) and `noaa-nwm-retrospective-2-1-zarr-pds` (Zarr). v2.0: Jan 1993 to Dec 2018. v1.2: 1993-2017 in `nwm-archive` — [AWS Registry nwm-archive](https://registry.opendata.aws/nwm-archive/)
- Forcing: v2.1 used AORC v1.0 for 1979-2006 and AORC v1.1 for 2007-2020, and v3.0 used AORC v1.1 throughout. The v3.0 forcing files carry a metadata tag of "v2.1" although "the files are in fact v3.0 forcing files" — [AWS Registry](https://registry.opendata.aws/nwm-archive/)
- Retrospective skill, 2013-2016 validation, AORC forcing, no streamflow DA (v2.1):
  - **Bias:** 52% of all USGS sites (~10,000) and 68% of GAGES-II reference sites (~1,600) are within ±20% bias.
  - **Correlation:** 52% of GAGES-II reaches have hourly correlation ≥0.8 (vs 18% in v1.0).
  - **Spatial pattern:** "wet biases more prevalent in the middle of the CONUS."
  - Source: [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf)
- A Texas study of storm events (610 USGS gauges) found NWM 3.0 "underestimates streamflow, struggles with peak timing", with mean NNSE below 0.5 for both v2.1 and v3.0. This is from a search-result summary; the full text was not accessed — [J. Hydrology: Regional Studies 2025](https://www.sciencedirect.com/science/article/pii/S2214581825000205)
- The retrospective has been restructured for web access — [Sci. Data 2023](https://www.nature.com/articles/s41597-023-02316-7); NWM reanalysis copies are also at RENCI — [HydroShare RENCI](https://www.hydroshare.org/resource/89b0952512dd4b378dc5be8d2093310f/)

### Inferences
- In a forecast paper, the retrospective is best used as (a) an "observed-forcing simulation" reference showing how much NWM error comes from the hydrologic model versus the forcing, and (b) a pre-2020 training or climatology series. It cannot score the 2023-2025 part of the test period (v3.0 ends Jan 2023).

### Gaps
- No Tennessee-specific or Southeast-specific retrospective statistics were located. Cosgrove et al. present CONUS maps and histograms only.

## 3. Published evaluations of NWM forecast skill vs USGS observations

### Takeaway
Published operational-forecast verification is thin and mostly short-period or regional. No peer-reviewed study was found that evaluates NWM medium-range forecasts in the Tennessee basin or Southeast for 2020-2025. A new evaluation in this paper would fill a real gap.

### Cited Findings
- **Real-time operational verification, v2.0 vs v2.1 (1 Jul to 9 Aug 2020):**
  - **Short range:** the "sizable overestimation of peak discharge in v2.0 ... is greatly reduced in v2.1", and forecasts remain "too quick to peak".
  - **Medium range, member 1, ~3,000 sites:** "Peak bias and peak timing errors improve during the first 6 h, but change little in the hours that follow". FAR decreased and CSI increased. Days 4-10 show similar results "with the medium-range ensemble means exhibiting higher scores".
  - Source: [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf)
- "Unlike the historical runs which are forced with observed precipitation, real-time NWM forecasts are forced with NWP-based precipitation ... input precipitation now an increased source of error" — [Cosgrove et al. 2024](https://repository.library.noaa.gov/view/noaa/66536/noaa_66536_DS1.pdf)
- An early NWS briefing reported medium-range (0-10 d) results "by RFC" showing a "positive forecast bias in the Southeast River Forecast Center (SERFC)", linked to a high bias in that region's retrospective. This was preliminary and early-version — [NWS NWM briefing PDF](https://www.weather.gov/media/wrn/calendar/NationalWaterModel.pdf)
- Summer 2016 evaluation at more than 6,000 USGS gauges separated forcing bias (HRRR/GFS) from model bias — [AGU 2016 abstract](https://ui.adsabs.harvard.edu/abs/2016AGUFM.H43H1566R/abstract)
- A 2025 J. Hydrometeorology study ran an NWM-based ~7-day ensemble with post-processed precipitation in Northern California. It found streamflow "generally skillful out to day 4 (7) for precipitation and streamflow" and that post-processing improved spread and accuracy — [JHM 26(4) 2025](https://journals.ametsoc.org/view/journals/hydr/26/4/JHM-D-24-0111.1.xml)
- CBRFC published a comparison page, "National Water Model Medium Range and RFC 10 Day Forecast Performance" (2018) — [CBRFC](https://www.cbrfc.noaa.gov/outgoing/nwm/2018_10/ZSWA3H_H_nwm/forecastlocation.html)

### Inferences
- The lack of Southeast/Tennessee NWM medium-range verification for 2020-2025 means the authors can present the NWM benchmark comparison as a contribution in its own right. The early SERFC positive-bias finding suggests checking NWM bias in the region.

### Gaps
- No OWP public verification dashboard or report with 2020-2025 statistics was located.
- No CIROH or TEEHR published medium-range study for the Tennessee basin was found.
- The Texas v2.1 vs v3.0 paper and the JHM 2025 paper were read only through search summaries (full text not accessed).

## 4. NWS River Forecast Center (LMRFC/OHRFC) forecasts and archives

### Takeaway
There is no official public RFC forecast archive, but the Iowa Environmental Mesonet (IEM) keeps a searchable, scriptable archive of NWS text products. That includes LMRFC RVF river forecast products in SHEF for 2020-2025, mostly stage (HG) at RFC forecast points, with 6-hourly values out to ~5 days, often with 24-h QPF only. Using them is possible but limited: they are stage-based, sparse in location, and TVA-coordinated in the Tennessee basin. That may conflict with the authors' TVA restriction.

### Cited Findings
- **IEM API.** The IEM AFOS archive gives access by PIL (e.g., RVFORN), with an API at `/api/1/nwstext/...` and bulk retrieval at `/cgi-bin/afos/retrieve.py?help` — [IEM RVFORN page](https://mesonet.agron.iastate.edu/wx/afos/p.php?pil=RVFORN)
- **Example retrieval, RVFOHX on 2021-01-03 (direct retrieval):**
  - The header is "NWS Lower Mississippi River Forecast Center, Slidell, LA" with "FORECAST GROUP IS MAINSTEM_TN".
  - The text says "FORECASTS INCLUDE 24-HOUR FUTURE RAINFALL IN 6 HOUR INCREMENTS".
  - The point shown is Iron City, TN, on Shoal Creek, with stage forecast `.ER IRCT1 ... HGIFF/DIH6` running 6-hourly to day 5, followed by QPF lines (PPQFZ).
  - Source: [IEM retrieve.py](https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=RVFOHX&sdate=2021-01-01&edate=2021-01-04&limit=1&fmt=text)
- **Example retrieval, RVFORN on 2020-03-01 (direct retrieval):** a Delft-FEWS SHEF export with observed flows (QRIP) and 5-day forecast flows (QRIFF) for Red River points — [IEM retrieve.py](https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=RVFORN&sdate=2020-03-01&edate=2020-03-02&limit=2&fmt=text)
- **Archive volume, 2020-01-01 to 2026-01-01 (direct counts):** RVFMRX (Morristown WFO area) 9,577 products, RVFHUN (Huntsville) 4,948, RVFORN 2,120 — [IEM retrieve.py](https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=RVFMRX&sdate=2020-01-01&edate=2026-01-01&limit=9999&fmt=text)
- NWS Directive 10-913 documents RVF formats ("Example #1 - In Standard Hydrologic Exchange Format (SHEF) with Crest Forecast") — [NWS PD 10-9-13](https://www.weather.gov/directives/sym/pd01009013curr.pdf)
- The RFC Archive Database (RAX) is internal NWS infrastructure. Its documentation page gives no public access path — [OWP RAX docs](https://www.weather.gov/owp/oh_rfcdev_archive_datadoc). Legacy NWSRFS verification ran "on archive machines at the river forecast centers" — [NWS Verification intro](https://www.weather.gov/media/owp/oh/hrl/docs/VerificationIntro.pdf)

### Inferences
- RFC forecasts are public domain (US Government work), and IEM archives are openly accessible. However, in the Tennessee basin LMRFC/OHRFC forecasts for regulated points depend on TVA release schedules, and TVA operates its own forecasting for the basin. The authors should check whether RFC products there are effectively TVA-derived before using them. Stage-to-flow conversion would need USGS rating curves.
- RVF products cover only official forecast points (tens of points in the basin), not arbitrary COMIDs. At best they are a secondary benchmark at a handful of gauges.

### Gaps
- No public archive of HEFS ensemble forecasts (or NWPS hydrograph forecasts as time series) for 2020-2025 was found.
- The exact list of LMRFC/OHRFC RVF PILs that cover Tennessee basin points was not compiled.
- Whether TVA supplies the flows or stages in LMRFC RVFs for Tennessee points was not verified.

## 5. GloFAS (Copernicus Global Flood Awareness System)

### Takeaway
GloFAS operational forecasts are archived publicly on the Copernicus EWDS from 5 Nov 2019 (daily, 30-day, ECMWF-ENS-forced, GRIB2/NetCDF). The version changed from v3.1 (0.1°) to v4.0 (0.05°, from 2023-07-26) within the test period. Daily time step and 0.05°-0.1° grids are coarse for 100-3,000 km² Tennessee basins. GloFAS is a useful "global operational" benchmark only at larger gauges, with daily rather than 6-hourly scoring.

### Cited Findings
- **Operational forecast dataset (`cems-glofas-forecast`):**
  - **Coverage:** river discharge "5 November 2019 to near real time for operational version".
  - **Resolution:** "0.05° x 0.05° for version 4.0, 0.1° x 0.1° for version 3.1 and older".
  - **Time step and horizon:** "Daily data", with ECMWF extended range "up to 30 days".
  - **Format:** "GRIB2 and NetCDF-4".
  - **Version:** current is GloFAS v4.0, released 2023-07-26.
  - Source: [EWDS cems-glofas-forecast](https://ewds.climate.copernicus.eu/datasets/cems-glofas-forecast?tab=overview)
- **Reforecasts (`cems-glofas-reforecast`):** "0.05° x 0.05° for version 4.0". The period is "2003 to 2022 (inclusive)", run twice weekly at 00 UTC, with a 24-h step and 46-day lead time. The first reforecast reference date is 2023-03-27. DOI 10.24381/cds.2d78664e. Reforecasts have 11 members — [EWDS cems-glofas-reforecast](https://ewds.climate.copernicus.eu/datasets/cems-glofas-reforecast?tab=overview); [ECMWF CEMS Model Output wiki](https://confluence.ecmwf.int/spaces/CEMS/pages/242067364/Model+Output)
- **Forcing:** GloFAS 30-day uses the 00 UTC ECMWF ENS (51 members: 1 control plus 50 perturbed) for days 1-15 and extended range for days 16-30. Sources disagree on the ENS resolution: ~18 km per one page, ~9 km per another. The difference likely reflects the 2023 IFS upgrade but is unconfirmed — [GloFAS products](https://global-flood.emergency.copernicus.eu/technical-information/products/); [CEMS GloFAS meteorological forecasts](https://confluence.ecmwf.int/display/CEMS/GloFAS+meteorological+forecasts)
- **v4.0 reanalysis** (separate dataset, used to initialize reforecasts) — [ECMWF news 2022](https://www.ecmwf.int/en/about/media-centre/news/2022/copernicus-emergency-management-service-releases-glofas-v40); [EWDS historical](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical?tab=overview)
- **Skill:**
  - The GloFAS v4.0 performance page (modified KGE) says the lowest KGE "are spread across some tropical areas, often in central southern USA and Mexico ... often in the drier climate" — [GloFAS skill page](https://global-flood.emergency.copernicus.eu/technical-information/glofas-skill)
  - A 2026 study across 671 CAMELS catchments found GloFAS-ERA5 median RMSE improved "from 2.16 to 1.81 mm" (v2.1 to v4.0), with RMSE more sensitive to precipitation error in humid catchments (up to 2.5 mm) — [EGUsphere 2026-3321](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-3321/egusphere-2026-3321.pdf)
- **Practical:** one project reports that EWDS throughput is roughly "one day every five minutes and one request at a time". This is anecdotal, not official — [OCHA GitHub PR](https://github.com/OCHA-DAP/ds-aa-som-floods/pull/31)

### Inferences
- A 0.05° cell (~5 km × 4.5 km at 36°N, ~22 km²) means basins of 100-300 km² span only a few cells. The GloFAS river network's upstream-area matching to gauges will be error-prone there, and the 0.1° v3.1 grid (Nov 2019 to Jul 2023) is worse. GloFAS also has no Tennessee-specific reservoir operations. It is reasonable only for gauges larger than about 1,000 km², on daily aggregates, with the v3.1/v4.0 break flagged.
- Extracting 2020-2025 daily forecasts for a small bounding box via EWDS area subsetting is feasible but slow (one request per issue date).

### Gaps
- No official GloFAS minimum recommended catchment area was found.
- Ensemble member count for the operational forecast archive (51 assumed from the ENS) was not confirmed on the EWDS page itself.
- No Tennessee/Southeast GloFAS forecast skill study was found.

## 6. Other systems: Google Flood Hub, HEFS, EFAS, and other ensemble products

### Takeaway
Google Flood Hub forecasts are available only in real time through a waitlisted API. There is no public 2020-2025 operational forecast archive. A research reforecast dataset (Nearing et al. 2024, Zenodo) exists but is in mm/day at HydroBASINS/Caravan gauges, not NHDPlus. HEFS has no public archive. EFAS is Europe-only and not applicable.

### Cited Findings
- **Google Flood Forecasting API:** it "provides an interface to access real time riverine flood forecasts". Access is via waitlist and requires a Google account, use is "primarily limited to non-commercial use", and the license is CC BY 4.0 — [Google Developers Flood Forecasting](https://developers.google.com/flood-forecasting?hl=en); [Google Flood Forecasting resources](https://sites.research.google/gr/floodforecasting/resources/)
- **Google Runoff Reanalysis & Reforecast (GRRR):** "global river discharge estimates" for 1980-2023, accessed via a Colab notebook. Format and license are not stated on the page — [Google resources page](https://sites.research.google/gr/floodforecasting/resources/)
- **Nearing et al. 2024 (Nature 627:559-563):** AI forecasts in ungauged basins at up to 5-day lead are as reliable as or better than GloFAS nowcasts. The Zenodo record (v3, 2023-12-17) `model_data.tgz` (~18.9 GB) contains Google model reforecasts (mm/day) and GloFAS reanalyses (m³/s), daily — [Zenodo 10397664](https://zenodo.org/records/10397664); [Nature abstract via x-mol](https://ma.x-mol.com/paper/1771580390897016832)
- The model code is open (Apache 2.0) at github.com/google-research/flood-forecasting. Caravan (including US CAMELS basins) and Caravan MultiMet ("enriches streamflow data with weather forecasts") are listed — [Google resources page](https://sites.research.google/gr/floodforecasting/resources/)
- **EFAS:** a European system ("for version 4.0 and older, the open-source LISFLOOD hydrological model was forced at a 5x5km resolution"), not applicable to the US — [ECMWF EFAS dataset page](https://www.ecmwf.int/en/forecasts/datasets/river-discharge-and-related-forecasted-data-european-flood-awareness-system)
- **HEFS:** published skill studies exist (e.g., Upper Trinity, Texas), but no public archive was found — [NOAA IR HEFS Trinity](https://repository.library.noaa.gov/view/noaa/59892)

### Inferences
- Google's Flood Hub cannot serve as a 2020-2025 operational benchmark in the US with public data. The Zenodo reforecasts could be a research-model reference at CAMELS gauges overlapping CAMELS-H, but they are daily and only to ~2021-2023. Caravan MultiMet may be useful for a shared-forcing ML baseline.

### Gaps
- Whether Flood Hub covers Tennessee basin gauges in the US, and whether historical issued forecasts can be requested from Google, was not established.
- The exact temporal end of the Zenodo reforecasts was not confirmed.

## 7. Public forcing / precipitation forecast archives for a fair comparison (2020-2025)

### Takeaway
All main NOAA NWP forcings for 2020-2025 are on AWS NODD and can be read with Herbie: GEFS (incl. the v12 reforecast 2000-2019), HRRR, and NBM (from May 2020). One exception is GFS on the `noaa-gfs-bdp-pds` bucket, which starts only in 2021-01-01, so 2020 GFS needs another source. The NWM bucket's own `forcing_medium_range*` folders also provide the exact GFS- and NBM-derived forcing the NWM used, which allows a "same forcing" comparison.

### Cited Findings
- **GEFSv12 reforecast:** bucket `noaa-gefs-retrospective`, "spanning the period 2000-2019", generated once per day with "only 5 members". Once weekly an 11-member reforecast "extend[s] in lead time to +35 days", from 00 UTC, in GRIB2 — [AWS Registry GEFS reforecast](https://registry.opendata.aws/noaa-gefs-reforecast/)
- **Direct bucket listings (2026-10-12):**
  - `noaa-gefs-pds` starts `gefs.20170101/`. The directory layout changes on **2020-09-23** (new `atmos/`, `chem/`, `wave/` subfolders, consistent with the GEFSv12 operational upgrade). Before that it is only `pgrb2a/pgrb2b` — [S3 listing](https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&delimiter=/&prefix=gefs.20200923/00/)
  - `noaa-hrrr-bdp-pds` starts `hrrr.20140730/` — [S3 listing](https://noaa-hrrr-bdp-pds.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=6)
  - `noaa-nbm-grib2-pds` starts `blend.20200518/` — [S3 listing](https://noaa-nbm-grib2-pds.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=6)
  - `noaa-gfs-bdp-pds`: first `gfs.` prefix is `gfs.20210101/` — [S3 listing](https://noaa-gfs-bdp-pds.s3.amazonaws.com/?list-type=2&delimiter=/&prefix=gfs.&max-keys=2)
- **Herbie** (`herbie-data` on PyPI) "downloads recent and archived numerical weather prediction (NWP) model output from different cloud archive sources", including GFS, GEFS, NBM and HRRR via NODD — [PyPI herbie-data](https://pypi.python.org/pypi/herbie-data); [Herbie GFS docs](https://herbie.readthedocs.io/en/2024.3.0/user_guide/_model_notebooks/gfs.html)
- **NWM forcing folders** in `gs://national-water-model`: `forcing_medium_range/` (GFS-based) throughout 2020-2025, and `forcing_medium_range_blend/` (NBM) from v3.0 — direct listing ([GCS](https://storage.googleapis.com/storage/v1/b/national-water-model/o?delimiter=/&prefix=nwm.20240601/&fields=prefixes)); blend forcing files are 001-240 h, 00/06/12/18Z — [SCN 23-76](https://www.weather.gov/media/notification/pdf_2023_24/scn23-76_national_water_model_v3.0_aab.pdf)
- A Mid-Atlantic distributed-model study forced by GEFS reforecasts found ensemble streamflow forecasts "remain skillful for lead times of up to 7 days". This is a regional precedent for GEFS-forced hindcasts — [NOAA IR 59934](https://repository.library.noaa.gov/view/noaa/59934)

### Inferences
- **Cleanest "fair forcing" design:**
  - Drive the authors' model with the same GFS-derived forcing the NWM medium range used (NWM `forcing_medium_range`, 1 km grid, already downscaled), or with raw GFS/GEFS from NODD.
  - Report skill relative to NWM mem1 with identical issue times (00/06/12/18Z).
  - For 2020 GFS, use NCEI or NCAR RDA archives, because the AWS bdp bucket starts 2021.
- The GEFS v11 to v12 switch on 2020-09-23 is another break inside the test period, if GEFS is used.

### Gaps
- No primary source was retrieved for 2020 GFS 0.25° archive locations (NCEI/NCAR RDA) or WPC QPF archives.
- The NBM version history within 2020-2025 (v4.0 to v4.1 to v4.2; NBM v4.2 PNS 23-45 exists) was not detailed. Only NWS notice listings were seen: [PNS 23-45 NBM v4.2](https://www.weather.gov/media/notification/pdf_2023_24/pns23-45_nbm_v4.2.pdf), [SCN 20-78 NBM v4](https://weather.gov/media/notification/pdf2/scn20-78nbm_v4_aaa.pdf).
