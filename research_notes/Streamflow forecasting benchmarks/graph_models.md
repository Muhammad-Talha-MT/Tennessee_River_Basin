# GNN and spatiotemporal deep-learning models for streamflow prediction, forecasting and river routing (baselines and related work)

Scope note: these notes cover graph and spatiotemporal models, the Google graph-routing line of work, forecasting and simulation setups, code and licences, and the evidence on whether river topology helps. "PR" means peer-reviewed; "preprint" means arXiv, EGUsphere discussion or an EGU abstract. Licences for GitHub repos were read from the GitHub API (`api.github.com/repos/...`, `license.spdx_id`) on 2026-10-12. "None" means no licence file was detected, so the code is all-rights-reserved by default.

## Q1. Kratzert et al. GNN routing (EGU 2021) and Gauch/Kratzert "towards deep learning river network models": papers or code since?

### Takeaway
The Google/JKU graph-routing line of work is still documented only in EGU abstracts (2021, 2024). It has no peer-reviewed paper and no released graph code. Google's open-source operational framework (OpenHydroNet / google-research/flood-forecasting, Apache-2.0) and its Global Flood Forecasting Model v2 are lumped per-basin LSTMs with no routing or graph component. The closest peer-reviewed descendant of the "LSTM runoff + GNN routing" idea comes from ECMWF/Reading (Mosaffa et al., HESS 2026), not from Google.

### Cited Findings
- **Kratzert et al., EGU 2021 abstract, "Large-scale river network modeling using Graph Neural Networks" (preprint/abstract).**
  - The setup is decoupled. An LSTM estimates each sub-catchment's unimpaired runoff, and a GNN routes it "along the river network in hierarchical order", including human influences such as hydropower regulation.
  - Candidate routing modules were 1D convolution, attention modules and state-aware time-series models.
  - Stated benefits: per-basin outputs can be inspected, observed streamflow can be substituted where available, and any intermediate ungauged node can be simulated.
  - Data: LamaH-CE, more than 600 gauges, Danube above Bratislava (more than 130,000 km²), daily and hourly.
  - Results were only described qualitatively as "promising"; the model captured the 2002 Danube flood.
  - No code release is mentioned.
  - Sources: [neuralhydrology blog post](https://neuralhydrology.github.io/post/research/kratzert2021egu/); [ADS abstract](https://ui.adsabs.harvard.edu/abs/2021EGUGA..2313375K/abstract); [EGU21 presentation](https://presentations.copernicus.org/EGU21/EGU21-13375_presentation.pdf)
- **Gauch, Kratzert et al. (Google Research, with Klotz at UFZ), EGU24-8899, "Deep Learning for Spatially Distributed Rainfall–Runoff Modeling" (abstract).**
  - Graph-based DL models whose nodes are connected sub-basins.
  - Three motivations: learning travel times and separating runoff generation from routing in large basins at high temporal resolution; data assimilation, with upstream measurements informing downstream predictions; and placing reservoirs and extractions at their real graph locations.
  - Data: more than 600 LamaH-CE gauges. The abstract gives no quantitative results.
  - Source: [EGU24-8899](https://meetingorganizer.copernicus.org/EGU24/EGU24-8899.html)
- **No 2025 EGU abstract was found** under the exact title "Towards deep learning river network models". Kratzert convened, and Gauch co-convened, EGU25 session HS3.4 "Deep learning in hydrology". — [EGU25 HS3.4](https://meetingorganizer.copernicus.org/EGU25/session/52670)
- **Google Global Flood Forecasting Model v2 (Cohen, Nearing et al., EGUsphere egusphere-2026-2283, preprint, April 2026).**
  - Architecture: ME-LSTM (two stacked LSTMs: a hindcast state, then a forecast embedding) with a CMAL output, replacing v1's ED-LSTM. GraphCast is used only as a meteorological forcing.
  - Lead time: the reliable horizon extends "by 6 days in gauged basins and 1 day in ungauged basins" relative to the v1 nowcast.
  - Scale: about 1,223 test basins, and training gauges grew from 5,680 to 15,923 (per a reviewer).
  - Skill: median gauged NSE at 0-day lead is about 0.78 without GraphCast and 0.83 with it (per a reviewer).
  - The available text describes no routing or graph component.
  - Code: github.com/google-research/flood-forecasting; data on Zenodo 10.5281/zenodo.19676842.
  - Source: [EGUsphere 2026-2283](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-2283/)
- **OpenHydroNet (Google's open framework).**
  - Described as "a heavily modified fork of the NeuralHydrology Github repository".
  - Supports real forecasting with meteorological forecast inputs, includes pretrained weights trained on Caravan Multimet (more than 20,000 watersheds), and supports fine-tuning.
  - The blog post does not mention graph or routing models.
  - Sources: [EGU HS blog, Oct 2026](https://blogs.egu.eu/divisions/hs/2026/10/08/googles-hydrology-framework/); repo licence Apache-2.0, last pushed 2026-10-08, 352 stars ([GitHub API](https://api.github.com/repos/google-research/flood-forecasting))
- **NeuralHydrology library:** BSD-3-Clause, actively maintained (pushed 2026-10-10). — [GitHub API](https://api.github.com/repos/neuralhydrology/neuralhydrology)
- **Google Nature 2024 (Nearing et al.), "Global prediction of extreme floods in ungauged watersheds"** (PR) is a per-basin LSTM, not a graph model. — [Nature](https://www.nature.com/articles/s41586-024-07145-1)
- **Related Google/ECMWF technical note on probabilistic routing (Ruparell, …, Cloke, Prudhomme, Pappenberger; arXiv 2607.03217, July 2026, preprint).**
  - Basin-scale probabilistic LSTMs whose runoff is routed by a Hayami scheme (Japan).
  - Finding: randomly pairing upstream ensemble members gives "severely under-dispersed" downstream ensembles, and quantile matching restores the spread.
  - Source: [arXiv 2607.03217](https://arxiv.org/abs/2607.03217)

### Inferences
- Google has promoted graph-based distributed DL since 2021 but has not published it as a paper or code. The paper can cite these abstracts as the motivation for "upstream observations informing downstream predictions" but cannot benchmark against them.
- Google's operational model (v2 ME-LSTM, with an open codebase and pretrained weights) is the most credible open lumped-forecasting baseline at gauges. Running it per CAMELS-H/USGS gauge with the same forecast precipitation would give a strong "no-graph" baseline for 1–10-day leads.

### Gaps
- No Gauch/Kratzert EGU 2025 abstract with the exact title given was found. It may exist under another title in HS3.4, but this was not verified.
- The licence of the Google v2 model weights and data (Zenodo) was not checked.

## Q2. Inventory of GNN and spatiotemporal hydrology models (graph type, temporal model, task, data, skill, code, licence)

### Takeaway
Most published hydrology GNNs work on gauge graphs (tens to hundreds of nodes, e.g. LamaH-CE with 358–530 nodes) or small reach graphs (552 reaches; one sub-basin of the Delaware). Reported NSE/KGE values are not comparable across studies. The only learned models shown at 10^5-plus reaches are physics-embedded differentiable routers (δMC / dHBV2.0 / ddr: about 180k MERIT catchments; more than 800k CONUS reaches), plus gridded global models (RiverMamba at 0.05°; GraphRiverCast at 0.25°). No reach-level GNN at about 60k reaches with observed-flow inputs and 10-day forecast precipitation was found.

### Cited Findings

**A. LSTM + GNN routing (gauge/sub-basin graphs)**

- **LSTM-GNN, Upper Danube: Mosaffa, Pappenberger, Prudhomme, Chantry, Rüdiger, Cloke. "A GNN routing module is all you need for LSTM Rainfall–Runoff models", HESS 30, 2079, 2026 (PR; EGUsphere preprint 2025-5008).**
  - Graph: 530 subbasins (gauged outlets) from LamaH-CE, 1987–2017. Edges are directed downstream, weighted by inverse Kirpich travel time; the best model used 3 hops.
  - Model: a shared LSTM per subbasin produces node embeddings, a GNN (GCN, GAT, GraphSAGE or ChebNet) routes them, and a linear head predicts discharge. Trained end-to-end with MSE.
  - Task and inputs: daily next-day discharge from a 180-day window of ERA5-Land precipitation, soil moisture and temperature plus 59 static attributes. Observed streamflow is not an input, so this is effectively simulation.
  - Skill: LSTM-GAT is best, with mean NSE 0.61, KGE 0.65, CC 0.84, about 35% RMSE reduction, and better NSE than the LSTM at 78% of stations. Gains are largest at downstream, highly connected stations.
  - Code: github.com/hmosaffa/GNN_flow_routing. Data on request.
  - Sources: [HESS article](https://hess.copernicus.org/articles/30/2079/2026/); [EGUsphere preprint](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-5008/); repo has no licence, last push 2025-10-08 ([GitHub API](https://api.github.com/repos/hmosaffa/GNN_flow_routing))
- **RiverGraphNet: Farmani, Bennett, Moghisi, Gupta, Jawad, Behrangi, Tavakoly, Niu (U. Arizona). "Physics-Aware Routing of Gridded Runoff Through Directed River Networks", EGUsphere egusphere-2026-4157 (preprint in discussion, July–Oct 2026).**
  - Graph: a directed graph from the NextGen hydrofabric for the Salt–Verde (Arizona). Node and edge attributes include width, slope, travel-time proxy, conveyance, storage and reach length.
  - Model: Noah-MP gridded surface and subsurface runoff (AORC forcing) is mapped onto the nodes. The router is a 4-layer GAT (hidden size 64, 4 heads, about 17.8k parameters) with runoff lag windows of 3–30 days.
  - Task: daily retrospective simulation, not forecasting. Evaluated at 23 USGS gauges.
  - Skill: median KGE_ss about 0.72 (14-day lag, weighted MSE), against RAPID at about 0 (−0.012). Volume-corrected RAPID reached about 0.45, and RiverGraphNet still won at 21 of 23 gauges.
  - Cost: about 8 h of GPU training, under 10 min inference.
  - Sources: [EGUsphere 2026-4157](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4157/)
  - Code: PyTorch + PyTorch Geometric, with an optional shared LSTM/TCN runoff generator replacing precomputed runoff. Inputs are AORC Zarr, a graph cache and USGS CSVs, and no data are downloaded. The README self-describes it as "research software, not an operational flood warning system". Sources: [GitHub mfarmani95/RiverGraphNet](https://github.com/mfarmani95/RiverGraphNet); no licence, 0 stars, pushed 2026-09-25 ([GitHub API](https://api.github.com/repos/mfarmani95/RiverGraphNet))

**B. GNN forecasting on gauge graphs with observed discharge input (LamaH-CE hourly)**

- **Kirschstein & Sun, "The Merit of River Network Topology for Neural Flood Forecasting", ICML 2024, PMLR 235 (PR).**
  - Graph: 358 gauges from the largest connected Danube component of LamaH-CE, hourly, 2000–2017, with 2016–2017 as test years.
  - Task: 24 h of past discharge is used to predict discharge 6 h ahead.
  - Architectures: an encoder, N = 19 graph layers (GCNII, ResGCN or ResGAT, latent size 128) and a decoder.
  - Adjacency variants: isolated, binary, weighted by stream length, elevation difference or slope, and learned. Each was tested downstream, upstream and bidirected.
  - Skill (relevancy-weighted NSE): MLP baseline 85.37 ± 1.64%; isolated ResGCN 85.07%; best (bidirected, learned, GCNII) 85.56 ± 1.41%.
  - Code: github.com/nkirschi/neural-flood-forecasting.
  - Sources: [PMLR](https://proceedings.mlr.press/v235/kirschstein24a.html); [arXiv HTML](https://arxiv.org/html/2405.19836v1); repo has no licence, 17 stars ([GitHub API](https://api.github.com/repos/nkirschi/neural-flood-forecasting))
- **Wang, Chen, Zheng, Song, "Accelerating Flood Warnings by 10 Hours: The Power of River Network Topology in AI-enhanced Flood Forecasting".**
  - Status: arXiv 2410.05536 (v3, March 2025), with a Nature-portfolio journal version at s44304-025-00083-6.
  - Graph: the same 358-station LamaH-CE hourly set, with 2010–2015 for training and 2016–2017 for testing.
  - Task: 1–24 h horizons from 24 h of history.
  - Architectures: ChebNet, GAT, GraphSAGE, GCNII, GCN and GIN (3 layers, 32-dimensional), against an EA-LSTM baseline.
  - Graph transformation: a dense reachability graph, built by passing topological distance through a Gaussian RBF.
  - Claims: transformed GNNs at 24 h match EA-LSTM's 14-h accuracy (a "71%" longer horizon), GAT beats EA-LSTM up to 13.3 h, and the average lead advantage is about 10 h. The text gives no explicit NSE table.
  - Code: github.com/Dreamzz5/FloodGNNs.
  - Sources: [arXiv abs](https://arxiv.org/abs/2410.05536); [arXiv v3 HTML](https://arxiv.org/html/2410.05536v3); [journal page](https://www.nature.com/articles/s44304-025-00083-6); repo MIT ([GitHub API](https://api.github.com/repos/Dreamzz5/FloodGNNs))

**C. Reach-graph GNNs (physics-guided or surrogate)**

- **Sun, Jiang, Yang, Xie, Chen, "A graph neural network (GNN) approach to basin-scale river network learning: the role of physics-based connectivity and data fusion", HESS 26, 5163, 2022 (PR).**
  - Graph: 552 reaches in the East Taylor watershed (Colorado, snow-dominated), built from NHDPlus COMIDs.
  - Model: a GNN trained as a surrogate of a high-resolution vector river network model (NWM v2.0 retrospective outputs), then fine-tuned on observations. Graph-based data fusion corrects biases.
  - Skill: median KGE above 0.97 as a surrogate.
  - Graph coarsening gave comparable skill at a fraction of the training cost.
  - Sources: [HESS](https://hess.copernicus.org/articles/26/5163/2022/); [PNNL summary](https://www.pnnl.gov/publications/graph-neural-network-gnn-approach-basin-scale-river-network-learning-role-physics); NHDPlus/NWM detail from search snippet of [HESS](https://hess.copernicus.org/articles/26/5163/2022/)
- **Jia et al. (UMN + USGS), "Physics-Guided Recurrent Graph Model for Predicting Flow and Temperature in River Networks", SIAM SDM 2021 (PR; arXiv 2009.12575, 2020).**
  - Model: a recurrent graph network over river segments, pretrained on physics-based model outputs, with a loss balanced across segments.
  - Region: a subset of the Delaware River Basin, daily.
  - Skill with 0.1% training data: temperature improved by 33% over the physics-based model and 24% over LSTM; streamflow improved by 14% over both.
  - Sources: [USGS pubs](https://pubs.usgs.gov/publication/70247905); [arXiv](https://arxiv.org/abs/2009.12575)
- **USGS river-dl (RGCN and related models), Delaware River Basin.**
  - Repo: archived on GitHub (development moved to code.usgs.gov/wma/wp/river-dl); licence CC0-1.0. Described as "a deep graph convolutional network for predicting environmental variables on river networks", originally for stream temperature and streamflow, with Delaware-specific data.
  - Topp et al. 2023 compared RGCN with Graph WaveNet for stream temperature. Both did well in-distribution, but Graph WaveNet was significantly better in 4 of 5 out-of-distribution experiments. The data release (DOI 10.5066/P9HU7BLR) includes code.
  - Sources: [ecosyste.ms listing](https://ost.ecosyste.ms/projects/20472); [GitHub API](https://api.github.com/repos/USGS-R/river-dl); [USGS pub](https://pubs.usgs.gov/publication/70250381); [USGS data release](https://www.usgs.gov/data/examining-influence-deep-learning-architecture-generalizability-predicting-stream-temperature)

**D. Spatiotemporal GNNs for streamflow forecasting (small gauge graphs)**

- **Akkala et al., "Spatio-Temporal Graph Neural Networks for Streamflow Prediction in the Upper Colorado Basin", *Hydrology* (MDPI) 12(3):60, March 2025 (PR).**
  - Model: GCN + LSTM over 20 stations, monthly data, 30 years.
  - Skill: R² 0.78, RMSE 0.81 mm/month, KGE 0.79 at Lees Ferry, beating SARIMA, RF, LSTM and GRU. Best input/output window was (36, 36) months.
  - Follow-up: an SWE-augmented STGNN appeared in *Hydrology* 12(10):268, 2025.
  - Sources: [doi 10.3390/hydrology12030060](https://doi.org/10.3390/hydrology12030060); [SWE follow-up](https://doi.org/10.3390/hydrology12100268)
- **DynaSTG-Former, "Dynamic Graph Transformer with Spatio-Temporal Attention for Streamflow Forecasting", *Hydrology* 12(12):322 (MDPI, PR).**
  - Graph: topology from NHDPlus HR + HydroRIVERS over 45 gauges.
  - Skill: KGE 0.855 at a 6-step (72 h) horizon. The KGE drop of about 11% from 1-step is from a search snippet only.
  - Source: [doi 10.3390/hydrology12120322](https://doi.org/10.3390/hydrology12120322)
- **"Directed graph deep neural network for multi-step daily streamflow forecasting", J. Hydrology 2022 (PR).**
  - Graph: upper Yangtze gauge and precipitation stations, multi-step daily.
  - Reported to beat ANN, LSTM, GRU and CNN (snippet only).
  - Source: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169422000907)
- **HydroGAT: Sarkar, …, Demir, Jannesari, "Distributed Heterogeneous Graph Attention Transformer for Spatiotemporal Flood Prediction", ACM SIGSPATIAL 2025 (PR; arXiv 2509.02481).**
  - Graph: pixel-level heterogeneous basin graph ("every land and river pixel is a node") in two Midwestern US basins, hourly.
  - Skill: NSE up to 0.97, KGE up to 0.96.
  - Engineering: distributed data-parallel training.
  - Code: github.com/swapp-lab/HydroGAT.
  - Sources: [arXiv](https://arxiv.org/abs/2509.02481); repo Apache-2.0 ([GitHub API](https://api.github.com/repos/swapp-lab/HydroGAT))
- **Others found only as titles or snippets (unverified details):**
  - "Spatio-temporal graph neural networks for the forecast of natural water systems", J. Hydroinformatics 2026 (page returned 403). — [IWA](https://iwaponline.com/jh/article/doi/10.2166/hydro.2026.092/111680/Spatio-temporal-graph-neural-networks-for-the)
  - "Graph Transformer Network for Flood Forecasting with Heterogeneous Covariates" (Purdue I-GUIDE; 403). — [Purdue](https://docs.lib.purdue.edu/context/iguide/article/1007/viewcontent/Graph_Transformer_Network_for_Flood_Forecasting_with_Heterogeneou.pdf)
  - ST-GNN with feature-enhancement fusion (Symmetry 2026; 1–8 h leads; 6.5% RMSE reduction over ST-GCN). — [doi 10.3390/sym18020240](https://doi.org/10.3390/sym18020240)
  - "Flood Prediction with Graph Neural Networks" (NeurIPS 2022 workshop). — [NeurIPS](https://neurips.cc/virtual/2022/59320)

**E. Learned or differentiable routing at continental and global scale (most relevant for scale)**

- **δMC: Bindas et al., WRR 60(1), 2024 (PR).**
  - Model: differentiable Muskingum-Cunge, where an NN infers Manning's n and channel geometry from reach attributes, trained only on downstream hydrographs. First demonstrated on the Juniata.
  - Sources: [ESSOAr preprint](https://essopenarchive.org/doi/full/10.22541/essoar.168500246.67971832/v1); [NOAA repository](https://repository.library.noaa.gov/view/noaa/63586); [CIROH abstract](https://ciroh.ua.edu/abstracts/improving-large-basin-streamflow-simulation-using-a-modular-differentiable-learnable-graph-model-for-routing/)
  - The CONUS version improved results for basins larger than 10,000 km². MC routing was more computationally demanding than unit-hydrograph routing. — [arXiv 2504.10707](https://arxiv.org/pdf/2504.10707)
  - δMC on MERIT flowlines gives flows for every reach; a KAN maps reach attributes to geometry and roughness. — [arXiv 2504.10707](https://arxiv.org/pdf/2504.10707)
- **ddr ("Distributed Differentiable Routing", DeepGroundwater org).**
  - "Differentiable river routing methods for the NextGen Framework and Hydrofabric": NextGen hydrofabric v2.2 CONUS adjacency, MERIT configs, sparse COO adjacency, and a solver inspired by RAPID.
  - Lateral inflow comes from dHBV2.0 (Song et al. 2025 WRR, about 180,000 MERIT unit catchments).
  - Sources: [GitHub deepgroundwater/ddr](https://github.com/deepgroundwater/ddr); Apache-2.0, pushed 2026-09-11 ([GitHub API](https://api.github.com/repos/deepgroundwater/ddr))
- **dHBV2.0MTS-MC: Yang, Lonzarich, Song, Ji, Pan, Lawson, Shen, "Hourly U.S.-wide flood simulation beyond the limits of traditional and data-driven models", arXiv 2609.06794 (Sept 2026, preprint).**
  - Scale: more than 800,000 CONUS reaches, hourly, simulation.
  - Skill: median hourly NSE 0.683 at 2,831 gauges, against 0.461 for NWM v3.0. Peak-timing error fell from 7–8 h to 4.5–6 h, and the model captured 33% more floods of 50-year return period or larger.
  - Code availability was not stated on the abstract page.
  - Source: [arXiv 2609.06794](https://arxiv.org/abs/2609.06794)
- **Lima, Deck, Dunbar, Schneider, "Toward routing river water in land surface models with recurrent neural networks", HESS 29, 3145, 2025 (PR).**
  - Model: an RNN router fed instantaneous LSM runoff rather than precipitation, trained globally and tested in hindcast.
  - Generalizes across basins and across time. In the time-split experiment, median NSE rose from 0.56 to 0.64 over a physics-based model (per search summary).
  - Sources: [HESS](https://hess.copernicus.org/articles/29/3145/2025/); [arXiv 2404.14212](https://arxiv.org/abs/2404.14212v2)
- **GraphRiverCast: Ren, Zhao, Wang, Slater, Yamazaki, et al., "Topology enables learning-based hydrodynamic prediction of the global river system", arXiv 2602.22293 (v2 Aug 2026, preprint; a journal version is mentioned).**
  - Scale: global 0.25° river network, daily multivariate hydrodynamics at every reach, from sparse gauges and with no initial state.
  - Claims "no intrinsic skill decay with lead time" and 25% higher accuracy than leading global river models.
  - Code availability was not stated.
  - Source: [arXiv 2602.22293](https://arxiv.org/abs/2602.22293)

**F. Non-graph spatiotemporal, Transformer, SSM and foundation models**

- **RiverMamba: Shams Eddin, Zhang, Kollet, Gall, NeurIPS 2025 (PR; arXiv 2505.22535).**
  - Model: Mamba blocks over a global 0.05° grid, pretrained on long-term reanalysis, with ECMWF HRES forecasts integrated in forecast blocks.
  - Lead time: up to 7 days.
  - Claims to beat both AI and physics-based models across flood return periods.
  - Code and datasets are public via the project page.
  - Sources: [arXiv](https://arxiv.org/abs/2505.22535); [NeurIPS](https://neurips.cc/virtual/2025/poster/118456); project-page repo hakamshams/RiverMamba is CC-BY-SA-4.0 ([GitHub API](https://api.github.com/repos/hakamshams/RiverMamba))
- **"From RNNs to Transformers: benchmarking deep learning architectures for hydrologic prediction", HESS 29, 6811, 2025 (PR).** The LSTM reached KGE about 0.80 on CAMELS, and the second-ranked model, Crossformer, trailed it by 0.07. — [HESS](https://hess.copernicus.org/articles/29/6811/2025/)
- **"Probing the limit of hydrologic predictability with the Transformer network".** A vanilla Transformer was not competitive with LSTM on CAMELS. — [NSF PAR](https://par.nsf.gov/biblio/10541025-probing-limit-hydrologic-predictability-transformer-network)
- **HydroGEM (arXiv 2512.14106).** A TCN-Transformer foundation model pretrained on 3,724 USGS stations, for streamflow quality control rather than forecasting. — [arXiv](https://arxiv.org/html/2512.14106)
- **AIFL (arXiv 2602.16579).** A global daily LSTM pretrained on ERA5-Land and fine-tuned on IFS forecasts. — [arXiv](https://arxiv.org/pdf/2602.16579)
- **HydroGraphNet (NVIDIA PhysicsNeMo example).**
  - Model: a physics-informed autoregressive encoder–processor–decoder GNN on unstructured 2D meshes predicting water depth and volume. It uses a mass-conservation loss, the pushforward trick and KAN layers.
  - Training data: shallow-water solver output (HEC-RAS).
  - Reported results (White River near Muncie, IN): 67% lower error and 58% better CSI for major floods than a baseline GNN.
  - This is 2D inundation emulation, not river-network streamflow forecasting.
  - Sources: [PhysicsNeMo docs](https://docs.nvidia.com/physicsnemo/26.05/physicsnemo/examples/weather/flood_modeling/hydrographnet/README.html); [NVIDIA forum](https://forums.developer.nvidia.com/t/will-hydrographnet-be-updated-from-swe-to-navier-stokes-nse/356637)
- **Physics-informed GNNs for operational flood modeling (arXiv 2512.23964)** is another mesh/inundation GNN. — [arXiv](https://arxiv.org/html/2512.23964)

### Inferences
- The paper's design (whole-upstream routing with learned travel-time kernels per river-distance bin, and upstream-only attention) sits between several neighbours:
  - The dense reachability graph of Wang et al., which also connects all upstream nodes, weighted by an RBF of topological distance.
  - The travel-time-weighted edges of Mosaffa et al. (Kirpich).
  - The physics routers δMC/ddr, which learn celerity per reach.
  - These are the three most direct comparators to position against.
- Among graph models, only δMC/ddr/dHBV (more than 180k–800k reaches) and the gridded global models (RiverMamba, GraphRiverCast) have been shown at or above the paper's scale of about 59k reaches. All published GNN routers with message passing (Mosaffa, RiverGraphNet, Kirschstein, Wang, Sun) were shown on 23–552 nodes.
- Skill numbers are not comparable across these studies. Kirschstein and Wang report hourly 6–24 h NSE with observed discharge input (about 0.85); Mosaffa reports daily simulation without observed flow (NSE 0.61); RiverGraphNet reports KGE_ss in arid Arizona; Akkala reports monthly values. A common evaluation on CAMELS-H/USGS gauges would be needed.

### Gaps
- "FloodGNN", "GNRRM" and "HydroFM" could not be found as named hydrology models. Searches returned the "FloodGNNs" repo of Wang et al. and unrelated works. Some may exist under other names; not verified.
- TFT (Temporal Fusion Transformer) streamflow studies were not specifically searched.
- Not verified: the number of segments in the Jia et al. Delaware study, its exact skill values (RMSE), and whether its code sits in river-dl.
- Not verified: whether the Sun et al. 2022 code is public, and the number of nodes in HydroGAT.
- The licence of NVIDIA PhysicsNeMo was not checked here.
- The J. Hydroinformatics 2026 STGNN paper and the Purdue graph transformer paper could not be read (HTTP 403).

## Q3. Which do 1–10-day forecasting with observed flow inputs, and which are pure simulation?

### Takeaway
Graph models that take observed discharge as input forecast only hours ahead (6–24 h on LamaH-CE hourly: Kirschstein & Sun; Wang et al.; DynaSTG-Former up to 72 h). Graph routers at multi-day timescales are run as simulations driven by meteorology or LSM runoff (Mosaffa, RiverGraphNet, δMC/dHBV, Lima et al.). No graph or reach-level model was found that combines observed gauge flow with forecast precipitation at 1–10-day leads. The paper's setup appears to fill that gap. The multi-day forecasting systems with forecast precipitation are lumped or gridded (Google v2 ME-LSTM, RiverMamba at 7 days, AIFL).

### Cited Findings
- **Forecasting with observed discharge input, hourly gauge graphs:**
  - Kirschstein & Sun: 24 h of discharge history, 6 h lead. — [arXiv HTML](https://arxiv.org/html/2405.19836v1)
  - Wang et al.: 24 h history, 1–24 h leads. — [arXiv v3](https://arxiv.org/html/2410.05536v3)
- **DynaSTG-Former:** evaluated up to 6 steps (72 h), KGE 0.855 (snippet only). — [doi](https://doi.org/10.3390/hydrology12120322)
- **Akkala et al. STGNN:** a monthly forecast with a 36-month window, so not short-range. — [doi](https://doi.org/10.3390/hydrology12030060)
- **Mosaffa et al.:** next-day daily discharge, no observed streamflow input, so it is presented as applicable to ungauged basins (effectively simulation). — [HESS](https://hess.copernicus.org/articles/30/2079/2026/)
- **Simulations, not forecasts:**
  - RiverGraphNet: retrospective, held-out temporal blocks. — [EGUsphere](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4157/)
  - dHBV2.0MTS-MC. — [arXiv 2609.06794](https://arxiv.org/abs/2609.06794)
  - Lima et al.: hindcasts. — [HESS](https://hess.copernicus.org/articles/29/3145/2025/)
  - Sun et al. 2022: surrogate of NWM, with data fusion for bias correction. — [HESS](https://hess.copernicus.org/articles/26/5163/2022/)
- **Multi-day forecasting with meteorological forecasts, non-graph:**
  - Google v2: ME-LSTM with hindcast and forecast phases, extending the reliable horizon by 6 days in gauged basins. — [EGUsphere 2026-2283](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-2283/)
  - RiverMamba: up to 7 days with HRES. — [arXiv](https://arxiv.org/abs/2505.22535)
  - GraphRiverCast: daily, claims no intrinsic decay with lead time. — [arXiv](https://arxiv.org/abs/2602.22293)
- **Using observed flow in DL forecasts:**
  - Nearing et al., HESS 26, 5493, 2022 (PR): for near-real-time streamflow in LSTMs, autoregression (feeding lagged observations) was more accurate and cheaper than variational data assimilation. It is sensitive to missing data, but a simple training strategy mitigates that. Tested on CAMELS. — [HESS](https://hess.copernicus.org/articles/26/5493/2022/)
  - Saint-Fleur et al., HESS 30, 3497, 2026 (PR) tested discharge-assimilation strategies at 1–7-day leads on CAMELS-US (531 basins) and CAMELS-FR (338). At 1 day, NSE gains were about 8–12% for LSTM and up to 22% for SAC-SMA. With ensemble forecasts the LSTM was penalized by under-dispersion. — [HESS](https://hess.copernicus.org/articles/30/3497/2026/)
- **Gauch/Kratzert (EGU24)** listed upstream observations informing downstream predictions as a key motivation, but reported no results. — [EGU24-8899](https://meetingorganizer.copernicus.org/EGU24/EGU24-8899.html)

### Inferences
- With observed-flow inputs, the closest graph-forecasting baselines (Kirschstein; Wang) are hourly, short-lead, gauge-only graphs. Adapting them to 6-hourly, 10-day forecasts with forecast precipitation would change their setup substantially. Reviewers will likely expect the paper to show that observed-flow autoregression is handled without leakage (e.g. the forecast period has no observed flow, and gauge dropout follows Nearing et al. 2022).
- The skill benefit of gauge inputs should decay with lead time. Google's results (6 extra reliable days gauged vs 1 ungauged) and Wang's EA-LSTM crossover at about 13 h suggest that comparing against an autoregressive lumped LSTM at each lead is essential.

### Gaps
- No public study evaluates a reach-level GNN with observed-flow inputs at multi-day leads, so this could not be verified as absent beyond the searches performed.
- The exact lead-time NSE curves for Google v2 were seen only via reviewer summaries.

## Q4. Which have public code that could run on a large NHDPlus graph, and what would it take?

### Takeaway
Repos with permissive licences and code relevant to large graphs:
- **ddr** (Apache-2.0): differentiable routing on the NextGen hydrofabric and MERIT, built for CONUS scale with sparse adjacency.
- **HydroGAT** (Apache-2.0): distributed data-parallel training, but a pixel graph.
- **Google flood-forecasting/OpenHydroNet and NeuralHydrology** (Apache-2.0 / BSD-3): lumped, per-gauge baselines.
- **FloodGNNs** (MIT): gauge graph.
- **river-dl** (CC0, archived): Delaware segments.

The most directly comparable GNN-routing codes (Kirschstein & Sun, Mosaffa GNN_flow_routing, RiverGraphNet) have no licence file, so reuse needs author permission, and none was shown beyond about 550 nodes.

### Cited Findings
- **Licences, activity and stars (GitHub API, 2026-10-12):**

| Repo | Licence | Last push | Stars | Notes |
|---|---|---|---|---|
| google-research/flood-forecasting | Apache-2.0 | 2026-10-08 | 352 | |
| neuralhydrology/neuralhydrology | BSD-3-Clause | 2026-10-10 | 586 | |
| deepgroundwater/ddr | Apache-2.0 | 2026-09-11 | | |
| swapp-lab/HydroGAT | Apache-2.0 | 2025-11-03 | | |
| Dreamzz5/FloodGNNs | MIT | 2024-10-05 | | |
| USGS-R/river-dl | CC0-1.0 | | | archived |
| nkirschi/neural-flood-forecasting | none | 2025-03-31 | | |
| hmosaffa/GNN_flow_routing | none | 2025-10-08 | | |
| mfarmani95/RiverGraphNet | none | 2026-09-25 | | |
| hakamshams/RiverMamba | CC-BY-SA-4.0 | | | project page |

  - Sources: [flood-forecasting](https://api.github.com/repos/google-research/flood-forecasting), [neuralhydrology](https://api.github.com/repos/neuralhydrology/neuralhydrology), [ddr](https://api.github.com/repos/deepgroundwater/ddr), [HydroGAT](https://api.github.com/repos/swapp-lab/HydroGAT), [FloodGNNs](https://api.github.com/repos/Dreamzz5/FloodGNNs), [river-dl](https://api.github.com/repos/USGS-R/river-dl), [neural-flood-forecasting](https://api.github.com/repos/nkirschi/neural-flood-forecasting), [GNN_flow_routing](https://api.github.com/repos/hmosaffa/GNN_flow_routing), [RiverGraphNet](https://api.github.com/repos/mfarmani95/RiverGraphNet), [RiverMamba](https://api.github.com/repos/hakamshams/RiverMamba)
- **ddr** builds sparse COO adjacency from `conus_nextgen.gpkg` (Hydrofabric v2.2) and has MERIT configs. NHDPlus is not listed as a supported input. Lateral inflow comes from dHBV2.0. — [ddr README](https://github.com/deepgroundwater/ddr)
- **RiverGraphNet** needs a precomputed graph cache (edge index and weights, node features, gauge indices, runoff-transfer mapping) built from NextGen GeoPackage flowpaths, plus AORC Zarr forcing. It is PyTorch Geometric code and downloads no data. — [GitHub](https://github.com/mfarmani95/RiverGraphNet)
- **The δMC CONUS work** reports being limited by computational memory and hydrofabric scaling. — search summary of [EGU24-20602](https://meetingorganizer.copernicus.org/EGU24/EGU24-20602.html) and [arXiv 2504.10707](https://arxiv.org/pdf/2504.10707)
- **Kirschstein & Sun** used 19 graph layers on 358 nodes. Message passing to reach far upstream nodes needs depth on the order of graph diameter, and Wang et al. attribute weak topology gains to over-squashing on tree-like river graphs. — [arXiv HTML](https://arxiv.org/html/2405.19836v1); [arXiv 2410.05536](https://arxiv.org/abs/2410.05536)
- **USGS river-dl data** were "specific to the Delaware River Basin". — [ecosyste.ms](https://ost.ecosyste.ms/projects/20472)

### Inferences
- **Practical baseline set at about 59k NHDPlus reaches:**
  1. **Lumped autoregressive LSTM forecaster** (NeuralHydrology or OpenHydroNet), one per CAMELS-H/USGS gauge, with past observed flow, past meteorology and forecast precipitation. This is the strongest and cheapest no-topology baseline, and it addresses the Kirschstein & Sun critique.
  2. **LSTM + GNN routing** (Mosaffa-style GAT on the gauge/sub-basin graph). Rebuilding the graph on NHDPlus with the gauge set or HUC subbasins is feasible (hundreds to a few thousand nodes). Running it per reach needs about 59k LSTMs (shared weights), which fits batched GPUs at 6-hourly resolution. Code reuse requires permission (no licence).
  3. **Reachability or dense-upstream GNN** (Wang et al. FloodGNNs, MIT) on the gauge graph. The dense adjacency is N² at reach level, so it is only feasible on gauges or with sparse upstream-only masks.
  4. **Differentiable physics routing** (ddr / δMC, Apache-2.0) as a physics-structured learned router. NHDPlus v2 would need converting to ddr's adjacency format; ddr targets the NextGen hydrofabric, which is itself derived from NHDPlus/reference fabric. A cross-walk COMID → NextGen flowpath would be needed. It needs a lateral-inflow source, e.g. an LSTM per catchment or dHBV2.0.
- **Engineering for NHDPlus at about 59k reaches (any of the above):**
  - Build the COMID → ToCOMID edge list (NHDPlus PlusFlow / VAA), topologically sort it, and build sparse CSR adjacency.
  - Map AORC/forecast-precipitation grids to catchments.
  - Gauge-index CAMELS-H/USGS sites to COMIDs.
  - For deep message passing, either use the paper's whole-upstream aggregation (sparse upstream-closure matrices binned by river distance) or many layers. With 6-hourly steps over 10-day horizons (40 steps) at 59k nodes, memory is dominated by temporal state × nodes. Mini-batching by sub-basin (HUC4/HUC6 subgraphs, as in Sun et al.'s graph coarsening) is the typical remedy.

### Gaps
- None of these baselines' GPU memory or run times at about 10^4–10^5 nodes was found reported. RiverGraphNet's about 8 h of training is for the Salt–Verde only.
- Whether HydroGAT's distributed pipeline scales to river-reach graphs (rather than pixels) was not verified.
- No code was found for Sun et al. 2022, GraphRiverCast or dHBV2.0MTS-MC.

## Q5. Does river topology help? Common positive and negative findings

### Takeaway
The evidence is mixed and depends on setup:
- For short-lead hourly forecasting with observed discharge on gauge graphs, topology adds little. Kirschstein & Sun (ICML 2024) found an MLP and an isolated-node GNN as good as any topology.
- Benefits appear when (a) the graph is densified or reachability-based to overcome over-squashing, and gains grow with lead time and spike floods (Wang et al.); (b) no observed flow is given and routing must be learned, with gains concentrated at downstream, highly connected stations (Mosaffa et al.); or (c) the comparison is against physics routing (RiverGraphNet vs RAPID; dHBV-MC vs NWM).

### Cited Findings
- **Negative: Kirschstein & Sun, ICML 2024.**
  - Performance was nearly insensitive to topology across 6 adjacency definitions, 3 orientations and 3 architectures. An isolated configuration did about as well as, or slightly better than, topology-aware ones, and an MLP matched the best GNN (85.37% vs 85.56% weighted NSE, within one standard deviation).
  - Learned edge weights correlated weakly with physical weights (mostly below |0.4|), with sign flips between architectures. Four small subnetworks gave the same result.
  - GNNs struggled with sudden, narrow spikes.
  - Sources: [arXiv HTML](https://arxiv.org/html/2405.19836v1); [PMLR](https://proceedings.mlr.press/v235/kirschstein24a.html)
- **Positive with a reachability graph: Wang et al.**
  - They attribute the weak topology effect to over-squashing: tree-like river graphs give distant nodes high resistance distances.
  - A dense reachability transformation reduces this. Dense adjacency "consistently" outperformed topological and isolated counterparts and EA-LSTM, with about a 10 h lead advantage.
  - Sources: [arXiv](https://arxiv.org/abs/2410.05536); [arXiv v3 HTML](https://arxiv.org/html/2410.05536v3); dense-vs-isolated wording from search snippet of [journal page](https://www.nature.com/articles/s44304-025-00083-6)
- **Positive, LSTM + GNN routing: Mosaffa et al., HESS 2026.** All four GNN variants beat the lumped LSTM. GAT was best, improving NSE at 78% of stations, with gains largest downstream at highly connected stations with large contributing areas. — [HESS](https://hess.copernicus.org/articles/30/2079/2026/); snippet from search
- **Positive vs physics routing: RiverGraphNet.** Median KGE_ss about 0.72 against about 0 for RAPID (about 0.45 for volume-corrected RAPID), winning at 21 of 23 gauges. Note that the comparison is a learned router against an uncalibrated or lightly corrected physics router. — [EGUsphere](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4157/)
- **Positive, physics-embedded routing: δMC.** Muskingum-Cunge routing improved CONUS performance for basins larger than 10,000 km². dHBV2.0MTS-MC improved median hourly NSE to 0.683 against 0.461 for NWM v3.0 and reduced peak-timing errors. — [arXiv 2504.10707](https://arxiv.org/pdf/2504.10707); [arXiv 2609.06794](https://arxiv.org/abs/2609.06794)
- **Positive, global: GraphRiverCast.** The title claims topology "enables" global hydrodynamic prediction, with 25% higher accuracy than leading global river models (preprint). — [arXiv 2602.22293](https://arxiv.org/abs/2602.22293)
- **Architecture matters out of distribution:** Graph WaveNet beat RGCN in 4 of 5 out-of-distribution stream-temperature experiments (Delaware). — [USGS data release](https://www.usgs.gov/data/examining-influence-deep-learning-architecture-generalizability-predicting-stream-temperature)
- **Probabilistic caveat:** naive combination of upstream ensembles gives under-dispersed downstream ensembles. — [arXiv 2607.03217](https://arxiv.org/abs/2607.03217)

### Inferences
- Results converge on one pattern. When each gauge's own recent discharge is available and the lead is short, persistence dominates and topology adds little. Topology and routing matter more (a) as lead time grows beyond local travel times, (b) for large downstream basins, (c) for ungauged reaches or simulation, and (d) for spike floods, if long-range message passing is not over-squashed.
- This supports the paper's design: whole-upstream aggregation with travel-time kernels avoids deep message passing and over-squashing, and 10-day leads exceed most within-basin travel times. The paper should still include the Kirschstein-style ablations: an isolated-node or no-routing variant, an MLP/LSTM per gauge with observed flow, and lead-time-stratified and drainage-area-stratified skill. This would show where topology helps over Tennessee basin travel times, which are regulated by TVA reservoirs.

### Gaps
- No study was found that tests the topology benefit at reach level (10^4-plus nodes) with observed-flow inputs and multi-day leads.
- No study was found that tests topology in a heavily reservoir-regulated system like the TVA network. Gauch/Kratzert flagged reservoirs as a motivation but published no results. Kratzert 2021 mentions hydropower regulation only qualitatively.
