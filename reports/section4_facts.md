# Section 4 facts sheet: models, training and evaluation

Snapshot of the repository and the trained models on 2026-11-21. Everything below is read from code, saved configs and saved checkpoints; nothing is taken from the paper draft. There are no results or skill scores in this file. `UNKNOWN` means it cannot be determined from the repository.

**Path shorthand** (the Python package directory under `src/` is written `<pkg>` and the data root `<data>`, to keep names neutral):
`H/` = `src/<pkg>/hydro/`, `D/` = `src/<pkg>/data/`, `T/` = `src/<pkg>/training/`, `E/` = `src/<pkg>/evaluation/`, `S/` = `scripts/`, `R/` = `reports/v3_tables/`, `<data>/models/attr/` = the model folder. Line numbers refer to the files as of the snapshot.

---

## 1. Model family

### 1.1 Variants that can be run (flags of `H/train.py`)

| Variant | Command line (stage B) | Stage-A checkpoint | Source |
|---|---|---|---|
| **M0** | `--router manning --insert` (graph layers default 3) | `manning_A` (Manning router, 3 graph layers) | `S/attribution_pipeline.sh:116` |
| **M1** | `--router manning` (no `--insert`) | `manning_A` | `S/attribution_pipeline.sh:117` |
| **M2** | `--router manning --insert --graph-layers 0` | `manning0_A` (Manning router, 0 graph layers) | `S/attribution_pipeline.sh:118` |
| **M3** | does not exist: no definition in `variant_args`; only a plotting colour entry (`S/v3_figures.py:19`) | | `S/attribution_pipeline.sh:106-122` |
| **M4** | `--router instant` (no `--insert`) | `instant_A` (instant router, 3 graph layers) | `S/attribution_pipeline.sh:119` |
| LSTM (area-scaled, "L4") | `train-lstm ... --area-output --loss nse --antecedent` | none | `S/attribution_pipeline.sh:238, 242-254` |

All M variants share the stage-B arguments `--stage B --loss nse-q --head lstm --delta-weight 0 --accumulate 8 --select gauges --resume` and `--updates 3000 --lr 3e-4 --warmup 100 --validate-every 500` (`S/attribution_pipeline.sh:102-103`). Saved arguments of M0, M1, M2 differ only in `init`, `insert`, `graph_layers` and the derived parameter count (`R/config_audit.csv`). `M4_s1_f1` differs from `M1` only in `init` and `router` (`R/m4_config_diff.txt`).

### 1.2 Component matrix of the variants used in the paper

| Component | M0 | M1 | M2 | M4 | LSTM (L4) | Source |
|---|---|---|---|---|---|---|
| Learned message passing (graph layers; mean of upstream and of downstream neighbours) | ON, 3 layers | ON, 3 | **OFF**, 0 | ON, 3 | n/a | `H/model.py:97-114, 353`; `--graph-layers` `H/train.py:329` |
| Manning-type routing | ON | ON | ON | **OFF** (instantaneous sum, no travel time) | n/a | `H/model.py:358-369`; `H/channel.py:145` |
| Gauge insertion | **ON** | OFF | ON | OFF | n/a | `H/model.py:331-332, 350-351`; `H/train.py:328` |
| NWM pretraining (stage A, 5,000 updates) | ON | ON | ON (own stage A) | ON (own stage A) | **OFF** | `H/train.py:385-395`; `S/attribution_pipeline.sh:170-183`; `T/lstm_baseline.py` (no NWM target) |
| NWM prior term in stage B (weight 0.3) | ON | ON | ON | ON | OFF | `H/train.py:315, 461-467` |
| Area scaling | ON, structural: lateral inflow = runoff depth x local catchment area | ON | ON | ON | ON (`--area-output`: log specific discharge x area) | `H/model.py:417`; `T/lstm_baseline.py:277-297` |
| Per-reach static attributes | ON (48 inputs per reach) | ON | ON | ON | OFF: basin-mean attributes (48 + 7) | `H/model.py:40`; `T/lstm_baseline.py:291`; `D/basin_series.py:1-10` |
| Owner-gauge broadcast (specific discharge of the visible gauge owning each reach as step input) | ON | ON | ON | ON | OFF | `H/samples.py:124-130`; switch `--off broadcast` `H/model.py:439-448` (not used) |
| Own-gauge history at the gauge reach | ON | ON | ON | ON | ON (own gauge only) | `H/samples.py:121-123`; `T/lstm_baseline.py:83-96` |
| Gauge head (direct forecast at visible gauges) | ON | ON | ON | ON | (the LSTM is itself the forecast) | `H/model.py:246-307, 498-507` |

### 1.3 Other variants present in code (not used in the final ablation report)

| Variant | Flags | Source |
|---|---|---|
| A0 | default kernel router (27 learned shared kernels), 3 graph layers, no insertion | `S/attribution_pipeline.sh:108`; `H/model.py:196-243` |
| A1 | A0 with `--off mixing broadcast` | `S/attribution_pipeline.sh:109` |
| A2 | A0 with `--off routing` (kernel router fixed to the first tap) | `S/attribution_pipeline.sh:110`; `H/model.py:199-212` |
| A3 / A6 | `--no-nwm` from scratch (graph / lumped), 8,000 updates | `S/attribution_pipeline.sh:111, 114, 44` |
| A4 | A0 with `--nwm-weight 0` | `S/attribution_pipeline.sh:112` |
| A5 | `--lumped` (no network), own stage A `lumped_A` | `S/attribution_pipeline.sh:113, 143-145`; `H/model.py:547-628` |
| R0 | `--graph-layers 0`, kernel router, own stage A `nograph_A` | `S/attribution_pipeline.sh:115, 147-149` |
| Attention trunk | `--trunk attention` (`AttentionGNN`) | `H/train.py:320, 379-380`; `H/model.py:149-193, 538-544` |
| Legacy V1 | `--legacy` in evaluation | `H/legacy.py`; `H/evaluate.py:62-69` |
| LSTM ladder L1-L3, matched LSTM, tuned LSTM | L1 `--area-output`, L2 `--loss nse`, L3 `--antecedent` | `S/attribution_pipeline.sh:235-241` |
| `--off mixing/broadcast/routing`, `--keep`, `--head small`, `--loss bounded` | ablation and coverage switches | `H/train.py:318, 321, 326, 335`; `H/model.py:313` |

**Which to keep for the paper.** The reporting scripts load M0, M1, M2, the area-scaled LSTM (L4) and the NWM reference (`S/v3_common.py:31-47`). M4 is not yet in `MODELS`, `HYDRO_FILES` or the report scripts (`S/v3_common.py:31-39`, `S/v3_report.py:81, 196`), although its models are being trained. Keep M0, M1, M2, M4, LSTM (L4) and NWM; every A-variant, R0, L1-L3, the attention trunk and the legacy model exist in code but are not in the final ablation.

**Parameter counts** (trainable, from saved checkpoints): stage A `manning_A` 232,825; `manning0_A` 220,153; `instant_A` 229,132. Stage B: M0 932,347; M1 932,347; M2 919,675; M4 928,654 (the 3,693 fewer are the Manning channel network, 3,651, and the 42 kernel-spread values; `H/model.py:152-156`, `H/channel.py:152-156`). LSTM (L4): 810,369 (`R/inputs_audit.csv`).

---

## 2. Reach-scale architecture

Shapes use n = 58,834 reaches, T = 184 steps (144 before the issue, 40 after), F = 120 (inflow window), S = 72 (scored window), G = 128 catalog gauges, 1 step = 6 h. Window constants: `H/paths.py:32-40` (HORIZON 40, SPIN 64, ROUTE 48, INNOVATION 32, PAST 80, HISTORY 144, WINDOW 184, FLOW_STEPS 120, SCORED 72).

### 2.1 Input tensors

| Input | Name / shape | Units and normalization | Source |
|---|---|---|---|
| Observed weather (8 variables: APCP, TMP 2 m, SPFH 2 m, PRES, DSWRF, DLWRF, UGRD 10 m, VGRD 10 m) | `weather` [n, 184, 8] (float16 storage, float32 in the model); first 144 steps | six-hour values per reach from catchment-weighted AORC: precipitation summed (hours t+1..t+6), other variables averaged (hours t..t+5), all six hours required; precipitation log1p then z-score, all variables z-scored with training-period statistics | `D/schema.py:3-12`; `D/aggregate.py:245, 304-306, 350-353`; `D/prepare.py:6, 104-160`; `H/samples.py:42-46` |
| Weather validity flags | `valid` [n, 184, 8] | 0/1 | `H/samples.py:43` |
| Forecast precipitation | the precipitation channel of `weather` for the last 40 steps; other 7 variables set to 0 and invalid; known-rain flag from validity | six-hour totals (mm) from the GEFSv12 control member, mapped to reaches with the AORC weights; log1p z-score as observed rain; reforecast for training and validation, operational archive for testing | `H/samples.py:44-54`; `H/paths.py:17-20`; `H/evaluate.py:274, 287` |
| Future flag | [n, 184, 1] | 1 for the 40 steps after the issue | `H/samples.py:93`, `H/samples.py:133` |
| Season | [n, 184, 2] | sin, cos of the day of the year (period 365.25 d) | `H/samples.py:29-31` |
| Own gauge flow at a gauge reach | [n, 184, 2] | log1p(flow) z-scored with the NWM flow statistics (mean 0.3471, std 0.8551 in the saved head config) and a shown flag; non-zero only at visible gauges, only in the 144 steps before the issue | `H/samples.py:85, 117-123`; `R/config_audit.csv` (head) |
| Owner-gauge specific discharge (broadcast) | [n, 184, 2] | log1p(flow x 21.6 / gauge area) (mm per 6 h, not z-scored) and a valid flag, on every reach in the basin of the smallest visible gauge containing it | `H/samples.py:124-130`; `H/data.py:141-147` |
| Static attributes | `static` [n, 48] = 24 attributes + 24 validity masks | StreamCat and NHDPlus attributes; log1p on area, length, permeability and hydraulic-conductivity features, then z-score with training-node statistics | `D/statics.py:280-320`; `H/data.py:96` |
| Antecedent indices | `antecedent` [n, 7] | log1p EWMA of rain over 7, 30, 90, 180, 365 days and EWMA of normalized temperature over 30 and 90 days, state at the start of the issue day (earlier days only), z-scored over training days | `H/prepare.py:35-36, 141-181`; `H/data.py:168-179` |
| Owner area ratio and flag | `reach` [n, 2] | ln(basin area / owner gauge area) / 5, and 0/1 flag | `H/samples.py:142-145` |
| Observed discharge at gauges | `observed` [184, G] (NaN where missing or withheld) | m3/s six-hour means; used as step inputs only at visible gauges and only before the issue; after the issue it is only the target | `H/samples.py:67-77, 117-119` |

Step inputs: 8 + 8 + 1 + 2 + 2 + 2 = **23 channels**; reach inputs: 48 + 7 + 2 = **57** (`H/model.py:39-40`; `H/samples.py:114-146`).

### 2.2 Encoder and temporal model

- Per reach: `Linear(23 -> 128)` on the step inputs plus `MLP(57 -> 128 -> 128, SiLU)` on the reach inputs added at every step; one unidirectional LSTM layer, hidden 128, no dropout (no dropout module anywhere in the trunk), forget-gate bias initialized to 3.0; `Linear(128 -> 32)` on the last 120 steps. (`H/model.py:117-134, 126`; `--hidden 128` `H/train.py:316`.)
- Reaches are processed in chunks of 15,000 and the trunk uses gradient checkpointing in training (`H/model.py:129, 136-146`).
- Output: reach states [n, 120, 32].

### 2.3 Graph layers

| Property | Value | Source |
|---|---|---|
| Type | residual block: `h + MLP([LayerNorm(h), mean of upstream neighbours, mean of downstream neighbours])` with `MLP(96 -> 32 -> 32, SiLU)` | `H/model.py:97-114` |
| Number | 3 (M0, M1, M4); 0 (M2) | `H/train.py:329`; `S/attribution_pipeline.sh:116-119` |
| Hidden size | 32 channels | `H/model.py:326` |
| Direction | **both**: parents (upstream) and children (downstream) means at every step; not upstream-to-downstream only | `H/model.py:75-94, 112` |
| Edge features | none (unweighted means over the 60,267 directed links) | `H/model.py:82-94`; edges from `graph.npz`, `H/data.py:97-98` |
| Receptive field | 1 hop per layer, i.e. 3 hops up and down for 3 layers | `H/model.py:144-145` |
| Switch | `mixing=False` replaces neighbour means by zeros (ablation only, not used by M0-M4) | `H/model.py:99-114` |

### 2.4 Output head, units and leads

| Output | Definition | Source |
|---|---|---|
| Simulation head | runoff depth r = 0.5 softplus(w^T h + b) in mm per 6 h; lateral inflow q = r x a_j / 21.6 in m3/s (a_j local catchment area in km2, at least 0.01) | `H/model.py:417`; `H/prepare.py:51` |
| Observed head | q_obs = q exp(delta), delta = 2 tanh(w_o^T h + b_o) (zero-initialized, so delta = 0 at the start); the delta penalty weight is 0 in all M variants | `H/model.py:354-357, 418-419`; `H/train.py:317`; `R/config_audit.csv` |
| Routed flows | `sim` (routed simulation head) and `observed` (routed observed head) at every reach [n, S], `routed_gauge` at the G gauges [G, S], in m3/s six-hour means | `H/model.py:413-422, 478-485` |
| Gauge head (visible gauges) | LSTM encoder-decoder (hidden 256, log z-score scaling, with 30 days of basin-mean weather and the 48 basin-mean attributes); outputs a log correction c [g, 40] = sigmoid(keep) x last misfit + 3 tanh(Linear(decoder)); forecast = routed x exp(c) at the 40 lead steps | `H/model.py:246-307, 506-507`; `H/train.py:374-375` |
| Tail head | 0.5, 0.9, 0.99 flow quantiles [G, 40, 3] around the detached forecast (loss `nse-q`) | `H/model.py:310, 395-411` |
| Lead times | 40 six-hour steps: 6, 12, ..., 240 h (leads 24, 48, 72, 96, 120 h are indices 3, 7, 11, 15, 19) | `H/paths.py:32`; `H/train.py:63` |
| Scored window | last 72 of the 120 routed steps: 32 before the issue (history) and 40 leads | `H/paths.py:40`; `H/model.py:478` |

Output units are m3/s six-hour mean flow at every reach (not runoff per unit area); runoff depth per unit area is the intermediate quantity before routing.

---

## 3. Routing

### 3.1 What exists

Three routers share one mass-conserving form (`H/channel.py:1-36`; `H/prepare.py:1-9`):

Q_i(t) = sum over upstream j of f_ij sum_s h_ij(s) q_j(t - s), each h_ij summing to one over s = 0..47 (`P.ROUTE = 48` taps, `H/paths.py:34`), f_ij the share of j's water that reaches i (1 except below divergences, where water splits equally between branches; `D/river.py:5-9, 19-30`; `H/prepare.py:57`: `share = weight x basin_area[i] / area[j]` = f_ij). The convolution zero-pads on the left (`H/model.py:242`; `H/channel.py:292`), so the first 47 routed steps of the 120-step window have incomplete memory, which is why only the last 72 are scored.

| Router | Used by | Travel time | Source |
|---|---|---|---|
| `kernel` (default) | A-variants | 27 shared learned kernels per (river-distance bin x lake level) | `H/model.py:196-243`; `H/prepare.py:32-34, 66-74` |
| `manning` | M0, M1, M2 | per reach from channel physics at the sample's flow (below) | `H/channel.py:145-293` |
| `instant` | M4 | none: every kernel fixed to tap 0, not trainable | `H/model.py:362-367, 203-212`; test `tests/test_hydro.py:335-371` |

### 3.2 Manning-type router: equations as implemented

The Manning router is a hybrid: Manning's equation sets each pair's **delay**, and a kernel with a learned **spread** sets the attenuation. It is not a numerical solution of the kinematic-wave equation.

**Lateral inflow** (`H/model.py:417-419`):

```latex
q_j(t) = \frac{a_j}{21.6}\; 0.5\,\mathrm{softplus}\!\big(w^{\top} h_j(t) + b\big)\ \ [\mathrm{m^3\,s^{-1}}],
\qquad \tilde q_j(t) = q_j(t)\, e^{\delta_j(t)},\quad \delta_j = 2\tanh(w_o^{\top} h_j + b_o)
```

with a_j the local catchment area in km2 and h_j(t) the reach state. The factor 21.6 converts mm per six hours over km2 to m3/s (1 mm over 1 km2 in 21,600 s = 1/21.6 m3/s).

**Reach travel time** (`H/channel.py:201-217`). With reference flow \bar Q_r (m3/s):

```latex
d_r = p_r\,(\bar Q_r + 0.01)^{0.4},\qquad
v_r = \frac{d_r^{2/3}\, S_r^{1/2}}{n_r},\qquad
c_r = \begin{cases} \tfrac{5}{3}\, v_r & \text{river reach}\\ \hat c_r & \text{lake or reservoir reach}\end{cases},\qquad
t_r = \frac{L_r}{\mathrm{clip}(c_r,\,0.05,\,8)\cdot 21{,}600}\ \text{[steps]}
```

Symbols: S_r channel slope (NHDPlus, unknown slopes set to the median, clipped to [1e-5, 0.5]), L_r reach length (km x 1000 m, at least 0.01 km), d_r depth (m), v_r Manning velocity (m/s), c_r kinematic-wave celerity (m/s), 21,600 s per step (`H/channel.py:51, 65-77`). Lake or reservoir reaches are NHDPlus waterbody types `LakePond` and `Reservoir` (`H/channel.py:76`).

**Reference flow** \bar Q_i = sum_j f_ij mean_t q_j(t): the basin sum of the time-mean lateral inflow of the sample, detached from the gradient (`H/channel.py:276-280`; used at `H/model.py:420-421`).

**Pair travel time** along the main downstream path from the middle of reach j to the end of reach i (`H/channel.py:219-223`):

```latex
\tau_{ij} = \max\!\Big(0,\; T_j + \tfrac12\, t_j - T_i\Big),\qquad T_k = \sum_{m\ \text{below}\ k\ \text{on the main path}} t_m
```

**Kernels and delay classes** (`H/channel.py:48-49, 225-238`): 14 delay centres \mu_k = (0, 0.5, 1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 34, 44) steps and 3 lake classes \ell (none, 1-10, more than 10 lake reaches on the path; `H/prepare.py:33`), 42 kernels in all:

```latex
h_{\ell k}(s)=\mathrm{softmax}_s\!\Big(-\frac{(\ln(1+s)-\ln(1+\mu_k))^2}{2\,w_{\ell k}^2}\Big),\quad s=0,\dots,47,\qquad w_{\ell k}=\mathrm{softplus}(\rho_{\ell k})+0.1
```

A pair with travel time \tau_{ij} and lake class \ell is split between the two neighbouring delay classes by linear interpolation in ln(1+\tau) (clipped to [0, ln 45 - 1e-4]):

```latex
h_{ij} = (1-\omega)\,h_{\ell k} + \omega\,h_{\ell,k+1},\qquad \omega = \frac{\ln(1+\tau_{ij})-\ln(1+\mu_k)}{\ln(1+\mu_{k+1})-\ln(1+\mu_k)}
```

so the routing is differentiable in \tau (`H/channel.py:233-238, 251-274`).

### 3.3 Learned versus fixed

| Quantity | Status | Range / transform / initial value | Source |
|---|---|---|---|
| Manning's n_r | **learned** per reach from a channel network | n = 0.01 + 0.24 sigmoid(o_1 + ln((0.04 - 0.01)/(0.25 - 0.04))): range (0.01, 0.25), start 0.04 | `H/channel.py:148, 204` |
| Depth coefficient p_r | **learned** | p = 0.27 exp(2 tanh(o_2)): [0.0365, 1.995], start 0.27 (m per (m3/s)^0.4) | `H/channel.py:205` |
| Lake wave speed \hat c_r | **learned** | 1.0 exp(2 tanh(o_3)): [0.135, 7.39] m/s, start 1.0 | `H/channel.py:206` |
| Kernel spread \rho_{\ell k} | **learned** (42 values) | w = softplus(\rho) + 0.1, start 0.35 | `H/channel.py:156, 229` |
| Channel network | MLP 53 -> 64 -> 3 (SiLU); last layer zero-initialized, so all reaches start at the initial n, p, speed | inputs: ln L, ln S / 5, stream order / 8, ln(1 + basin area) / 10, lake flag, plus the 48 reach attributes | `H/channel.py:50, 152-155, 164-173` |
| Depth exponent | fixed 0.4 | | `H/channel.py:52` |
| Celerity clip | fixed [0.05, 8] m/s | | `H/channel.py:216` |
| Delay centres, taps, step | fixed (14 centres; 48 taps; 6 h) | | `H/channel.py:48`; `H/paths.py:34` |
| Slope, length, order, lake flag | fixed (NHDPlus) | | `H/channel.py:65-77` |
| Shares f_ij, basin membership | fixed | | `H/prepare.py:45-86` |

### 3.4 Time step, sub-steps, order of computation, inflow, area scaling

- **Time step** 6 h. **Sub-steps: none**; routing is one 1-D convolution over the 120-step window with 48 taps, not a recursion (`H/channel.py:282-293`).
- **Topological order.** There is no per-reach recursion at run time. Pair operators are sparse matrices over all (upstream j, downstream i) pairs, built from the main-path pairs of `main_path` (`H/channel.py:80-94, 177-198`) and rebuilt every forward from the travel times (`H/channel.py:241-274`). A topological order is used only offline, to count lake reaches on each path (`H/prepare.py:59-65`) and to build basin membership (`D/river.py:19-30`).
- **Lateral inflow** is formed as in 3.2 (runoff depth x local catchment area). **Area scaling** therefore enters twice: through a_j in q_j, and through the basin-mean weights a_j f_ij / A_i (`D/river.py:96-98`).
- **Caveat on divergences.** For pairs reached through a side branch, the pair time uses the difference of main-path travel times (`H/channel.py:219-223`), which is an approximation. The share f_ij is exact. UNKNOWN: how many pairs are affected.
- **Instant router (M4).** Q_i(t) = sum_j f_ij q_j(t): the same f_ij, all kernel weight on the first tap (`H/model.py:203-212, 362-367`); tests `tests/test_hydro.py:335, 348, 362`.

---

## 4. Gauge insertion

Implemented in `H/model.py:424-437` (operator `H/channel.py:193-197, 266-271`), called at `H/model.py:508-514`. Insertion requires the Manning router (`H/model.py:331-332`).

| Question | Answer | Source |
|---|---|---|
| What is corrected | **additive correction**, not replacement: a correction series Delta_g(t) at each visible gauge is routed down the network with the same pair travel times and kernels and added to the routed observed-head flow at every reach below the gauge and at every gauge below it; results clipped at 0 | `H/model.py:424-437, 509-514`; `H/channel.py:266-271` |
| Delta before the issue | for t in the 80 steps before the issue (PAST): Delta_g(t) = o_g(t) - \hat Q_g(t) where o_g is observed and finite and the gauge is visible; 0 where the observation is missing | `H/model.py:430-433` |
| Delta after the issue | for the 40 lead steps: Delta_g(t) = \hat y_g(t) - \hat Q_g(t), with \hat y_g the gauge-head forecast of the same model (not an observation) and \hat Q_g the routed observed head, for visible gauges only | `H/model.py:434-436` |
| Observations used | only flow at or before the issue (mask `past`); observations after the issue never enter; test `tests/test_hydro.py:268` | `H/model.py:431-432` |
| Missing observations | no interpolation: the correction is 0 at those steps | `H/model.py:432-433` |
| Availability rule | a gauge is visible only if it is not withheld and has at least 8 finite six-hour values in the last 32 steps before the issue: **confirmed** | `H/samples.py:33-36`; test `tests/test_hydro.py:299-324` |
| Blending or gating weight | **none**: full-weight correction (weight 1); no learned gate | `H/model.py:436-437` |
| Gauge's own reach | **not** corrected by its own Delta: pairs from a gauge to itself are excluded (`members != rows`); a visible gauge keeps its own gauge-head forecast at the lead steps | `H/channel.py:194-197`; `H/model.py:513-514`; test `tests/test_hydro.py:206` |
| Reaches below the gauge | every reach and gauge downstream of a visible gauge receives its routed correction; reaches not below any gauge are unchanged; test `tests/test_hydro.py:236` | `H/channel.py:193-197`; `H/model.py:508-514` |
| In training | active in M0 and M2 with the training visibility draw (below), so the correction is learned end to end | `H/train.py:106-108, 383` |
| Mass | the correction is water added or removed at the gauge and routed with mass-conserving kernels; test `tests/test_hydro.py:206` | `H/channel.py:30-35` |

Equations as implemented (visible gauge set V; \hat Q the routed observed head before insertion; the correction is routed by the pairs from each gauge reach to every reach below it, with shares f and kernels h_{gi} as in 3.2):

```latex
\Delta_g(t)=\begin{cases} o_g(t)-\hat Q_g(t) & t\le 0,\ o_g(t)\ \text{observed}\\ 0 & t\le 0,\ o_g(t)\ \text{missing}\\ \hat y_g(t)-\hat Q_g(t) & t=1,\dots,40\end{cases}\qquad g\in V,
\qquad
Q'_i(t)=\max\!\Big(0,\ \tilde Q_i(t)+\sum_{g\in V,\,g\ne i}\ f_{gi}\sum_{s}h_{gi}(s)\,\Delta_g(t-s)\Big)
```

For gauges, the output is \max(0, \hat Q_g + corrections from other visible gauges above it), except that a visible gauge at the lead steps keeps \hat y_g (`H/model.py:511-514`).

### 4.1 Exclusion of withheld gauges (for Text S1)

| Stage | Mechanism | Source |
|---|---|---|
| Which gauges are withheld | `splits.withheld`: fold gauges (random scheme) or every catalog gauge inside the blocked basin, scored or not | `H/splits.py:81-96` |
| Inputs and targets in training | `Samples.build` / `gauge_view` set the withheld gauges' observed flow to NaN at every step and mark them not visible, so they are neither step inputs, nor owner-broadcast sources, nor gauge-head history, nor insertion sources, nor targets | `H/samples.py:38-77`; `H/train.py:106-109` |
| Loss | restricted to `trained = gauges.trained & ~withheld` | `H/train.py:414-415, 472` |
| Validation and model selection | the same mask; visibility from `eligible(step, withheld)` | `H/train.py:249-251, 271-278` |
| Test | `withheld[k]` for each fold model; view built with `gauge_view(step, eligible(step, withheld[k]), withheld[k])` | `H/evaluate.py:220-225, 285` |
| LSTM | withheld gauges' observed flow, normalized flow and finite flag are set to NaN/0/False before any use; selection scores of withheld gauges become NaN | `T/lstm_baseline.py:208-216`; `T/selection.py:25`; test-time history hidden `E/benchmark.py:312-314` |
| Unit tests | `test_gauge_view_hides_withheld_gauges_and_shows_only_eligible_ones` (`tests/test_hydro.py:299`), `test_withheld_gauge_observation_never_enters_insertion` (`:250`), `test_insertion_uses_only_flow_before_the_issue` (`:268`) | `tests/test_hydro.py` |
| Leakage check on real fold models | forecasts do not change when the withheld gauge's true flow or the visible gauges' future flow is altered | `S/v3_leak_check.py:1-8`; `R/leak_check.csv` |

Statistics computed on all gauges (training-period mean, std, 90th percentile, observation counts) exist in `H/data.py:133-139`; they enter the loss only through the `trained & ~withheld` mask (`H/train.py:414-415`) and `gauge_std_raw` of non-masked gauges.

---

## 5. LSTM baseline (audit)

The paper's LSTM is the area-scaled "L4" model, `train-lstm --area-output --loss nse --antecedent` with the arguments below (`S/attribution_pipeline.sh:235-254`; saved config `<data>/models/attr/ladder_L4_s1/run_config.json`).

### 5.1 Inputs

| Input | Definition | Source |
|---|---|---|
| Weather before the issue | 120 steps (30 days) x 16 channels: 8 AORC variables + 8 validity flags, **basin-mean** (area-weighted over the gauge's upstream basin) | `D/schema.py:13`; `D/basin_series.py:1-10, 81-95`; `T/lstm_baseline.py:89-96` |
| Observed flow history | 120 steps of z-scored log1p flow and an availability flag, **own gauge only**; hidden in 57.5% of training samples | `T/lstm_baseline.py:90-96, 379` |
| Forecast precipitation | 40 leads x 2 channels (basin-mean rain value, known share); GEFSv12 reforecast in training (`--forecast-fraction 1`, no unknown-rain or cut draws), operational archive at test | `T/lstm_baseline.py:98-100`; `S/attribution_pipeline.sh:250`; `E/benchmark.py:311-316` |
| Static attributes | 48 basin-mean attributes (24 attributes + 24 masks) | `D/basin_series.py:95`; `T/lstm_baseline.py:291` |
| Antecedent indices | 7 basin-mean indices appended to the attributes (static_dim 55) | `T/lstm_baseline.py:76-81, 103-104` |
| Season | none | `src/<pkg>/models/lstm.py:16-47` (18 dynamic channels = 16 weather + flow + flag) |
| Any NWM-derived input | none as a feature. The normalization constants of the observed-flow input (log-flow mean and std) are read from the shared dynamic normalization of the NWM-based flow arrays (`mean[8]`, `std[8]`); the output normalization is fitted on observed training flow (specific discharge) | `T/lstm_baseline.py:204-205, 277-286` |
| Observations of other gauges | none | `T/lstm_baseline.py:83-105` |

### 5.2 Settings

| Setting | Value | Source |
|---|---|---|
| Lookback / horizon | 120 steps / 40 steps | `D/schema.py:13` |
| Target | six-hour observed flow at the next 40 steps at the gauge; the network outputs normalized log specific discharge, flow = expm1(...) x area / 21.6 | `T/lstm_baseline.py:101, 277-297` |
| Area scaling | ON: NHDPlus total drainage area `totdasqkm` of the gauge reach (km2); the M family uses the routing basin area (`routing.npz`) | `T/lstm_baseline.py:279-286`; `H/data.py:132` |
| Architecture | encoder LSTM (input 18 + 55, hidden 256), linear state handoff, decoder LSTM (input 2 + 55 + 16, hidden 256), lead embedding 16, dropout 0, linear head | `src/<pkg>/models/lstm.py:16-47`; `T/lstm_baseline.py:292` |
| Loss | `--loss nse`: sum over leads of weight x ((forecast - truth) / (gauge std + 0.1))^2; no log term, no high-flow weight; lead weights "balanced" (three bands 6-24 h, 30-72 h, 78-240 h each given equal total weight) | `T/lstm_baseline.py:415-426`; `T/forecast.py:43-55` |
| Optimizer / schedule | AdamW (weight decay 1e-4), lr 5e-4, warmup min(500, updates/10), cosine to 5% of lr, gradient clip 1.0, batch 48 (issue, gauge) pairs drawn gauge-balanced | `T/lstm_baseline.py:298-307, 372-378, 429`; `S/attribution_pipeline.sh:252-254` |
| Training volume | 3,300,807 sample budget -> 68,767 updates; 3,300,807 usable training pairs | `S/attribution_pipeline.sh:252`; `run_config.json` |
| Selection | 10 evaluations (every 6,877 updates); score = mean of the shown-history and hidden-history selection scores, each the mean over the 6-120 h leads of the median NSE over CAMELS-H gauges (min 100 values), on 2019 validation issues of the validation split (356 issues) | `T/lstm_baseline.py:337-369, 444-456`; `T/selection.py:1-33`; `run_config.json` |
| Selected update | gauged model: 13,754 (second of ten); fold 1: 34,385; blocked basin 1: 61,893 | `<data>/models/attr/ladder_L4_s1*/validation_progress.csv` |
| Hardware / time | one GPU per run; 17.0 min logged for the gauged model's 68,600 updates | `S/attribution_pipeline.sh:242-247`; `<data>/logs/attr/ladder_L4_s1.log` |

### 5.3 Is the "same inputs" sentence true?

**No: the LSTM does not receive exactly the same inputs as the M-family models.** It sees the same weather variables and the same forecast-rain source, but as basin means over a 30-day window, and it differs in the following.

| Difference | M family | LSTM (L4) | Source |
|---|---|---|---|
| Spatial detail | per-reach inputs over the whole network | one basin mean per gauge | `H/model.py:117-146`; `D/basin_series.py:1-10` |
| Weather window | 144 steps (36 days) | 120 steps (30 days) | `H/paths.py:37`; `D/schema.py:13` |
| Observations of other gauges | yes: visible gauges' own and owner-broadcast flow, gauge-head history, insertion (M0, M2) | **none** | `H/samples.py:117-146`; `H/model.py:498-514` |
| NWM | pretraining (5,000 updates) and a prior term (weight 0.3) in stage B | **none** | `H/train.py:385-395, 461-467` |
| Season (sin, cos) | yes | no | `H/samples.py:29-31`; `src/<pkg>/models/lstm.py` |
| Network structure and routing | river graph, routing | none | |
| Training loss | NSE* on gauge forecast and routed forecast + pinball tail + 0.3 NWM prior; equal lead weights | NSE*; lead weights by three bands | `H/train.py:184-213, 471-474`; `T/lstm_baseline.py:415-418` |
| Training volume | 3,000 updates x 8 whole-network samples (24,000 draws of 6,931 issues) | 68,767 updates x 48 pairs | `H/train.py:403-404`; `run_config.json` |
| Validation set | 355 issues (2019-01-01 to 2019-12-21) | 356 issues | `H/train.py:130-133`; `run_config.json` |
| Gauge visibility in training | 15% of samples none, otherwise each eligible gauge shown with probability 0.5 (overall hidden about 57.5%) | own history hidden in 57.5% of samples | `H/train.py:106-108`; `T/lstm_baseline.py:379` |
| Gauges in the training loss | gauges with more than 1,460 values before 2019, not withheld | every (issue, gauge) pair with an observation at the issue and a target before 2019 | `H/data.py:139`; `T/lstm_baseline.py:218-227` |
| Output | runoff depth x area, routed (a visible gauge's forecast is the gauge head's) | specific discharge x area | `H/model.py:417, 506-507`; `T/lstm_baseline.py:294-297` |
| Parameters | M0 932,347 | 810,369 | `R/inputs_audit.csv` |

At test the LSTM sees its own gauge's history unless the gauge is withheld (`E/benchmark.py:312-314`), as does an M model for a visible gauge (`H/evaluate.py:285`).

---

## 6. NWM reference

| Role | Product | Source |
|---|---|---|
| (a) Pretraining target | NWM v3.0 **retrospective** simulation streamflow at every reach, hourly values averaged to six-hour means over [t, t+6 h) (all six hours required), stored as log1p z-scored `flow` arrays with validity; stage-A loss on the 72-step scored window (8 days before and 10 days after each issue) | `D/aggregate.py:245, 320-324, 350-353`; `D/prepare.py:129, 290`; `H/train.py:159-165, 452-454` |
| Also in stage B | the same retrospective, as a prior term (weight 0.3) on the simulation head everywhere and on the observed head at 10% weight inside basins of visible gauges | `H/train.py:456-467` |
| (b) Reach-level reference | the retrospective at every reach for every second test issue up to 2023-01-20 (`P.RETRO_UNTIL + 19 days`); the NWM operational forecast at every reach is a further reference (`S/retrospective_comparison.py`, named in `H/evaluate.py:15-20, 99-102`) | `H/evaluate.py:15-20, 99-117, 276-281`; `H/paths.py:30` |
| (c) Gauge-level forecast reference | NWM operational medium-range forecast, deterministic member `medium_range_mem1`, 00Z cycle, `channel_rt_1` streamflow at each gauge's reach (feature_id = NHDPlus COMID) | `src/<pkg>/downloads/nwm_forecasts.py:1-18`; `H/paths.py:15` |

**Alignment to six-hour means** (`src/<pkg>/downloads/nwm_forecasts.py:196-203`; `FLOW_LEADS = 3, 6, ..., 240 h`, line 36): the archive output is instantaneous; the six-hour mean over [t+6k, t+6k+6) is the trapezoid of the values at 6k, 6k+3 and 6k+6 h, and for k = 0 the mean of the 3 h and 6 h values (the 0 h value is not in the files). **The forecast is therefore sampled every 3 h in all versions, not hourly.**

**Alignment to issue times** (`E/benchmark.py:70-81`; `H/evaluate.py:99-110`): one record per 00Z issue date; test issues are days from 2020-01-01 that have both an NWM forecast and GEFSv12 operational rain.

**Missing issues and files** (`src/<pkg>/downloads/nwm_forecasts.py:177-182`): missing archive files are recorded and left as NaN; days without any NWM forecast are not test issues; an issue with NaN NWM values at a gauge or lead is dropped from the scored set for **all** models at that gauge and lead because the scoring mask requires a finite NWM forecast (`H/evaluate.py:265`; `E/benchmark.py:292`). Version changes (v2.0 to 2021-04-20, v2.1 to 2023-09-20, then v3.0) are recorded per day but not used in scoring (`src/<pkg>/downloads/nwm_forecasts.py:17-18, 166-183`).

---

## 7. Training

### 7.1 Stages

| Item | Stage A (NWM pretraining) | Stage B (forecast) | Source |
|---|---|---|---|
| Data | no gauge observations | 30 days of flow at a random half of the eligible gauges (none in 15% of samples); loss over all training gauges | `H/train.py:6-19, 106-109` |
| Loss | bounded NSE of the simulation head against the NWM retrospective at every reach (log(1 + MSE/var) and log(1 + MSLE), halved each, per reach) | `nse-q` (below) | `H/train.py:136-165, 471-474` |
| Updates | 5,000 | 3,000 | `S/attribution_pipeline.sh:103, 170, 176` |
| Learning rate | 5e-4, warmup 500 | 3e-4, warmup 100 | `S/attribution_pipeline.sh:103, 170` |
| Samples per update | 8 | 8 | `S/attribution_pipeline.sh:102, 143-145, 172-183` |
| Validation | every 1,000 updates on every 8th 2019 issue; selection on the NWM loss | every 500 updates on every 2019 issue (355) | `H/train.py:130-133, 253-263` |
| What is frozen | nothing | nothing: AdamW over all parameters; only the stage-A parts `trunk.`, `simulation.`, `router.` are loaded from stage A, heads start fresh | `H/train.py:392-395, 402` |

**Stage-B loss (`--loss nse-q`)** (`H/train.py:205-213, 471-474`): loss = forecast term + 0.3 x NWM prior + 1.0 x tail term + 0.0 x delta penalty, with forecast term = 0.5 (NSE*(gauge forecast) + NSE*(routed gauge)) over the 40 leads, NSE* per gauge = MSE / (std + 0.1)^2 with std the gauge's training-period std (`H/train.py:184-191`; `H/samples.py:94-96`), tail term the pinball loss of the 0.5, 0.9, 0.99 quantiles scaled the same way. **Weighting:** equal weight per gauge (mean over gauges having at least 10 of 40 valid leads); equal weight per lead; no area weighting; no flood weighting in `nse-q` (the x3 above the 90th percentile belongs to the unused `bounded` loss, `H/train.py:168-181`). The prior is 0.5 (NWM loss of the simulation head + NWM loss of the observed head at weight 1 - 0.9 x owned) (`H/train.py:459-465`). Gauges in the loss: `trained & ~withheld` (`H/train.py:414-415`).

### 7.2 Optimization

| Item | Value | Source |
|---|---|---|
| Optimizer | AdamW, weight decay 1e-4 | `H/train.py:402` |
| Schedule | linear warmup then cosine, floor at 1% of the base rate | `H/train.py:442-444` |
| Batch | 8 whole-network samples per update (`--accumulate` is per process: 2 per GPU x 4 GPUs, 4 x 2, or 8 x 1; the sample list is the same) | `H/train.py:403-404`; `<data>/models/attr/M0_s1/arguments.json` (accumulate 2, 4 GPUs) |
| Gradient clipping | global norm 1.0 | `H/train.py:483` |
| Mixed precision | bf16 autocast in training, validation and evaluation (routing in float32) | `H/train.py:252, 449`; `H/evaluate.py:257`; `H/model.py:235, 241` |
| Epochs | not epoch-based: 24,000 (stage B) or 40,000 (stage A) issue draws, uniformly with replacement from 6,931 training issues | `H/train.py:122-127, 403-404` |
| Validation set | 2019 reforecast issues, 2019-01-01 to 2019-12-21 (355 days) | `H/train.py:130-133` |
| Selection rule | stage B `--select gauges`: mean of (1) median NSE at training gauges with history shown and (2) with every history hidden, each over leads 24-120 h; the default `all` adds a third score against the NWM retrospective but is not used (`--select gauges` in `B_COMMON`) | `H/train.py:237-241, 287-294, 333`; `S/attribution_pipeline.sh:102` |
| Checkpoints | every validation (500 updates): 6 candidates per stage-B run; `best.pt` is the highest score | `H/train.py:497-521` |
| Seeds | `--seed 1` for every run; torch seed = seed + rank; issue list seeded by seed + 1000 x fold | `H/train.py:324, 363, 404`; `S/attribution_pipeline.sh:127` |

### 7.3 Seeds, hardware and time

- **Seeds run:** one (seed 1) per model and protocol (`<data>/models/attr/*_s1*`). Seed 2 on the blocked folds is planned and has not been run; no other seeds of M0-M4 or the L4 LSTM exist.
- **Hardware:** the machine has four NVIDIA L40S GPUs (46 GB); the pipeline script comments give GPU-hour estimates for one L40S (`S/attribution_pipeline.sh:24-27`). The checkpoints record only the number of GPUs (`gpus` field). UNKNOWN whether any reported model was trained on another system (the script has switches for a different cluster, `S/attribution_pipeline.sh:33, 47-48`).
- **Time** (wall-clock minutes of the logged runs, number of GPUs): stage A `manning_A` 238.5 (4), `manning0_A` 544.1 (1), `instant_A` 358.6 (2). Stage B: M0 gauged 177.8 (4), folds 156.6-595.6 (1 or 4); M1 162.2-167.4 (4); M2 101.6-119.8 (4); M4 folds 246.1-247.0 (2); blocked M0 177.7-178.0 (4), M1 331.9-332.6 (2), M2 118.9-119.4 (4) (`<data>/models/attr/*/validation.json`, `gpus`). Roughly 7-12 GPU-hours per stage-B model.

### 7.4 Number of trained models (DONE markers on 2026-11-21)

| Protocol | M0 | M1 | M2 | M4 | LSTM (L4) |
|---|---|---|---|---|---|
| Gauged (1 all-gauge model) | 1 | 1 | 1 | 0 (not planned) | 1 |
| Random folds (5) | 5 | 5 | 5 | 5 | 5 |
| Blocked basins (3) | 3 | 3 | 3 | 0 done: `S1_M4_s1_b1` running, b2 and b3 queued | 3 |
| Stage A (shared by all folds and protocols of a variant) | `manning_A` (M0, M1) | | `manning0_A` | `instant_A` | none |

Stage A has no gauge data and is selected on the NWM loss, so one stage-A model per variant family is used for every fold (`H/train.py:253-263`; `S/attribution_pipeline.sh:170-183`).

---

## 8. Metrics and aggregation

| Item | Definition as implemented | Source |
|---|---|---|
| Per gauge and lead | computed over **all test issues** (1,917 for the gauge protocols) of the pairs where the observation, the issue-time observation and the NWM forecast are finite (and the gauge is a CAMELS-H site); sums over issues, so NSE is pooled over issues | `H/evaluate.py:265, 300-307`; `E/benchmark.py:290-292`; `S/v3_common.py:100-120` |
| Scored gauge | at least 100 pairs, positive observed mean, observed standard deviation at least 1e-3 m3/s | `S/v3_common.py:117`; `E/hydrology.py:105-120`; `H/evaluate.py:170` |
| NSE | 1 - MSE / Var(obs), population variances (m3/s) | `S/v3_common.py:114-115`; `E/hydrology.py:125` |
| KGE and components | Gupta et al. (2009): r = Pearson correlation, alpha = std_sim / std_obs, beta = mean_sim / mean_obs, KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2); not the 2012 variant; checked against an independent numpy implementation (1,010 values, max abs difference 1.8e-14) | `S/v3_common.py:114-119`; `S/v3_kge_highflow.py:1-8`; `R/kge_validation.csv` |
| Normalized or log variants | `normalized_log_mse` exists in `E/metrics.py:35-39` but is not used by the v3 reports; no log-NSE | `E/metrics.py` |
| Bias | beta (mean ratio); peak-flow bias, timing and magnitude error for flood events (observed 10-day maximum at least the 2-year flow = median of annual maxima over water years with at least 70% observed, at least 10 such years) | `S/v3_kge_highflow.py:70-125` |
| Skill relative to NWM or LSTM | **no skill-score ratio**; differences of NSE (paired per gauge, or difference of medians) | `S/v3_stats.py:44`; `S/v3_stats.py:175` |
| Lead summary | leads 24, 48, 72, 96, 120 h (indices 3, 7, 11, 15, 19): per gauge, mean over the five leads of per-lead NSE (paired tests); for a median, the median over gauges per lead, then the mean over the five leads | `S/v3_common.py:29-30, 123-135` |
| Gauge aggregation | median over gauges (headline); the gauge set is the gauges scored by all models of the comparison (101, 98, 29) | `S/v3_stats.py:25-31`; `S/v3_common.py:33` |
| Missing observations | masked pairwise (no filling); a gauge-lead is NaN if it fails the scoring rule | `S/v3_common.py:100-120` |

**Uncertainty and tests** (all `S/v3_stats.py` unless noted): gauge-level bootstrap of the mean paired difference, 10,000 resamples, seed 0, 5-95% percentile interval (`S/v3_common.py:199-206`; `v3_stats.py:45`); lead curves: bootstrap of the difference of medians and of the mean difference, 4,000 resamples, 2.5-97.5% (`v3_stats.py:175-178`); Wilcoxon signed-rank test on per-gauge days 1-5 differences (zeros dropped, more than 5 pairs; `S/v3_common.py:220-224`); paired TOST on the mean difference with margins +-0.01 and +-0.02 NSE, larger of the two one-sided t-test p-values (`S/v3_common.py:226-234`; `v3_stats.py:49`); Holm correction across all six pairs per protocol for the difference tests and across the three equivalence pairs per margin (`v3_stats.py:22-23, 51-54`; `S/v3_common.py:237-247`); area strata at 200, 600, 2,000 km2 (`v3_stats.py:20`). **Seeds are not used for uncertainty: one seed per model.**

---

## 9. Nested versus headwater gauges

Computed from `H/data.py:117-139` basins and `H/evaluate.py:127-137` (`nested = another catalog gauge in the upstream basin`); the evaluation set is the 101 gauges scored in the gauged protocol.

| Definition | Nested (count, median area km2) | Headwater (count, median area km2) |
|---|---|---|
| (a) another of the **128 matched gauges** upstream | **44**, 1,237.9 | **57**, 275.2 |
| (b) another of the **101 evaluation gauges** upstream | **39**, 1,658.7 | **62**, 276.3 |

**Which definition the insertion model can use at test time.** Definition (a). A model sees every catalog gauge that is not withheld and has at least 8 values in the last 32 steps, not only the 101 evaluation gauges (`H/samples.py:33-36`; `H/evaluate.py:285`).

| Protocol | Scored gauges | With at least one non-withheld upstream catalog gauge | Of these with an upstream evaluation gauge |
|---|---|---|---|
| Gauged (nothing withheld) | 101 | 44 | 39 |
| Random hold-out (the scored gauge's fold withheld) | 98 | 40 | 36 |
| Blocked (the whole basin withheld) | 29 | **0** | 0 |

(`R/insertion_upstream_visible_summary.csv` gives 44, 40, 0 and the share of issues with a visible upstream gauge: 39.2%, 37.2% of issues for the gauged and held-out protocols.) Insertion can therefore act on at most 44 (gauged), 40 (held-out) and 0 (blocked) scored gauges, and only at issues where the upstream gauge meets the 8-of-32 rule.

---

## 10. Figure facts

### 10.1 Component on/off matrix (for a schematic)

Rows are the components of 1.2: graph layers (3/3/0/3), Manning routing (Y/Y/Y/N), insertion (Y/N/Y/N), NWM pretraining (all Y; own stage A for M2 and M4), NWM stage-B prior (Y), area scaling (Y), per-reach statics (Y), owner broadcast (Y), gauge head (Y) for M0/M1/M2/M4 respectively. Parameter counts: 932,347 / 932,347 / 919,675 / 928,654.

### 10.2 Tensor flow from inputs to forecast (M0)

1. Inputs: step tensor [58,834, 184, 23], reach tensor [58,834, 57] (`H/samples.py:114-146`).
2. Trunk: `Linear(23->128)` + `MLP(57->128->128)` -> LSTM(128) over 184 steps -> `Linear(128->32)` on the last 120 steps -> [58,834, 120, 32] (`H/model.py:131-134`).
3. Graph layers: 3 x residual block mixing the means of upstream and downstream neighbours -> [58,834, 120, 32] (`H/model.py:144-145`).
4. Heads: simulation `Linear(32->1)` -> runoff depth (mm per 6 h) -> x a_j / 21.6 -> lateral inflow q [58,834, 120] m3/s; observed head -> delta [58,834, 120], q_obs = q exp(delta) (`H/model.py:417-419`).
5. Manning router: travel times from the channel network at the reference flow; pair operators; routed `sim` and `observed` [58,834, 120], `gauge` [128, 120] (`H/model.py:420-422`).
6. Scored window: last 72 steps (`H/model.py:478-485`).
7. Gauge head at visible gauges: reads the last 120 steps of observed flow, the routed and simulated flow over the last 32 steps and the 40 leads, basin-mean forecast rain and weather -> correction c [g, 40]; forecast = routed x exp(c) (`H/model.py:498-507`).
8. Insertion (M0, M2): Delta [G, 120] from observed-minus-routed before the issue and gauge-forecast-minus-routed after it, routed down the network -> added to `observed` [n, S] and `routed_gauge` [G, S] (`H/model.py:508-514`).
9. Output: `gauge` [128, 72] (forecast at lead steps 32-71), `reach` [n, 72] when `full` (`H/model.py:515-535`); spread head or tail head [G, 40, 3] (`H/model.py:516-530`).

### 10.3 Example reach with a gauge upstream (computed from the repository)

Upstream gauge USGS 03453500 (French Broad River at Marshall; reach index 54,338; COMID 22161966; basin area 3,450.7 km2) and downstream gauge USGS 03455000 (French Broad River near Newport; reach index 54,672; COMID 22162876; basin area 4,800.5 km2; outlet of blocked basin B1).

| Quantity | Value | How it was obtained |
|---|---|---|
| Main-path reaches from the upstream gauge reach to the downstream gauge reach | 102 (including both ends) | `H/channel.py:80-94` applied to the network |
| River length below the upstream gauge reach to the end of the downstream reach | 77.8 km | `H/channel.py:65-77` |
| Share f_ij | 1.0 | `H/data.py:102` routing table |
| Pair travel time with the trained M0 channel network at the mean NWM flow (sum over path reaches below the upstream gauge reach plus half the upstream gauge reach) | 1.50 steps = 9.0 h | `<data>/models/attr/M0_s1/best.pt`, `H/channel.py:209-223` |
| Reaches within 3 hops upstream of the downstream gauge reach (message-passing receptive field, including itself) / downstream | 10 / 4 | `H/model.py:75-94, 144-145` |
| Is the upstream gauge reach within 3 hops of the downstream gauge reach | **No**: message passing does not carry the upstream gauge's observation to the downstream gauge | same |

- **Reaches that pass messages to the downstream gauge reach:** the reaches within 3 hops up and down, by 3 layers; upstream neighbours and downstream neighbours both.
- **Observation inserted:** if the upstream gauge is visible (at least 8 of the last 32 values), its observed flow of the previous 80 steps minus the routed observed head there is routed down the 101 reaches below it, arriving at the downstream reach after about 9 h at mean flow; after the issue the gauge-head forecast minus the routed flow at the upstream gauge is routed the same way (M0, M2 only). The upstream gauge's own reach is not corrected by its own Delta; the downstream gauge, if itself visible, keeps its own gauge-head forecast at the lead steps and takes the corrected routed flow only at pre-issue steps (`H/model.py:511-514`).
- **Also available to every reach in the upstream gauge's basin** (all M variants): the owner-broadcast specific discharge of that gauge as a step input (`H/samples.py:124-130`).

---

## 11. Open issues

### 11.1 Code versus earlier reports

1. **LSTM loss.** `R/inputs_audit.csv` states for the LSTM "nse on gauge flow; high-flow weight 2.0, raw weight 1.0". With `--loss nse` the code applies no high-flow weight and no log term (`T/lstm_baseline.py:415-418`); the value 2.0 is only the unused default argument stored in `run_config.json`. The earlier report is misleading on this point.
2. **LSTM normalization.** `R/inputs_audit.csv` says "NWM: not used (no NWM pretraining, no NWM input)". The LSTM does read NWM-derived normalization constants for its flow input (`T/lstm_baseline.py:204-205`); they are statistics, not features, but the statement is not literally true.
3. **Samples per update.** `R/config_audit.csv` lists `accumulate 2` for M0, M1 and M2; this is per process on 4 GPUs. The effective batch is 8 whole-network samples (`H/train.py:403-404`).
4. **Selection rule wording.** The module docstring of `H/train.py:30-32` describes selection as the mean of three scores including the NWM retrospective; the pipeline passes `--select gauges` (two scores) (`S/attribution_pipeline.sh:102`), and the saved checkpoints record `select = gauges`.
5. **Router description.** The docstring of `H/model.py:1-25` and `H/prepare.py:1-9` describe 27 learned kernels (the `kernel` router); the paper's models use the Manning router with 42 kernels and a learned channel network (`H/channel.py:1-36`).
6. **Section 3 versus code.** (a) Section 3 says models are trained "with all gauges that have training-period records"; the loss uses gauges with more than 1,460 values before 2019 (`H/data.py:139`), i.e. 122, not 123. (b) The 8-of-32 rule, the 25% area cap, the 8-30 gauge range, the 36-day look-back, the 40 leads, the 6,931 / 355 / 1,917 issue counts and the NWM 3-hourly sampling agree with the code (`H/samples.py:36`; `H/splits.py:26-27`; `H/paths.py:32-37`; `H/train.py:130-133`; `src/<pkg>/downloads/nwm_forecasts.py:36`).
7. **Validation sets differ between the M family and the LSTM** (355 versus 356 issues; different issue rules: all reforecast days of 2019 versus the validation split's issues with an observed latest value and target at a selection gauge) (`H/train.py:130-133`; `T/lstm_baseline.py:233-245`).

### 11.2 Dead or unused options

`--trunk attention`, `--lumped`, `--lumped-reaches`, `--no-nwm`, `--keep`/`--keep-seed`, `--head small`, `--loss bounded`, `--whole`, `--off` (mixing, broadcast, routing), router `kernel`, `--legacy` (V1), `AttentionGNN` (cannot take a router, insertion or ablations; `H/model.py:538-544`); the x3 flood weight (`bounded` loss only); the `delta` penalty (weight 0 in all M variants); the M3 colour entry (`S/v3_figures.py:19`). The routed-observed head `delta` is still computed and bounded to (-2, 2) in every M variant.

### 11.3 Hard-coded paths and magic numbers that affect results

- Data root from an environment variable with a fixed default path (`H/paths.py:8`); model folder `<data>/models/attr` fixed in `S/v3_common.py:24`; absolute data paths in `S/attribution_pipeline.sh:33-42`.
- Window and routing constants: 8-day history for the gauge head's routed tail (INNOVATION 32), 80-step insertion window (PAST), 30-day gauge history (120), 48 taps, SPIN 64 (`H/paths.py:32-40`; `H/model.py:249-253, 459-463`).
- Model constants: runoff factor 0.5 softplus, delta bound 2, learned correction bound 3, misfit clamp +-5, keep initialization 3.0 - 0.25 x lead, forget bias 3.0, EPS 0.01, conversion 21.6 (`H/model.py:38, 126, 275, 303-306, 417-418`).
- Routing constants: initial n 0.04, depth coefficient 0.27, lake speed 1.0 m/s, depth exponent 0.4, celerity clip [0.05, 8], delay centres, spread init 0.25 + 0.1 (`H/channel.py:48-52, 148, 156, 216`).
- Loss and training constants: NSE* offset 0.1, NWM prior weight 0.3, tail weight 1.0, 15% no-gauge samples, visibility probability 0.5, `Gauges.trained` threshold 1,460 values, quarter-of-steps validity rule, 8 samples per update, learning rates, 3,000 and 5,000 updates (`H/train.py:106-108, 184-191, 315, 319`; `H/data.py:139`).
- Scoring: 100 pairs, std >= 1e-3, validation `min_count` 30 (`H/train.py:288`), unlike the test-time 100.
- Block selection: BLOCKS 3, 8-30 gauges, 25% area, deterministic order (`H/splits.py:25-27, 47-56`).

### 11.4 Threats to validity to disclose

1. **The LSTM is not an input-matched baseline** (section 5.3): no other gauges' observations, no NWM pretraining or prior, no season input, basin-mean 30-day inputs, different loss and lead weights, different training volume, validation set and checkpoint-selection granularity. Selected LSTM checkpoints are early in training (13,754 of 68,767 for the gauged model).
2. **NWM enters the M models in three places**: the stage-A target, the stage-B prior and, if `--select all` were used, selection. Whether the NWM retrospective used observations of the withheld gauges (calibration) cannot be determined from the repository: UNKNOWN. It is therefore a possible indirect pathway of information about withheld gauges, for the M family only.
3. **Observations of non-withheld gauges reach the M models by four routes** (own channel, owner broadcast, gauge-head history, insertion); only insertion is ablated by M0 versus M1, and M2 removes the graph layers but keeps broadcast, insertion and the gauge head.
4. **Insertion after the issue injects the model's own gauge-head forecast**, not observations (`H/model.py:434-436`).
5. **Blocked protocol.** Three basins; downstream gauges stay in training and see the withheld basin's outflow (`H/splits.py:9-10`); insertion cannot act (0 of 29 scored gauges have a non-withheld upstream gauge).
6. **Visibility at test differs from training**: all eligible gauges are visible at test, a random half in training (`H/evaluate.py:285`; `H/train.py:106-109`). In the gauged protocol the scored gauge's own history is visible.
7. **Scoring conditions on an observation at the issue time and an NWM forecast** for every model and the NWM (`H/evaluate.py:265`; `E/benchmark.py:292`).
8. **Model selection** uses 2019 scores at training gauges only (`trained`, non-withheld), with history shown and hidden; no withheld gauge is used, but the hidden-history score is measured on gauges the model was trained on (`H/train.py:249-251, 287-289`).
9. **One seed per model and protocol**; bf16 inference and training are not bitwise reproducible (`H/evaluate.py:257`, `H/train.py:449`); best checkpoint picked among 6 (M) or 10 (LSTM) candidates; M2 fold 3 selected the 500-update checkpoint (`<data>/models/attr/M2_s1_f3/best.pt`).
10. **Training draws with replacement** from 6,931 reforecast issues (24,000 draws in stage B, 40,000 in stage A); one forecast source (the GEFSv12 control member, no ensemble) (`H/train.py:122-127`).
11. **Manning router approximations**: pair times through divergences use main-path differences; reference flow is the sample's time mean; the router is not a PDE solution (`H/channel.py:219-223, 276-280`).
12. **A restricted dam inventory is used only to stratify results** (`H/evaluate.py:22-24, 127-137`); no model, loss, input or selection step reads it (no reference in `H/model.py`, `H/train.py`, `H/samples.py`, `H/prepare.py`, `H/data.py`, `H/channel.py`, `T/lstm_baseline.py`). The dam-category results cannot be reproduced from public data.
13. **Training period and gauges.** Training issues are 2000-01-01 to 2018-12-22 only (reforecast availability); three scored gauges have no training-period data (`H/train.py:122-127`; Section 3 facts sheet).
