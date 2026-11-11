# Attribution experiments (manuscript Exp.1-Exp.3)

Question: the GNN beats the tuned LSTM at held-out gauges (median NSE days 1-5: 0.378 vs 0.111) but ties it at
gauged sites (0.415 vs 0.414). Is the held-out gain from the river graph, the NWM supervision, or how the
baseline was trained? Every model here follows the GEFS rule and the current GNN recipe (stage B, `--loss nse-q
--head lstm`, 3000 updates of 8 samples), selected on 2019 gauge scores only (`--select gauges`).

Runner: `scripts/attribution_pipeline.sh PHASE`; on Frontier see `scripts/frontier/README.md` (portable via `TVA_ROOT`, resumes cut-off runs,
several nodes share one job list) (each phase started by hand; `DRY_RUN=1` lists the jobs).
Outputs: models in `/data/tva_data/models/attr/`, reports in `/data/tva_data/figures/attribution/`.

## What existed before this work

| Item | Status |
|---|---|
| A0, seed 1 (full GNN, all gauges + 5 folds) and its evaluation | `models/gnn_B*`, `gnn_evaluation`; linked as `attr/A0_s1*`. Both selection rules pick the same checkpoints, so it is not retrained |
| Shown / hidden / held-out gauge history, realized-rain diagnostic, headwater subset | done (`figures/gnn_final`, `final_report`) |
| Routing / message-passing ablations | run once on the older `scratch_graph` model under the old rain mix, then deleted; not valid for the current model |
| Seeds, matched controls, LSTM masking check, leakage, blocked folds, coverage, conservation | none |

## Matrix

| Phase | ID | Model | Graph | Routing | NWM pretrain | NWM prior | Runs | Answers |
|---|---|---|---|---|---|---|---|---|
| 0 | P1 | LSTM, the GNN's masking (57.5% history hidden) and selection (shown + hidden) | no | - | no | no | 3 seeds x 6 | Is 0.111 a baseline handicap? |
| 0 | P2 | NWM retrospective skill at each gauge (no training) | | | | | | Is the held-out test leaky through stage A? |
| 0 | Exp.3 | Conservation audit of A0 (no training) | | | | | | Mass balance of routing; does the gauge head break it? |
| 1 | A0 | Full GNN | yes | learned | yes | yes | existing | reference |
| 1 | A1 | GNN, no exchange (`--off mixing broadcast`) | no | learned | yes | yes | 6 | message passing + owner-gauge broadcast |
| 1 | A3 | GNN without NWM (`--no-nwm`, from scratch) | yes | learned | no | no | 6 | graph without the simulation |
| 1 | A5 | Lumped reach model, same NWM training (`--lumped`) | no | none | yes | yes | 1 + 6 | **graph vs NWM supervision** |
| 1 | A6 | Lumped reach model, no NWM | no | none | no | no | 6 | observation-only, same code path as A5 |
| seeds | | A0, A5 seeds 2, 3 (`SEEDS`, `SEED_VARIANTS`) | | | | | 24 | seed variance |
| 2 | S1 | A0, A5 on 3 blocked tributary basins (`--scheme blocked`) | | | | | 6 | transfer without neighbours |
| 2 | S2 | A0, A5 with 0, 25, 50, 100% of training gauges (`--keep`) | | | | | 20 | value of the monitoring network |
| 2 | S3 | Skill vs river distance to a training gauge, gauged share, area, dams | | | | | | what predicts transfer |
| extra | A2, A4 | instantaneous routing (`--off routing`); no stage-B prior (`--nwm-weight 0`) | | | | | 12 | deferred |

A0, A3, A5 and A6 form the 2x2 graph x NWM table; the report gives the interaction (A0 - A5) - (A3 - A6).

## Design choices

- **A1** switches off every path by which a reach learns about another reach: graph layers see zero neighbours
  (same parameters) and no reach gets the owner gauge's specific discharge. Water is still routed downstream.
- **A5/A6** (`model.LumpedModel`): the same LSTM trunk, heads, losses and gauge head, but every reach is a lumped
  basin. It reads area-weighted basin-mean weather and attributes, forecasts its outlet flow directly (no routing)
  and sees no other reach. A5's stage A trains on the NWM retrospective at all 58,834 reaches, exactly the targets
  A0 had. Parameters: 915,982 vs 929,950.
- **A3/A6** start from scratch with 8,000 updates (A0's stage A + B budget; `SCRATCH_UPDATES=3000` halves their cost).
  Without NWM no prior is defined: the term is not computed.
- **Selection**: `--select gauges` (mean of shown and hidden 2019 gauge NSE). The old score also used NWM
  agreement at every reach, which a no-NWM model must not see. A checkpoint is saved at every validation.
- **Blocked folds** (`hydro.splits`): complete upstream basins of CAMELS-H gauges holding 8-30 scored gauges,
  disjoint, largest first: 03455000 (15 gauges, 4,801 km2), 03495500 (13), 03566000 (12). Every catalog gauge
  inside is withheld from targets, inputs and selection. No water enters a block from outside; gauges below a
  block stay in training (a caveat to report).
- **Coverage**: `--keep f` also withholds a random 1 - f of the remaining training gauges (seeded per fold). Only
  the fold's gauges are scored. 0% is the stage-A model written as fold models that see no gauge.
- **Statistics** (`scripts/attribution_report.py`): every model is scored on A0's (issue, lead, gauge) mask. Per-gauge
  seed means; paired median/mean differences with 95% bootstrap intervals (gauges, and seeds when there are several),
  Wilcoxon signed-rank p, share of gauges improved; NSE, KGE and CSI above the training 95th percentile, days 1-5.

## Cost (GPU-hours on one L40S; 4 GPUs run 4 jobs)

| Phase | GPU-h | Wall time |
|---|---:|---:|
| phase0 | ~4 (LSTM) + CPU benchmarks | ~3 h |
| phase1 | ~275 (A3 123, A6 64, A1 46, A5 24 + stage A 7, evaluations 11) | ~3 days (~2 days with `SCRATCH_UPDATES=3000`) |
| seeds | ~145 | ~1.5 days |
| phase2 | ~160 | ~1.7 days |
| extra | ~95 | ~1 day |

Measured: GNN stage B 7.7 h, lumped stage B 4 h (all reaches) per run.

## Decision gates

1. After phase0: if the matched LSTM closes most of the held-out gap, or held-out skill is explained by the
   retrospective's skill, rethink phase 1 before spending ~3 GPU-days.
2. After phase1 (seed 1): if A5 is about A0 at held-out gauges, the paper becomes a multi-fidelity training study;
   run seeds only for the comparisons that matter.

## Checks made while building

- The refactored model reproduces the saved `gnn_evaluation` forecasts within the GPU's run-to-run bf16 noise
  (the unchanged code differs from them as much). The report reproduces 0.415 / 0.414 / 0.378 / 0.111.
- Routing conserves mass end to end (worst relative error 4e-7 in float32). Basin continuity holds at all 35,662
  confluences once divergence shares are counted.
- Preliminary (existing models only; regenerated by the phases):
  - At gauged sites the GNN ties the LSTM on NSE but is below it on KGE (-0.029) and CSI (-0.032), with 23-29%
    of gauges improved.
  - Held-out GNN skill correlates with the retrospective's skill at the gauge (Spearman +0.39, p 8e-5); the
    LSTM's does not (+0.14).
  - The gauge head changes the 10-day volume at visible gauges by 0.81-1.15 (5-95%) relative to the routed flow
    and is not passed downstream.
