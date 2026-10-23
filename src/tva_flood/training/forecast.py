"""Ten-day forecasting on NWM targets in forecast mode, with the shared encoder.

At issue time the model sees what a forecaster sees: weather up to the issue
time, flow history only at gauges (--history-gauges: the NWM flow at those
reaches stands in for the observation; every other reach has no flow history),
and future precipitation known only up to --rain-hours and only where a forecast
exists (random coverage during training). Targets are NWM flow at every reach,
so the model learns to forecast ungauged reaches from gauged neighbours.

Use --init with an ssl-pretrain checkpoint to transfer its encoder, or
--architecture-from with the same checkpoint to take only its architecture.
Without either, flags such as --hidden set the architecture. The head starts at
persistence where the latest flow is visible and has a separate branch for
reaches without history. Training hides some visible histories
(--history-hide-fraction) and, for some windows, all of them (--no-history-prob).

The default loss gives the 6-24h, 30-72h and 78-240h lead bands equal total
weight, and selection uses the same band-balanced normalized log MSE with full
history. Validation reports raw and log metrics against persistence for both
full-history and no-history scenarios. Inputs never include data after issue.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tva_flood.common import sha
from tva_flood.data.dataset import lineage
from tva_flood.data.gauges import Catalog
from tva_flood.data.rain_error import degrade
from tva_flood.data.river import DISTANCE_BINS
from tva_flood.data.schema import HORIZON
from tva_flood.data.ssl import FLOW, FLOW_VALID, mask_future, mask_history
from tva_flood.evaluation.metrics import metrics, sums

from .encoder_loop import derived_seed, resolve_architecture
from .encoder_loop import main as run_main
from .runtime import reduced

BANDS = {"6-24h": slice(0, 4), "30-72h": slice(4, 12), "78-240h": slice(12, 40)}
SCENARIOS = ("full", "none")
GROUPS = ("model_all_valid", "model_matched", "persistence_matched")
# Upper bound for log1p flow before inversion (about 1e7 m3/s): avoids overflow only.
MAX_LOG_FLOW = 16.2


def lead_weights(kind):
    weights = np.ones(HORIZON)
    if kind == "balanced":
        for band in BANDS.values():
            weights[band] = HORIZON / (len(BANDS) * len(range(HORIZON)[band]))
    return weights


def finite(value):
    return None if value is None or pd.isna(value) else float(value)


def raw_flow(z, mean, std):
    return np.maximum(np.expm1(np.minimum(np.asarray(z, np.float64) * std + mean, MAX_LOG_FLOW)), 0)


def load_reference(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") not in ("ssl-reconstruction-v1", "encoder-forecast-v1"):
        raise ValueError(f"{path} is not an ssl-pretrain or train-forecast checkpoint")
    return saved


class Forecast:
    format = "encoder-forecast-v1"
    horizon = HORIZON

    def arguments(self, p):
        from .encoder_loop import add_arguments

        add_arguments(p, updates=6000, batch_size=4, subgraph_nodes=6000, lr=1e-3)
        source = p.add_mutually_exclusive_group()
        source.add_argument(
            "--init", type=Path, help="Checkpoint whose encoder weights initialize this model"
        )
        source.add_argument(
            "--architecture-from",
            type=Path,
            help="Checkpoint whose encoder architecture (not weights) is used",
        )
        p.add_argument(
            "--encoder-lr-scale",
            type=float,
            default=1.0,
            help="Encoder LR = scale x --lr; 0 freezes the encoder",
        )
        p.add_argument("--history-hide-fraction", type=float, default=0.1)
        p.add_argument("--no-history-prob", type=float, default=0.1)
        p.add_argument("--lead-weighting", choices=["balanced", "uniform"], default="balanced")
        p.add_argument(
            "--future-rain",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Give the decoder future precipitation (realized AORC, degraded as below)",
        )
        p.add_argument(
            "--rain-hours",
            type=int,
            default=96,
            help="Future precipitation is known only up to this lead (TVA's forecast rain is "
            "real to about 72 h and a fifth of the rain to 96 h; unknown after)",
        )
        p.add_argument(
            "--rain-error",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Give the realized training rain the errors of TVA's rain forecast "
            "(data.rain_error), in training and validation",
        )
        p.add_argument(
            "--history-gauges",
            type=Path,
            help="Gauge catalog (build-gauges): flow history is visible only at these reaches",
        )
        p.add_argument(
            "--future-none-prob",
            type=float,
            default=0.2,
            help="Probability that no future precipitation is known",
        )
        p.add_argument(
            "--future-region-prob",
            type=float,
            default=0.4,
            help="Probability that it is known only in connected regions",
        )
        p.add_argument(
            "--future-cutoff-prob",
            type=float,
            default=0.5,
            help="Probability that it is known only up to a random shorter lead",
        )
        p.add_argument("--decoder-layers", type=int, default=5)
        p.add_argument("--decoder-graph-layers", type=int, default=2)
        p.add_argument(
            "--routing",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Route every upstream reach's decoder state to each reach with a learned "
            "travel-time kernel per river-distance bin (needs --direction upstream)",
        )
        p.add_argument("--routing-lags", type=int, default=12, help="Longest travel time, steps")
        p.add_argument(
            "--basin-rain",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Also give the decoder basin-mean future rain (needs --direction upstream)",
        )

    def check(self, p, a):
        if a.encoder_lr_scale < 0:
            p.error("Encoder LR scale must be nonnegative")
        if not 0 <= a.history_hide_fraction < 1 or not 0 <= a.no_history_prob <= 1:
            p.error("History hide fraction in [0,1), no-history probability in [0,1]")
        reference = a.init or a.architecture_from
        self.reference = load_reference(reference) if reference else None
        self.weights = lead_weights(a.lead_weighting)
        if a.evaluate_checkpoint:
            # The scored model's own decoder settings, whatever flags were passed.
            stored = load_reference(a.evaluate_checkpoint)["config"]
            decoder = stored.get("decoder")
            a.future_rain = bool(decoder)
            a.rain_hours = stored.get("rain_hours", a.rain_hours)
            a.rain_error = stored.get("rain_error", False)
            self.gauge_comids = stored.get("history_gauge_comids")
            if decoder:
                a.decoder_layers = stored["decoder"]["decoder_layers"]
                a.decoder_graph_layers = stored["decoder"]["decoder_graph_layers"]
                a.routing = bool(stored["decoder"].get("routing_bins", 0))
                a.routing_lags = stored["decoder"].get("routing_lags", 12)
                a.basin_rain = stored["decoder"].get("basin_rain", False)
        elif a.history_gauges:
            self.gauge_comids = Catalog(a.history_gauges).comid.tolist()
        else:
            self.gauge_comids = None
        if not 6 <= a.rain_hours <= 6 * HORIZON or a.rain_hours % 6:
            p.error(f"--rain-hours must be a multiple of 6 in [6, {6 * HORIZON}]")
        self.rain_steps = a.rain_hours // 6
        if a.decoder_layers < 1 or a.decoder_graph_layers < 0:
            p.error("Decoder needs >= 1 temporal layer and >= 0 graph layers")
        if a.routing_lags < 0:
            p.error("--routing-lags must be >= 0")
        if (a.routing or a.basin_rain) and not a.evaluate_checkpoint:
            if (a.direction or "both") != "upstream" and not (a.init or a.architecture_from):
                p.error("--routing and --basin-rain need --direction upstream")
        probabilities = (a.future_none_prob, a.future_region_prob, a.future_cutoff_prob)
        if min(probabilities) < 0 or max(probabilities) > 1:
            p.error("Future coverage probabilities must be in [0,1]")
        if a.future_none_prob + a.future_region_prob > 1:
            p.error("--future-none-prob + --future-region-prob must be <= 1")
        self.future_rain = a.future_rain
        self.rain_error = bool(a.future_rain and a.rain_error)
        self.decoder = None
        if self.future_rain:
            self.decoder = dict(
                decoder_layers=a.decoder_layers,
                decoder_graph_layers=a.decoder_graph_layers,
                future_channels=2,
                routing_bins=len(DISTANCE_BINS) - 1 if a.routing else 0,
                routing_lags=a.routing_lags,
                basin_rain=bool(a.basin_rain),
            )
        self.future_masking = (
            dict(
                none_prob=a.future_none_prob,
                region_prob=a.future_region_prob,
                cutoff_prob=a.future_cutoff_prob,
                region_fraction=[0.05, 0.5],
                region_hops=[5, 30],
                cutoff_steps=[min(4, self.rain_steps), self.rain_steps],
            )
            if self.future_rain
            else None
        )

    def halo(self, architecture):
        extra = self.decoder["decoder_graph_layers"] if self.decoder else 0
        return architecture["graph_layers"] + extra

    def architecture(self, p, a):
        inherited = self.reference["config"]["architecture"] if self.reference else None
        return resolve_architecture(p, a, inherited)

    def build(self, encoder):
        from tva_flood.models.ssl import ForecastModel, WeatherForecastModel

        if self.decoder:
            return WeatherForecastModel(encoder, HORIZON, **self.decoder)
        return ForecastModel(encoder, horizon=HORIZON)

    def forward(self, model, batch, supports=None):
        if self.decoder:
            return model(batch["dynamic"], batch["static"], batch["future"], supports)
        return model(batch["dynamic"], batch["static"], supports)

    def initialize(self, model, run):
        a = run.a
        self.rain = (run.train.precip_transform, float(run.train.mean[0]), float(run.train.std[0]))
        self.visible = None
        if self.gauge_comids is not None:
            self.visible = torch.as_tensor(np.isin(run.ids, self.gauge_comids))
            if not self.visible.any():
                raise ValueError("No history gauge is a reach of the graph")
        if self.reference is None:
            return
        reference = self.reference["config"]
        if reference["dataset_identity"] != lineage(run.train.manifest):
            raise ValueError("Reference checkpoint was trained on different prepared data")
        if reference["normalization_sha256"] != sha(run.prep / "dynamic_normalization.json"):
            raise ValueError("Reference checkpoint used a different normalization")
        if a.init:
            state = self.reference["model"]
            encoder = {
                k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")
            }
            model.encoder.load_state_dict(encoder, strict=True)
        if a.encoder_lr_scale == 0:
            model.encoder.requires_grad_(False)

    def parameter_groups(self, model, a):
        head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
        groups = [dict(params=head, lr=a.lr)]
        if a.encoder_lr_scale > 0:
            groups.append(
                dict(params=list(model.encoder.parameters()), lr=a.lr * a.encoder_lr_scale)
            )
        return groups

    def config(self, a):
        reference = a.init or a.architecture_from
        return dict(
            task="forecast",
            initialization="ssl encoder" if a.init else "scratch",
            reference_checkpoint=str(reference.resolve()) if reference else None,
            reference_sha256=sha(reference) if reference else None,
            reference_update=self.reference.get("update") if self.reference else None,
            encoder_lr_scale=a.encoder_lr_scale,
            history_hide_fraction=a.history_hide_fraction,
            no_history_prob=a.no_history_prob,
            lead_weighting=a.lead_weighting,
            future_rain="realized AORC precipitation, known to rain_hours and degraded by "
            "coverage masking in training; replace it with a forecast for operation"
            if self.future_rain
            else "none",
            rain_hours=a.rain_hours,
            rain_error=self.rain_error,
            history_gauges=str(a.history_gauges.resolve()) if a.history_gauges else None,
            history_gauge_comids=self.gauge_comids,
            decoder=self.decoder,
            future_masking=self.future_masking,
            loss="lead-weighted MSE of normalized log flow at valid targets of core reaches",
            selection="band-balanced normalized log MSE, full history, all valid targets",
        )

    def restrict(self, dynamic, nodes=None):
        """Flow history only at gauges (in place on [..., node, channel]); no-op without gauges."""
        if self.visible is not None:
            hidden = ~(self.visible if nodes is None else self.visible[torch.as_tensor(nodes)])
            dynamic[..., hidden, FLOW : FLOW_VALID + 1] = 0
        return dynamic

    def prepare(self, batch, graph, seed, a):
        rng = np.random.default_rng(seed)
        self.restrict(batch["dynamic"], graph["nodes"])
        if "future" in batch:
            batch["future"][:, self.rain_steps :] = 0
        dynamic = []
        for sample in range(batch["dynamic"].shape[0]):
            current = batch["dynamic"][sample : sample + 1]
            valid = batch["valid"][sample : sample + 1]
            draw = rng.random()
            if draw < a.no_history_prob:
                current, _ = mask_history(
                    current, valid, seed=derived_seed(seed, sample), mode="all"
                )
            elif a.history_hide_fraction > 0:
                current, _ = mask_history(
                    current,
                    valid,
                    seed=derived_seed(seed, sample),
                    mode="reach",
                    reach_fraction=a.history_hide_fraction,
                )
            dynamic.append(current)
            if self.rain_error:
                self.degrade(batch["future"][sample], derived_seed(seed, sample, 3))
            if self.future_masking:
                batch["future"][sample] = mask_future(
                    batch["future"][sample],
                    seed=derived_seed(seed, sample, 2),
                    adjacency=graph["adjacency"],
                    **{
                        k: tuple(v) if isinstance(v, list) else v
                        for k, v in self.future_masking.items()
                    },
                )
        batch["dynamic"] = torch.cat(dynamic)
        batch["target_mask"] = batch.pop("target_valid") & batch["core"][None, None]
        del batch["valid"], batch["flow"]

    def degrade(self, future, seed):
        """Forecast-like rain errors in place on one sample's future [lead, node, 2]."""
        rng = np.random.default_rng(seed)
        future[..., 0] = degrade(future[..., 0], future[..., 1] > 0.5, rng, self.rain)

    def loss(self, model, batch, supports, run):
        prediction = self.forward(model, batch, supports).float()
        target, mask = batch["target"], batch["target_mask"]
        weights = torch.as_tensor(self.weights, dtype=torch.float32, device=target.device)
        squared = (prediction - target).square()
        sse = (squared * weights[None, :, None])[mask].sum()
        latest = batch["dynamic"][:, -1, :, FLOW].float()
        available = batch["dynamic"][:, -1, :, FLOW_VALID] > 0.5
        matched = mask & available[:, None, :]
        persistence = (latest[:, None, :] - target).square()
        return (
            sse,
            mask.sum(),
            [
                squared[mask].sum(),
                squared[matched].sum(),
                persistence[matched].sum(),
                matched.sum(),
            ],
        )

    def log_values(self, statistics, total):
        matched = max(statistics[5].item(), 1)
        return dict(
            mse=statistics[2].item() / max(total, 1),
            matched_mse=statistics[3].item() / matched,
            matched_persistence_mse=statistics[4].item() / matched,
        )

    def validate(self, run):
        data, mean, std = run.validation, run.mean, run.std
        totals = np.zeros((len(SCENARIOS), len(GROUPS), HORIZON, 7))
        local = run.items[run.rank :: run.world]
        for position, item in enumerate(local):
            sample = data.load(int(item))
            target = sample["target"].numpy().astype(np.float64)
            valid = sample["target_valid"].numpy()
            actual = raw_flow(target, mean, std)
            for s, scenario in enumerate(SCENARIOS):
                dynamic = sample["dynamic"][None].clone()
                if scenario == "none":
                    dynamic[..., FLOW : FLOW_VALID + 1] = 0
                else:
                    self.restrict(dynamic)
                inputs = dict(dynamic=dynamic.to(run.device), static=run.full_static)
                if self.decoder:
                    future = sample["future"][None].clone()
                    future[:, self.rain_steps :] = 0
                    if self.rain_error:
                        self.degrade(future[0], derived_seed(int(item), 3))
                    inputs["future"] = future.to(run.device)
                prediction = self.forward(run.model, inputs)[0]
                prediction = prediction.float().cpu().numpy().astype(np.float64)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite validation forecast")
                latest = dynamic[0, -1, :, FLOW].numpy().astype(np.float64)
                available = dynamic[0, -1, :, FLOW_VALID].numpy() > 0.5
                matched = valid & available[None]
                persistence = np.broadcast_to(latest[None], target.shape)
                forecast_raw = raw_flow(prediction, mean, std)
                for g, (pn, pr, mask) in enumerate(
                    (
                        (prediction, forecast_raw, valid),
                        (prediction, forecast_raw, matched),
                        (persistence, raw_flow(persistence, mean, std), matched),
                    )
                ):
                    totals[s, g] += sums(pn, pr, target, actual, mask, mean, std, axis=1).T
            if run.rank == 0 and (position % 8 == 0 or position + 1 == len(local)):
                print(f"validation windows {position + 1}/{len(local)} (rank 0)", flush=True)
        totals = reduced(torch.as_tensor(totals, device=run.device), run.world).cpu().numpy()
        summary, leads = [], []
        for s, scenario in enumerate(SCENARIOS):
            for g, group in enumerate(GROUPS):
                for band, indices in {"all": slice(None), **BANDS}.items():
                    summary.append(
                        dict(
                            scenario=scenario,
                            group=group,
                            lead_band=band,
                            **metrics(totals[s, g, indices].sum(axis=0)),
                        )
                    )
                for lead in range(HORIZON):
                    leads.append(
                        dict(
                            scenario=scenario,
                            group=group,
                            lead_hours=6 * (lead + 1),
                            **metrics(totals[s, g, lead]),
                        )
                    )
        frame = pd.DataFrame(summary)

        def value(scenario, group, band, column):
            row = frame[
                (frame.scenario == scenario) & (frame.group == group) & (frame.lead_band == band)
            ]
            return row[column].iloc[0]

        def balanced(scenario):
            scores = [
                value(scenario, "model_all_valid", band, "normalized_log_mse") for band in BANDS
            ]
            return None if any(v is None or pd.isna(v) for v in scores) else float(np.mean(scores))

        def ratio(band):
            model = value("full", "model_matched", band, "rmse_m3_s")
            base = value("full", "persistence_matched", band, "rmse_m3_s")
            if model is None or base is None or pd.isna(model) or pd.isna(base) or not base:
                return None
            return float((model / base) ** 2)

        selection = balanced("full")
        if selection is None:
            raise ValueError("Validation produced no targets in some lead band")
        return dict(
            selection=selection,
            balanced_log_mse_full=selection,
            balanced_log_mse_no_history=balanced("none"),
            **{f"raw_mse_ratio_vs_persistence_{band}": ratio(band) for band in BANDS},
            **{
                f"log_mse_full_{band}": finite(
                    value("full", "model_all_valid", band, "normalized_log_mse")
                )
                for band in BANDS
            },
            validation_windows=len(run.items),
            tables=dict(summary=frame, by_lead=pd.DataFrame(leads)),
        )


def main():
    run_main(Forecast(), __doc__)


if __name__ == "__main__":
    main()
