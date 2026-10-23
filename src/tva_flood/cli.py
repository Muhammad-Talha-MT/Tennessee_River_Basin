"""Lazy command dispatch: download commands do not import the trainer."""

import importlib
import sys

COMMANDS = {
    "build-gauges": (
        "data.gauges",
        "Catalog of observed-flow gauges: TVA headwater and all CAMELS-H gauges in Tennessee",
    ),
    "build-routing": (
        "data.river",
        "Whole upstream basin of every reach: area weights and travel distances",
    ),
    "build-array-cache": (
        "data.arrays",
        "Write decoded monthly arrays once for fast encoder training",
    ),
    "build-public-gauges": (
        "data.public_gauges",
        "Gauge catalog of public series only: CAMELS-H before the USGS start, USGS after",
    ),
    "build-basin-series": (
        "data.basin_series",
        "Basin-mean weather and attributes per gauge, for lumped baselines (LSTM)",
    ),
    "evaluate-hydrology": (
        "evaluation.hydrology",
        "Per-reach NSE, KGE, RMSE and peak error against NWM of a train-forecast checkpoint",
    ),
    "evaluate-observed": (
        "evaluation.observed",
        "Score checkpoints, persistence and TVA's forecasts against observed flow at TVA gauges",
    ),
    "evaluate-events": (
        "evaluation.events",
        "Flood-threshold and observed-peak skill from saved evaluate-observed forecasts",
    ),
    "evaluate-benchmark": (
        "evaluation.benchmark",
        "Graph models, LSTM baseline and NOAA NWM vs USGS observations (no TVA data)",
    ),
    "ssl-pretrain": (
        "training.ssl_pretrain",
        "Masked flow-reconstruction pretraining of the shared encoder",
    ),
    "train-forecast": (
        "training.forecast",
        "Ten-day forecast-mode training on NWM targets (gauge-only flow history)",
    ),
    "finetune-observed": (
        "training.observed_finetune",
        "Fine-tune on observed flow at every catalog gauge, in forecast mode",
    ),
    "hydro-prepare": (
        "hydro.prepare",
        "Distributed model: routing groups, per-reach NWM statistics, antecedent storage",
    ),
    "hydro-dams": (
        "hydro.dams",
        "Distributed model: TVA dams on the graph, observed releases and inflows, TVA forecasts",
    ),
    "hydro-train": (
        "hydro.train",
        "Distributed model: stage A (simulation) or B (forecast) on all GPUs (torchrun)",
    ),
    "hydro-evaluate": (
        "hydro.evaluate",
        "Distributed model: gauged, ungauged and every-reach test protocols (torchrun)",
    ),
    "download-usgs": (
        "downloads.usgs",
        "USGS observed flow for the benchmark test period (all catalog gauges)",
    ),
    "download-nwm-forecasts": (
        "downloads.nwm_forecasts",
        "NWM medium-range streamflow at the gauges and its rain forcing per reach (2020-2025)",
    ),
    "train-lstm": (
        "training.lstm_baseline",
        "Lumped forecast LSTM baseline on observed flow (same inputs and periods)",
    ),
    "download-gefs": (
        "downloads.gefs",
        "GEFSv12 rain forecasts (reforecast 2000-2019, operational 2020-) per reach",
    ),
    "download-boundaries": ("downloads.boundaries", "Download Tennessee boundary layers"),
    "download-nhdplus": ("downloads.nhdplus", "Download and extract NHDPlus Region 06"),
    "inventory": ("data.inventory", "Match NHDPlus reaches to NWM coordinates"),
    "build-graph": ("data.graph", "Build full-network graph and catchment tables"),
    "download-forcing": ("downloads.forcing", "Download hourly AORC and NWM"),
    "download-camelsh": ("downloads.camelsh", "Optional CAMELS-H reference data"),
    "build-weights": ("data.weights", "Build area-weighted AORC-to-reach mapping"),
    "aggregate": ("data.aggregate", "Aggregate hourly inputs into six-hour files"),
    "prepare-statics": ("data.statics", "Download and align static reach attributes"),
    "prepare": ("data.prepare", "Fit training normalization and write dataset manifest"),
    "prepare-test": ("data.prepare_test", "Prepare the held-out test period (2020 onward)"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: tva-flood COMMAND [OPTIONS]\n")
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:24} {help_text}")
        print("\nUse tva-flood COMMAND --help for options.")
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 2
    module, _ = COMMANDS[command]
    sys.argv = [f"tva-flood {command}", *sys.argv[2:]]
    return importlib.import_module(f"tva_flood.{module}").main()
