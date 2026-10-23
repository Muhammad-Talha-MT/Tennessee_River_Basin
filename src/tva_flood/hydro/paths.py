"""Data locations and fixed windows (six-hour steps)."""

from pathlib import Path

import pandas as pd

ROOT = Path("/data/tva_data")
PROCESSED = ROOT / "processed"
TRAIN_PREP = PROCESSED / "pretraining_full"
TRAIN_ARRAYS = PROCESSED / "pretraining_full_arrays"
TEST_PREP = PROCESSED / "test_preparation_2020_2025"
TEST_ARRAYS = PROCESSED / "test_preparation_2020_2025_arrays"
GAUGES = PROCESSED / "gauges_public/gauges.nc"
NWM_GAUGES = PROCESSED / "benchmarks/nwm_medium_range"  # forecasts at gauges + their rain
NWM_REACHES = PROCESSED / "benchmarks/nwm_medium_range_reaches"  # forecasts at every reach
# GEFSv12 control rain to 240 h: the only future weather any model sees (training,
# selection and test alike). Reforecast 2000-2019; operational from 2020-09-23.
GEFS_REFORECAST = PROCESSED / "rain_forecasts/gefs_reforecast_240"
GEFS_OPERATIONAL = PROCESSED / "rain_forecasts/gefs_operational_240"
HYDRO = PROCESSED / "hydro"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

START = pd.Timestamp("1979-02-01")  # step 0 of the timeline, 00Z
TEST_FROM = pd.Timestamp("2020-01-01")  # test arrays from here on
TRAIN_FROM = pd.Timestamp("1980-02-01")  # first issue: a year of antecedent spin-up
TRAIN_UNTIL = pd.Timestamp("2019-01-01")  # targets end before this
VALID_UNTIL = pd.Timestamp("2020-01-01")
RETRO_UNTIL = pd.Timestamp("2023-01-01")  # NWM v3.0 retrospective ends

HORIZON = 40  # forecast steps (10 days)
SPIN = 64  # runoff-network warm-up (receptive field 63)
ROUTE = 48  # routing kernel taps (12 days)
INNOVATION = 32  # assimilation window before the issue (8 days)
PAST = ROUTE + INNOVATION  # simulated steps before the issue (lateral inflow window)
HISTORY = SPIN + PAST  # weather steps before the issue
WINDOW = HISTORY + HORIZON  # weather steps per sample
FLOW_STEPS = PAST + HORIZON  # lateral-inflow and routed-flow window
SCORED = INNOVATION + HORIZON  # routed steps with a full routing memory

STEPS_PER_DAY = 4
SECONDS = 6 * 3600
