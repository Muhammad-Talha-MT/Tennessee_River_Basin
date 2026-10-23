"""Forecast-mode inputs on gauge subgraphs, shared by fine-tuning and evaluation.

At issue time t the model may use everything observed up to t: weather history
on every reach, observed flow at every catalog gauge (no other flow history),
and future precipitation known only up to a rain window and only where a
forecast exists. This module builds those inputs, exactly as in operation, on
the subgraph around chosen gauges: the gauges' reaches plus a halo of as many
hops as the model has graph blocks (or, for upstream-only models with river
routing, the gauges' whole upstream basins), so their outputs equal full-graph
outputs.

Future rain is described by a ``spec``:

- ``None``: no future rain;
- ``"qpf"``: TVA's basin forecast at TVA gauges (where it exists), nothing elsewhere;
- ``"aorc"``: realized AORC rain over every TVA gauge basin, nothing elsewhere
  (the same coverage with a perfect forecast: an upper bound);
- a boolean mask over gauges: realized AORC over those gauges' basins (a proxy
  for a forecast where none exists), plus TVA's forecast wherever it exists.

``rain_field`` instead takes a whole rain forecast per reach (e.g. the rain NWM
was forced with), and ``folder=None`` builds the inputs without TVA's archive
(no TVA forecasts), for benchmarks that must not use TVA data.

With ``error`` (a random generator) the realized AORC rain gets the errors of
TVA's rain forecast first (``data.rain_error``); TVA's own forecast is unchanged.
"""

import numpy as np
import pandas as pd
import torch

from . import tva
from .rain_error import degrade
from .schema import HISTORY, HORIZON
from .ssl import FLOW, FLOW_VALID
from .subgraph import SubgraphSampler
from .transforms import transform_precipitation


class ForecastInputs:
    def __init__(self, data, catalog, folder, edges, supports, halo, river=None):
        self.data, self.catalog = data, catalog
        ids, n = data.ids, len(data.ids)
        self.outlet = pd.Index(ids).get_indexer(catalog.comid)
        if (self.outlet < 0).any():
            raise ValueError("A catalog gauge is not a reach of the graph")
        self.mean, self.std = float(data.mean[8]), float(data.std[8])
        self.rain = (data.precip_transform, float(data.mean[0]), float(data.std[0]))
        self.observed = catalog.on(data.times).astype(np.float64)
        self.observed_z = np.where(
            np.isfinite(self.observed),
            (np.log1p(np.maximum(self.observed, 0)) - self.mean) / self.std,
            0.0,
        ).astype(np.float32)
        if folder is None:
            self.station_gauge = np.empty(0, dtype=np.int64)
            self.stations = pd.DataFrame()
            self.forecast, self.qpf = {}, {}
        else:
            archive = tva.gauges(folder, ids, forecasts=True)
            known = set(archive.comid)
            self.station_gauge = np.array(
                [i for i, c in enumerate(catalog.comid) if c in known], dtype=np.int64
            )
            self.stations = archive.set_index("comid").loc[catalog.comid[self.station_gauge]]
            self.stations = self.stations.reset_index()
            _, self.forecast, self.qpf = tva.align(folder, self.stations, data.times)
        self.is_tva = np.zeros(len(catalog), dtype=bool)
        self.is_tva[self.station_gauge] = True
        # A reach belongs to the smallest gauge basin containing it. Ownership for
        # TVA's forecast uses TVA gauges only, so it covers nested basins too.
        self.owner = tva.basin_owner(edges, n, self.outlet)[0]
        if len(self.station_gauge):
            tva_owner = tva.basin_owner(edges, n, self.outlet[self.station_gauge])[0]
            self.forecast_owner = np.where(tva_owner >= 0, self.station_gauge[tva_owner], -1)
        else:
            self.forecast_owner = np.full(n, -1, dtype=np.int64)
        # With a river (upstream-only models) each gauge needs its whole basin, else a halo.
        self.sampler = SubgraphSampler(n, edges, supports, halo, river)
        core = np.zeros(n, dtype=bool)
        self.around = []
        for g in range(len(catalog)):
            core[:] = False
            core[self.outlet[g]] = True
            self.around.append(np.flatnonzero(self.sampler.closed(core)))

    def withhold(self, mask):
        """Remove gauges' observations entirely (held-out gauges): no target, no history."""
        mask = np.asarray(mask, dtype=bool)
        self.observed[:, mask] = np.nan
        self.observed_z[:, mask] = 0.0

    def graph(self, gauges):
        """Exact subgraph around the given catalog gauges, with local positions."""
        gauges = np.asarray(gauges, dtype=np.int64)
        nodes = np.unique(np.concatenate([self.around[g] for g in gauges]))
        core = np.zeros(len(self.data.ids), dtype=bool)
        core[self.outlet[gauges]] = True
        graph = self.sampler.subgraph(nodes, core[nodes])
        graph["gauges"] = gauges
        graph["local"] = np.searchsorted(nodes, self.outlet[gauges])
        inside = np.flatnonzero(np.isin(self.outlet, nodes))
        graph["visible_gauges"] = inside
        graph["visible_local"] = np.searchsorted(nodes, self.outlet[inside])
        graph["owner"] = torch.as_tensor(self.owner[nodes])
        graph["forecast_owner"] = torch.as_tensor(self.forecast_owner[nodes])
        graph["static"] = torch.as_tensor(self.data.static[nodes], dtype=torch.float32)
        return graph

    def load(self, data, item, graph, hidden=None):
        """History with observed flow at the graph's catalog gauges (``hidden`` [gauge] are out)."""
        sample = data.load(int(item), graph["nodes"])
        issue = int(data.issues[item])
        dynamic = sample["dynamic"]
        dynamic[..., FLOW : FLOW_VALID + 1] = 0
        shown = np.arange(len(graph["visible_gauges"]))
        if hidden is not None:
            shown = shown[~hidden[graph["visible_gauges"]]]
        if len(shown):
            gauge = graph["visible_gauges"][shown]
            local = graph["visible_local"][shown]
            window = slice(issue - HISTORY, issue)
            valid = np.isfinite(self.observed[window][:, gauge])
            dynamic[:, local, FLOW] = torch.as_tensor(
                self.observed_z[window][:, gauge], dtype=dynamic.dtype
            )
            dynamic[:, local, FLOW_VALID] = torch.as_tensor(valid, dtype=dynamic.dtype)
        sample["issue"] = issue
        return sample

    def forecast_at(self, issue):
        """TVA basin forecast [lead, gauge] and which gauges have one at this issue time."""
        if issue not in self.qpf:
            return None, np.zeros(len(self.catalog), dtype=bool)
        forecast = np.full((HORIZON, len(self.catalog)), np.nan)
        forecast[:, self.station_gauge] = self.qpf[issue]
        return forecast, self.is_tva & np.isfinite(forecast).any(axis=0)

    def future_rain(self, sample, graph, spec, steps, error=None):
        """Future precipitation [lead, node, 2] for a rain ``spec`` within ``steps`` leads."""
        future = sample["future"]
        value, flag = future[..., 0], future[..., 1] > 0.5
        if error is not None:
            value = degrade(value, flag, error, self.rain)
        none = torch.zeros(len(graph["nodes"]), dtype=torch.bool)
        known, forecast, use = none, None, np.zeros(len(self.catalog), dtype=bool)
        if isinstance(spec, str) and spec == "aorc":
            known = graph["forecast_owner"] >= 0
        elif isinstance(spec, str) and spec == "qpf":
            forecast, use = self.forecast_at(sample["issue"])
        elif spec is not None:
            owner = graph["owner"]
            safe = torch.where(owner >= 0, owner, torch.zeros_like(owner))
            known = (owner >= 0) & torch.as_tensor(spec)[safe]
            forecast, use = self.forecast_at(sample["issue"])
        return tva.basin_rain(
            value, flag, known, forecast, graph["forecast_owner"], use, steps, self.rain
        )

    def rain_field(self, sample, graph, rain_mm, steps):
        """Future rain [lead, node, 2] from a forecast per reach.

        ``rain_mm`` [lead, reach of the full graph] holds six-hour totals (NaN where
        unknown). Rain is known within the first ``steps`` leads at reaches with
        AORC coverage (as in training); everything else is unknown.
        """
        transform, mean, std = self.rain
        values = np.asarray(rain_mm, dtype=np.float64)[:, graph["nodes"]]
        leads = min(steps, values.shape[0], HORIZON)
        covered = sample["future"][0, :, 1].numpy() > 0.5
        known = np.zeros((HORIZON, len(graph["nodes"])), dtype=bool)
        known[:leads] = np.isfinite(values[:leads]) & (values[:leads] >= 0) & covered[None]
        padded = np.zeros((HORIZON, len(graph["nodes"])))
        padded[:leads] = np.where(known[:leads], values[:leads], 0.0)
        z = (transform_precipitation(padded, known, transform) - mean) / std
        z = np.where(known, z, 0.0)
        future = torch.zeros(HORIZON, len(graph["nodes"]), 2, dtype=sample["future"].dtype)
        future[..., 0] = torch.as_tensor(z, dtype=future.dtype)
        future[..., 1] = torch.as_tensor(known, dtype=future.dtype)
        return future

    def target(self, issue, gauges):
        """Observed raw flow [lead, gauge] after the issue time (NaN where unobserved)."""
        values = self.observed[issue : issue + HORIZON][:, gauges]
        return torch.as_tensor(values, dtype=torch.float32)
