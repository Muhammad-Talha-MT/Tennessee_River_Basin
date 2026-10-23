"""One forecast sample over the whole network, and its assembly into model inputs on the GPU.

A sample at issue step t holds, from t - ``HISTORY`` to t + ``HORIZON``:

- weather [T, reach, 8] and validity: observed (AORC) up to t; after t only forecast
  rain, the GEFSv12 control to 240 h, and every other variable unknown. ``rain="aorc"``
  puts the realized rain there instead: a perfect-forecast diagnostic, never trained on;
- observed flow at every gauge [T, gauge] (NaN where missing or withheld), which visible
  gauges show before t and which gives the targets after t;
- NWM retrospective flow over the scored window [t-32, t+40) (normalized log, validity);
- which gauges are visible and the visible gauge owning each reach (smallest visible
  basin containing it); antecedent storage at the issue day; season of every step;
- at every TVA dam the observed release and inflow [T, dam] (the model reads them only
  before t; after t they are targets), the storage change over the last 30 and 90 days,
  and TVA's forecast release issued at t, where one exists (``plan``).
"""

import numpy as np
import torch

from . import paths as P
from .data import PRECIP, Antecedent, Arrays, Dams, Gauges, Network, Rain


class Samples:
    def __init__(self, network=None):
        self.arrays = Arrays()
        self.network = network or Network()
        self.gauges = Gauges(self.network, self.arrays.times)
        self.dams = Dams(self.network, self.arrays.times)
        self.antecedent = Antecedent()
        self.rain = Rain()
        day = self.arrays.times.dayofyear.to_numpy() + self.arrays.times.hour.to_numpy() / 24
        angle = 2 * np.pi * day / 365.25
        self.season = np.stack([np.sin(angle), np.cos(angle)]).astype(np.float32)

    def eligible(self, step, withheld):
        """Gauges that may be shown at this issue: not withheld, flow in the last 8 days."""
        recent = self.gauges.observed[step - P.INNOVATION : step]
        return (~withheld) & (np.isfinite(recent).sum(axis=0) >= 8)

    def build(self, step, rain, visible=None, withheld=None):
        """numpy sample at global issue step ``step``; ``rain``: a GEFS source or "aorc"."""
        a = self.arrays
        start, stop = step - P.HISTORY, step + P.HORIZON
        weather = a.read("weather", start, stop)
        valid = a.read("weather_valid", start, stop)
        future = slice(P.HISTORY, P.WINDOW)
        weather[future, :, 1:] = 0
        valid[future, :, 1:] = False
        if rain != "aorc":
            mm = self.rain.get(rain, a.times[step].normalize())
            if mm is None:
                raise KeyError(f"No {rain} rain for {a.times[step]}")
            precip, known = weather[future, :, PRECIP], valid[future, :, PRECIP]
            forecast = known & np.isfinite(mm) & (mm >= 0)
            precip[:] = np.where(forecast, a.rain_z(np.nan_to_num(mm)), 0)
            known[:] = forecast
        window = slice(step - P.INNOVATION, step + P.HORIZON)
        return dict(
            step=np.int64(step),
            weather=weather.astype(np.float16),
            valid=valid,
            nwm=a.read("flow", window.start, window.stop),
            nwm_valid=a.read("flow_valid", window.start, window.stop),
            antecedent=self.antecedent.at(step),
            season=self.season[:, start:stop],
            release=self.dams.release[start:stop],
            dam_inflow=self.dams.inflow[start:stop],
            storage=self.dams.storage(step),
            plan=self.dams.planned(step),
            **self.gauge_view(step, visible, withheld),
        )

    def gauge_view(self, step, visible=None, withheld=None):
        """Observed flow over the window without withheld gauges, visibility and owners."""
        observed = self.gauges.observed[step - P.HISTORY : step + P.HORIZON].copy()
        withheld = np.zeros(self.gauges.count, bool) if withheld is None else withheld
        observed[:, withheld] = np.nan
        visible = np.zeros(self.gauges.count, bool) if visible is None else visible & ~withheld
        return dict(
            observed=observed.astype(np.float32),
            visible=visible,
            owner=self.gauges.owners(visible, self.network.n),
        )


class Device:
    """Fixed network tensors on the device; samples become model inputs there."""

    def __init__(self, network, gauges, arrays, device, dams):
        self.device = device
        self.flow_mean, self.flow_std = float(arrays.mean[8]), float(arrays.std[8])
        self.rain_mean, self.rain_std = float(arrays.mean[PRECIP]), float(arrays.std[PRECIP])
        self.fixed = dict(
            static=torch.as_tensor(network.static, device=device),
            area=torch.as_tensor(network.area, device=device),
            basin_area=torch.as_tensor(network.basin_area, device=device),
            gauge_reach=torch.as_tensor(gauges.reach, device=device),
            gauge_area=torch.as_tensor(gauges.area, device=device),
            future=(torch.arange(P.WINDOW, device=device) >= P.HISTORY).float(),
            std_raw=torch.as_tensor(network.std_raw, device=device),
            gauge_std_raw=torch.as_tensor(gauges.std_raw, device=device),
            gauge_p90=torch.as_tensor(gauges.p90, device=device),
            # Log errors use log(Q + eps), eps 1% of the mean flow: scale-free, and a
            # forecast near zero still feels the full pull of the target.
            eps=torch.as_tensor(np.maximum(0.01 * network.mean_raw, 1e-4), device=device),
            gauge_eps=torch.as_tensor(np.maximum(0.01 * gauges.mean_raw, 1e-4), device=device),
            regulated=torch.as_tensor(dams.regulated, device=device),
            dam_reach=torch.as_tensor(dams.reach, device=device),
            dam_mean_raw=torch.as_tensor(dams.mean_raw, device=device),
            dam_std_raw=torch.as_tensor(dams.std_raw, device=device),
            dam_eps=torch.as_tensor(np.maximum(0.01 * dams.mean_raw, 1e-3), device=device),
        )

    def batch(self, s):
        move = lambda x: torch.as_tensor(x).to(self.device, non_blocking=True)  # noqa: E731
        b = dict(self.fixed)
        b["weather"] = move(s["weather"]).permute(1, 0, 2).float()  # [reach, step, 8]
        b["valid"] = move(s["valid"]).permute(1, 0, 2).float()
        for key in ("season", "antecedent", "nwm", "nwm_valid", "observed", "visible", "owner",
                    "release", "dam_inflow", "storage", "plan"):
            b[key] = move(s[key])
        return self.assemble(b)

    def assemble(self, b):
        """Model inputs ``steps`` [n, T, 23] and ``reach`` [n, 57] from the batch parts."""
        n = b["weather"].shape[0]
        observed, visible, owner = b["observed"], b["visible"].bool(), b["owner"]
        history = (torch.arange(P.WINDOW, device=self.device) < P.HISTORY)[:, None]
        shown = torch.isfinite(observed) & history & visible[None]
        flow = torch.nan_to_num(observed).clamp_min(0)
        own_z = torch.where(shown, (torch.log1p(flow) - self.flow_mean) / self.flow_std, 0.0)
        own = torch.zeros(n, P.WINDOW, 2, device=self.device)
        own[b["gauge_reach"]] = torch.stack([own_z.T, shown.T.float()], dim=-1)
        specific = torch.where(shown, torch.log1p(flow * 21.6 / b["gauge_area"]), 0.0)  # mm per 6 h
        owned = owner >= 0
        g = owner.clamp(min=0)
        owner_valid = shown[:, g].T & owned[:, None]
        basin = torch.stack(
            [torch.where(owner_valid, specific[:, g].T, 0.0), owner_valid.float()], dim=-1
        )
        steps = torch.cat(
            [
                b["weather"],
                b["valid"],
                b["future"][None, :, None].expand(n, -1, 1),
                b["season"].T[None].expand(n, -1, -1),
                own,
                basin,
            ],
            dim=-1,
        )
        ratio = torch.where(owned, torch.log(b["basin_area"] / b["gauge_area"][g]) / 5, 0.0)
        reach = torch.cat(
            [b["static"], b["antecedent"], ratio[:, None], owned[:, None].float()], dim=-1
        )
        return dict(b, steps=steps, reach=reach, owned=owned.float())

    def limit_rain(self, b, steps):
        """Batch ``b`` with future rain known only for the first ``steps`` leads (the rest unknown),
        for models trained with a shorter forecast-rain window."""
        if steps >= P.HORIZON:
            return b
        weather, valid = b["weather"].clone(), b["valid"].clone()
        weather[:, P.HISTORY + steps :, PRECIP] = 0
        valid[:, P.HISTORY + steps :, PRECIP] = 0
        return self.assemble(dict(b, weather=weather, valid=valid))

    def with_gauges(self, b, view):
        """Batch ``b`` with another gauge view (``Samples.gauge_view``)."""
        c = dict(b)
        for key in ("observed", "visible", "owner"):
            c[key] = torch.as_tensor(view[key]).to(self.device)
        return self.assemble(c)

    def with_rain(self, b, mm):
        """Batch ``b`` (built with realized rain) with forecast rain ``mm`` [lead, reach],
        as in ``Samples.build``: rain counts where it is a finite total and AORC covers the reach."""
        c = dict(b)
        weather, valid = b["weather"].clone(), b["valid"].clone()
        lead = slice(P.HISTORY, P.WINDOW)
        rain = torch.as_tensor(mm, dtype=torch.float32, device=self.device).T
        ok = torch.isfinite(rain) & (rain >= 0) & (valid[:, lead, PRECIP] > 0)
        z = (torch.log1p(torch.nan_to_num(rain).clamp_min(0)) - self.rain_mean) / self.rain_std
        weather[:, lead, PRECIP] = torch.where(ok, z, torch.zeros_like(z))
        valid[:, lead, PRECIP] = ok.float()
        c["weather"], c["valid"] = weather, valid
        return self.assemble(c)
