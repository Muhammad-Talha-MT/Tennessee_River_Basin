"""Helpers for the insertion tests: HydroModel.insertion only reads ``self.router``, so a stub with the router is enough."""

from types import SimpleNamespace

import torch

from tva_flood.hydro import paths as P
from tva_flood.hydro.model import HydroModel


def stub_for_insertion(router):
    q = torch.ones(5, P.FLOW_STEPS)
    router.prepare(router.reference_flow(q))
    return SimpleNamespace(router=router)


def run_insertion(stub, batch, routed_full, forecast, shown):
    return HydroModel.insertion(stub, batch, routed_full, forecast, shown)
