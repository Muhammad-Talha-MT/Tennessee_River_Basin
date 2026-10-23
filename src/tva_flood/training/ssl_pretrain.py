"""Masked flow-reconstruction pretraining of the shared temporal/graph encoder.

Each training window receives one structured mask, drawn from --mask-mix:
contiguous blocks per reach, a forecast-shaped tail (optionally without
weather), independent whole reaches, connected patches of whole reaches, or
all flow. The causal encoder reconstructs only hidden valid values from earlier
flow, weather, statics and connected reaches. Validation uses fixed masks of
every kind on the full graph and reports skill against causal persistence
(each reach's last visible earlier value; the training mean if none).
"""

import numpy as np
import pandas as pd
import torch

from tva_flood.data.ssl import FLOW, FLOW_VALID, causal_persistence, mask_history

from .encoder_loop import VALIDATION_SEED, derived_seed, resolve_architecture
from .encoder_loop import main as run_main
from .runtime import reduced

DEFAULT_MIX = "block=0.25,tail=0.15,tail_noweather=0.15,reach=0.1,patch=0.25,all=0.1"
# Validation always scores every kind with fixed settings so runs are comparable.
VALIDATION_KINDS = ("block", "tail", "tail_noweather", "reach", "patch", "all")
VALIDATION_SETTINGS = dict(
    block_min=28, block_max=60, reach_fraction=0.5, tail_min=4, tail_max=40, patch_hops=(3, 8)
)


def parse_mix(text):
    mix = {}
    for part in text.split(","):
        name, _, weight = part.partition("=")
        if name.strip() not in VALIDATION_KINDS:
            raise ValueError(f"Unknown mask kind {name!r}; use {', '.join(VALIDATION_KINDS)}")
        mix[name.strip()] = float(weight)
    if any(w < 0 for w in mix.values()) or not sum(mix.values()):
        raise ValueError("Mask weights must be nonnegative with a positive sum")
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items() if v > 0}


def masked_sample(dynamic, valid, seed, kind, adjacency, settings):
    mode = "tail" if kind == "tail_noweather" else kind
    return mask_history(
        dynamic,
        valid,
        seed=seed,
        mode=mode,
        adjacency=adjacency,
        hide_weather=kind == "tail_noweather",
        **settings,
    )


class Reconstruction:
    format = "ssl-reconstruction-v1"
    horizon = 0

    def arguments(self, p):
        from .encoder_loop import add_arguments

        add_arguments(p, updates=6000, batch_size=4, subgraph_nodes=6000, lr=1e-3)
        p.add_argument(
            "--mask-mix",
            default=DEFAULT_MIX,
            help="Comma-separated kind=weight; kinds: " + ", ".join(VALIDATION_KINDS),
        )
        p.add_argument("--block-min", type=int, default=28)
        p.add_argument("--block-max", type=int, default=60)
        p.add_argument("--tail-min", type=int, default=4)
        p.add_argument("--tail-max", type=int, default=40)
        p.add_argument("--reach-fraction", type=float, default=0.5)
        p.add_argument("--patch-min-hops", type=int, default=3)
        p.add_argument("--patch-max-hops", type=int, default=8)

    def check(self, p, a):
        try:
            self.mix = parse_mix(a.mask_mix)
        except ValueError as error:
            p.error(str(error))
        if not 1 <= a.block_min <= a.block_max <= 120 or not 1 <= a.tail_min <= a.tail_max <= 120:
            p.error("Block/tail bounds must satisfy 1 <= min <= max <= 120")
        if not 0 < a.reach_fraction <= 1 or not 0 <= a.patch_min_hops <= a.patch_max_hops:
            p.error("Reach fraction in (0,1]; 0 <= patch min hops <= max hops")
        self.settings = dict(
            block_min=a.block_min,
            block_max=a.block_max,
            reach_fraction=a.reach_fraction,
            tail_min=a.tail_min,
            tail_max=a.tail_max,
            patch_hops=(a.patch_min_hops, a.patch_max_hops),
        )

    def architecture(self, p, a):
        return resolve_architecture(p, a)

    def build(self, encoder):
        from tva_flood.models.ssl import SSLReconstructor

        return SSLReconstructor(encoder)

    def initialize(self, model, run):
        pass

    def parameter_groups(self, model, a):
        return [dict(params=list(model.parameters()), lr=a.lr)]

    def config(self, a):
        return dict(
            task="masked flow reconstruction",
            mask_mix=self.mix,
            mask_settings={
                k: list(v) if isinstance(v, tuple) else v for k, v in self.settings.items()
            },
            validation_kinds=list(VALIDATION_KINDS),
            validation_mask_settings={
                k: list(v) if isinstance(v, tuple) else v for k, v in VALIDATION_SETTINGS.items()
            },
            loss="MSE of normalized log flow at hidden valid targets of core reaches",
            selection="mean normalized MSE over validation mask kinds",
        )

    def prepare(self, batch, graph, seed, a):
        kinds, weights = zip(*self.mix.items())
        rng = np.random.default_rng(seed)
        dynamic, loss_mask = [], []
        for sample in range(batch["dynamic"].shape[0]):
            kind = kinds[rng.choice(len(kinds), p=weights)]
            masked, selected = masked_sample(
                batch["dynamic"][sample : sample + 1],
                batch["valid"][sample : sample + 1],
                derived_seed(seed, sample),
                kind,
                graph["adjacency"],
                self.settings,
            )
            dynamic.append(masked)
            loss_mask.append(selected)
        batch["dynamic"] = torch.cat(dynamic)
        batch["loss_mask"] = torch.cat(loss_mask) & batch["core"][None, None]
        del batch["valid"]

    def loss(self, model, batch, supports, run):
        prediction = model(batch["dynamic"], batch["static"], supports).float()
        mask = batch["loss_mask"]
        target = batch["flow"]
        visible = batch["dynamic"][..., FLOW_VALID] > 0.5
        baseline = causal_persistence(batch["dynamic"][..., FLOW].float(), visible)
        sse = (prediction - target)[mask].square().sum()
        baseline_sse = (baseline - target)[mask].square().sum()
        return sse, mask.sum(), [baseline_sse]

    def log_values(self, statistics, total):
        return dict(persistence_mse=statistics[2].item() / max(total, 1))

    def validate(self, run):
        model, data, device = run.model, run.validation, run.device
        totals = torch.zeros((len(VALIDATION_KINDS), 3), dtype=torch.float64, device=device)
        local = run.items[run.rank :: run.world]
        for position, item in enumerate(local):
            sample = data.load(int(item))
            flow = sample["flow"][None].to(device)
            for k, kind in enumerate(VALIDATION_KINDS):
                masked, mask = masked_sample(
                    sample["dynamic"][None],
                    sample["valid"][None],
                    derived_seed(VALIDATION_SEED, item, k),
                    kind,
                    run.full_graph["adjacency"],
                    VALIDATION_SETTINGS,
                )
                masked, mask = masked.to(device), mask.to(device)
                prediction = model(masked, run.full_static).float()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite validation reconstruction")
                visible = masked[..., FLOW_VALID] > 0.5
                baseline = causal_persistence(masked[..., FLOW].float(), visible)
                totals[k, 0] += (prediction - flow)[mask].double().square().sum()
                totals[k, 1] += (baseline - flow)[mask].double().square().sum()
                totals[k, 2] += mask.sum()
            if run.rank == 0 and (position % 8 == 0 or position + 1 == len(local)):
                print(f"validation windows {position + 1}/{len(local)} (rank 0)", flush=True)
        values = reduced(totals, run.world).cpu().numpy()
        rows = []
        for k, kind in enumerate(VALIDATION_KINDS):
            model_sse, baseline_sse, count = values[k]
            mse = model_sse / count if count else None
            base = baseline_sse / count if count else None
            rows.append(
                dict(
                    kind=kind,
                    targets=int(count),
                    mse=mse,
                    persistence_mse=base,
                    skill_vs_persistence=1 - mse / base if count and base else None,
                )
            )
        scored = [r["mse"] for r in rows if r["mse"] is not None]
        if not scored:
            raise ValueError("Validation produced no hidden targets")
        result = dict(
            selection=float(np.mean(scored)),
            **{f"mse_{r['kind']}": r["mse"] for r in rows},
            **{f"skill_{r['kind']}": r["skill_vs_persistence"] for r in rows},
            validation_windows=len(run.items),
        )
        result["tables"] = dict(by_kind=pd.DataFrame(rows))
        return result


def main():
    run_main(Reconstruction(), __doc__)


if __name__ == "__main__":
    main()
