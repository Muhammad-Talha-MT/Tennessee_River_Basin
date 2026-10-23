"""Historical split isolation, target-independent masking and input ownership."""

import numpy as np
import pandas as pd
import pytest
import torch

from tva_flood.data.dataset import CachedWindows
from tva_flood.data.schema import HISTORY
from tva_flood.data.ssl import EncoderWindows, history_issues, mask_history


def inputs(batch=2, steps=120, nodes=5):
    generator = torch.Generator().manual_seed(3)
    dynamic = torch.randn(batch, steps, nodes, 18, generator=generator)
    valid = torch.ones(batch, steps, nodes, dtype=torch.bool)
    dynamic[..., 17] = 1
    return dynamic, valid


def test_windows_reconstruct_only_inside_their_declared_split(monkeypatch):
    times = pd.date_range("2000-01-01", periods=500, freq="6h")
    split = dict(
        train=[str(times[0]), str(times[240])], validation=[str(times[240]), str(times[480])]
    )
    requests = []

    def initialize(self, root, out, selected, **kwargs):
        self.manifest = {"splits": split}
        self.times = times
        self.static = np.ones((3, 48), dtype=np.float32)
        self.ids = np.arange(3)

    def window(self, start, stop):
        requests.append((start, stop))
        shape = (stop - start, 3)
        flow = np.broadcast_to(np.arange(start, stop, dtype=np.float32)[:, None], shape).copy()
        known = np.ones(shape, dtype=bool)
        known[5, 1] = False
        return np.ones(shape + (8,), np.float32), np.ones(shape + (8,), bool), flow, known

    monkeypatch.setattr(CachedWindows, "__init__", initialize)
    monkeypatch.setattr(CachedWindows, "window", window)
    training = EncoderWindows("unused", "unused", "train")
    validation = EncoderWindows("unused", "unused", "validation")
    assert np.array_equal(training.issues, np.arange(HISTORY, 241))
    assert np.array_equal(validation.issues, np.arange(240 + HISTORY, 481))
    assert training.static.shape == (3, 48)
    for data, bounds in ((training, (0, 240)), (validation, (240, 480))):
        for index in (0, len(data) - 1):
            item = data[index]
            start, stop = requests[-1]
            assert bounds[0] <= start < stop <= bounds[1]
            assert stop - start == HISTORY
            assert item["issue"] == stop
            assert item["dynamic"].shape == (HISTORY, 3, 18)
            assert item["dynamic"].dtype == item["flow"].dtype == torch.float32
            assert item["valid"].dtype == torch.bool
            assert torch.equal(item["dynamic"][..., 17] > 0, item["valid"])
            assert item["dynamic"][5, 1, 16] == 0
            assert torch.equal(item["flow"][:, 0], torch.arange(start, stop).float())
            assert set(item) == {"dynamic", "flow", "valid", "issue"}


def test_history_indices_support_exact_length_and_final_endpoint():
    times = pd.date_range("2000-01-01", periods=HISTORY, freq="6h")
    end = times[-1] + pd.Timedelta(hours=6)
    assert history_issues(times, times[0], end).tolist() == [HISTORY]
    assert history_issues(times, times[0], end - pd.Timedelta(seconds=1)).size == 0
    assert history_issues(times, times[1], end).size == 0
    with pytest.raises(ValueError, match="contiguous"):
        history_issues(times.delete(10), times[0], end)


@pytest.mark.parametrize("mode", ["block", "reach", "all"])
def test_masking_is_repeatable_and_hidden_values_cannot_leak(mode):
    dynamic, valid = inputs()
    valid[:, 3:8, 1] = False
    dynamic[..., 17] = valid.float()
    before = dynamic.clone()
    state = torch.get_rng_state().clone()
    masked, selected = mask_history(dynamic, valid, seed=11, mode=mode)
    again, selected_again = mask_history(dynamic, valid, seed=11, mode=mode)
    assert torch.equal(masked, again) and torch.equal(selected, selected_again)
    assert torch.equal(dynamic, before)
    assert torch.equal(torch.get_rng_state(), state)
    assert not (selected & ~valid).any()
    assert selected.flatten(1).any(dim=1).all()
    assert torch.equal(masked[..., :16], dynamic[..., :16])
    assert torch.count_nonzero(masked[..., 16:18][selected | ~valid]) == 0
    perturbed = dynamic.clone()
    perturbed[..., 16][selected | ~valid] = 123456
    perturbed[..., 17][selected | ~valid] = 1
    changed, changed_selection = mask_history(perturbed, valid, seed=11, mode=mode)
    assert torch.equal(changed_selection, selected)
    assert torch.equal(changed, masked)
    if mode != "all":
        _, next_epoch = mask_history(dynamic, valid, seed=12, mode=mode)
        assert not torch.equal(next_epoch, selected)


def test_blocks_are_contiguous_and_reach_masks_hide_whole_histories():
    dynamic, valid = inputs(nodes=40)
    _, blocks = mask_history(dynamic, valid, seed=5, mode="block")
    for row in blocks.permute(0, 2, 1).reshape(-1, 120):
        positions = torch.flatnonzero(row) if hasattr(torch, "flatnonzero") else row.nonzero()[:, 0]
        assert 28 <= len(positions) <= 60
        assert int(positions[-1] - positions[0]) + 1 == len(positions)
    _, reaches = mask_history(dynamic, valid, seed=6, mode="reach", reach_fraction=0.5)
    assert torch.equal(reaches, reaches[:, :1].expand_as(reaches))
    assert reaches.any() and (~reaches).any()


@pytest.mark.parametrize("mode", ["block", "reach", "all"])
def test_sparse_validity_always_gets_a_loss_target_without_masking_invalid_targets(mode):
    dynamic, valid = inputs(batch=3, steps=8, nodes=2)
    valid[:] = False
    valid[0, 7, 1] = True
    valid[1, 0, 0] = True
    masked, selected = mask_history(
        dynamic, valid, seed=8, mode=mode, block_min=2, block_max=2, reach_fraction=1e-12
    )
    assert torch.equal(selected, valid)
    assert not masked[2, ..., 16:18].any()
    assert not selected[2].any()


def test_bad_mask_contracts_fail():
    dynamic, valid = inputs()
    with pytest.raises(ValueError, match="shaped"):
        mask_history(dynamic[0], valid[0], seed=1, mode="all")
    with pytest.raises(ValueError, match="boolean"):
        mask_history(dynamic, valid.float(), seed=1, mode="all")
    with pytest.raises(ValueError, match="Block bounds"):
        mask_history(dynamic, valid, seed=1, mode="block", block_max=121)
