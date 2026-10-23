"""Causality, sparse graph context, encoder transfer and reconstruction gradients."""

import numpy as np
import torch

from tva_flood.models.sparse import sparse_supports
from tva_flood.models.ssl import ForecastModel, SSLReconstructor, TemporalGraphEncoder


def inputs(nodes=4, steps=20):
    dynamic = torch.randn(2, steps, nodes, 18)
    dynamic[..., 8:16] = 1
    dynamic[..., 17] = 1
    static = torch.randn(nodes, 48)
    return dynamic, static


def encoder(nodes=4, **kwargs):
    supports = sparse_supports(nodes, np.array([[0, 1], [1, 2]]))
    return TemporalGraphEncoder(supports, hidden=8, **kwargs)


def test_causal_outputs_in_training_and_evaluation():
    torch.manual_seed(1)
    model = SSLReconstructor(encoder())
    dynamic, static = inputs()
    changed = dynamic.clone()
    changed[:, 10:] = torch.randn_like(changed[:, 10:]) * 8
    for training in (True, False):
        model.train(training)
        with torch.no_grad():
            baseline = model(dynamic, static)
            result = model(changed, static)
        torch.testing.assert_close(result[:, :10], baseline[:, :10], rtol=0, atol=0)
        assert not torch.equal(result[:, 10:], baseline[:, 10:])
    assert model.encoder.receptive_field == 127
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )


def test_graph_context_changes_connected_nodes_only():
    torch.manual_seed(2)
    model = encoder().eval()
    dynamic, static = inputs()
    changed = dynamic.clone()
    changed[:, :, 0, 0] += 4
    with torch.no_grad():
        baseline = model(dynamic, static)
        result = model(changed, static)
    assert (result[:, :, 1] - baseline[:, :, 1]).abs().max() > 1e-5
    assert (result[:, :, 2] - baseline[:, :, 2]).abs().max() > 1e-5
    torch.testing.assert_close(result[:, :, 3], baseline[:, :, 3], rtol=0, atol=0)


def test_reconstruction_updates_all_encoder_parts_and_checkpoint_matches():
    torch.manual_seed(3)
    plain = SSLReconstructor(encoder(dropout=0.1))
    recomputed = SSLReconstructor(encoder(dropout=0.1, checkpoint_blocks=True))
    recomputed.load_state_dict(plain.state_dict())
    dynamic, static = inputs(steps=80)
    target = torch.randn(dynamic.shape[:3])
    predictions = []
    for model in (plain, recomputed):
        torch.manual_seed(40)
        result = model(dynamic, static)
        predictions.append(result.detach())
        (result - target).square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    for (name, left), (_, right) in zip(plain.named_parameters(), recomputed.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, msg=name)


def test_encoder_transfer_and_forecast_paths():
    torch.manual_seed(4)
    reconstruction = SSLReconstructor(encoder())
    transferred = encoder()
    transferred.load_state_dict(reconstruction.encoder.state_dict(), strict=True)
    forecast = ForecastModel(transferred, horizon=6)
    dynamic, static = inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            transferred(dynamic, static), reconstruction.encoder(dynamic, static), rtol=0, atol=0
        )
        result = forecast(dynamic, static)
        expected = dynamic[:, -1, :, 16][:, None].expand(-1, 6, -1)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # Supports are supplied by the target dataset, never silently replaced by
    # source-checkpoint graph buffers when moving the shared encoder to a graph.
    different_graph = encoder(nodes=5)
    different_graph.load_state_dict(reconstruction.encoder.state_dict(), strict=True)
    assert different_graph.forward_support.shape == (5, 5)
    dynamic[..., 16:18] = 0
    assert torch.isfinite(reconstruction(dynamic, static)).all()
    result = forecast(dynamic, static)
    assert result.shape == (2, 6, 4) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert forecast.missing_prediction.weight.grad.abs().sum() > 0
    assert forecast.encoder.dynamic_projection.weight.grad.abs().sum() > 0
