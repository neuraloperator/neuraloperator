import pytest
import torch
from torch.testing import assert_close

from ..meta_losses import Relobralo, SoftAdapt


@pytest.mark.parametrize("aggregator_class", [SoftAdapt, Relobralo])
@pytest.mark.parametrize(
    "previous,current",
    [
        ([0.0, 1.0], [0.0, 2.0]),
        ([0.0, 1.0], [0.5, 2.0]),
        ([1e-14, 1.0], [1e-8, 2.0]),
        ([1.0, 2.0], [3.0, 4.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_adaptive_losses_regularized_ratios(aggregator_class, previous, current):
    """Zero and small loss histories must retain finite, correctly weighted gradients."""
    current = torch.tensor(current, requires_grad=True)
    kwargs = {"alpha": 0.5, "beta": 0.0} if aggregator_class is Relobralo else {}
    aggregator = aggregator_class([current], num_losses=2, **kwargs)
    aggregator(dict(enumerate(torch.tensor(previous))), step=0)

    loss, weights = aggregator(dict(enumerate(current)), step=1)
    ratios = current.detach() / (torch.tensor(previous) + aggregator.eps)
    expected_weights = 2 * torch.softmax(ratios, dim=0)
    expected_loss = torch.dot(expected_weights, current)

    assert torch.isfinite(loss)
    assert torch.isfinite(weights).all()
    assert_close(loss, expected_loss)
    loss.backward()
    assert_close(current.grad, expected_weights)


@pytest.mark.parametrize("aggregator_class", [SoftAdapt, Relobralo])
def test_adaptive_losses_zero_static_weight(aggregator_class):
    """A disabled objective must not poison later optimization steps."""
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    aggregator = aggregator_class([parameter], num_losses=2, weights={"physics": 0.0})
    for step in range(3):
        loss, weights = aggregator(
            {"data": parameter.square(), "physics": (parameter - 1).square()},
            step=step,
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(weights).all()
        loss.backward()
        assert torch.isfinite(parameter.grad)
        assert parameter.grad > 0
        parameter.grad = None
