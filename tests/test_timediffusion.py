import pytest

import numpy as np

import torch
from torch import nn

from timediffusion import count_params, TimeDiffusionProjector, TimeDiffusion, TD


@pytest.mark.parametrize(
    "in_features,out_features",
    [(100, 100), (200, 200)],
)
def test_count_params_linear(in_features, out_features):
    linear = nn.Linear(in_features, out_features)
    assert count_params(linear) == in_features * out_features + out_features

@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,groups",
    [
        (3, 24, 3, 1),
        (1, 10, 2, 1),
        (3, 12, 4, 3)
     ],
)
def test_count_params_conv(in_channels, out_channels, kernel_size, groups):
    conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
         kernel_size=kernel_size, groups=groups)
    assert count_params(conv) == out_channels * (in_channels // groups) * kernel_size + out_channels


@pytest.mark.parametrize("model_init", [TimeDiffusionProjector, TimeDiffusion])
@pytest.mark.parametrize("dims", [[1, 35], [1, 7, 7], [1, 5, 5, 5], [2, 35], [2, 7, 7], [2, 5, 5, 5]])
class TestTimeDiffusion:
    def test_forward_pass(self, model_init, dims):
        model = model_init(input_dims=dims)

        # unbatched forward pass
        data = torch.ones(*dims)
        try:
            res = model(data)
        except Exception as e:
            pytest.fail(f"Unbatched forward pass of {type(model).__name__} with {dims = } failed with exception: {e}")
        assert data.shape == res.shape
        
        # batched forward pass
        data = torch.ones(1, *dims)
        try:
            res = model(data)
        except Exception as e:
            pytest.fail(f"Batched forward pass of {type(model).__name__} with {dims = } failed with exception: {e}")
        assert data.shape == res.shape

    def test_backward_pass(self, model_init, dims):
        model = model_init(input_dims=dims)

        # unbatched backward pass
        data = torch.ones(*dims)
        try:
            res = model(data)
            loss = (res - 1).mean().backward()
        except Exception as e:
            pytest.fail(f"Unbatched backward pass of {type(model).__name__} with {dims = } failed with exception: {e}")
        
        # batched backward pass
        data = torch.ones(1, *dims)
        try:
            res = model(data)
            loss = (res - 1).mean().backward()
        except Exception as e:
            pytest.fail(f"Batched backward pass of {type(model).__name__} with {dims = } failed with exception: {e}")


@pytest.mark.parametrize("dims", [[1, 35], [1, 7, 7], [1, 5, 5, 5], [2, 35], [2, 7, 7], [2, 5, 5, 5]])
@pytest.mark.parametrize("mask_dropout", [None, 0.2])
class TestTD:
    def test_fit(self, dims, mask_dropout):
        model = TD(input_dims=dims)

        data = np.ones(dims)
        if mask_dropout is None:
            mask = None
        else:
            np.random.seed(42)
            mask = np.random.uniform(low=0., high=1.0, size=data.shape) < mask_dropout
        
        try:
            model.fit(data, mask=mask, epochs=1, batch_size=1, steps_per_epoch=2)
        except Exception as e:
            pytest.fail(f"TD fit with {dims = } failed with exception: {e}")
