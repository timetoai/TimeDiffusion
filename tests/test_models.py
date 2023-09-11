import pytest

import numpy as np

import torch
from torch import nn

from timediffusion import TimeDiffusionModel, TimeDiffusionProjector,\
                         TimeDiffusionAttention, TimeDiffusionLiquid


@pytest.mark.parametrize("model_init", [TimeDiffusionProjector, TimeDiffusionAttention, TimeDiffusionLiquid])
@pytest.mark.parametrize("dims", [[1, 35], [1, 7, 7], [1, 5, 5, 5], [2, 35], [2, 7, 7], [2, 5, 5, 5]])
class TestTimeDiffusionModel:
    def test_forward_pass(self, model_init, dims):
        model = model_init(input_dims=dims)
        assert isinstance(model, TimeDiffusionModel)

        # unbatched forward pass
        data = torch.ones(*dims)
        try:
            res = model(data)
        except Exception as e:
            pytest.fail(f"Unbatched forward pass of {type(model).__name__} with {dims = } failed with exception: {e}")

        assert data.shape == res.shape
        assert (~ torch.isnan(res)).all()
        
        # batched forward pass
        data = torch.ones(1, *dims)
        try:
            res = model(data)
        except Exception as e:
            pytest.fail(f"Batched forward pass of {type(model).__name__} with {dims = } failed with exception: {e}")

        assert data.shape == res.shape
        assert (~ torch.isnan(res)).all()

    def test_backward_pass(self, model_init, dims):
        model = model_init(input_dims=dims)
        assert isinstance(model, TimeDiffusionModel)

        # unbatched backward pass
        data = torch.ones(*dims)
        try:
            res = model(data)
            loss = (res - 1).mean()
            loss.backward()
        except Exception as e:
            pytest.fail(f"Unbatched backward pass of {type(model).__name__} with {dims = } failed with exception: {e}")

        assert (~ torch.isnan(loss)).all()
        
        # batched backward pass
        data = torch.ones(1, *dims)
        try:
            res = model(data)
            loss = (res - 1).mean()
            loss.backward()
        except Exception as e:
            pytest.fail(f"Batched backward pass of {type(model).__name__} with {dims = } failed with exception: {e}")

        assert (~ torch.isnan(loss)).all()
