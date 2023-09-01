import pytest

import numpy as np

import torch
from torch import nn

from timediffusion import TD


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

    def test_restore(self, dims, mask_dropout):
        model = TD(input_dims=dims)

        data = np.ones(dims)
        if mask_dropout is None:
            mask = None
        else:
            np.random.seed(42)
            mask = np.random.uniform(low=0., high=1.0, size=data.shape) < mask_dropout

        try:
            model.fit(example=data, mask=mask, epochs=1, batch_size=1, steps_per_epoch=2)
            res = model.restore(data, mask=mask, steps=2)
        except Exception as e:
            pytest.fail(f"TD restore with {dims = } failed with exception: {e}")

        if mask is not None:
            assert np.allclose(res.numpy()[mask], data[mask])

    def test_forecast(self, dims, mask_dropout):
        if len(dims) > 2:
            return
        horizon = 3

        model = TD(input_dims=dims)

        data = np.ones(dims)
        if mask_dropout is None:
            mask = None
        else:
            np.random.seed(42)
            mask = np.random.uniform(low=0., high=1.0, size=data.shape) < mask_dropout

        try:
            model.fit(example=data, mask=mask, epochs=1, batch_size=1, steps_per_epoch=2)
            res = model.forecast(horizon, steps=2)
        except Exception as e:
            pytest.fail(f"TD restore with {dims = } failed with exception: {e}")

        assert len(res.shape) == 2
        assert res.shape[0] == dims[0]
        assert res.shape[1] == horizon
