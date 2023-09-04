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
            model.fit(data, mask=mask, early_stopping_epochs=5, epochs=1, batch_size=1, steps_per_epoch=2)
        except Exception as e:
            pytest.fail(f"TD.fit with {dims = } failed with exception: {e}")

    def test_restore(self, dims, mask_dropout):
        model = TD(input_dims=dims)

        data = np.ones(dims)
        if mask_dropout is None:
            mask = None
        else:
            np.random.seed(42)
            mask = np.random.uniform(low=0., high=1.0, size=data.shape) < mask_dropout
            data[mask] = np.nan

        try:
            model.fit(example=data, mask=mask, epochs=1, batch_size=1, steps_per_epoch=2)
            res = model.restore(data, mask=mask, steps=2)
        except Exception as e:
            pytest.fail(f"TD.restore with {dims = } failed with exception: {e}")

        assert (~ torch.isnan(res)).all()
        if mask is not None:
            # restoring with mask shouldn't change unmasked tokens
            assert np.allclose(res.numpy()[~ mask], data[~ mask])

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
            pytest.fail(f"TD.forecast with {dims = } failed with exception: {e}")

        assert len(res.shape) == 2
        assert res.shape[0] == dims[0]
        assert res.shape[1] == horizon
        assert (~ torch.isnan(res)).all()

    def test_synth(self, dims, mask_dropout):
        # using mask_dropout as a marker to start from noise or not
        start = mask_dropout

        model = TD(input_dims=dims)

        data = np.ones(dims)
        if start is not None:
            start = np.repeat(np.expand_dims(data, 0), 2, axis=0)
        samples = 2
        try:
            model.fit(example=data, epochs=1, batch_size=1, steps_per_epoch=2)
            if start is None:
                res = model.synth(proximity=0.7, step_granulation=2, samples=samples, batch_size=1)
            else:
                res = model.synth(start=start, proximity=0.7, step_granulation=2, samples=samples, batch_size=1)
        except Exception as e:
            pytest.fail(f"TD.synth with {dims = } and {start is None } failed with exception: {e}")
        
        assert len(res) == samples
        assert list(res[0].shape) == dims
        assert (~ torch.isnan(res)).all()
