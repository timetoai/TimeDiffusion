import pytest

import numpy as np

import torch
from torch import nn

from timediffusion import count_params, kl_div


@pytest.mark.parametrize(
    "x,y",
    [
        (np.sin(np.arange(10)), np.arange(10)),
        (torch.sin(torch.arange(10)), torch.arange(10)),
        pytest.param(np.arange(10), torch.arange(10), marks=pytest.mark.xfail)
    ],
)
def test_kl_div(x, y):
    kl_div(x, y)

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
