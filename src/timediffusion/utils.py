from typing import Union

import numpy as np

import torch
from torch import nn


def count_params(model: nn.Module) -> int:
    """
    counts number of model parameters
    """
    res = 0
    for param in model.parameters():
        res += np.prod(param.shape)
    return res


def get_appropriate_conv_layer(dims: int) -> nn.Module:
    """
    returns appropriate convolutional layer for certain number of dimensionalities
    """
    if dims not in (1, 2, 3):
        raise NotImplementedError(f"Convolutional layer for dimensionalty {dims} not implemented")
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]


def kl_div(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
           eps: float = 1e-6):
    """
    Calculates kl_div on min-max version of x, y
    
    args:
        `x` - input array
        `y` - reference array
    """
    if type(x) is not type(y):
        raise ValueError(f"input arrays for kl_div should be same type, got {type(x)} and {type(y)}")

    if type(x) is np.ndarray:
        clip = lambda arr: np.clip(arr, a_min=eps, a_max=1)
        log = np.log
    elif type(x) is torch.Tensor:
        clip = lambda arr: torch.clip(arr, min=eps, max=1)
        log = torch.log
    else:
        raise NotImplementedError(f"kl_div array type should be numpy.array or torch.tensor`, got {type(x)}")
    
    # biased min-max to get rid off zeros
    x = (x - x.min())  / clip(x.max() - x.min())
    y = (y - y.min()) / clip(y.max() - y.min())
    x = clip(x)
    y = clip(y)

    # to probs
    x /= x.sum()
    y /= y.sum()

    return (log(x / y) * x)


class DimUniversalStandardScaler:
    """
    Universal class for normal scaling data
    """
    def __init__(self, eps=1e-9):
        self.eps = eps

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            mask = ~ torch.isnan(data)
            self.mu = data[mask].mean().item()
            self.std = data[mask].std().item()
        else:
            mask = ~ np.isnan(data)
            self.mu = data[mask].mean()
            self.std = data[mask].std()

    def transform(self, data):
        return (data - self.mu) / (self.std + self.eps)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * (self.std + self.eps) + self.mu
