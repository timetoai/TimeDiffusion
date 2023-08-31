from typing import Union

import torch
from torch import nn

from .utils import get_appropriate_conv_layer


class Chomp(nn.Module):
    """
    cuts padding part of sequence
    inspired by https://github.com/locuslab/TCN
    """
    def __init__(self, chomp_size: int, dims: int=1):
        """
        args:
            `chomp_size` - padding size to cut off
            `dims` - number of working dimensionalities, which needed to be chomped
        """
        super().__init__()
        self.chomp_size = chomp_size
        if dims not in (1, 2, 3):
            raise NotImplementedError(f"Chomp layer for {dims = } not implemented")
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 1:
            return x[..., : - self.chomp_size].contiguous()
        if self.dims == 2:
            return x[..., : - self.chomp_size, : - self.chomp_size].contiguous()
        if self.dims == 3:
            return x[..., : - self.chomp_size, : - self.chomp_size,  : - self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    combination of (convolutional layer, chomp, relu, dropout) repeated `layers` times
    adds additional convolutional layer if needed to downsample number of channels
    inspired by https://github.com/locuslab/TCN
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: Union[int, tuple[int]],
                  stride: Union[int, tuple[int]], dilation: Union[int, tuple[int]], padding: Union[int, tuple[int]],
                  dropout: int = 0.2, dims: int = 1, layers: int = 2):
        super().__init__()

        conv_layer = get_appropriate_conv_layer(dims)
        self.padding = padding
        self.dropout = dropout

        net = []
        for i in range(layers):
            net.append(torch.nn.utils.weight_norm(conv_layer(
                (n_inputs if i == 0 else n_outputs), n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
            if padding > 0:
                net.append(Chomp(padding, dims))
            net.append(nn.ReLU())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        self.net = nn.ModuleList(net)

        self.downsample = conv_layer(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        sets normal weight distribution for convolutional layers
        """
        for i in range(0, len(self.net), 2 + (self.dropout > 0) + (self.padding > 0)):
            self.net[i].weight.data.normal_(0, 0.5)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `x` in format [batch_size, channels, *other_dims]
        """
        out = x
        for i in range(len(self.net)):
            out = self.net[i](out)

        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)
