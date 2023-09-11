from typing import Union
from abc import ABC, abstractmethod

import numpy as np

import torch
from torch import nn

from .utils import get_appropriate_conv_layer
from .layers import TemporalBlock


class TimeDiffusionModel(nn.Module, ABC):
    """
    Abstract class for all base models
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwards):
        pass


class TimeDiffusionProjector(TimeDiffusionModel):
    """
    convolutional network, used as projector in TD
    consists of temporal blocks with exponentially increasing padding/dilation parameters
    """
    def __init__(self, input_dims: Union[list[int], tuple[int]], max_deg_constraint: Union[int, None] = None,
                  conv_filters: int = 128, base_dropout: float = 0.05):
        """
        args:

            `input_dims` - [channels, *dims]
                needed for dynamical network building
                best way to pass it as `x.shape` (without batches)

            `max_deg_constraint` - constraint to lessen network size, if not big enough will worsen model quality
                number of temporal blocks in network will be (1 + max_deg_constraint) maximum
                if None: ignores parameter

            `conv_filters` - number of convolutional filters for each layer

            `base_dropout` - dropout for first temporal block
        """
        super().__init__()

        self.input_dims = input_dims
        self.dims = len(input_dims) - 1
        self.channels = input_dims[0]
        self.max_seq = max(input_dims[1:])
        self.max_deg = int(np.ceil(np.log2(self.max_seq)))
        if max_deg_constraint is not None and max_deg_constraint < self.max_deg:
            print(f"For better TimeDiffusion performance it's recommended to use max_deg_constraint ", end="")
            print(f"with value{self.max_deg} for input with shape {input_dims}")
            self.max_deg = max_deg_constraint
            print(f"Setting current {self.max_deg = }")

        self.tcn = nn.ModuleList(
            [TemporalBlock(self.channels, conv_filters, 
                           kernel_size=1, stride=1, dilation=1, padding=0, dropout=base_dropout, dims=self.dims),
             *[TemporalBlock(conv_filters, conv_filters, 
                             kernel_size=2, stride=1, dilation=i, padding=i, dropout=0.0, dims=self.dims)
                                        for i in [2 ** i for i in range(self.max_deg + 1)]]
                                    ])
        
        self.last = get_appropriate_conv_layer(self.dims)(conv_filters, self.channels, kernel_size=1, stride=1, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_acc = None
        for layer in self.tcn:
            skip, x = layer(x)
            if skip_acc is None:
                skip_acc = skip
            else:
                skip_acc += skip
        x = self.last(x + skip_acc)
        return x


class TimeDiffusionAttention(TimeDiffusionModel):
    """
    attention model, uses projectors to create (q, k, v) for vanilla attention layer
    """
    def __init__(self, *args, **params):
        """
        `args`, `params` - parameters for projectors
        """
        super().__init__()
        self.key_proj = TimeDiffusionProjector(*args, **params)
        self.val_proj = TimeDiffusionProjector(*args, **params)
        self.query_proj = TimeDiffusionProjector(*args, **params)

        self.input_dims = self.key_proj.input_dims
        self.dims = self.key_proj.dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # projections
        key = self.key_proj(x)
        val = self.val_proj(x)
        query = self.query_proj(x)

        is_batched = self.dims + 2 == len(key.size())
        mat_mul = torch.bmm if is_batched else torch.matmul

        # flattening last dimensionalities in case of 2D and 3D input
        # TODO: think of better solution
        if self.dims > 1:
            orig_shape = key.shape
            new_shape = list(key.size()[: - self.dims]) + [np.prod(key.size()[ - self.dims:])]

            key = key.view(new_shape)
            val = val.view(new_shape)
            query = query.view(new_shape)

        # vanilla attenion
        scores = mat_mul(query, key.transpose(- 2, - 1))
        weights = torch.nn.functional.softmax(scores, dim=1)
        attention = mat_mul(weights, val)

        # back to original shape in case of 2D and 3D input
        if self.dims > 1:
            attention = attention.view(orig_shape)

        return attention


class TimeDiffusionLiquid(TimeDiffusionModel):
    def __init__(self, input_dims: Union[list[int], tuple[int]], max_deg_constraint: Union[int, None] = None,
                  conv_filters: int = 128, base_dropout: float = 0.05):
        """
        args:

            `input_dims` - [channels, *dims]
                needed for dynamical network building
                best way to pass it as `x.shape` (without batches)

            `max_deg_constraint` - constraint to lessen network size, if not big enough will worsen model quality
                number of temporal blocks in network will be (1 + max_deg_constraint) maximum
                if None: ignores parameter

            `conv_filters` - number of convolutional filters for each layer

            `base_dropout` - dropout for first temporal block
        """
        super().__init__()
        
        self.in_channels = input_dims[0]
        self.max_seq = max(input_dims[1:])
        self.max_deg = int(np.ceil(np.log2(self.max_seq)))
        if max_deg_constraint is not None and max_deg_constraint < self.max_deg:
            print(f"For better TimeDiffusion performance it's recommended to use max_deg_constraint ", end="")
            print(f"with value{self.max_deg} for input with shape {input_dims}")
            self.max_deg = max_deg_constraint
            print(f"Setting current {self.max_deg = }")

        self.projector = nn.Conv1d(self.in_channels, conv_filters, kernel_size=1)
        self.base_dropout = nn.Dropout(base_dropout)
        self.conv = nn.Conv1d(conv_filters, conv_filters, 2)
        self.out_projector = nn.Conv1d(conv_filters, self.in_channels, kernel_size=1)

        self.__init_weights()

    def __init_weights(self):
        nn.init.kaiming_uniform_(self.projector.weight, a=0.01)
        nn.init.kaiming_uniform_(self.conv.weight, a=0.01)
        nn.init.kaiming_uniform_(self.out_projector.weight, a=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        x = self.base_dropout(x)

        # as network is much smaller, it's implemented in more functional way
        for i in range(self.max_deg + 1):
            x = nn.functional.conv1d(x, self.conv.weight, self.conv.bias,
                                     padding=2 ** i, dilation=2 ** i)
            x = nn.functional.relu(x)
            x = x[..., : - 2 ** i]

        x = self.out_projector(x)
        return x
