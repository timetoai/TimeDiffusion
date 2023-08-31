from typing import Union

from tqdm import tqdm

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
        raise NotImplementedError("Convolutional layer for dimensionalty {dims} not implemented")
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]


class DimUniversalStandardScaler:
    """
    Universal class for normal scaling data
    """
    def __init__(self, eps=1e-9):
        self.eps = eps

    def fit(self, data):
        self.mu = data.mean()
        self.std = data.std()
        if isinstance(data, torch.Tensor):
            self.mu = self.mu.item()
            self.std = self.std.item()
    
    def transform(self, data):
        return (data - self.mu) / (self.std + self.eps)
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.std + self.mu


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


class TimeDiffusionProjector(nn.Module):
    """
    convolutional network, used as projector in TD
    consists of temporal blocks with exponentially increasing padding/dilation parameters
    """
    def __init__(self, input_dims: Union[list[int], tuple[int]], max_deg_constraint: int = 13,
                  conv_filters: int = 128, base_dropout: float = 0.05):
        """
        args:

            `input_dims` - [channels, *dims]
                needed for dynamical network building
                best way to pass it as `x.shape` (without batches)

            `max_deg_constraint` - constraint to lessen network size, if not big enough will worsen model quality
                number of temporal blocks in network will be (1 + max_deg_constraint) maximum

            `conv_filters` - number of convolutional filters for each layer

            `base_dropout` - dropout for first temporal block
        """
        super().__init__()

        self.input_dims = input_dims
        self.dims = len(input_dims) - 1
        self.channels = input_dims[0]
        self.max_seq = max(input_dims[1:])
        self.max_deg = int(np.ceil(np.log2(self.max_seq)))
        if max_deg_constraint < self.max_deg:
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


class TimeDiffusion(nn.Module):
    """
    main model, uses projectors to create (q, k, v) for vanilla attention layer
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
    

class TD(nn.Module):
    """
    Class provides a convenient framework for effectively working with TimeDiffusion, encompassing all essential functions.
    """
    def __init__(self, verbose: bool = False, seed=42, *args, **params):
        """
        args (mostly same as TimeDiffusionProjector):

            `verbose` - whether to report number of model parameters

            `seed` - random seed for model parameters initialization

            `input_dims` - [channels, *dims]
                needed for dynamical network building
                best way to pass it as `x.shape` (without batches)

            `max_deg_constraint` - constraint to lessen network size, if not big enough will worsen model quality
                number of temporal blocks in network will be (1 + max_deg_constraint) maximum

            `conv_filters` - number of convolutional filters for each layer

            `base_dropout` - dropout for first temporal block
        """
        super().__init__()
        torch.random.manual_seed(seed)
        self.model = TimeDiffusion(*args, **params)
        self.input_dims = self.model.input_dims
        self.is_fitted = False
        if verbose:
            print(f"Created model with {count_params(self):.1e} parameters")

    def dtype(self):
        return next(self.model.parameters()).dtype

    def device(self):
        return  next(self.model.parameters()).device

    def fit(self, example: Union[np.array, torch.Tensor], mask: Union[None, np.array, torch.Tensor] = None,
            epochs: int = 20, batch_size: int = 2, steps_per_epoch: int = 32,
            lr: float = 4e-4, distance_loss: Union[str, nn.Module] = "MAE",
            distribution_loss: Union[str, nn.Module] = "kl_div", distrib_loss_coef = 1e-2,
            verbose: bool = False, seed=42) -> list[float]:
        """
        simulates diffusion process for model training

        args:

            `example` - [sequence | image | video] in format (channels, *dims)

            `mask` - None for full model fitting on `example`
                or same shape as `example` for not fitting in points, that masked with 1

            `epochs` - number of training epochs

            `batch_size` - number of random noises to train on
                balance between (epochs - batch_size) means balance between (time - memory)
                more batch_size usually gives better results, but increasing epochs gives more significant improvement

            `steps_per_epoch` - number of diffusion steps to train each epoch

            `lr` - learning rate

            `distance_loss` - main loss for fitting into input exampls
                should be either "MAE", "MSE"
                or pytorch nn.Module, that produces some kind of distance loss without dimensionality reduction

            `distribution_loss` - additional loss
                should be either `kl_div` for using built-in Kullback–Leibler divergence
                or pytorch nn.Module, that produces some kind of distributional loss without dimensionality reduction
                if pytorch module, it should take tensors with shape [batch_size, channels, *dims]

            `distrib_loss_coef` - scale of distribution loss in total loss

            `verbose` - whether to output training progress or not

            `seed` - random seed for fit method reproducibility

        returns:
            list of training losses (per step for each epoch)
        """
        def _kl_div(x, y, eps=1e-3):
            x = (x - x.min())  / torch.clip(x.max() - x.min(), min=eps) + 1e-12
            y = (y - y.min()) / torch.clip(y.max() - y.min(), min=eps) + 1e-12
            return (torch.log(x / torch.clip(y, min=eps)) * x)
        
        _mae = lambda x, y: (x - y).abs()
        _mse = lambda x, y: ((x - y) ** 2)

        if isinstance(distance_loss, str):
            if distance_loss not in ("MAE", "MSE"): 
                raise NotImplementedError(f"Distance loss {distance_loss} doesn't exist")
            distance_loss = {"MAE": _mae, "MSE": _mse}[distance_loss]
        elif not isinstance(distance_loss, nn.Module):
            raise NotImplementedError(f"Distance loss should be 'MAE', 'MSE' or nn.Module, got {type(distance_loss)}")

        if isinstance(distribution_loss, str):
            if distribution_loss != "kl_div":
                raise NotImplementedError(f"Distribution loss {distribution_loss} doesn't exist")
            distribution_loss = _kl_div
        elif not isinstance(distribution_loss, nn.Module):
            raise NotImplementedError(f"Distribution loss should be 'kl_div' or nn.Module got {type(distribution_loss)}")
        
        if mask is not None and mask.shape != example.shape:
            raise ValueError(f"Mask should None or the same shape as example, got {example.shape = } and {mask.shape = }")

        self.scaler = DimUniversalStandardScaler()
        train_tensor = torch.tensor(self.scaler.fit_transform(example), dtype=self.dtype(), device=self.device()).unsqueeze(0)
        X = train_tensor.repeat(batch_size, *[1] * (len(train_tensor.shape) - 1))

        if mask is not None:
            mask_tensor = ~ torch.tensor(mask, dtype=torch.bool, device=self.device())

        optim = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        torch.random.manual_seed(seed)
        for epoch in (tqdm(range(1, epochs + 1)) if verbose else range(1, epochs + 1)):
            self.model.train()

            noise = torch.rand(*X.shape, device=self.device(), dtype=self.dtype())
            # noise_level = torch.rand(X.shape).to(device=self.device(), dtype=self.dtype())
            # noise *= noise_level
            # scaling random noise with noise level gives additional training diversity and stability in some cases
            # TODO: further research in this area

            for step in range(steps_per_epoch):
                optim.zero_grad()
                y_hat = self.model(noise)
                # noise - y_hat -> X
                loss = distance_loss(noise - y_hat, X) + distrib_loss_coef * distribution_loss(y_hat, noise)
                loss = loss.mean() if mask is None else (loss * mask_tensor).mean()
                loss.backward()
                optim.step()

                with torch.no_grad():
                    noise -= y_hat
                losses.append(loss.item())

        self.training_steps_per_epoch = steps_per_epoch
        self.training_example = example
        self.is_fitted = True

        return losses
    
    def restore(self, example: Union[None, np.array, torch.Tensor] = None, shape: Union[None, list[int], tuple[int]] = None,
                    mask: Union[None, np.array, torch.Tensor] = None, steps: Union[None, int] = None,
                    seed: int = 42, verbose: bool = False) -> torch.Tensor:
        """
        recreates data using fitted model

        either `example` or `shape` should be provided

        3 possible workflows

            1 case of `shape`: model starts with random noise

            2 case of `example`: ignores `shape` and model starts with `example`

            3 case of `example` and `mask`: same as 2 case, but masked values persistent through diffusion process

        args:

            `example` - None or in format [channels, *dims], channels should be the same as in training example

            `shape` - None or in format [channels, *dims], channels should be the same as in training example

            `mask` - None or in format of `example`, zeros in positions, that needed to be persistent

            `steps` - steps for diffusion process, if None uses same value as in fit method

            `seed` - random seed, only necessary in case of providing only `shape`

            `verbose` - whether to output progress of diffusion process or not

        returns:
            result of diffusion process (torch.Tensor)
        """
        if not self.is_fitted:
            raise RuntimeError("Model isn't fitted")

        if example is None:
            if shape is None:
                raise ValueEror("Either `example` or `shape` should be passed")

            torch.random.manual_seed(seed)
            X = torch.rand(*dims).to(device=self.device(), dtype=self.dtype())
        else:
            if len(self.input_dims) != len(example.shape):
                raise ValueError(f"Model fitted with {len(self.input_dims)} dims, but got {len(example.shape)}")

            if self.input_dims[0] != example.shape[0]:
                raise ValueError(f"Model fitted with {self.input_dims[0]} channels, but got {example.shape[0]}")

            X = torch.tensor(example, device=self.device(), dtype=self.dtype())

            if mask is not None:
                if mask.shape != example.shape:
                    raise ValueError(f"Mask should be same shape as example, got {example.shape = } {mask.shape = }")
                
                mask = ~ torch.tensor(mask, device=self.device(), dtype=torch.bool)

        steps = self.training_steps_per_epoch if steps is None else steps
        for step in (tqdm(range(steps)) if verbose else range(steps)):
            with torch.no_grad():
                preds = self.model(X)
                if mask is None:
                    X -= preds
                else:
                    X[mask] -= preds[mask]

        return X

    def forecast(self, horizon: Union[int, tuple[int], list[int]], steps: Union[None, int] = None,
                    seed: int = 42, verbose: bool = False) -> torch.Tensor:
        """
        convinient version of `restore` to only get prediction for forecasting `horizon`
        uses trained sequence as masked (persistent) reference to forecast next values

        WORKS ONLY FOR 1D DATA

        args:

            `horizon` - forecasting horizon (e.g. number of values to forecast)

            `steps` - steps for diffusion process, if None uses same value as in fit method

            `seed` - random seed for reproducability

            `verbose` - whether to output progress of diffusion process or not

        returns:
            forecasted values (torch.Tensor in shape [channels, horizon])
        """
        if not self.is_fitted:
            raise RuntimeError("Model isn't fitted")
        if len(self.input_dims) != 2:
            raise NotImplementedError("forecast method works only for 1D data")
        
        np.random.seed(seed)
        example = np.append(self.training_example, np.zeros((self.input_dims[0], horizon)), axis=1)
        mask = np.zeros_like(example)
        mask[:, - horizon:] = 1

        res = self.restore(example=example, mask=mask, steps=steps, seed=seed, verbose=verbose)
        return res[:, - horizon:]
