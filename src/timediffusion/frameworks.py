from typing import Union

from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from .utils import count_params, DimUniversalStandardScaler, kl_div as _kl_div
from .models import TimeDiffusion
    

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

    def fit(self, example: Union[np.ndarray, torch.Tensor], mask: Union[None, np.ndarray, torch.Tensor] = None,
            epochs: int = 20, batch_size: int = 2, steps_per_epoch: int = 32,
            early_stopping_epochs: Union[None, int] = None,
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

            `early_stopping_epochs` - whether to validate model after each epoch
                and stop model after `early_stopping_epochs` restoring quality decrease without improvement

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
        # distance loss definition
        if isinstance(distance_loss, str):
            if distance_loss not in ("MAE", "MSE"): 
                raise NotImplementedError(f"Distance loss {distance_loss} doesn't exist")
            _mae = lambda x, y: (x - y).abs()
            _mse = lambda x, y: ((x - y) ** 2)
            distance_loss = {"MAE": _mae, "MSE": _mse}[distance_loss]
        elif not isinstance(distance_loss, nn.Module):
            raise NotImplementedError(f"Distance loss should be 'MAE', 'MSE' or nn.Module, got {type(distance_loss)}")

        # distribution loss definition
        if isinstance(distribution_loss, str):
            if distribution_loss != "kl_div":
                raise NotImplementedError(f"Distribution loss {distribution_loss} doesn't exist")
            distribution_loss = _kl_div
        elif not isinstance(distribution_loss, nn.Module):
            raise NotImplementedError(f"Distribution loss should be 'kl_div' or nn.Module got {type(distribution_loss)}")
        
        # mask check
        if mask is not None and mask.shape != example.shape:
            raise ValueError(f"Mask should None or the same shape as example, got {example.shape = } and {mask.shape = }")

        # scaling
        self.scaler = DimUniversalStandardScaler()
        train_tensor = torch.tensor(example, dtype=self.dtype(), device=self.device()).unsqueeze(0)
        train_tensor = self.scaler.fit_transform(train_tensor)
        X = train_tensor.repeat(batch_size, *[1] * (len(train_tensor.shape) - 1))

        if mask is not None:
            mask_tensor = ~ torch.tensor(mask, dtype=torch.bool, device=self.device()).unsqueeze(0)
            mask_tensor = mask_tensor.repeat(batch_size, *[1] * (len(mask_tensor.shape) - 1))

        optim = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        if early_stopping_epochs is not None:
            val_losses = []
            val_noise = torch.rand(*X.shape, device=self.device(), dtype=self.dtype())

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
                loss = loss.mean() if mask is None else loss[mask_tensor].mean()
                loss.backward()
                optim.step()

                with torch.no_grad():
                    noise -= y_hat
                losses.append(loss.item())

            # validation
            if early_stopping_epochs is not None:
                with torch.no_grad():
                    cur = val_noise.clone()
                    for step in range(steps_per_epoch):
                        cur -= self.model(cur)
                    val_losses.append(distance_loss(cur, X).mean().item())
                
                best_val_epoch = np.argmin(val_losses)
                if epoch - best_val_epoch - 1 >= early_stopping_epochs:
                    if verbose:
                        print(f"Due to early stopping fitting stops after {epoch}")
                        print(f"Val quality of {best_val_epoch} epoch {val_losses[best_val_epoch]: .1e}")
                        print(f"\tof current {val_losses[- 1]: .1e}")
                    break

        # saving some training parameters, could be useful in inference
        self.training_steps_per_epoch = steps_per_epoch
        self.training_example = example
        self.distance_loss = distance_loss
        self.distribution_loss = distribution_loss
        self.is_fitted = True

        return losses
    
    @torch.no_grad()
    def restore(self, example: Union[None, np.ndarray, torch.Tensor] = None, shape: Union[None, list[int], tuple[int]] = None,
                    mask: Union[None, np.ndarray, torch.Tensor] = None, steps: Union[None, int] = None,
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
                raise ValueError("Either `example` or `shape` should be passed")

            torch.random.manual_seed(seed)
            X = torch.rand(*shape).to(device=self.device(), dtype=self.dtype())

            # no real meaning behind masking random noise
            # maybe for fun, but setting it here as None for stability
            mask = None
        else:
            if len(self.input_dims) != len(example.shape):
                raise ValueError(f"Model fitted with {len(self.input_dims)} dims, but got {len(example.shape)}")

            if self.input_dims[0] != example.shape[0]:
                raise ValueError(f"Model fitted with {self.input_dims[0]} channels, but got {example.shape[0]}")

            X = torch.tensor(example, device=self.device(), dtype=self.dtype())
            X = self.scaler.transform(X)

            if mask is not None:
                if mask.shape != example.shape:
                    raise ValueError(f"Mask should be same shape as example, got {example.shape = } {mask.shape = }")
                
                mask = torch.tensor(mask, device=self.device(), dtype=torch.bool)

            # provided example could have nan values
            nan_mask = torch.isnan(X)
            X[nan_mask] = torch.randn(nan_mask.sum(), device=X.device, dtype=X.dtype)

        steps = self.training_steps_per_epoch if steps is None else steps
        for step in (tqdm(range(steps)) if verbose else range(steps)):
            preds = self.model(X)
            if mask is None:
                X -= preds
            else:
                X[mask] -= preds[mask]

        X = self.scaler.inverse_transform(X)
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

    @torch.no_grad()
    def synth(self, start=None, proximity: float = 0.9, samples: int = 8, batch_size: int = 8, 
            step_granulation : int = 10,
            seed: int = 42, verbose: bool = False) -> torch.Tensor:
        """
        generates synthetic data according to the proximity to the original example
            could start with random noise and denoise it to certain degree
            or start with provided samples and work with them

        args:

            `start` - diffusion process starts with 
                random noise in case None
                with provided samples [samples, channesl, *dims] in other cases

            `proximity` - similarity of syntetic samples to the fitted example
                should be in [0.0, 1.0] where 0.0 - random noise, 1.0 - full restored

            `samples` - number of synthetic samples to generate

            `batch_size` - number of sequences to generate at one diffusion process
                sets tradeoff between speed and memory of generation

            `step_granulation` - number of substeps to split usual step
                significantly increases computation time 
                    in tradeoff of better closeness of synthetic samples to selected proximity

            `seed` - random seed for reproducability

            `verbose` - whether to output progress or not

        returns:
            torch.Tensor of shape [samples, channels, *dims]
        """
        if not self.is_fitted:
            raise RuntimeError("Model isn't fitted")
        if not 0. <= proximity <= 1.:
            raise ValueError(f"proximity should be in [0, 1], got {proximity = }")
        dist = _kl_div
        gran_coef = 1 / step_granulation
        torch.random.manual_seed(seed)
        
        # proximity estimation of fitted model
        # TODO: think of better proximity mechanism
        if start is None:
            x = torch.rand(*self.input_dims, device=self.device(), dtype=self.dtype())
        else:
            x = torch.tensor(start[0], device=self.device(), dtype=self.dtype())
        ref = self.training_example.numpy() if type(self.training_example) is torch.Tensor else self.training_example
        scores = [dist(x.cpu().numpy(), ref).mean()]
        if verbose:
            print("Estimating fitted proximity...")

        _range = range(self.training_steps_per_epoch * step_granulation)
        for _ in (tqdm(_range) if verbose else _range):
            preds = self.model(x) * gran_coef
            x -= preds
            scores.append(dist(x.cpu().numpy(), ref).mean())
        
        scores = 1 - np.array(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        best_step = np.argmin(np.abs(scores - proximity))
        if verbose:
            print(f"Best granulated step is {best_step}")

        # generation
        res = []
        if verbose:
            print("Generating...")
        _range = range(0, samples, batch_size)
        for i in (tqdm(_range) if verbose else _range):
            if start is None:
                x = torch.rand(min(batch_size, samples - i),
                    *self.input_dims, device=self.device(), dtype=self.dtype())
            else:
                x = torch.tensor(start[i: i + batch_size], device=self.device(), dtype=self.dtype())
            for _ in range(best_step - 1):
                x -= self.model(x) * gran_coef
            res.append(x)

        res = torch.concat(res, dim=0)
        res = self.scaler.inverse_transform(res)

        return res
