# TimeDiffusion - Unified time series framework for multiple tasks

Supports 2D (image) and 3D (video) data, but is currently not suitable for working with them.

**Install**

```
pip install timediffusion
```


**Quick Start**

Forecasting time seires

```
# train sequence in shape [channels, sequence_length]
model = TD(input_dims=train.shape).to(device=device)
training_losses = model.fit(train)
predictions = model.forecast(horizon)
```

Creating synthetic time series

```
# sequence in shape [channels, sequence_length]
model = TD(input_dims=seq.shape).to(device=device)
training_losses = model.fit(seq)
# proximity - how close to original, samples - total synthetic time series
synthetic_data = model.synth(proximity=0.9, samples=3, batch_size=2, step_granulation=100)
```

Time series Imputation

```
# sequence in shape [channels, sequence_length]
model = TD(input_dims=seq.shape).to(device=device)
# mask - binary array of same shape, as sequence, with 1 in positions, that needed to be overlooked
training_losses = model.fit(seq, mask=mask)
restored_seq = model.restore(example=seq, mask=mask)
```

**Examples**

[Time series tasks example](./examples/example_1d_data.ipynb)

**Philosophy****

TODO

**Model architecture**

TODO
