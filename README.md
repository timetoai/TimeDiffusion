# TimeDiffusion - Unified Framework for Multiple Time Series Tasks

Supports 2D (image) and 3D (video) data as input for research purposes.

## Install

```bash
pip install timediffusion
```


## Quick Start

**Forecasting time seires**

```python
# train sequence in shape [channels, sequence_length]
model = TD(input_dims=train.shape).to(device=device)
training_losses = model.fit(train)
predictions = model.forecast(horizon)
```

**Creating synthetic time series**

```python
# sequence in shape [channels, sequence_length]
model = TD(input_dims=seq.shape).to(device=device)
training_losses = model.fit(seq)
# proximity - how close to original, samples - total synthetic time series
synthetic_data = model.synth(proximity=0.9, samples=3, batch_size=2, step_granulation=100)
```

**Time series Imputation**

```python
# sequence in shape [channels, sequence_length]
model = TD(input_dims=seq.shape).to(device=device)
# mask - binary array of same shape, as sequence, with 1 in positions, that needed to be overlooked
training_losses = model.fit(seq, mask=mask)
restored_seq = model.restore(example=seq, mask=mask)
```

## Examples

[Time series tasks example](./examples/example_1d_data.ipynb)

## Philosophy

Main synopsis behind TimeDiffusion model is that in reality, when working with time series we don’t have many samples, as it could be in other machine learning fields (e.g. cv, nlp). Thus, classical autoregressive approaches like ARIMA has the most suitable approach of fitting / training only on original sequence (maybe with some exogenous data).

TimeDiffusion takes inspiration from these established methods and only trains on the input sample. Model incorporates most powerful modern deep learning techniques such as diffusion process, exponential dilated convolutions, residual connections and attention mechanism

## Model architecture

TODO
