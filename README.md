# TimeDiffusion - Unified Framework for Multiple Time Series Tasks

Supports 2D (image) and 3D (video) data as input for research purposes.

## Contents
* [Install](#install)
* [Quick Start](#quick-start)
* [Examples](#examples)
* [Philosophy](#philosophy)
* [Base Models](#base-models)
* [Model architecture](#model-architecture)

## Install

```bash
pip install timediffusion
```

## Quick Start

**Forecasting time series**

```python
# train sequence in shape [channels, sequence_length]
model = TD(input_dims=train.shape).to(device=device)
training_losses = model.fit(train)
# horizon : int - how many future values to forecast
predictions = model.forecast(horizon)
```

**Creating synthetic time series**

```python
# sequence in shape [channels, sequence_length]
model = TD(input_dims=seq.shape).to(device=device)
training_losses = model.fit(seq)
# proximity - how close to original, samples - total synthetic time series
synthetic_data = model.synth(proximity=0.9, samples=3, batch_size=2, step_granulation=10)
```

**Time series Imputation**

```python
# sequence in shape [channels, sequence_length]
model = TD(input_dims=seq.shape).to(device=device)
# mask - binary array of same shape, as sequence, with 1 in positions, that are unknown
training_losses = model.fit(seq, mask=mask)
restored_seq = model.restore(example=seq, mask=mask)
```

## Examples

[Time series: multiple tasks example](./examples/example_1d_data.ipynb)

[Forecasting bitcoin price example](./examples/bitcoin_forecast.ipynb)

## Philosophy

Main synopsis behind TimeDiffusion model is that in reality, when working with time series we don’t have many samples, as it could be in other machine learning fields (e.g. cv, nlp). Thus, classical autoregressive approaches like ARIMA has the most suitable approach of fitting / training only on original sequence (maybe with some exogenous data).

TimeDiffusion takes inspiration from these established methods and only trains on the input sample. Model incorporates most powerful modern deep learning techniques such as diffusion process, exponential dilated convolutions, residual connections and attention mechanism (in one of the versions)

## Base Models

* `TimeDiffusionProjector` - exponential dilated convolutions + residual connections.

Currently main model in use

* `TimeDiffusionAttention` - attention mechanism on top of `TimeDiffusionProjector` (q, k, v) projectors.

Currently not viable

* `TimeDiffusionLiquid` - exponential dilated convolutions with shared middle convolutional layer weight. 

Lightweight, fast, but less accurate than the main model.

## Model architecture

Presented below are diagrams depicting model components, each new scheme representing a higher level of abstraction.

* Temporal Block

![Temporal Block Architecture image](https://github.com/timetoai/TimeDiffusion/blob/main/handout/imgs/TemporalBlock.png)

* TimeDiffusionProjector

![TimeDiffusionProjector Architecture image](https://github.com/timetoai/TimeDiffusion/blob/main/handout/imgs/TDP_architecture.png)

* TimeDiffusionAttention

![TimeDiffusion Architecture image](https://github.com/timetoai/TimeDiffusion/blob/main/handout/imgs/TDA_architecture.png)
