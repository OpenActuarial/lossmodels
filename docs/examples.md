# Examples

This page summarizes the example scripts included in the repository.

## Aggregate Modeling

### Panjer vs Simulation

Compare aggregate loss estimates from Panjer recursion and Monte Carlo simulation.

Relevant script:
- `examples/panjer_vs_simulation.py`

### Panjer vs FFT vs Simulation

Compare aggregate loss distributions across Panjer recursion, FFT, and simulation.

Relevant script:
- `examples/panjer_vs_fft_vs_simulation.py`

## Estimation and Model Selection

### Fit and Simulate

Fit a model from data and then simulate aggregate losses.

Relevant script:
- `examples/fit_and_simulate.py`

### Fit and Compare Models

Fit multiple candidate models and compare them.

Relevant script:
- `examples/fit_and_compare_models.py`

## Coverage Examples

### Ordinary Deductible

Apply an ordinary deductible to a severity model.

Relevant script:
- `examples/deductible_example.py`

### Policy Limit

Apply a policy limit to a severity model.

Relevant script:
- `examples/limit_example.py`

### Layer

Apply a deductible plus payment limit to create a layer.

Relevant script:
- `examples/layer_example.py`

## Risk Measures

### Stop-Loss

Compute stop-loss quantities from simulated losses.

Relevant script:
- `examples/stop_loss_example.py`