# ML Algorithms Comparison — Gas Turbine Emission Prediction

Comparison of different ML approaches for multi-target regression on real sensor data from a gas turbine plant (Turkey, 2011–2015).

## Dataset

[UCI ML Repository #551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set) — 36,733 hourly sensor readings

**Targets:** TEY (energy yield), CO (carbon monoxide), NOX (nitrogen oxides)

## Methods Compared

| Method | Description |
|---|---|
| Decision Tree (CART) | Per-target, hyperparameter-optimized via cross-validation |
| Single Output NN | Per-target neural network (128-64-32, Scaling + BatchNorm) |
| Multi Output NN | One network, 3 outputs — fair comparison with matched parameter count |

## Key Results (Test RMSE)

| Target | Baseline | Decision Tree | Single NN | Multi NN |
|---|---|---|---|---|
| TEY | 15.61 | 0.95 | **0.65** | 0.80 |
| CO | 2.33 | 1.18 | **1.16** | 1.15 |
| NOX | 11.51 | 4.60 | 3.82 | **3.71** |

## Stack

Python · scikit-learn · Keras/TensorFlow · PyTorch backend · pandas · matplotlib · seaborn

## Usage

Open `superprojekt(final).ipynb` in Jupyter and run all cells.
