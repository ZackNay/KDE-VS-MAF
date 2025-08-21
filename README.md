# KDE-VS-MAF

A comparison between Kernel Density Estimation (KDE) and Masked Autoregressive Flows (MAF) for density modeling across multiple dimensions and distribution types.

## Overview

This project compares two density modeling approaches:

- Kernel Density Estimation (KDE): Non-parametric method using Epanechnikov kernel
- Masked Autoregressive Flows (MAF): Normalizing flow with autoregressive transformations

The comparison evaluates both methods across:
- Dimensions: 1D to 8D
- Target Distributions: Gaussian, Mixture of Gaussians, Skew Normal
- Metrics: KL divergence, negative log-likelihood, sample quality

## Features

- Systematic evaluation across multiple dimensions and distribution types
- Reproducible experimental setup with fixed random seeds
- Automatic hyperparameter scaling based on dimensionality
- KL divergence and negative log-likelihood evaluation
- Sample quality assessment
- Minimum data size analysis to achieve KL divergence ≤ 0.5
- Training time and convergence tracking
- Visualization tools for results and flow transformations

## Project Structure

```
├── main.py                     # Main experiment runner for minimum data size analysis
├── kde_experiment.py           # KDE implementation and experiments
├── maf_experiment.py          # MAF implementation using normalizing flows
├── simple_KDE_experiment.py    # Basic KDE analysis
├── visual_for_kl_divergence.py # KL divergence visualization
├── flow_visual_gif_creation.py # Flow transformation animations
└── README.md                   # This file
```

## Dependencies

- torch
- numpy
- matplotlib
- pandas
- normflows
- scipy
- sklearn
- tqdm

## Usage

### Running Full Experiments

Run complete comparison across all dimensions and distributions:

```python
# KDE experiments
from kde_experiment import KDEExperiment
experiment = KDEExperiment(output_file="kde_results.xlsx", seed=42)
results = experiment.run_experiments()

# MAF experiments  
from maf_experiment import MAF_Multidimensional_Comparison_Experiment
experiment = MAF_Multidimensional_Comparison_Experiment(output_file="maf_results.xlsx", seed=42)
results = experiment.run_experiments()
```

### Minimum Data Size Analysis

Find minimum dataset sizes for target KL divergence:

```python
from main import MinimumDataSizeExperiment
experiment = MinimumDataSizeExperiment()
experiment.run_all_experiments()
experiment.visualize_results()
```

### Specific Experiments

Run experiments for specific configurations:

```python
# Single dimension and distribution
results = experiment.run_experiments(dimensions=3, distribution_types='gaussian')

# Range of dimensions
results = experiment.run_experiments(dimensions=(1, 5), distribution_types='mixture')

# Multiple specific dimensions
results = experiment.run_experiments(dimensions=[2, 4, 6], distribution_types='skewnormal')
```

## Distribution Types

### Gaussian
Standard multivariate Gaussian with diagonal covariance matrix.

### Mixture of Gaussians
Three-component mixture with evenly spaced means and equal weights.

### Skew Normal
Asymmetric distribution with configurable skewness parameters.

## Evaluation Metrics

- **KL Divergence**: Measures distributional similarity between target and learned model
- **Negative Log-Likelihood**: Model fit quality on test data
- **Sample Quality**: Statistical property comparison (mean/std errors)
- **Training Time**: Computational efficiency comparison
- **Convergence**: Number of iterations to reach optimal performance

## Output Files

Results are automatically saved to Excel files:
- `kde_results.xlsx`: KDE experiment results
- `maf_results.xlsx`: MAF experiment results  
- `min_data_size_results.xlsx`: Minimum data size analysis

## Key Classes

### KDE Implementation
- `KDE`: Core KDE class with Epanechnikov kernel
- `KDEExperiment`: Experiment runner for KDE evaluation
- Optimal bandwidth selection using Leave-One-Out Maximum Likelihood Cross-Validation

### MAF Implementation
- `MAF_Multidimensional_Comparison_Experiment`: Experiment runner for MAF evaluation
- Uses normflows library for masked autoregressive flows
- Automatic hyperparameter scaling based on dimension

### Target Distributions
- `KDEGaussianDistribution` / `Gaussian_Distribution`: Gaussian distributions
- `KDEMixtureOfGaussians` / `Mixture_of_Gaussians`: Mixture models
- `KDESkewNormal` / `Skew_Normal`: Skew normal distributions

## Reproducibility

All experiments use fixed random seeds for reproducible results. Seeds are deterministically generated based on:
- Base seed (default: 42)
- Distribution type hash
- Dimension multiplier

## Notes

- CUDA support automatically detected and used when available
- Early stopping implemented for MAF training to prevent overfitting
- Gradient clipping applied for training stability
- Results include both accuracy metrics and computational efficiency measures
