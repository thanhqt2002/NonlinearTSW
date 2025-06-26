# Tree-Sliced Wasserstein Distance with Nonlinear Projection

[![arXiv](https://img.shields.io/badge/arXiv-2505.00968-red)](https://arxiv.org/abs/2505.00968)
[![Conference](https://img.shields.io/badge/ICML-2025-blue)](https://icml.cc/Conferences/2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **"Tree-Sliced Wasserstein Distance with Nonlinear Projection"** (ICML 2025).

## Overview

This repository introduces a novel **nonlinear projectional framework** for Tree-Sliced Wasserstein (TSW) distances, generalizing linear projections to nonlinear mappings while preserving theoretical guarantees. The method provides efficient metrics for both Euclidean spaces and spherical manifolds, demonstrating significant improvements over recent SW and TSW variants. Applications include gradient flows, self-supervised learning, and generative modeling.

## Installation

### Prerequisites

- Python >= 3.9
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/thanhqt2002/nonlinear-tsw
cd nonlinear-tsw

# Complete setup (environment + installation + testing)
make setup
```

### Manual Setup

```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate nonlinear-tsw

# Install the package
pip install -e .

# Verify installation
make check
```

## Quick Start

### Basic Usage

```python
import torch
from tsw import TSW, generate_trees_frames

# Initialize Tree-Sliced Wasserstein Distance
tsw_obj = TSW(
    ntrees=250,     # Number of trees
    nlines=4,       # Lines per tree
    p=2,            # Norm level
    delta=2,        # Temperature parameter
    device='cuda'
)

# Generate sample data
N, M, d = 100, 100, 3
X = torch.randn(N, d, device='cuda')
Y = torch.randn(M, d, device='cuda')

# Generate tree frames
theta, intercept = generate_trees_frames(
    ntrees=250, 
    nlines=4, 
    dim=d, 
    gen_mode="gaussian_orthogonal"
)

# Compute Tree-Sliced Wasserstein Distance
distance = tsw_obj(X, Y, theta, intercept)
print(f"TSW Distance: {distance:.4f}")
```

### Spherical Data

```python
import torch
from tsw import SphericalTSW

# Initialize Spherical TSW
stsw_obj = SphericalTSW(
    ntrees=200,
    nlines=5,
    p=2,
    delta=2,
    device='cuda',
    ftype='normal'  # or 'generalized'
)

# Generate spherical data (normalized to unit sphere)
X = torch.randn(100, 3, device='cuda')
X = X / torch.norm(X, dim=1, keepdim=True)
Y = torch.randn(100, 3, device='cuda') 
Y = Y / torch.norm(Y, dim=1, keepdim=True)

# Compute Spherical TSW Distance
distance = stsw_obj(X, Y)
print(f"Spherical TSW Distance: {distance:.4f}")
```

## Core Components

### TSW Class

The main `TSW` class supports multiple nonlinear projection types:

```python
from tsw import TSW

# Linear projection (default)
tsw_linear = TSW(ftype='linear')

# Polynomial projection  
tsw_poly = TSW(ftype='poly', d=3, degree=2)

# Power projection
tsw_power = TSW(ftype='pow', degree=3, pow_beta=0.5)

# Circular projection
tsw_circular = TSW(ftype='circular', radius=2.0)

# Circular concentric projection
tsw_circular_r0 = TSW(ftype='circular_r0')
```

**Key Parameters:**
- `ntrees`: Number of trees (default: 1000)
- `nlines`: Number of lines per tree (default: 5)
- `p`: Norm level (default: 2)
- `delta`: Temperature parameter for mass division (default: 2)
- `mass_division`: `'uniform'` or `'distance_based'` (default: `'distance_based'`)
- `ftype`: Projection type - `'linear'`, `'poly'`, `'circular'`, `'circular_r0'`, `'pow'`
- `device`: Computation device (default: `'cuda'`)

### SphericalTSW Class

Specialized for data on the unit sphere:

```python
from tsw import SphericalTSW

stsw = SphericalTSW(
    ntrees=200,
    nlines=5, 
    p=2,
    delta=2,
    ftype='normal'  # 'normal' or 'generalized'
)
```

### Tree Generation

```python
from tsw import generate_trees_frames, generate_spherical_trees_frames

# Euclidean space
theta, intercept = generate_trees_frames(
    ntrees=100,
    nlines=5,
    dim=3,
    gen_mode="gaussian_orthogonal"  # or "gaussian_raw"
)

# Spherical space  
root, intercept = generate_spherical_trees_frames(
    ntrees=100,
    nlines=5,
    d=3
)
```

## Experiments

The repository includes comprehensive experiments demonstrating the method's effectiveness across various domains:

### 📊 Euclidean Experiments

| Experiment | Location |
|------------|----------|
| **Gradient Flow** | `experiments/euclidean/Gradient_flow/` |
| **Denoising Diffusion** | `experiments/euclidean/denoising-diffusion-gan/` |

### 🌐 Spherical Experiments  

| Experiment | Location |
|------------|----------|
| **Gradient Flow** | `experiments/spherical/gradient_flow/` |
| **Self-Supervised Learning** | `experiments/spherical/ssl/` |
| **Spherical WAE** | `experiments/spherical/swae/` |

### ⚡ Performance Analysis

| Experiment | Location |
|------------|----------|
| **Runtime Comparison** | `experiments/runtime/` |

### 🚀 Running Experiments

Each experiment directory contains detailed README files with specific setup instructions and parameter configurations.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{tran2025nonlinear,
    title={Tree-Sliced Wasserstein Distance with Nonlinear Projection},
    author={Tran, Thanh and Tran, Viet-Hoang and Chu, Thanh and Pham, Trang and El Ghaoui, Laurent and Le, Tam and Nguyen, Tan M.},
    booktitle={Forty-second International Conference on Machine Learning}
    year={2025},
    url={https://arxiv.org/abs/2505.00968}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.