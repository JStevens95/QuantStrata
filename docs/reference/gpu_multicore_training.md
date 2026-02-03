# GPU and Multi-Core Training Guide

**Last Updated:** 2026-01-27  
**Status:** Implementation Guide

---

## Overview

This guide covers how to optimize ML training in QuantStrata for:
1. **Multi-core CPU** execution (all systems)
2. **GPU acceleration** (when available)
3. **Apple Silicon optimization** (M1/M2/M3 Macs)

---

## Current System Detection

Your system: **Intel Mac (x86_64)**

```python
# Check system
import platform
print(platform.machine())  # x86_64 = Intel, arm64 = Apple Silicon
```

---

## Framework Support Matrix

| Framework | Intel Mac | Apple Silicon | NVIDIA GPU |
|-----------|-----------|---------------|------------|
| TensorFlow | CPU + XLA | Metal (tensorflow-metal) | CUDA |
| PyTorch | CPU | MPS (Metal) | CUDA |
| JAX | CPU | Metal (jax-metal) | CUDA |
| Numba | CPU parallel | CPU parallel | CUDA (numba-cuda) |

---

## 1. TensorFlow Optimization (Current Framework)

### 1.1 Enable XLA Compilation

XLA (Accelerated Linear Algebra) provides significant speedups via:
- Kernel fusion
- Memory optimization
- Automatic parallelization

```python
from src.machine_learning.core.config import TrainingConfig, OptimizerConfig

# Enable XLA for faster training
config = TrainingConfig(
    epochs=100,
    batch_size=256,
    xla_compile=True,  # Enable XLA JIT compilation
    mixed_precision=False,  # Enable for Apple Silicon
    optimizer=OptimizerConfig(
        name='adam',
        learning_rate=1e-3,
    ),
)

# Train with XLA
from src.machine_learning.training.trainer import Trainer
trainer = Trainer(model, config)
result = trainer.fit(train_data, val_data)
```

### 1.2 Multi-Core Parallelization

TensorFlow automatically uses multiple cores, but you can tune:

```python
import tensorflow as tf

# Set thread counts
tf.config.threading.set_intra_op_parallelism_threads(8)  # Within ops
tf.config.threading.set_inter_op_parallelism_threads(4)  # Between ops

# Or use environment variables
import os
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
```

### 1.3 Data Pipeline Optimization

```python
import tensorflow as tf

# Optimize data pipeline
train_ds = tf.data.Dataset.from_tensor_slices((X, y))
train_ds = (
    train_ds
    .shuffle(buffer_size=10000)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)  # Prefetch while training
    .cache()  # Cache in memory
)
```

### 1.4 Apple Silicon Setup (M1/M2/M3)

For Apple Silicon Macs, install tensorflow-metal:

```bash
# Install for Apple Silicon
pip install tensorflow-macos tensorflow-metal

# Verify GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 2. JAX Optimization

### 2.1 Current JAX Kernels

Located in `src/core/performance/jax_kernels.py`:

```python
from src.core.performance.jax_kernels import (
    gbm_paths_jax,
    gbm_terminal_spots_jax,
    vanilla_payoff_jax,
    digital_payoff_jax,
)

import jax.random as jr

# GPU-accelerated Monte Carlo
key = jr.PRNGKey(42)
terminal_spots = gbm_terminal_spots_jax(
    spot0=100.0,
    drift=0.05,
    vol=0.2,
    n_paths=1_000_000,  # 1M paths on GPU
    n_steps=252,
    dt=1/252,
    key=key,
)
```

### 2.2 Adding More JAX Kernels

**Recommendation:** Add JAX versions of:

```python
# src/core/performance/jax_kernels.py

def heston_paths_jax(
    spot0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    key: Any,
) -> Tuple[Any, Any]:
    """
    Heston model path simulation using JAX.
    
    Returns:
        Tuple of (spot_paths, variance_paths)
    """
    _require_jax()
    import jax.numpy as jnp
    import jax.random as jr
    
    # Cholesky for correlation
    sqrt_dt = jnp.sqrt(dt)
    
    def step(carry, _):
        S, v, key = carry
        key, k1, k2 = jr.split(key, 3)
        
        z1 = jr.normal(k1, shape=S.shape)
        z2 = rho * z1 + jnp.sqrt(1 - rho**2) * jr.normal(k2, shape=S.shape)
        
        v_plus = jnp.maximum(v, 0)
        sqrt_v = jnp.sqrt(v_plus)
        
        S_new = S * jnp.exp((drift - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * z1)
        v_new = v + kappa * (theta - v_plus) * dt + sigma * sqrt_v * sqrt_dt * z2
        
        return (S_new, v_new, key), (S_new, v_new)
    
    # Initial
    S0 = jnp.full(n_paths, spot0)
    v_init = jnp.full(n_paths, v0)
    
    carry_final, (S_path, v_path) = _jax_scan(step, (S0, v_init, key), n_steps)
    
    return S_path, v_path
```

### 2.3 JAX on Apple Silicon

```bash
# Install JAX with Metal support (Apple Silicon)
pip install jax-metal

# Verify
import jax
print(jax.devices())
# Should show Metal device
```

---

## 3. Numba Multi-Core Optimization

### 3.1 Current Configuration

```python
from numba import njit, prange

@njit(parallel=True, cache=True, fastmath=True)
def simulate_paths_parallel(spot0, drift, vol, dt, n_paths, n_steps, randoms):
    """Parallel GBM simulation using all CPU cores."""
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = spot0
    
    sqrt_dt = np.sqrt(dt)
    drift_dt = (drift - 0.5 * vol * vol) * dt
    
    for i in prange(n_paths):  # Parallel loop
        for t in range(n_steps):
            paths[i, t + 1] = paths[i, t] * np.exp(
                drift_dt + vol * sqrt_dt * randoms[i, t]
            )
    
    return paths
```

### 3.2 Verify Parallel Execution

```python
from numba import config

# Check parallel settings
print(f"Numba threads: {config.NUMBA_NUM_THREADS}")
print(f"Threading layer: {config.THREADING_LAYER}")

# Set thread count
import os
os.environ['NUMBA_NUM_THREADS'] = '8'  # Use 8 cores
```

---

## 4. PyTorch Backend (Recommended Addition)

### 4.1 Why Add PyTorch?

1. **Apple Silicon MPS**: Native GPU support on M1/M2/M3
2. **Better gradients**: For deep hedging autodiff
3. **Industry standard**: Widely used in ML finance

### 4.2 Deep Hedging PyTorch Implementation

```python
# src/deep_hedging/training/pytorch_trainer.py

import torch
import torch.nn as nn
from typing import Dict, Any

class MLPPolicyTorch(nn.Module):
    """PyTorch MLP policy for deep hedging."""
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_layers: list = [64, 64],
        output_dim: int = 1,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepHedgingTrainerTorch:
    """PyTorch trainer for deep hedging with GPU support."""
    
    def __init__(
        self,
        policy: nn.Module,
        risk_measure: str = "mean_variance",
        learning_rate: float = 0.001,
        device: str = "auto",
    ):
        self.policy = policy
        self.risk_measure = risk_measure
        self.learning_rate = learning_rate
        
        # Auto-detect device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")  # Apple Silicon
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.policy = self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )
    
    def train_epoch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        pnl: torch.Tensor,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()
        
        # Move to device
        states = states.to(self.device)
        pnl = pnl.to(self.device)
        
        # Forward pass
        predicted_actions = self.policy(states)
        
        # Compute risk measure loss
        if self.risk_measure == "mean_variance":
            loss = pnl.mean() + 0.5 * pnl.var()
        elif self.risk_measure == "cvar":
            alpha = 0.95
            var = torch.quantile(pnl, alpha)
            loss = pnl[pnl >= var].mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
```

### 4.3 Install PyTorch

```bash
# For Intel Mac
pip install torch torchvision torchaudio

# For Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio

# Verify MPS (Apple Silicon GPU)
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

---

## 5. Benchmarking

### 5.1 MC Simulation Benchmark

```python
import time
import numpy as np
from src.core.performance.backend import get_backend_info

def benchmark_mc(n_paths: int = 100_000, n_steps: int = 252):
    """Benchmark MC simulation across backends."""
    results = {}
    
    # NumPy
    start = time.time()
    randoms = np.random.randn(n_paths, n_steps)
    # ... simulation
    results["numpy"] = time.time() - start
    
    # Numba (if available)
    try:
        from src.core.performance.mc_kernels import simulate_gbm_paths
        start = time.time()
        # ... simulation
        results["numba"] = time.time() - start
    except ImportError:
        pass
    
    # JAX (if available)
    try:
        from src.core.performance.jax_kernels import gbm_terminal_spots_jax
        import jax.random as jr
        start = time.time()
        key = jr.PRNGKey(42)
        # ... simulation
        results["jax"] = time.time() - start
    except ImportError:
        pass
    
    return results

# Run benchmark
results = benchmark_mc(n_paths=1_000_000)
print("Benchmark Results (seconds):")
for backend, time_sec in results.items():
    print(f"  {backend}: {time_sec:.3f}s")
```

---

## 6. Recommended Setup

### For Intel Mac (Your System)

```bash
# Install optimized packages
pip install tensorflow numba jax jaxlib

# Environment variables for multi-core
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=4
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8
```

### For Apple Silicon Mac

```bash
# Install with Metal support
pip install tensorflow-macos tensorflow-metal
pip install torch torchvision torchaudio
pip install jax-metal
pip install numba

# PyTorch MPS is automatic on M1/M2/M3
```

---

## 7. Configuration Presets

### Add to `src/core/performance/presets.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PerformancePreset:
    """Performance configuration preset."""
    backend: str
    parallel: bool
    n_threads: int
    use_gpu: bool
    
    @classmethod
    def auto_detect(cls) -> "PerformancePreset":
        """Auto-detect optimal settings."""
        import platform
        import os
        
        n_cpus = os.cpu_count() or 4
        
        # Check for GPU
        use_gpu = False
        backend = "numba"
        
        try:
            import torch
            if torch.backends.mps.is_available():
                use_gpu = True
                backend = "pytorch"
        except ImportError:
            pass
        
        try:
            import jax
            if len(jax.devices("gpu")) > 0:
                use_gpu = True
                backend = "jax"
        except:
            pass
        
        return cls(
            backend=backend,
            parallel=True,
            n_threads=n_cpus,
            use_gpu=use_gpu,
        )

# Usage
preset = PerformancePreset.auto_detect()
print(f"Backend: {preset.backend}")
print(f"GPU: {preset.use_gpu}")
print(f"Threads: {preset.n_threads}")
```

---

## Summary

| Optimization | Intel Mac | Apple Silicon | Effort |
|--------------|-----------|---------------|--------|
| TensorFlow XLA | ✅ Ready | ✅ Ready | Config only |
| Multi-core threads | ✅ Ready | ✅ Ready | Config only |
| JAX CPU | ✅ Ready | ✅ Ready | Existing |
| JAX GPU | ❌ N/A | ✅ jax-metal | Install only |
| PyTorch CPU | ✅ Ready | ✅ Ready | Add backend |
| PyTorch MPS | ❌ N/A | ✅ Native | Add backend |
| Numba parallel | ✅ Ready | ✅ Ready | Existing |

**Recommended Actions:**
1. Enable XLA by default (`xla_compile=True`)
2. Add PyTorch backend for deep hedging
3. Add more JAX kernels (Heston, local vol)
4. Create performance benchmark suite
