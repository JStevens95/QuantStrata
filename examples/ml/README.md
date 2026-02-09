# Machine Learning & Reinforcement Learning Examples

Production-grade examples demonstrating ML and RL applications in quantitative finance.

## Examples

### 1. `01_hedging_environment.py` - RL Hedging Environment

**Learning Objectives:**
- Understand the Gymnasium-compatible hedging environment
- State space: spot, time, delta, gamma, position, P&L
- Action space: continuous hedge ratio
- Benchmark comparison: delta hedge vs no hedge vs over-hedge

**Key Concepts:**
- Environment design for financial RL
- Episode structure (reset → step → terminate)
- Risk-adjusted reward functions

### 2. `02_rl_hedging_agent.py` - Training an RL Hedging Agent

**Learning Objectives:**
- Implement a simple policy gradient (REINFORCE) agent
- Train using collected episodes
- Compare trained agent vs delta hedging benchmark

**Key Concepts:**
- Policy networks for continuous actions
- REINFORCE algorithm with baseline
- Entropy regularization for exploration

### 3. `03_model_validation.py` - BSM vs Monte Carlo vs Finite Difference

**Learning Objectives:**
- Compare analytical, MC, and FD pricing methods
- Study convergence properties (O(1/√N) for MC, O(h²) for FD)
- Validate Greeks via bump-and-reprice

**Key Concepts:**
- Model validation workflow
- Error quantification and tolerance setting
- When to use each method

## Running the Examples

```bash
cd /path/to/QuantStrata

# Run with plots
PYTHONPATH=. python examples/ml/01_hedging_environment.py

# Run without plots
PYTHONPATH=. python examples/ml/01_hedging_environment.py --no-plot

# Reproducibility and audit trail (production)
PYTHONPATH=. python examples/ml/01_hedging_environment.py --seed 42 --output-dir output/ml_hedging
PYTHONPATH=. python examples/ml/02_rl_hedging_agent.py --seed 42 --output-dir output/ml_rl_agent
PYTHONPATH=. python examples/ml/03_model_validation.py --seed 42 --output-dir output/model_validation
```

Use `--seed` for reproducible runs and `--output-dir DIR` to save run config, metrics, and plots for audit.

## Production Context

At a hedge fund:
- **RL Hedging**: Active research area for transaction cost optimization
- **Model Validation**: Required before any model goes to production
- **Deep Hedging**: Learns to hedge under realistic market conditions

## Prerequisites

- `examples/fundamentals/` - Market data basics
- `examples/pricing/` - Option pricing methods
- `examples/risk/04_delta_hedging.py` - Traditional delta hedging

## Dependencies

All examples use only NumPy for portability. For production RL:
- PyTorch or TensorFlow for deep RL agents
- Stable-Baselines3 for off-the-shelf algorithms
- Ray RLlib for distributed training

## Related Documentation

- `src/q_learning/environments/hedging.py` - HedgingEnvironment source
- `src/q_learning/core/protocols.py` - RL protocol definitions
