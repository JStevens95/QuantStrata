# Hybrid GNN-RNN: Architecture Design Principles

> **Target audience**: Front-office ML / quant library standards  
> **Purpose**: Canonical reference for mathematically correct layer design and placement of nonlinearities

---

## 0. How Does a "Linear" GNN Learn Non-Linear Structure?

**The sublayers are NOT redundant.** Non-linearity arises from **composition** of linear layers with activations between them.

### Effective flow (2-layer example)

$$\mathbf{H}^{(0)} = X$$

$$\mathbf{Z}^{(0)} = \text{GNN}^{(0)}(X, \mathbf{A}) \quad \text{(linear: message passing + } \mathbf{W}_1)$$

$$\mathbf{H}^{(1)} = \sigma(\mathbf{Z}^{(0)}) \quad \text{(non-linear!)}$$

$$\mathbf{Z}^{(1)} = \text{GNN}^{(1)}(\mathbf{H}^{(1)}, \mathbf{A}) \quad \text{(linear in } \mathbf{H}^{(1)} \text{, but } \mathbf{H}^{(1)} \text{ is non-linear in } X\text{)}$$

$$\mathbf{H}^{(2)} = \sigma(\mathbf{Z}^{(1)} + \mathbf{W}_{\text{proj}} X) \quad \text{(output)}$$

### Why this is non-linear

- Sublayer 1 operates on raw features $X$; its output $\mathbf{Z}^{(0)}$ is linear in $X$.
- After $\sigma(\mathbf{Z}^{(0)})$, we have $\mathbf{H}^{(1)} = \sigma(\mathbf{W}_1 \cdot \text{AGG}(X))$, which is **non-linear in $X$**.
- Sublayer 2 computes $\mathbf{Z}^{(1)} = \mathbf{W}_2 \cdot \text{AGG}(\mathbf{H}^{(1)})$. This is linear in $\mathbf{H}^{(1)}$, but $\mathbf{H}^{(1)}$ is non-linear in $X$, so $\mathbf{Z}^{(1)}$ is **non-linear in $X$**.

The full composition is $\sigma\bigl(\mathbf{W}_2 \cdot \text{AGG}(\sigma(\mathbf{W}_1 \cdot \text{AGG}(X))) + \text{residual}\bigr)$ — a non-linear function of $X$ and the graph structure.

### Role of each sublayer

| Sublayer | Input | Learns | Output |
|----------|-------|--------|--------|
| 1 | Raw features $X$ | Low-level graph patterns (local neighbors) | $\mathbf{Z}^{(0)} \to \sigma \to \mathbf{H}^{(1)}$ |
| 2 | Activated $\mathbf{H}^{(1)}$ | Higher-level patterns (multi-hop structure) | $\mathbf{Z}^{(1)} \to +\text{residual} \to \sigma \to \mathbf{H}^{(2)}$ |

Each sublayer has **different learnable weights** ($\mathbf{W}_1 \neq \mathbf{W}_2$). Without activation between them, $\mathbf{Z}^{(1)} = \mathbf{W}_2 \cdot \mathbf{W}_1 \cdot \ldots = \mathbf{W}_{\text{eff}} \cdot X$ would collapse to a single linear transform — that would be redundant. The activations prevent this collapse and allow each layer to learn distinct representations.

**Analogy**: An MLP with `Dense(128) → ReLU → Dense(64) → ReLU` has linear Dense layers, but the stack is non-linear. Same idea: linear layers + activations between = non-linear composition.

---

## 1. Design Standard: Linear Primitives + Block-Level Activation

### 1.1 Industry Convention (PyG, DGL, Spektral)

Major GNN libraries use **linear convolution layers**; activation is a model-level choice:

| Library | Layer | Activation |
|---------|-------|------------|
| PyTorch Geometric | `SAGEConv`, `GCNConv` | None (user adds `ReLU` in model) |
| DGL | `SAGEConv`, `GATConv` | None |
| Spektral | `GraphConv`, `GCNConv` | None |

**Rationale**: Separating linear message passing from nonlinearity allows:
- Flexible activation placement (e.g. pre-norm vs post-norm)
- Easy experimentation (ReLU → GELU, etc.) in one place
- Correct residual-block formulation (see below)

### 1.2 Residual Block Mathematics (He et al.)

Standard residual block:
$$\text{output} = \sigma\bigl(F(\mathbf{x}) + \mathbf{W}_s \mathbf{x}\bigr)$$

- $F$ is the residual branch (convolutions, linear transforms)
- **$F$'s last layer is LINEAR** before the add
- $\sigma$ is applied **after** the add

If $F$ ended with activation, we would have $\sigma(\sigma(\cdot) + \mathbf{x})$, which differs from the intended formulation and changes gradient flow.

### 1.3 Our Implementation

- **GraphSage / MixedGraphSage**: Act as **linear** primitives when used inside GnnBlock (we pass `activation=None` in config)
- **GnnBlock**: Applies all activation:
  - Between sublayers: $\mathbf{H}^{(l+1)} = \text{Dropout}(\sigma(\text{LN}(\mathbf{Z}^{(l)})))$ for $l < L-1$
  - After residual: $\mathbf{H}^{(L)} = \sigma(\mathbf{Z}^{(L-1)} + \mathbf{W}_{\text{proj}} \mathbf{H}^{(0)})$

---

## 2. "Sublayer Activation" vs "Block Activation" — When Are They the Same?

**For intermediate layers** (layer 1 → layer 2), they **are** equivalent:

| Design | Layer 1 output | What layer 2 receives |
|--------|-----------------|------------------------|
| Activation in sublayer | GraphSage outputs $\sigma(\mathbf{W}_1 x)$ | $\sigma(\mathbf{W}_1 x)$ |
| Activation in block | GraphSage outputs $\mathbf{W}_1 x$, block applies $\sigma$ | $\sigma(\mathbf{W}_1 x)$ |

So the next layer gets the same input either way. Functionally identical.

**The difference only matters at the residual boundary** (last layer before the add):

| Design | Before add | After add + final $\sigma$ |
|--------|------------|----------------------------|
| **Activation in last sublayer** | $\sigma(\mathbf{W}_L \mathbf{H}^{(L-1)})$ | $\sigma\bigl(\sigma(\mathbf{W}_L \mathbf{H}^{(L-1)}) + \text{residual}\bigr)$ |
| **Linear last sublayer, block $\sigma$** | $\mathbf{W}_L \mathbf{H}^{(L-1)}$ | $\sigma\bigl(\mathbf{W}_L \mathbf{H}^{(L-1)} + \text{residual}\bigr)$ |

These are **different**. Example with ReLU: $\sigma(\sigma(-1) + 2) = \sigma(0+2) = 2$, but $\sigma(-1+2) = 1$. The standard ResNet formulation uses the second form ($\sigma(F(x) + x)$), so $F$ must end linear.

**Summary**: For a block without residual, either design works. The linear-primitive convention matters for correct residual-block math and alignment with PyG/DGL.

---

## 3. Why RnnBlock Is Different (No Changes Needed)

**RnnBlock does NOT follow the same "linear primitive" pattern**, and it shouldn't.

| Block | Sublayers | Why different? |
|-------|-----------|-----------------|
| **GnnBlock** | GraphSage = simple linear (msg + agg + Wx) | Easy to strip activation; activation is "optional" |
| **RnnBlock** | LSTM / GRU = gated cells | Activation (sigmoid, tanh) is **built into** the cell design |

An LSTM has:
- Forget gate: $\sigma(\ldots)$
- Input gate: $\sigma(\ldots)$
- Cell state: $\tanh(\ldots)$
- Output gate: $\sigma(\ldots)$, output: $\tanh(\ldots)$

You can't meaningfully "turn off" these activations and apply them from outside; they're part of how the gates work. LSTM/GRU are atomic units—you use them as-is. Keras passes `activation` and `recurrent_activation` directly to the cell.

**No change needed**: RnnBlock correctly uses Keras LSTM/GRU with their built-in activations.

---

## 4. Why Not "Sublayers Nonlinear, Block Linear"?

A plausible alternative: each sublayer applies its own activation; the block only adds residual and optionally activates afterward.

**Issues**:
1. **Last sublayer**: For the residual $\sigma(F(\mathbf{x}) + \mathbf{x})$, $F$ must end with a linear layer. So the last sublayer would need to be special-cased as linear.
2. **Inconsistency with libraries**: GraphSage in PyG has no activation; adding it inside breaks the "layer = linear primitive" convention.
3. **Dual activation points**: If sublayers and block both apply activation, you get redundant or incorrectly placed nonlinearities.

**Conclusion**: Keep sublayers linear; block owns all activation. This matches PyG/DGL and ResNet.

---

## 5. Config Copy and `activation=None`

```python
layer_config = copy.deepcopy(self.layer_config)
layer_config['parameters']['activation'] = None
```

- **Deepcopy**: Avoid mutating the original config (affects serialization, shared configs).
- **`activation=None`**: Force sublayers to behave as linear primitives. Without this, GraphSage/MixedGraphSage would apply activation internally, conflicting with block-level control.

---

## 6. Layer Design Summary

| Component | Linear / Nonlinear | Rationale |
|-----------|--------------------|------------|
| GraphSage | Linear (in GnnBlock) | Match PyG; block applies σ |
| MixedGraphSage | Linear (in GnnBlock) | Same |
| GnnBlock | Applies σ between layers, after residual | ResNet convention; full control |
| RnnBlock | Keras LSTM (internal activations) | Standard RNN unit |
| FusionLayer | Linear + attention + gate | Cross-modal fusion |
| ProjectionLayer | Linear + optional baseline norm | Output projection |

---

## 7. Impact on Exotic / Vanilla PnL Accuracy

### When Would Our Design Differ From a "Work Version"?

| Scenario | Accuracy impact |
|----------|------------------|
| **No residual in GnnBlock** | **None.** Intermediate-layer placement (sublayer vs block activation) is equivalent; the model learns the same mapping. |
| **Residual + activation in last sublayer** | **Possible improvement.** Our formulation $\sigma(F(x) + x)$ (with $F$ linear) gives cleaner gradient flow through the shortcut. |
| **Residual + linear last sublayer (our design)** | Correct ResNet formulation; best-studied choice. |

### Why It Might Help for Real Exotics (When Residuals Are Used)

1. **Deeper GnnBlock**: Exotic PnL depends on multi-hop structure in the trade graph (vanilla ↔ exotic ↔ hedging basket). Deeper GNNs can model that, but they degrade without good residuals. The standard $\sigma(F(x) + x)$ formulation keeps a clean identity path for gradients.
2. **Training stability**: ResNet-style residuals are designed to avoid gradient vanishing in deep stacks. More stable optimization can lead to better convergence on noisy PnL targets.
3. **Reproducibility**: Matches published architectures; easier to replicate results and compare with baselines.

### What We Cannot Claim

- **Guaranteed accuracy gain**: Your work version may already be close (e.g. if residuals use a similar pattern). The only way to know is to run both on your data.
- **Domain-specific magic**: The benefit is from correct residual formulation and gradient flow, not from anything special to exotics.
- **Substitute for data/features**: Accuracy depends mainly on graph quality, attributes (delta, vega, moneyness, etc.), scenario coverage, and RNN/fusion design—not on this activation-placement detail alone.

### Recommendation

Use the standard design for correctness and maintainability. If you have a work version with residuals and a different last-layer setup, run an A/B comparison on your exotic/vanilla PnL dataset; the improvement (if any) will show up in validation loss and downstream metrics.

---

## 8. References

- [1] Hamilton et al., "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- [2] He et al., "Deep Residual Learning for Image Recognition"
- [3] PyTorch Geometric: `torch_geometric.nn.conv.SAGEConv` (no built-in activation)
- [4] DGL: `dgl.nn.pytorch.conv.SAGEConv` (no built-in activation)
