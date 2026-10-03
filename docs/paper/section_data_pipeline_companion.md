# Data Pipeline Section -- Companion Guide

A plain-language walkthrough of the paper section, plus a Q&A bank for defending it.

---

## Part 1: What Each Section Actually Says (In Plain English)

### Section 1 -- Introduction

**What it says**: We have a portfolio of exotic derivatives. We want to predict each trade's daily P&L using a neural network. The challenge is that the raw data is messy -- different trades have wildly different scales, some are missing data, the number of time steps is huge, and the trades are interconnected. Before feeding anything to the model, we need to clean, compress, and structure the data.

**Why it matters**: This section tells the reader "there are 5 non-trivial preprocessing steps, here they are." The TikZ diagram on page 1 is the visual anchor -- reviewers will refer back to it throughout.

**Key point to remember**: The pipeline is *modular*. Each stage can be swapped out without touching the model. This is a deliberate design choice for experimentation.

---

### Section 2 -- Elementary P&L Construction

**What it says**: We take each trade's daily P&L (today's value minus yesterday's value) and put it into a big matrix: rows = days, columns = trades. Some trades aren't live for the whole period, so we zero-pad those gaps and keep a mask so the model knows which entries are real.

Then we standardise: for each trade, subtract its mean and divide by its standard deviation. This makes all trades comparable in scale.

**The formula (Eq. 3)**:
```
standardised = (raw - mean) / (std + epsilon)
```
That's a z-score. The epsilon stops division by zero for trades with near-zero variance (like a deeply OTM option that barely moves).

**What we don't reveal**: Exactly which robust estimator we use, the calibration window length, or the epsilon value. The paper says "robust estimators" generically. In practice we use sklearn's StandardScaler fitted on training data only.

---

### Section 3 -- Trade Graph Construction

**What it says**: We build a graph where each node is a trade and edges connect "similar" trades. Similarity is measured in attribute space (product type, underlying, Greeks, etc.). For each trade, we find its k nearest neighbours and draw edges to them. Edge weights are normalised so they sum to 1 per row.

**The formula (Eq. 4)**:
```
A[i][j] = similarity(i,j) / sum_of_similarities_for_all_k_neighbours_of_i
```
This is just "normalise the weights so they're a proper average."

**Why row-normalised**: When the GNN does `A @ X` (matrix multiply), each node gets the weighted *average* of its neighbours' features. If A weren't row-normalised, nodes with many neighbours would get inflated values.

**What we don't reveal**: The exact similarity function, which attributes get what weight, or the specific value of k. The paper says "domain-specific and not detailed here."

---

### Section 4 -- Attribute Encoding

**What it says**: Each trade has two kinds of features:
- **Categorical** (product type, currency): turned into learned embeddings (dense vectors)
- **Continuous** (delta, gamma, vega): standardised and optionally clipped

These are concatenated into one vector per trade, giving the attribute matrix X.

**What we don't reveal**: Which specific Greeks we include, the embedding dimensions, or the clipping thresholds. The paper describes the *class* of encoding (embeddings + standardisation) without the specifics.

---

### Section 5 -- Dimensionality Reduction

**What it says**: The P&L matrix has too many time steps. We compress it using SVD (essentially PCA). We keep enough components to explain a target fraction of the variance (e.g., 95%), throwing away the noisy trailing modes.

**The formulas**:
- **Eq. 6**: Standard SVD decomposition P = UΣV^T
- **Eq. 7**: Projection: multiply P by the top-m right singular vectors to get a compressed version
- **Eq. 8**: Choose m as the smallest number of components that explains ≥ τ of the total variance
- **Eq. 9**: Gavish-Donoho optimal threshold -- a theoretically principled alternative to the ad-hoc τ

**Why this works for derivatives**: A portfolio's P&L is driven by a small number of risk factors (spot, vol surface, rates). So most of the information lives in the top 10-30 components. The rest is noise.

**What we don't reveal**: The paper describes standard SVD projection. The actual implementation uses a more sophisticated approach (pivoted QR basis selection with tail-weighted scenarios). This is intentional -- the paper gives the general method, not the production recipe.

---

### Section 6 -- Model Input Assembly

**What it says**: We cut the compressed P&L into overlapping windows of length w. Each window is one training sample. We package everything into 5 tensors:
- X (trade attributes), P (windowed P&L), A (sparse graph), target indices, mask

**Key architectural insight**: The graph and attributes are *shared across the batch* -- the GNN processes them once and the result is broadcast. Only the P&L windows vary per sample.

---

## Part 2: Common Questions and Answers

### On Standardisation

**Q: "Why per-trade standardisation instead of global standardisation?"**

A: Exotic derivatives have vastly different P&L scales -- a near-maturity OTM barrier might move pennies while an ATM vanilla moves thousands. Global standardisation would be dominated by the high-variance trades, effectively zeroing out the low-variance ones. Per-trade standardisation ensures every instrument contributes meaningfully to the loss function and gradient updates.

**Q: "Why not use log returns instead of standardised P&L?"**

A: Log returns assume multiplicative dynamics and require strictly positive prices. Derivative P&L can be negative (losses), zero (expired worthless), or exhibit discontinuous jumps (barrier knock events). The additive standardisation framework handles all of these cases without requiring sign or continuity assumptions.

**Q: "What happens if a trade has zero variance?"**

A: The epsilon in the denominator prevents division by zero. In practice, a zero-variance trade (e.g., a fully expired instrument) contributes no information and is effectively a constant input to the model. The standardisation maps it to zero, which is numerically harmless.

---

### On Graph Construction

**Q: "Why k-NN instead of a fully connected graph or a learned graph?"**

A: Three reasons. First, sparsity: a full graph has O(T²) edges which is infeasible for T > 1000. Second, inductive bias: in a derivative portfolio, each trade's P&L is primarily influenced by a small number of structurally similar instruments, not all 2000 trades on the book. Third, stability: jointly learning the graph topology alongside the model parameters creates a bilevel optimisation that is notoriously unstable, especially at scale.

**Q: "Is the graph directed or undirected?"**

A: Directed. Trade i having trade j as a neighbour does not imply the reverse. This asymmetry is appropriate because similarity in attribute space is not always symmetric after k-NN truncation -- a rare exotic might point to nearby vanillas, but those vanillas have closer neighbours among other vanillas.

**Q: "Why not use correlation-based edges instead of attribute-based?"**

A: Correlation measures require sufficient overlapping history and are unstable for short-lived or newly booked trades. Attributes are available from day one (booking time). Furthermore, correlation captures statistical co-movement but not structural causation -- two trades might be uncorrelated historically but share the same underlying, meaning a future shock would affect both. The attribute-based graph captures this structural relationship.

**Q: "How sensitive are results to k?"**

A: Moderately. Small k (5-10) gives very local message passing -- good for concentrated books. Large k (30-50) gives broader propagation -- better for diversified books. In a multi-layer GNN with L layers, the receptive field grows as O(k^L), so even moderate k provides global reach. We tune k as a hyperparameter during model selection.

---

### On Dimensionality Reduction

**Q: "Why not just truncate the time series to the most recent N days?"**

A: Truncation discards long-horizon information entirely. For exotics with long maturities (5y+ swaps, multi-year autocallables), the P&L dynamics over the full history are informative of the instrument's risk profile. SVD-based compression retains the dominant modes across the *entire* history while discarding noise -- it compresses without discarding.

**Q: "How do you choose the variance threshold τ?"**

A: We inspect the eigenvalue spectrum. Derivative portfolios typically show a clear "elbow" where eigenvalues transition from systematic (driven by risk factors) to idiosyncratic (noise). We set τ to capture the systematic portion, typically 0.90-0.97. As a robustness check, we also consider the Gavish-Donoho optimal hard threshold, which provides a distribution-free criterion.

**Q: "Doesn't PCA assume linear structure? Exotic payoffs are highly non-linear."**

A: PCA compresses the *time series*, not the payoff function. Even though exotic payoffs are non-linear functions of market variables, the resulting daily P&L series are approximately linear in the underlying risk factors over short horizons (via the Greeks). The non-linearity is captured by the neural network downstream, not by the compression step. The SVD step is purely a variance-preserving dimensionality reduction of the input representation.

**Q: "What is the Gavish-Donoho threshold and why mention it?"**

A: It's a mathematically optimal rule for deciding how many singular values to keep when the data is signal plus noise. Instead of picking an arbitrary variance threshold like 95%, Gavish-Donoho gives a formula based on the matrix aspect ratio and the median singular value. We mention it for academic rigour -- it shows we're aware of principled alternatives to the heuristic threshold. Both give similar results in practice for our data.

---

### On Attribute Encoding

**Q: "Why learned embeddings instead of one-hot encoding for categoricals?"**

A: One-hot encoding creates very sparse, high-dimensional vectors (e.g., 200+ product types → 200-dim binary vector). Learned embeddings compress this into a dense vector of 10-50 dimensions where similar categories are mapped to nearby points. This is critical for the k-NN graph construction -- cosine similarity in a dense embedding space is far more meaningful than Hamming distance in a sparse one-hot space.

**Q: "Are the embeddings trained jointly with the model?"**

A: No, the attribute encoding is a preprocessing step. The embeddings are fitted as part of the data pipeline, not backpropagated through during model training. This is a deliberate choice -- it decouples the data representation from the model training, allowing the same encoded features to be reused across different model configurations.

---

### On the Overall Pipeline

**Q: "What is the computational cost of the pipeline?"**

A: The most expensive step is the SVD for dimensionality reduction, which is O(T · S · min(T,S)) -- a one-time cost per calibration cycle. Graph construction is O(T² · p) for the pairwise distances. Both are negligible compared to model training time. The pipeline runs in seconds for portfolios up to ~5000 trades.

**Q: "How often does the pipeline need to be re-run?"**

A: The graph and attributes are recomputed when the portfolio composition changes materially (new trades booked, trades maturing). The standardisation and SVD are refit when the training window shifts. In practice, we rerun the full pipeline daily or weekly, depending on portfolio turnover.

**Q: "Why is the mask described but not in the model's required inputs?"**

A: The mask is a design-level concept -- it describes how we handle missing data at the pipeline level. In the current implementation, zero-padding combined with the standardisation (which centres at zero) means the model implicitly treats missing entries as "no information." An explicit mask tensor is a natural extension for future work but is not required for the current architecture to function correctly.

**Q: "Could this pipeline be applied to other asset classes (rates, credit, commodities)?"**

A: Yes, the pipeline is asset-class agnostic. The only components that change are: (1) the specific attributes used for encoding and graph construction, and (2) the choice of standardisation method (e.g., robust scaler for credit where defaults create extreme outliers). The architecture and data flow remain identical.

---

## Part 3: Key Numbers to Remember

- Typical portfolio size: T = 500-2000 trades
- Typical observation window: S = 250-500 business days (1-2 years)
- Typical basis components after SVD: m = 10-30 (explains 90-97% variance)
- Typical k for graph: 10-50 neighbours
- Sparse adjacency storage: O(T·k) << O(T²) dense
- Pipeline runtime: seconds for T < 5000
