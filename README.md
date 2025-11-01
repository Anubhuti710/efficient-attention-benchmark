# Implementation and Benchmark of Efficient Attention Mechanisms

### [View the Full Benchmark Notebook](./main_benchmark.ipynb)

This project implements and analyzes three Transformer attention mechanisms from scratch in PyTorch to empirically validate their performance-compute trade-offs.

## Summary

The standard attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017) has a computational complexity of $O(n^2)$ with respect to sequence length $n$. This quadratic scaling makes it computationally infeasible for very long sequences.

This project implements and benchmarks two "efficient" alternatives against the standard:

1.  **Standard (Full) Attention:** The $O(n^2)$ baseline, where every token attends to every other token.
2.  **Sparse (Top-k) Attention:** An $O(n \cdot k)$ implementation where each token only attends to the $k$ most similar tokens.
3.  **Local (Sliding Window) Attention:** An $O(n \cdot w)$ implementation where each token only attends to tokens within a local window of size $w$.

## Methodology

1.  **Implementation:** All three mechanisms were implemented as `nn.Module` classes in PyTorch. The code is available in [`attention_mechanisms.py`](./attention_mechanisms.py).
2.  **Performance Benchmark:** Each model was benchmarked on a CPU/GPU by measuring the average execution time over 20 runs for various sequence lengths (from 64 to 1024).
3.  **Qualitative Analysis:** Attention heatmaps were generated for each model to visually inspect their behavior and confirm that the masking was implemented correctly.

## Key Findings

### 1. Performance-Compute Trade-off (The $O(n^2)$ Bottleneck)

The benchmark empirically validates the theoretical complexity.

![Performance Benchmark Plot](./download (1).png)

The log-log plot clearly shows the classic **overhead vs. scaling** trade-off:

* **For short sequences ($n < 128$)**: The `Full O(n^2)` attention is actually the fastest. This is because the "efficient" methods have a fixed *overhead* (e.g., creating the mask, finding the top-k) that outweighs the benefits at a small scale.
* **For long sequences ($n > 128$)**: The `Full O(n^2)` line's slope becomes much steeper, proving its quadratic scaling. The `Sparse` and `Local` methods, despite their initial overhead, have a shallower slope (more linear) and quickly become far more performant as the sequence length grows.

### 2. Qualitative Heatmap Analysis

The heatmaps visually confirm that the attention mechanisms function as designed.

#### Heatmap: Full O(n^2)
![Full Attention Heatmap](./download (2).png)
**Analysis:** Correct. The plot is a solid block, showing that every token (Y-axis) is attending to every other token (X-axis).

#### Heatmap: Local (w=32)
![Local Attention Heatmap](./Screenshot 2025-10-31 at 10.02.46 PM.png)
**Analysis:** Correct. The plot shows a perfect diagonal band. This proves that each token is only attending to tokens within its local window, and all tokens outside this window (the dark purple areas) are correctly masked.

#### Heatmap: Sparse (k=32)
![Sparse Attention Heatmap](./Screenshot 2025-10-31 at 10.02.34 PM.png)
**Analysis:** Correct. The plot shows that each token (Y-axis) is only attending to a fixed number of 32 tokens (X-axis). Because the input was a simple, non-random tensor, all tokens "agreed" on which 32 tokens were the "top-k" most important, creating this blocky pattern. This confirms the top-k masking is working.

## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/anubhuti710/efficient-attention-benchmark.git](https://github.com/anubhuti710/efficient-attention-benchmark.git)
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the benchmark:
    Open and run the `main_benchmark.ipynb` notebook in Jupyter or Google Colab.
