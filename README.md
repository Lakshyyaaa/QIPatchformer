# QI-Patchformer

## Quantum-Inspired Patch Transformer for Long-Term Financial Time-Series Forecasting

QI-Patchformer is an encoder-only Transformer architecture for long-term
financial time-series forecasting. The model combines patch-based
representation learning from PatchTST with a Quantum-Inspired Attention
(QIA) mechanism based on Hilbert-space representations and Born's Rule.

Instead of measuring query-key similarity with the conventional scaled
dot product, QI-Patchformer represents queries and keys as
complex-valued wavefunctions and computes their similarity using quantum
transition probability.

## Overview

Long-term financial forecasting is challenging because financial series
are noisy, non-stationary, nonlinear, and subject to changing market
regimes. QI-Patchformer is designed to preserve local temporal
information through patching while providing an alternative nonlinear
similarity function for Transformer attention.

The core pipeline is:

``` text
Raw Multivariate Time Series
          |
          v
      RevIN
          |
          v
 Channel-Independent Processing
          |
          v
      Patching
          |
          v
 Patch Embedding + Positional Encoding
          |
          v
 Quantum-Inspired Transformer Encoder
          |
          v
 Linear Forecasting Head
          |
          v
 Long-Term Forecast
```

![QI-Patchformer Architecture](assets/architecture.png)

## Key Idea

QI-Patchformer keeps the patch-based Transformer structure of PatchTST
and changes the similarity function used by self-attention.

For each query and key vector:

1.  Split the feature representation into amplitude and phase
    components.
2.  L2-normalize the amplitude component.
3.  Map the phase component into `[-π, π]`.
4.  Construct a complex-valued wavefunction.
5.  Compute the overlap between query and key wavefunctions.
6.  Convert the overlap into an attention score using Born's Rule.
7.  Normalize the scores with softmax and apply them to the value
    vectors.

The resulting attention score is:

$$
P(q,k) = |\langle \psi_q | \psi_k \rangle|^2
$$

This replaces the scaled dot-product similarity used in standard
Transformer attention.

![Quantum-Inspired Attention](assets/quantum-inspired-attention.png)

## Why Patching?

Given an input sequence of length `L`, patching partitions each channel
into overlapping sub-series of patch length `P` and stride `S`.

This reduces the effective sequence length from approximately `L` to:

$$
N \approx \frac{L}{S}
$$

The approach preserves local temporal structure while reducing the
number of tokens processed by the Transformer.

## Model Architecture

QI-Patchformer follows the channel-independent, patch-based design of
PatchTST.

The encoder contains:

-   Multi-head Quantum-Inspired Attention
-   Residual connections
-   Feed-forward networks
-   GELU activation
-   Batch normalization
-   Dropout
-   Reversible Instance Normalization (RevIN)

The attention mechanism is the primary architectural modification: the
original dot-product similarity is replaced by Born's Rule attention
while the remaining PatchTST-style encoder components are retained for a
controlled comparison.

## Computational Complexity

Both standard scaled dot-product attention and the proposed
Quantum-Inspired Attention have the same asymptotic attention
complexity:

$$
O(N^2 d_k)
$$

With patching:

$$
O\left(\left(\frac{L}{S}\right)^2 d_k\right)
$$

The QIA mechanism introduces additional state-construction operations
and a larger constant factor, but it does not change the asymptotic
complexity class.

![Complexity Comparison](assets/complexity.png)

In the reported experiments, QIA required approximately twice the
wall-clock time per epoch compared with standard attention on the Apple
Silicon MPS backend.

## Forecasting Setup

The experiments use:

  Parameter                   Value
  ------------------------- -------
  Look-back window              336
  Forecast horizon               96
  Patch length                   16
  Patch stride                    8
  Model dimension                16
  Attention heads                 4
  Encoder layers                  3
  Head dimension                  4
  Feed-forward dimension        128
  Dropout                       0.3
  Activation                   GELU
  Normalization               RevIN
  Optimizer                    Adam
  Initial learning rate        1e-4
  Batch size                    128
  Maximum epochs                 50
  Early stopping patience        10

The benchmark experiments use five random seeds for the reported QIA
results.

## Datasets

QI-Patchformer is evaluated on both standard long-term forecasting
benchmarks and financial datasets.

### Benchmark datasets

-   ETTh1
-   ETTh2
-   ETTm1

### Financial datasets

-   S&P 500
-   NIFTY 50
-   AAPL
-   BTC/USD

The financial datasets use daily OHLCV data. The five input channels
are:

-   Open
-   High
-   Low
-   Close
-   Volume

The Close channel is used as the univariate target while all five
channels are predicted jointly.

## Results

### Benchmark Results

Mean Squared Error (MSE), where lower is better.

  --------------------------------------------------------------------------------------
  Dataset      Horizon   QI-Patchformer   Informer   Autoformer   Crossformer   PatchTST
  --------- ---------- ---------------- ---------- ------------ ------------- ----------
  ETTh1             96  0.3781 ± 0.0024        ---        0.384         0.305      0.375

  ETTh1            192  0.4153 ± 0.0030        ---        0.392         0.352      0.414

  ETTh1            336  0.4270 ± 0.0017      1.128        0.505         0.440      0.431

  ETTh1            720  0.4426 ± 0.0035      1.215        0.498         0.519      0.449

  ETTh2             96  0.2756 ± 0.0008        ---        0.261           ---      0.274

  ETTh2            192  0.3389 ± 0.0009        ---        0.312           ---      0.339

  ETTh2            336  0.3300 ± 0.0005      2.723        0.471           ---      0.331

  ETTh2            720  0.3805 ± 0.0005      3.467        0.474           ---      0.379
  --------------------------------------------------------------------------------------

### Financial Results

96-step forecasting horizon.

  --------------------------------------------------------------------------
  Dataset               QI-Patchformer             PatchTST Better Model
  --------------- -------------------- -------------------- ----------------
  S&P 500              0.4736 ± 0.0043      0.4711 ± 0.0052 PatchTST
                                                            (+0.53%)

  NIFTY 50             0.3100 ± 0.0043      0.3067 ± 0.0055 PatchTST
                                                            (+1.08%)

  AAPL                 2.7517 ± 0.0216      2.8134 ± 0.0616 QI-Patchformer
                                                            (+2.19%)

  BTC/USD              0.9676 ± 0.0221      0.9711 ± 0.0157 QI-Patchformer
                                                            (+0.36%)
  --------------------------------------------------------------------------

The results indicate that the effectiveness of Born's Rule attention is
dataset-dependent. QI-Patchformer improves over PatchTST on AAPL and
BTC/USD, while PatchTST remains slightly better on the S&P 500 and NIFTY
50.

## Training and Evaluation

The reported implementation uses a PatchTST codebase with the attention
module replaced by the proposed Quantum-Inspired Attention mechanism.
This keeps the rest of the architecture aligned with PatchTST for a
controlled comparison.

For the benchmark datasets, ETTh1 and ETTh2 experiments were repeated
using five random seeds.

For financial data, the model is evaluated using daily OHLCV data and a
chronological 70/10/20 train/validation/test split.

## Repository Structure

The exact structure should match the implementation in this repository.
A recommended organization is:

``` text
QI-Patchformer/
├── assets/
│   ├── architecture.png
│   ├── quantum-inspired-attention.png
│   └── complexity.png
├── data/
├── models/
│   ├── attention/
│   ├── encoder/
│   └── patchformer.py
├── scripts/
├── experiments/
├── requirements.txt
├── train.py
├── evaluate.py
└── README.md
```

## Installation

Add the installation commands corresponding to the current repository
implementation here.

For example, if the project uses a Python environment:

``` bash
git clone <repository-url>
cd <repository-directory>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Add the exact training and evaluation commands from the repository here.

A typical workflow is:

``` bash
# Train
python train.py

# Evaluate
python evaluate.py
```

Replace these commands with the actual scripts and arguments provided by
the implementation.

## Reproducibility

The reported experiments use:

-   Look-back length: 336
-   Forecast horizon: 96
-   Patch length: 16
-   Patch stride: 8
-   Model dimension: 16
-   4 attention heads
-   3 encoder layers
-   Batch size: 128
-   Maximum 50 epochs
-   Early stopping patience: 10
-   Five random seeds for the reported QIA benchmark results

## Limitations and Future Work

The paper identifies several directions for improving the approach:

-   Larger attention heads
-   Learnable amplitude-phase conversion
-   Hybrid QIA and dot-product attention
-   Ablation studies of the QIA components
-   Evaluation on intraday financial data
-   Evaluation on order-book data
-   Reduction of the additional computational overhead of QIA

## Citation

If you use this work, please cite:

``` bibtex
@article{masand2025qipatchformer,
  title   = {QI-Patchformer: A Quantum-Inspired Patch Transformer for Long-Term Financial Time-Series Forecasting},
  author  = {Masand, Harshika and Santani, Lakshya and Manek, Jugal Jeetendra and Va, Venkataramanan},
  year    = {2025},
  journal = {Procedia Computer Science}
}
```

## Authors

**Harshika Masand**\
**Lakshya Santani**\
**Jugal Jeetendra Manek**\
**Venkataramanan Va**

Department of Information Technology\
K J Somaiya School of Engineering, Somaiya Vidyavihar University\
Mumbai, Maharashtra, India

## License

The paper is published under the CC BY-NC-ND 4.0 license. The software
license for this repository should be specified separately according to
the license chosen for the codebase.
