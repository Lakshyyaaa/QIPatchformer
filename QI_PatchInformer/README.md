# QI-PatchInformer: A Quantum-Inspired Patch Transformer for Long-Term Financial Time-Series Forecasting

Official PyTorch implementation of the research paper:  
**"QI-PatchInformer: A Quantum-Inspired Patch Transformer for Long-Term Financial Time-Series Forecasting"**  
*Harshika Masand & Lakshya Santani*  
*Department of Information Technology, K J Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai, Maharashtra, India.*

---

## 🌟 Overview

**QI-PatchInformer** is an encoder-only time-series forecasting architecture designed for financial market modeling and long-term time-series forecasting (LTSF). It combines subseries patching with a novel **Quantum-Inspired Attention (QIA)** mechanism based on Quantum Mechanics and **Born's Rule**.

Standard attention mechanisms use classical dot-product similarity ($\mathbf{Q}\mathbf{K}^T / \sqrt{d_k}$), which assumes linear relationships between tokens. QI-PatchInformer models each temporal patch as a quantum wavefunction in complex Hilbert space, computing attention weights from Born's rule state overlaps:

$$\psi(x) = a(x) \cdot e^{i \theta(x)}$$

$$S_{ij} = \left| \langle \psi_i \mid \psi_j \rangle \right|^2 = \text{Re} \left[ \langle \psi_i \mid \psi_j \rangle \right]^2 + \text{Im} \left[ \langle \psi_i \mid \psi_j \rangle \right]^2$$

```
                               ┌────────────────────────────────┐
                               │     Raw Time-Series Input      │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  Instance Normalization (RevIN) │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Subseries Patching & Embed   │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Quantum-Inspired Attention   │
                               │   (Born's Rule Wavefunctions)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │     Linear Projection Head     │
                               └────────────────────────────────┘
```

---

## 📊 Datasets

The repository includes preprocessed datasets for both standard LTSF benchmarks and real financial market assets:

### 1. Financial Datasets (Daily Open-High-Low-Close-Volume)
- **AAPL** (`dataset/aapl.csv`): Single-asset equity daily OHLCV prices.
- **S&P 500** (`dataset/sp500.csv`): U.S. Equity Market Benchmark Index.
- **NIFTY 50** (`dataset/nifty50.csv`): Indian Equity Market Benchmark Index.
- **BTC/USD** (`dataset/btcusd.csv`): High-volatility Cryptocurrency daily prices.

### 2. Standard LTSF Benchmarks
- **ETTh1** (`dataset/ETTh1.csv`): Electricity Transformer Temperature (Hourly).
- **ETTm1** (`dataset/ETTm1.csv`): Electricity Transformer Temperature (15-min).
- **Exchange Rate** (`dataset/exchange_rate.csv`): Foreign Exchange rates of 8 countries.

---

## 🛠️ Repository Structure

```
QI_PatchInformer/
├── QI_PatchInformer__A_Quantum_Inspired_Patch_Transformer_for_Long_Term_Financial_Time_Series_Forecasting.pdf  # Research Paper
├── models/
│   └── PatchTST.py                 # Core PatchTST & QI-PatchInformer model wrapper
├── layers/
│   ├── PatchTST_backbone.py        # Quantum-Inspired Attention (QIA) & backbone layers
│   ├── RevIN.py                    # Reversible Instance Normalization
│   └── PatchTST_layers.py          # Helper transformer sub-modules
├── exp/
│   ├── exp_main.py                 # Training & evaluation pipeline
│   └── exp_basic.py                # Abstract base experiment class
├── data_provider/
│   ├── data_factory.py             # Dataset factory dispatcher
│   └── data_loader.py              # Custom & ETT dataset loader classes
├── utils/
│   ├── tools.py                    # Early stopping, LR adjustments & visualizers
│   └── metrics.py                  # Evaluation metrics (MSE, MAE, RSE)
├── dataset/                        # Preprocessed CSV data files
├── run_longExp.py                  # Main CLI execution entrypoint
├── run_qi_experiment.sh            # Example single-run shell script
└── requirements.txt                # Dependencies
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Install PyTorch and project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Single QI-PatchInformer Experiment
To run a sample experiment on the Apple (AAPL) dataset with $L=336$ lookback and $T=96$ forecast horizon:

```bash
bash run_qi_experiment.sh
```

Or execute via Python CLI:
```bash
python3 run_longExp.py \
  --random_seed 2022 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path aapl.csv \
  --model_id AAPL_336_96 \
  --model PatchTST \
  --version QIA \
  --data custom \
  --features M \
  --target Close \
  --freq d \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.3 \
  --patch_len 16 \
  --stride 8 \
  --train_epochs 50 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001
```

### 3. Run Standard Softmax Baseline
To train the baseline PatchTST model with standard softmax attention for direct comparison, simply pass `--version Standard`:

```bash
python3 run_longExp.py \
  --model PatchTST \
  --version Standard \
  --data custom \
  --data_path aapl.csv \
  --seq_len 336 \
  --pred_len 96 \
  --is_training 1
```

---

## ⚙️ Model Hyperparameters

As specified in the research paper:

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| Look-back Window ($L$) | `336` | Historical input timesteps |
| Forecast Horizons ($T$) | `96, 192, 336, 720` | Output prediction horizons |
| Patch Length ($P$) | `16` | Subseries patch size |
| Stride ($S$) | `8` | Patch extraction stride |
| Model Dimension ($D$) | `16` | Hidden feature embedding dimension |
| Attention Heads ($H$) | `4` | Number of multi-head attention heads |
| Head Dimension ($d_k$) | `4` | $D/H = 4$ ($d_k$ is even for amplitude/phase split) |
| Encoder Layers ($e$) | `3` | Number of stacked encoder layers |
| Feed-forward Dim ($d_{ff}$) | `128` | Dimension of FCN hidden layer |
| Normalization | `RevIN` | Reversible Instance Normalization (without affine) |
| Optimizer / Schedule | `Adam` / `OneCycleLR` | Initial learning rate = $10^{-4}$ |

---

## 📜 Citation

If you find this codebase or paper useful in your research, please consider citing:

```bibtex
@article{masand2025qipatchinformer,
  title={QI-PatchInformer: A Quantum-Inspired Patch Transformer for Long-Term Financial Time-Series Forecasting},
  author={Masand, Harshika and Santani, Lakshya},
  journal={Department of Information Technology, K J Somaiya School of Engineering, Somaiya Vidyavihar University},
  year={2025}
}
```

---

## 📄 License
This repository is released under the [MIT License](LICENSE).
