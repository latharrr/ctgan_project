# CTGAN — Conditional Tabular GAN

A clean, from-scratch PyTorch implementation of **CTGAN** from the NeurIPS 2019 paper:

> *Modeling Tabular Data using Conditional GAN*  
> Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni  
> NeurIPS 2019

---

## Project Structure

```
ctgan/
├── data_transformer.py   # VGM fitting, mode-specific normalization, inverse transform
├── conditional.py        # Conditional vector construction, training-by-sampling
├── models.py             # Generator (residual) and Critic (PacGAN) architectures
├── ctgan.py              # Main CTGAN class: fit(), sample(), save(), load()
├── train.py              # WGAN-GP training loop, gradient penalty, cond loss
├── demo.py               # UCI Adult dataset demo + ML efficacy evaluation
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
import pandas as pd
from ctgan import CTGAN

# Load your data
df = pd.read_csv('your_data.csv')

# Specify which columns are discrete/categorical
discrete_columns = ['col_A', 'col_B', 'target']

# Fit CTGAN
model = CTGAN(epochs=300, batch_size=500, verbose=True)
model.fit(df, discrete_columns)

# Generate synthetic rows
synthetic = model.sample(n=1000)
print(synthetic.head())

# Save / load
model.save('ctgan_model.pkl')
loaded = CTGAN.load('ctgan_model.pkl')
```

---

## Run the Demo

```bash
cd ctgan
python demo.py
```

This will:
1. Download the UCI Adult (Census) dataset
2. Train CTGAN for 300 epochs
3. Generate synthetic data of the same size
4. Evaluate ML Efficacy (TSTR vs TRTR) using 3 classifiers

---

## Architecture Summary

### Data Preprocessing (Mode-Specific Normalization)

| Column type | Encoding |
|---|---|
| Continuous | `[alpha ∈ [-1,1] | beta (one-hot mode)]` via Bayesian GMM |
| Discrete | One-hot encoding via LabelEncoder |

### Generator

```
h0 = z(128) ⊕ cond
h1 = [h0 || ReLU(BN(FC(h0 → 256)))]   ← residual block 1
h2 = [h1 || ReLU(BN(FC(h1 → 256)))]   ← residual block 2

Heads (all from h2):
  α̂_i  = tanh(FC → 1)
  β̂_i  = gumbel_softmax(FC → n_modes, τ=0.2)
  d̂_i  = gumbel_softmax(FC → n_cats,  τ=0.2)
```

### Critic (PacGAN, pac_size=10)

```
Input = [r₁ ⊕ … ⊕ r₁₀ ⊕ cond₁ ⊕ … ⊕ cond₁₀]
h1 = Dropout(LeakyReLU(0.2)(FC → 256))
h2 = Dropout(LeakyReLU(0.2)(FC → 256))
out = FC → 1  (scalar, no sigmoid)
```

### Training

| Hyperparameter | Value |
|---|---|
| Loss | WGAN-GP (λ=10) |
| Optimizer | Adam, β=(0.5, 0.9) |
| LR (G and D) | 2×10⁻⁴ |
| Batch size | 500 |
| Epochs | 300 |
| Pac size | 10 |
| z dim | 128 |
| Gumbel τ | 0.2 |
| VGM components | ≤10 (weight > 0.005) |

---

## ML Efficacy Evaluation

| Metric | Description |
|---|---|
| **TRTR** | Train on Real, Test on Real (oracle baseline) |
| **TSTR** | Train on Synthetic, Test on Real (quality proxy) |
| **Delta** | TSTR − TRTR (closer to 0 = better synthetic quality) |

Three classifiers are used: `DecisionTree`, `LinearSVM`, `MLP`.

---

## Key Implementation Notes

- **VGM**: `BayesianGaussianMixture(n_components=10, weight_concentration_prior_type='dirichlet_process')`
- **Gumbel-Softmax**: `F.gumbel_softmax(logits, tau=0.2, hard=False)` — differentiable discrete relaxation
- **Gradient Penalty**: Interpolate real/fake, penalise `(||∇C(x̃)||₂ - 1)²`
- **Conditional Loss**: CrossEntropy between generator output for conditioned column and sampled category
- **BatchNorm**: Only in Generator — never in Critic
- **Dropout(0.5)**: Only in Critic hidden layers
