"""
demo.py
-------
End-to-end demonstration of CTGAN on the UCI Adult (Census Income) dataset.

Pipeline:
  1. Download & prepare Adult dataset
  2. Train/test split
  3. Fit CTGAN on training set
  4. Sample synthetic dataset of the same size
  5. Evaluate with Machine Learning Efficacy:
       - Train classifiers on SYNTHETIC data
       - Test on REAL held-out test set
       - Compare against training on REAL data
  6. Print full comparison report
"""

from __future__ import annotations

import sys, os
# Ensure ctgan package directory is on path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from ctgan import CTGAN


# ─────────────────────────────────────────────────────────────
#  1. Load Adult Dataset
# ─────────────────────────────────────────────────────────────

ADULT_URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
)

ADULT_COLS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

DISCRETE_COLUMNS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country', 'income'
]

TARGET_COL = 'income'


def load_adult() -> pd.DataFrame:
    """
    Load the UCI Adult dataset. Tries local cache first, then downloads.
    """
    local_path = os.path.join(os.path.dirname(__file__), 'adult.csv')
    if os.path.exists(local_path):
        print('[Demo] Loading Adult dataset from local cache...')
        df = pd.read_csv(local_path)
    else:
        print('[Demo] Downloading Adult dataset from UCI...')
        try:
            df = pd.read_csv(ADULT_URL, header=None, names=ADULT_COLS,
                             na_values=' ?', sep=', ', engine='python')
            df.to_csv(local_path, index=False)
        except Exception as e:
            print(f'[Demo] Download failed: {e}')
            print('[Demo] Generating a synthetic stand-in dataset for demo purposes.')
            df = _make_demo_dataset(n=5000)
            df.to_csv(local_path, index=False)

    df = df.dropna().reset_index(drop=True)
    # Strip whitespace from string columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    return df


def _make_demo_dataset(n: int = 5000) -> pd.DataFrame:
    """Fallback: generate a plausible-schema dataset when UCI is unreachable."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'age':            rng.integers(18, 90, n),
        'workclass':      rng.choice(['Private', 'Self-emp', 'Gov', 'Other'], n),
        'fnlwgt':         rng.integers(10000, 1000000, n),
        'education':      rng.choice(['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'PhD'], n),
        'education-num':  rng.integers(1, 16, n),
        'marital-status': rng.choice(['Married', 'Single', 'Divorced'], n),
        'occupation':     rng.choice(['Tech', 'Sales', 'Service', 'Craft', 'Other'], n),
        'relationship':   rng.choice(['Husband', 'Wife', 'Not-in-family', 'Own-child'], n),
        'race':           rng.choice(['White', 'Black', 'Asian', 'Other'], n),
        'sex':            rng.choice(['Male', 'Female'], n),
        'capital-gain':   rng.integers(0, 100000, n),
        'capital-loss':   rng.integers(0, 5000, n),
        'hours-per-week': rng.integers(1, 99, n),
        'native-country': rng.choice(['United-States', 'Mexico', 'Other'], n),
        'income':         rng.choice(['<=50K', '>50K'], n, p=[0.75, 0.25]),
    })
    return df


# ─────────────────────────────────────────────────────────────
#  2. Pre-processing helpers for ML Efficacy evaluation
# ─────────────────────────────────────────────────────────────

def prepare_for_ml(
    df: pd.DataFrame,
    discrete_cols: list,
    target: str,
    fit_encoders=None,
):
    """
    One-hot encode discrete columns (except target), label-encode target.
    Returns X (np.ndarray), y (np.ndarray), and fitted encoders dict.
    """
    df = df.copy()
    encoders = fit_encoders or {}

    # Encode target
    if 'target_le' not in encoders:
        le = LabelEncoder()
        le.fit(df[target].astype(str))
        encoders['target_le'] = le
    y = encoders['target_le'].transform(df[target].astype(str))

    # One-hot encode categoricals (except target)
    cat_cols = [c for c in discrete_cols if c != target]
    num_cols = [c for c in df.columns if c not in discrete_cols]

    # Encode categoricals
    cat_parts = []
    for c in cat_cols:
        if f'le_{c}' not in encoders:
            le = LabelEncoder()
            le.fit(df[c].astype(str))
            encoders[f'le_{c}'] = le
        lbl = encoders[f'le_{c}'].transform(df[c].astype(str))
        ohe = np.zeros((len(lbl), len(encoders[f'le_{c}'].classes_)), dtype=float)
        ohe[np.arange(len(lbl)), lbl] = 1.0
        cat_parts.append(ohe)

    num_arr = df[num_cols].to_numpy(dtype=float) if num_cols else np.zeros((len(df), 0))

    X = np.concatenate(cat_parts + [num_arr], axis=1) if cat_parts else num_arr

    return X, y, encoders


# ─────────────────────────────────────────────────────────────
#  3. ML Efficacy Evaluation
# ─────────────────────────────────────────────────────────────

CLASSIFIERS = {
    'DecisionTree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'LinearSVM':    LinearSVC(max_iter=2000, random_state=42),
    'MLP':          MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
}


def evaluate_ml_efficacy(
    train_df:   pd.DataFrame,
    test_df:    pd.DataFrame,
    syn_df:     pd.DataFrame,
    discrete_cols: list,
    target:     str,
):
    """
    For each classifier:
      - Train on REAL train → test on REAL test   (baseline)
      - Train on SYNTHETIC  → test on REAL test   (TSTR)

    Prints a comparison table.
    """
    print('\n' + '='*65)
    print('  MACHINE LEARNING EFFICACY — TSTR vs TRTR')
    print('='*65)
    print(f'  {"Classifier":<18} {"TRTR F1":>10} {"TSTR F1":>10} {"Delta":>10}')
    print('-'*65)

    X_train_r, y_train_r, encoders = prepare_for_ml(train_df, discrete_cols, target)
    X_test,    y_test,    _        = prepare_for_ml(test_df,   discrete_cols, target, encoders)

    # Synthetic data — re-use same encoders
    X_syn, y_syn, _ = prepare_for_ml(syn_df, discrete_cols, target, encoders)

    for name, clf_template in CLASSIFIERS.items():
        import copy

        # TRTR: train on real
        clf_real = copy.deepcopy(clf_template)
        clf_real.fit(X_train_r, y_train_r)
        pred_real = clf_real.predict(X_test)
        f1_real = f1_score(y_test, pred_real, average='weighted', zero_division=0)

        # TSTR: train on synthetic
        clf_syn = copy.deepcopy(clf_template)
        clf_syn.fit(X_syn, y_syn)
        pred_syn = clf_syn.predict(X_test)
        f1_syn = f1_score(y_test, pred_syn, average='weighted', zero_division=0)

        delta = f1_syn - f1_real
        print(f'  {name:<18} {f1_real:>10.4f} {f1_syn:>10.4f} {delta:>+10.4f}')

    print('='*65)
    print('  TRTR = Train on Real, Test on Real (oracle)')
    print('  TSTR = Train on Synthetic, Test on Real')
    print('  Delta = TSTR - TRTR  (closer to 0 is better)')
    print('='*65 + '\n')


# ─────────────────────────────────────────────────────────────
#  4. Main
# ─────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*65)
    print('  CTGAN — Conditional Tabular GAN Demo')
    print('  Xu et al., NeurIPS 2019')
    print('='*65 + '\n')

    # ── Load data ─────────────────────────────────────────────
    df = load_adult()
    print(f'[Demo] Dataset shape: {df.shape}')
    print(f'[Demo] Columns: {list(df.columns)}')

    # ── Train/test split ───────────────────────────────────────
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f'[Demo] Train: {len(train_df)} rows  |  Test: {len(test_df)} rows\n')

    # ── Fit CTGAN ─────────────────────────────────────────────
    ctgan = CTGAN(
        epochs     = 300,
        batch_size = 500,
        z_dim      = 128,
        hidden_dim = 256,
        lr_g       = 2e-4,
        lr_d       = 2e-4,
        pac_size   = 10,
        lambda_gp  = 10.0,
        tau        = 0.2,
        verbose    = True,
    )
    ctgan.fit(train_df, DISCRETE_COLUMNS)

    # ── Save model ────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), 'ctgan_adult.pkl')
    ctgan.save(model_path)

    # ── Generate synthetic data ───────────────────────────────
    print(f'\n[Demo] Sampling {len(train_df)} synthetic rows...')
    syn_df = ctgan.sample(len(train_df))
    print(f'[Demo] Synthetic data shape: {syn_df.shape}')

    # Show a few rows
    print('\n[Demo] Synthetic samples (first 5 rows):')
    print(syn_df.head().to_string(index=False))

    # ── Distribution comparison ────────────────────────────────
    print('\n[Demo] Discrete column distributions (income):')
    print('  Real:')
    print(train_df['income'].value_counts(normalize=True).to_string())
    print('  Synthetic:')
    print(syn_df['income'].value_counts(normalize=True).to_string())

    print('\n[Demo] Continuous column stats (age):')
    stats = pd.DataFrame({
        'Real':      train_df['age'].describe(),
        'Synthetic': syn_df['age'].describe(),
    })
    print(stats.to_string())

    # ── ML Efficacy ───────────────────────────────────────────
    evaluate_ml_efficacy(
        train_df      = train_df,
        test_df       = test_df,
        syn_df        = syn_df,
        discrete_cols = DISCRETE_COLUMNS,
        target        = TARGET_COL,
    )

    # ── Training loss plot ────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')          # headless — no display required
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ctgan.g_losses, label='Generator',   color='#4C72B0', linewidth=1.5)
    ax.plot(ctgan.d_losses, label='Critic (D)',  color='#DD8452', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('CTGAN Training Losses (WGAN-GP) — UCI Adult Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'training_losses.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'[Demo] Loss plot saved to: {plot_path}')

    # ── Save synthetic data to CSV ────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), 'synthetic_adult.csv')
    syn_df.to_csv(out_path, index=False)
    print(f'[Demo] Synthetic data saved to: {out_path}')


if __name__ == '__main__':
    main()
