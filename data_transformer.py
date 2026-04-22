"""
data_transformer.py
-------------------
Implements Mode-Specific Normalization for CTGAN (Xu et al., NeurIPS 2019).

For each continuous column:
    - Fit a Bayesian Gaussian Mixture Model (VGM) with up to 10 components.
    - Keep only components with weight > 0.005.
    - Encode each value as (alpha, beta) where:
        alpha = (x - mu_k) / (4 * sigma_k)  clipped to [-1, 1]
        beta  = one-hot vector of the chosen mode index

For each discrete column:
    - Label-encode then one-hot encode.

Final row vector:
    r = [alpha_1 | beta_1 | ... | alpha_Nc | beta_Nc | d_1 | ... | d_Nd]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  Column meta-data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class ContinuousColumnInfo:
    name: str
    model: BayesianGaussianMixture          # fitted VGM
    active_indices: np.ndarray              # indices of "active" components (weight > 0.005)
    means: np.ndarray                       # means of active components
    stds: np.ndarray                        # std-devs of active components
    weights: np.ndarray                     # weights of active components (sum ≈ 1)
    n_modes: int                            # = len(active_indices)
    output_dim: int                         # 1 (alpha) + n_modes (beta)


@dataclass
class DiscreteColumnInfo:
    name: str
    encoder: LabelEncoder
    n_categories: int
    output_dim: int                         # = n_categories


# ─────────────────────────────────────────────────────────────
#  DataTransformer
# ─────────────────────────────────────────────────────────────

class DataTransformer:
    """
    Fits a mode-specific normaliser on training data and
    provides transform / inverse_transform.
    """

    VGM_MAX_COMPONENTS = 10
    VGM_WEIGHT_THRESHOLD = 0.005

    def __init__(self):
        self.cont_columns: List[ContinuousColumnInfo] = []
        self.disc_columns: List[DiscreteColumnInfo] = []
        self.output_dim: int = 0           # total row vector length
        self._column_order: List[str] = [] # ordered column names as given during fit
        self._column_types: dict = {}      # name -> 'continuous' | 'discrete'

    # ──────────────────────────────────────────────────────────
    #  fit
    # ──────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame, discrete_columns: List[str]):
        """
        Fit the transformer on a training DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
        discrete_columns : list[str]
            Names of columns that should be treated as discrete.
            All other columns are treated as continuous.
        """
        self.cont_columns = []
        self.disc_columns = []
        self._column_order = list(data.columns)
        discrete_set = set(discrete_columns)

        for col in self._column_order:
            if col in discrete_set:
                self._fit_discrete(data[col], col)
                self._column_types[col] = 'discrete'
            else:
                self._fit_continuous(data[col], col)
                self._column_types[col] = 'continuous'

        self.output_dim = (
            sum(c.output_dim for c in self.cont_columns) +
            sum(d.output_dim for d in self.disc_columns)
        )

    def _fit_continuous(self, series: pd.Series, name: str):
        values = series.to_numpy(dtype=float).reshape(-1, 1)

        vgm = BayesianGaussianMixture(
            n_components=self.VGM_MAX_COMPONENTS,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            max_iter=100,
            n_init=1,
            random_state=42,
        )
        vgm.fit(values)

        # Keep only components whose weight exceeds the threshold
        active = np.where(vgm.weights_ > self.VGM_WEIGHT_THRESHOLD)[0]
        if len(active) == 0:
            # Fall-back: keep the single highest-weight component
            active = np.array([np.argmax(vgm.weights_)])

        means   = vgm.means_[active].flatten()
        stds    = np.sqrt(vgm.covariances_[active]).flatten()
        weights = vgm.weights_[active]
        weights = weights / weights.sum()   # re-normalise

        n_modes    = len(active)
        output_dim = 1 + n_modes

        info = ContinuousColumnInfo(
            name=name,
            model=vgm,
            active_indices=active,
            means=means,
            stds=stds,
            weights=weights,
            n_modes=n_modes,
            output_dim=output_dim,
        )
        self.cont_columns.append(info)

    def _fit_discrete(self, series: pd.Series, name: str):
        le = LabelEncoder()
        le.fit(series.astype(str))
        n_cat = len(le.classes_)

        info = DiscreteColumnInfo(
            name=name,
            encoder=le,
            n_categories=n_cat,
            output_dim=n_cat,
        )
        self.disc_columns.append(info)

    # ──────────────────────────────────────────────────────────
    #  transform
    # ──────────────────────────────────────────────────────────

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Convert a DataFrame into the CTGAN row-vector representation.

        Returns
        -------
        np.ndarray, shape (N, output_dim)
        """
        N = len(data)
        parts = []

        # --- continuous ---
        cont_dict = {c.name: c for c in self.cont_columns}
        disc_dict  = {d.name: d for d in self.disc_columns}

        for col in self._column_order:
            if self._column_types[col] == 'continuous':
                info  = cont_dict[col]
                alpha, beta = self._transform_continuous(data[col].to_numpy(dtype=float), info)
                parts.append(alpha.reshape(-1, 1))
                parts.append(beta)
            else:
                info = disc_dict[col]
                ohe  = self._transform_discrete(data[col].astype(str), info)
                parts.append(ohe)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def _transform_continuous(
        self,
        values: np.ndarray,
        info: ContinuousColumnInfo,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns alpha (N,) and beta (N, n_modes).
        Mode is sampled proportional to component responsibility rho_k.
        """
        N       = len(values)
        n_modes = info.n_modes

        # Compute probability density under each active component
        # rho shape: (N, n_modes)
        rho = np.zeros((N, n_modes))
        for k, (mu, sigma, w) in enumerate(zip(info.means, info.stds, info.weights)):
            rho[:, k] = w * _gaussian_pdf(values, mu, sigma)

        rho_sum = rho.sum(axis=1, keepdims=True)
        rho_sum = np.where(rho_sum == 0, 1e-10, rho_sum)
        rho     = rho / rho_sum                             # normalise to probabilities

        # Sample mode index for each row
        mode_idx = np.array([
            np.random.choice(n_modes, p=rho[i]) for i in range(N)
        ])

        # Compute alpha
        mu_k    = info.means[mode_idx]
        sigma_k = info.stds[mode_idx]
        alpha   = (values - mu_k) / (4.0 * sigma_k)
        alpha   = np.clip(alpha, -1.0, 1.0)

        # One-hot encode mode
        beta = np.zeros((N, n_modes), dtype=np.float32)
        beta[np.arange(N), mode_idx] = 1.0

        return alpha.astype(np.float32), beta

    def _transform_discrete(
        self,
        series: pd.Series,
        info: DiscreteColumnInfo,
    ) -> np.ndarray:
        labels = info.encoder.transform(series)
        ohe    = np.zeros((len(labels), info.n_categories), dtype=np.float32)
        ohe[np.arange(len(labels)), labels] = 1.0
        return ohe

    # ──────────────────────────────────────────────────────────
    #  inverse_transform
    # ──────────────────────────────────────────────────────────

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """
        Convert generator output back to a DataFrame with original column names.

        Parameters
        ----------
        data : np.ndarray, shape (N, output_dim)
        """
        N      = len(data)
        cursor = 0
        result = {}

        cont_dict = {c.name: c for c in self.cont_columns}
        disc_dict  = {d.name: d for d in self.disc_columns}

        for col in self._column_order:
            if self._column_types[col] == 'continuous':
                info = cont_dict[col]

                # alpha is the first scalar
                alpha = data[:, cursor]
                alpha = np.clip(alpha, -1.0, 1.0)
                cursor += 1

                # beta is the next n_modes values
                beta_logits = data[:, cursor : cursor + info.n_modes]
                cursor += info.n_modes

                mode_idx = np.argmax(beta_logits, axis=1)
                mu_k     = info.means[mode_idx]
                sigma_k  = info.stds[mode_idx]

                recovered = alpha * 4.0 * sigma_k + mu_k
                result[col] = recovered

            else:
                info = disc_dict[col]
                ohe  = data[:, cursor : cursor + info.n_categories]
                cursor += info.n_categories

                cat_idx = np.argmax(ohe, axis=1)
                result[col] = info.encoder.inverse_transform(cat_idx)

        return pd.DataFrame(result, columns=self._column_order)

    # ──────────────────────────────────────────────────────────
    #  Helpers for conditional vector builder
    # ──────────────────────────────────────────────────────────

    def get_output_info(self) -> List[Tuple[str, str, int]]:
        """
        Returns list of (col_name, type, dim) for each column segment
        in the row vector, in order.
        type is 'alpha', 'beta', or 'discrete'.
        """
        info_list = []
        cont_dict = {c.name: c for c in self.cont_columns}
        disc_dict  = {d.name: d for d in self.disc_columns}
        for col in self._column_order:
            if self._column_types[col] == 'continuous':
                c = cont_dict[col]
                info_list.append((col, 'alpha', 1))
                info_list.append((col, 'beta',  c.n_modes))
            else:
                d = disc_dict[col]
                info_list.append((col, 'discrete', d.n_categories))
        return info_list


# ─────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────

def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate N(x; mu, sigma) element-wise."""
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
