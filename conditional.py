"""
conditional.py
--------------
Implements the Conditional Vector (cond) construction and
Training-by-Sampling logic for CTGAN (Xu et al., NeurIPS 2019).

The conditional vector:
    - Has total length = sum of all discrete column cardinalities.
    - For one training sample, exactly one bit is set to 1.
    - The column i* is chosen uniformly at random from Nd discrete columns.
    - The category k* within D_i* is sampled proportional to
      log-frequency: prob(k) ∝ log(freq(k) + 1).

Training-by-Sampling:
    - Real rows returned are guaranteed to satisfy D_i* == k*.
    - This ensures the real-data distribution is also conditioned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from data_transformer import DataTransformer, DiscreteColumnInfo


class ConditionalVectorBuilder:
    """
    Pre-computes log-frequency tables for each discrete column and
    provides fast batch sampling of (cond, selected_col_idx, selected_cat_idx).
    """

    def __init__(self, transformer: DataTransformer, data: pd.DataFrame):
        """
        Parameters
        ----------
        transformer : DataTransformer
            A fitted DataTransformer (discrete column info extracted from it).
        data : pd.DataFrame
            The original (un-transformed) training DataFrame.
        """
        self.transformer     = transformer
        self.disc_columns: List[DiscreteColumnInfo] = transformer.disc_columns
        self.n_discrete      = len(self.disc_columns)
        self.cond_dim        = sum(d.n_categories for d in self.disc_columns)

        # Pre-compute offsets in the cond vector for each discrete column
        self.offsets: List[int] = []
        offset = 0
        for d in self.disc_columns:
            self.offsets.append(offset)
            offset += d.n_categories

        # Pre-compute log-frequency tables (for training-by-sampling)
        # and raw frequency tables (for inference sampling)
        self._log_freq_pmfs: List[np.ndarray] = []
        self._raw_freq_pmfs: List[np.ndarray] = []

        # Also store row indices per (col_idx, category)
        # row_indices[col_idx][cat_idx] = array of row positions
        self._row_indices: List[List[np.ndarray]] = []

        for col_info in self.disc_columns:
            labels = col_info.encoder.transform(
                data[col_info.name].astype(str)
            )
            n_cat  = col_info.n_categories

            counts = np.zeros(n_cat, dtype=float)
            for lbl in labels:
                counts[lbl] += 1.0

            # log-frequency PMF (used during training)
            log_freq = np.log(counts + 1.0)
            log_freq_pmf = log_freq / log_freq.sum()
            self._log_freq_pmfs.append(log_freq_pmf)

            # raw-frequency PMF (used during inference / sampling)
            raw_pmf = counts / counts.sum()
            self._raw_freq_pmfs.append(raw_pmf)

            # row indices
            rows_per_cat: List[np.ndarray] = []
            for cat in range(n_cat):
                rows_per_cat.append(np.where(labels == cat)[0])
            self._row_indices.append(rows_per_cat)

    # ──────────────────────────────────────────────────────────
    #  Batch sampler (used inside training loop)
    # ──────────────────────────────────────────────────────────

    def sample_train(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of conditional vectors for training.

        Returns
        -------
        cond : np.ndarray, shape (batch_size, cond_dim)
            One-hot conditional vectors.
        col_idx : np.ndarray, shape (batch_size,) int
            Which discrete column was selected for each sample.
        cat_idx : np.ndarray, shape (batch_size,) int
            Which category was selected within that column.
        row_indices : np.ndarray, shape (batch_size,) int
            Row positions in the training dataset that match D_i* == k*.
        """
        if self.n_discrete == 0:
            # No discrete columns — return zero vectors
            cond = np.zeros((batch_size, 0), dtype=np.float32)
            col_idx = np.zeros(batch_size, dtype=int)
            cat_idx = np.zeros(batch_size, dtype=int)
            row_idx = np.random.randint(0, 1, size=batch_size)
            return cond, col_idx, cat_idx, row_idx

        cond    = np.zeros((batch_size, self.cond_dim), dtype=np.float32)
        col_idx = np.random.randint(0, self.n_discrete, size=batch_size)
        cat_idx = np.zeros(batch_size, dtype=int)
        row_idx = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            ci = col_idx[i]
            pmf = self._log_freq_pmfs[ci]
            k   = np.random.choice(len(pmf), p=pmf)
            cat_idx[i] = k

            # Set the corresponding bit in cond
            cond[i, self.offsets[ci] + k] = 1.0

            # Sample a matching real row
            candidates = self._row_indices[ci][k]
            if len(candidates) == 0:
                # Edge case: no rows — pick any row randomly
                row_idx[i] = np.random.randint(0, sum(
                    len(self._row_indices[ci][kk]) for kk in range(self.disc_columns[ci].n_categories)
                ))
            else:
                row_idx[i] = candidates[np.random.randint(0, len(candidates))]

        return cond, col_idx, cat_idx, row_idx

    # ──────────────────────────────────────────────────────────
    #  Inference sampler (used in CTGAN.sample())
    # ──────────────────────────────────────────────────────────

    def sample_inference(self, n: int) -> np.ndarray:
        """
        Build conditional vectors for inference (sampling synthetic rows).
        Uses raw marginal frequency (not log-frequency).

        Returns
        -------
        cond : np.ndarray, shape (n, cond_dim)
        """
        if self.n_discrete == 0:
            return np.zeros((n, 0), dtype=np.float32)

        cond    = np.zeros((n, self.cond_dim), dtype=np.float32)
        col_idx = np.random.randint(0, self.n_discrete, size=n)

        for i in range(n):
            ci  = col_idx[i]
            pmf = self._raw_freq_pmfs[ci]
            k   = np.random.choice(len(pmf), p=pmf)
            cond[i, self.offsets[ci] + k] = 1.0

        return cond

    # ──────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self.cond_dim
