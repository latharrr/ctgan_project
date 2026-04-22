"""
ctgan.py
--------
Main CTGAN class — public API: fit() and sample().

Ties together DataTransformer, ConditionalVectorBuilder,
Generator, Critic, and the training loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import List, Optional

from data_transformer import DataTransformer
from conditional import ConditionalVectorBuilder
from models import Generator, Critic, ColumnOutputSpec
from train import train_ctgan


class CTGAN:
    """
    Conditional Tabular GAN (Xu et al., NeurIPS 2019).

    Parameters
    ----------
    epochs      : int    — number of training epochs (default 300)
    batch_size  : int    — training batch size (default 500)
    z_dim       : int    — latent noise dimension (default 128)
    hidden_dim  : int    — generator/critic hidden layer size (default 256)
    lr_g        : float  — Generator Adam learning rate (default 2e-4)
    lr_d        : float  — Critic Adam learning rate (default 2e-4)
    pac_size    : int    — PacGAN packing size (default 10)
    lambda_gp   : float  — gradient penalty coefficient (default 10.0)
    tau         : float  — Gumbel-Softmax temperature (default 0.2)
    verbose     : bool   — print tqdm progress bar (default True)
    device      : str | None — 'cpu', 'cuda', or None (auto-detect)
    """

    def __init__(
        self,
        epochs:     int   = 300,
        batch_size: int   = 500,
        z_dim:      int   = 128,
        hidden_dim: int   = 256,
        lr_g:       float = 2e-4,
        lr_d:       float = 2e-4,
        pac_size:   int   = 10,
        lambda_gp:  float = 10.0,
        tau:        float = 0.2,
        verbose:    bool  = True,
        device:     Optional[str] = None,
    ):
        self.epochs     = epochs
        self.batch_size = batch_size
        self.z_dim      = z_dim
        self.hidden_dim = hidden_dim
        self.lr_g       = lr_g
        self.lr_d       = lr_d
        self.pac_size   = pac_size
        self.lambda_gp  = lambda_gp
        self.tau        = tau
        self.verbose    = verbose

        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        # Set during fit()
        self._transformer:   Optional[DataTransformer]         = None
        self._cond_builder:  Optional[ConditionalVectorBuilder] = None
        self._generator:     Optional[Generator]                = None
        self._critic:        Optional[Critic]                   = None
        self._col_specs:     Optional[List[ColumnOutputSpec]]   = None
        self._discrete_cols: Optional[List[str]]                = None

        self.g_losses: List[float] = []
        self.d_losses: List[float] = []

    # ──────────────────────────────────────────────────────────
    #  fit
    # ──────────────────────────────────────────────────────────

    def fit(
        self,
        data:            pd.DataFrame,
        discrete_columns: List[str],
    ) -> "CTGAN":
        """
        Fit CTGAN to training data.

        Parameters
        ----------
        data : pd.DataFrame
            Training dataset (no NaN values).
        discrete_columns : list[str]
            Names of categorical columns. All others treated as continuous.

        Returns
        -------
        self
        """
        self._discrete_cols = list(discrete_columns)

        # ── 1. Fit DataTransformer ────────────────────────────
        if self.verbose:
            print('[CTGAN] Fitting data transformer (VGM)...')
        transformer = DataTransformer()
        transformer.fit(data, discrete_columns)
        self._transformer = transformer

        # ── 2. Transform data ─────────────────────────────────
        transformed = transformer.transform(data)   # (N, row_dim)
        row_dim     = transformer.output_dim

        # ── 3. Build column specs for Generator heads ─────────
        col_specs = [
            ColumnOutputSpec(name, typ, dim)
            for name, typ, dim in transformer.get_output_info()
        ]
        self._col_specs = col_specs

        # ── 4. Build Conditional Vector Builder ───────────────
        cond_builder = ConditionalVectorBuilder(transformer, data)
        self._cond_builder = cond_builder
        cond_dim = cond_builder.dim

        # ── 5. Initialise Generator & Critic ──────────────────
        generator = Generator(
            z_dim     = self.z_dim,
            cond_dim  = cond_dim,
            col_specs = col_specs,
            hidden    = self.hidden_dim,
            tau       = self.tau,
        )
        critic = Critic(
            row_dim  = row_dim,
            cond_dim = cond_dim,
            pac_size = self.pac_size,
            hidden   = self.hidden_dim,
        )
        self._generator = generator
        self._critic    = critic

        if self.verbose:
            n_g = sum(p.numel() for p in generator.parameters())
            n_c = sum(p.numel() for p in critic.parameters())
            print(f'[CTGAN] Generator params : {n_g:,}')
            print(f'[CTGAN] Critic    params : {n_c:,}')
            print(f'[CTGAN] Row dim   : {row_dim}  |  Cond dim : {cond_dim}')
            print(f'[CTGAN] Device    : {self._device}')
            print(f'[CTGAN] Training for {self.epochs} epochs ...')

        # ── 6. Train ──────────────────────────────────────────
        g_losses, d_losses = train_ctgan(
            generator    = generator,
            critic       = critic,
            transformer  = transformer,
            cond_builder = cond_builder,
            real_data    = transformed,
            col_specs    = col_specs,
            epochs       = self.epochs,
            batch_size   = self.batch_size,
            z_dim        = self.z_dim,
            lr_g         = self.lr_g,
            lr_d         = self.lr_d,
            pac_size     = self.pac_size,
            lambda_gp    = self.lambda_gp,
            n_critic     = 1,
            device       = self._device,
            verbose      = self.verbose,
        )
        self.g_losses = g_losses
        self.d_losses = d_losses

        if self.verbose:
            print('[CTGAN] Training complete.')

        return self

    # ──────────────────────────────────────────────────────────
    #  sample
    # ──────────────────────────────────────────────────────────

    def sample(self, n: int) -> pd.DataFrame:
        """
        Generate n synthetic rows.

        Parameters
        ----------
        n : int — number of rows to generate.

        Returns
        -------
        pd.DataFrame with original column names and dtypes (approximately).
        """
        if self._generator is None:
            raise RuntimeError('CTGAN has not been fitted yet. Call .fit() first.')

        self._generator.eval()

        results = []
        remaining = n

        while remaining > 0:
            batch = min(remaining, self.batch_size)

            # Sample cond for inference (raw marginal frequency)
            cond_np = self._cond_builder.sample_inference(batch)
            z_np    = np.random.randn(batch, self.z_dim).astype(np.float32)

            z_t    = torch.tensor(z_np,    device=self._device)
            cond_t = torch.tensor(cond_np, device=self._device)

            with torch.no_grad():
                fake = self._generator(z_t, cond_t)

            results.append(fake.cpu().numpy())
            remaining -= batch

        raw = np.concatenate(results, axis=0)[:n]

        df = self._transformer.inverse_transform(raw)
        self._generator.train()
        return df

    # ──────────────────────────────────────────────────────────
    #  persistence
    # ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save generator and transformer state to a file."""
        import pickle
        state = {
            'generator_state': self._generator.state_dict(),
            'transformer':     self._transformer,
            'cond_builder':    self._cond_builder,
            'col_specs':       self._col_specs,
            'config': {
                'z_dim':      self.z_dim,
                'hidden_dim': self.hidden_dim,
                'tau':        self.tau,
                'epochs':     self.epochs,
                'batch_size': self.batch_size,
                'pac_size':   self.pac_size,
            },
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f'[CTGAN] Saved to {path}')

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "CTGAN":
        """Load a previously saved CTGAN model."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        cfg = state['config']
        model = cls(
            z_dim      = cfg['z_dim'],
            hidden_dim = cfg['hidden_dim'],
            tau        = cfg['tau'],
            device     = device,
        )
        model._transformer  = state['transformer']
        model._cond_builder = state['cond_builder']
        model._col_specs    = state['col_specs']

        transformer = model._transformer
        cond_builder = model._cond_builder
        col_specs    = model._col_specs
        row_dim      = transformer.output_dim
        cond_dim     = cond_builder.dim

        generator = Generator(
            z_dim     = cfg['z_dim'],
            cond_dim  = cond_dim,
            col_specs = col_specs,
            hidden    = cfg['hidden_dim'],
            tau       = cfg['tau'],
        )
        generator.load_state_dict(state['generator_state'])
        model._generator = generator
        print(f'[CTGAN] Loaded from {path}')
        return model
