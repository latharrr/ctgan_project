"""
models.py
---------
Generator and Critic architectures for CTGAN (Xu et al., NeurIPS 2019).

Generator
---------
Residual fully-connected network:
    h0 = z ⊕ cond                         (input)
    h1 = h0 ⊕ ReLU(BN(FC(h0 → 256)))     (residual block 1)
    h2 = h1 ⊕ ReLU(BN(FC(h1 → 256)))     (residual block 2)
Output heads from h2:
    - alpha_hat_i = tanh(FC(dim_h2 → 1))
    - beta_hat_i  = gumbel_softmax(FC(dim_h2 → n_modes_i), tau=0.2)
    - d_hat_i     = gumbel_softmax(FC(dim_h2 → n_cats_i),  tau=0.2)

Critic (PacGAN, pac_size=10)
----------------------------
Input: 10 rows + 10 cond vectors concatenated.
    h0 = input
    h1 = Dropout(LeakyReLU(0.2)(FC(in_dim → 256)))
    h2 = Dropout(LeakyReLU(0.2)(FC(256 → 256)))
    out = FC(256 → 1)    (no sigmoid — WGAN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─────────────────────────────────────────────────────────────
#  Column output spec  (passed from DataTransformer to models)
# ─────────────────────────────────────────────────────────────

class ColumnOutputSpec:
    """
    Describes one segment of the generator output vector.
    type_str : 'alpha' | 'beta' | 'discrete'
    dim      : length of that segment
    """
    def __init__(self, col_name: str, type_str: str, dim: int):
        self.col_name = col_name
        self.type_str = type_str
        self.dim      = dim


# ─────────────────────────────────────────────────────────────
#  Residual block used in Generator
# ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.fc(x))
        h = F.relu(h)
        # Skip connection: pad with zeros if dimensions differ
        if x.shape[-1] != h.shape[-1]:
            pad  = h.shape[-1] - x.shape[-1]
            skip = F.pad(x, (0, pad))
        else:
            skip = x
        return torch.cat([skip, h], dim=-1)  # concatenation-based residual


# ─────────────────────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    CTGAN Generator.

    Parameters
    ----------
    z_dim     : int   — latent noise dimension (128)
    cond_dim  : int   — conditional vector dimension
    col_specs : list  — ordered list of ColumnOutputSpec
    hidden    : int   — hidden layer size (256)
    tau       : float — Gumbel-Softmax temperature (0.2)
    """

    def __init__(
        self,
        z_dim:     int,
        cond_dim:  int,
        col_specs: List[ColumnOutputSpec],
        hidden:    int   = 256,
        tau:       float = 0.2,
    ):
        super().__init__()
        self.z_dim     = z_dim
        self.cond_dim  = cond_dim
        self.col_specs = col_specs
        self.tau       = tau

        # ── Trunk: two residual blocks ──────────────────────
        in_dim0 = z_dim + cond_dim          # h0 dimension
        in_dim1 = in_dim0 + hidden          # h1 dimension after cat-residual
        in_dim2 = in_dim1 + hidden          # h2 dimension

        self.block1 = ResidualBlock(in_dim0, hidden)
        self.block2 = ResidualBlock(in_dim1, hidden)

        # ── Output heads (one Linear per segment in col_specs) ──
        self.output_heads = nn.ModuleList([
            nn.Linear(in_dim2, spec.dim) for spec in col_specs
        ])

    def forward(
        self,
        z:    torch.Tensor,   # (N, z_dim)
        cond: torch.Tensor,   # (N, cond_dim)
    ) -> torch.Tensor:
        # h0
        h = torch.cat([z, cond], dim=-1)   # (N, z_dim + cond_dim)

        # h1 = h0 || ReLU(BN(FC(h0)))
        h = self.block1(h)                 # (N, in_dim0 + hidden)

        # h2 = h1 || ReLU(BN(FC(h1)))
        h = self.block2(h)                 # (N, in_dim1 + hidden)

        # Output heads
        outputs = []
        for head, spec in zip(self.output_heads, self.col_specs):
            logits = head(h)
            if spec.type_str == 'alpha':
                out = torch.tanh(logits)                        # (N, 1)
            else:
                # 'beta' or 'discrete' → Gumbel-Softmax
                out = F.gumbel_softmax(logits, tau=self.tau, hard=False)
            outputs.append(out)

        return torch.cat(outputs, dim=-1)                       # (N, row_dim)


# ─────────────────────────────────────────────────────────────
#  Critic  (PacGAN with pac_size = 10)
# ─────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    CTGAN Critic with PacGAN packing.

    Parameters
    ----------
    row_dim   : int   — dimension of one row vector
    cond_dim  : int   — dimension of one cond vector
    pac_size  : int   — number of samples packed together (10)
    hidden    : int   — hidden layer size (256)
    dropout   : float — dropout probability (0.5)
    """

    def __init__(
        self,
        row_dim:  int,
        cond_dim: int,
        pac_size: int   = 10,
        hidden:   int   = 256,
        dropout:  float = 0.5,
    ):
        super().__init__()
        self.pac_size = pac_size
        in_dim = pac_size * (row_dim + cond_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (N, pac_size * (row_dim + cond_dim))
            Pre-packed concatenation of pac_size rows + pac_size cond vectors.

        Returns
        -------
        score : torch.Tensor, shape (N, 1)
        """
        return self.net(x)
