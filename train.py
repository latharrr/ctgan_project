"""
train.py
--------
Training loop for CTGAN (Xu et al., NeurIPS 2019).

Implements:
  - WGAN-GP (Wasserstein GAN with Gradient Penalty, λ=10)
  - PacGAN packing (pac_size=10)
  - Generator cross-entropy penalty on the conditioned discrete column
  - 5 Critic steps per Generator step (WGAN-GP standard)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Tuple, Optional
from tqdm import tqdm

from models import Generator, Critic, ColumnOutputSpec
from conditional import ConditionalVectorBuilder
from data_transformer import DataTransformer


# ─────────────────────────────────────────────────────────────
#  Gradient Penalty
# ─────────────────────────────────────────────────────────────

def compute_gradient_penalty(
    critic:     Critic,
    real_pack:  torch.Tensor,    # (M, pac_size*(row+cond))
    fake_pack:  torch.Tensor,    # (M, pac_size*(row+cond))
    device:     torch.device,
    lambda_gp:  float = 10.0,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty.
    Interpolates between real and fake packed inputs, computes
    ||∇C(x̃)||_2 and penalises (||∇C(x̃)||_2 - 1)^2.
    """
    M     = real_pack.size(0)
    alpha = torch.rand(M, 1, device=device)
    alpha = alpha.expand_as(real_pack)

    interp     = (alpha * real_pack + (1.0 - alpha) * fake_pack).detach()
    interp.requires_grad_(True)

    score_interp = critic(interp)

    grads = torch.autograd.grad(
        outputs=score_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(score_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # grads shape: (M, in_dim)
    grads_norm = grads.view(M, -1).norm(2, dim=1)  # (M,)
    gp = lambda_gp * ((grads_norm - 1.0) ** 2).mean()
    return gp


# ─────────────────────────────────────────────────────────────
#  Pack batches into PacGAN format
# ─────────────────────────────────────────────────────────────

def pack_batch(
    rows: torch.Tensor,    # (N, row_dim)
    cond: torch.Tensor,    # (N, cond_dim)
    pac_size: int,
) -> torch.Tensor:
    """
    Concatenate rows and cond vectors, then reshape into packs of pac_size.

    Returns shape: (N // pac_size, pac_size * (row_dim + cond_dim))
    Any trailing samples that don't fill a full pack are dropped.
    """
    N = rows.size(0)
    # Trim to multiple of pac_size
    N_trimmed = (N // pac_size) * pac_size
    rows = rows[:N_trimmed]
    cond = cond[:N_trimmed]

    # (N_trimmed, row_dim + cond_dim)
    combined = torch.cat([rows, cond], dim=-1)

    # (N_trimmed // pac_size, pac_size * (row_dim + cond_dim))
    return combined.view(N_trimmed // pac_size, -1)


# ─────────────────────────────────────────────────────────────
#  Cross-entropy penalty for Generator
# ─────────────────────────────────────────────────────────────

def generator_cond_loss(
    fake_rows:   torch.Tensor,   # (N, row_dim)
    cond:        torch.Tensor,   # (N, cond_dim)
    col_idx:     np.ndarray,     # (N,) which discrete col was conditioned on
    cat_idx:     np.ndarray,     # (N,) which category was sampled
    col_specs:   List[ColumnOutputSpec],
    transformer: DataTransformer,
    device:      torch.device,
) -> torch.Tensor:
    """
    Cross-entropy between the mask vector (one-hot over D_i*) and
    the generator's output distribution for D_i*.
    """
    disc_cols = transformer.disc_columns
    n_disc    = len(disc_cols)

    if n_disc == 0:
        return torch.tensor(0.0, device=device)

    # Build offset lookup for each discrete column in fake_rows
    # (follows the same ordering as col_specs)
    disc_offsets: List[int] = []
    cursor = 0
    for spec in col_specs:
        if spec.type_str == 'discrete':
            disc_offsets.append(cursor)
        cursor += spec.dim

    total_loss = torch.tensor(0.0, device=device)
    N = fake_rows.size(0)

    # Group by col_idx for efficiency
    for ci in range(n_disc):
        mask = col_idx == ci
        if not mask.any():
            continue

        offset = disc_offsets[ci]
        n_cat  = disc_cols[ci].n_categories

        # Generator outputs Gumbel-softmax probabilities (not raw logits),
        # so use NLL loss with log-probabilities instead of cross_entropy
        # (cross_entropy internally applies softmax, which would double-softmax).
        gen_probs = fake_rows[mask, offset : offset + n_cat]   # (K, n_cat)
        gen_probs = torch.clamp(gen_probs, min=1e-8)           # numerical safety

        # Target: the sampled category index
        target = torch.tensor(cat_idx[mask], dtype=torch.long, device=device)  # (K,)

        loss_i = F.nll_loss(torch.log(gen_probs), target)
        total_loss = total_loss + loss_i

    return total_loss / n_disc


# ─────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────

def train_ctgan(
    generator:   Generator,
    critic:      Critic,
    transformer: DataTransformer,
    cond_builder: ConditionalVectorBuilder,
    real_data:   np.ndarray,         # (N_total, row_dim) — transformed
    col_specs:   List[ColumnOutputSpec],
    *,
    epochs:      int   = 300,
    batch_size:  int   = 500,
    z_dim:       int   = 128,
    lr_g:        float = 2e-4,
    lr_d:        float = 2e-4,
    pac_size:    int   = 10,
    lambda_gp:   float = 10.0,
    n_critic:    int   = 1,
    device:      Optional[torch.device] = None,
    verbose:     bool  = True,
) -> Tuple[List[float], List[float]]:
    """
    Full CTGAN training loop.

    Returns
    -------
    g_losses : list of generator losses per epoch (averaged)
    d_losses : list of critic   losses per epoch (averaged)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = generator.to(device)
    critic    = critic.to(device)

    opt_g = Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_d = Adam(critic.parameters(),    lr=lr_d, betas=(0.5, 0.9))

    N_total    = real_data.shape[0]
    steps_per_epoch = max(N_total // batch_size, 1)

    g_losses: List[float] = []
    d_losses: List[float] = []

    epoch_iter = tqdm(range(epochs), desc='CTGAN Training', unit='epoch') if verbose else range(epochs)

    for epoch in epoch_iter:
        epoch_g = 0.0
        epoch_d = 0.0

        for step in range(steps_per_epoch):

            # ── 1. Sample cond + matching real rows ──────────
            cond_np, col_idx, cat_idx, row_idx = cond_builder.sample_train(batch_size)

            real_rows_np = real_data[row_idx]                    # (B, row_dim)

            real_rows = torch.tensor(real_rows_np, dtype=torch.float32, device=device)
            cond_t    = torch.tensor(cond_np,      dtype=torch.float32, device=device)

            # ── 2. Sample z ───────────────────────────────────
            z = torch.randn(batch_size, z_dim, device=device)

            # ── 3. Generate fake rows ─────────────────────────
            with torch.no_grad():
                fake_rows = generator(z, cond_t)                 # (B, row_dim)

            # ── 4. Pack for PacGAN ────────────────────────────
            real_pack = pack_batch(real_rows, cond_t, pac_size)  # (M, pac_in)
            fake_pack = pack_batch(fake_rows, cond_t, pac_size)

            # ── 5. Update Critic (n_critic steps) ────────────
            d_step_loss = 0.0
            for _ in range(n_critic):
                opt_d.zero_grad()

                d_real = critic(real_pack).mean()
                d_fake = critic(fake_pack.detach()).mean()
                gp     = compute_gradient_penalty(critic, real_pack, fake_pack, device, lambda_gp)

                loss_d = d_fake - d_real + gp
                loss_d.backward()
                opt_d.step()
                d_step_loss += loss_d.item()

            epoch_d += d_step_loss / n_critic   # average across critic steps

            # ── 6. Update Generator ───────────────────────────
            opt_g.zero_grad()

            z2        = torch.randn(batch_size, z_dim, device=device)
            fake_rows2 = generator(z2, cond_t)

            fake_pack2 = pack_batch(fake_rows2, cond_t, pac_size)
            d_fake2    = critic(fake_pack2).mean()

            loss_cond = generator_cond_loss(
                fake_rows2, cond_t, col_idx, cat_idx,
                col_specs, transformer, device
            )

            loss_g = -d_fake2 + loss_cond
            loss_g.backward()
            opt_g.step()

            epoch_g += loss_g.item()

        avg_g = epoch_g / steps_per_epoch
        avg_d = epoch_d / steps_per_epoch
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if verbose:
            epoch_iter.set_postfix({'G': f'{avg_g:.4f}', 'D': f'{avg_d:.4f}'})

    return g_losses, d_losses
