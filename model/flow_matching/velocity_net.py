"""
velocity_net.py — MDM-style transformer that predicts the flow-matching
velocity field ``v_theta(x_t, t)`` for skeletal sequences.

Design
------
Input  : ``x_t`` of shape ``(B, T, J, C)`` (default J=17 joints, C=3 coords),
         plus a flow time ``t`` of shape ``(B,)`` with values in [0, 1].
Output : predicted velocity of shape ``(B, T, J, C)`` — same geometry as x_t,
         which is exactly what the target ``u_t = x_1 - x_0`` (from
         ``flow_matching.path.AffineProbPath(CondOTScheduler())``) requires.

Two separate conditioning signals:

* **Flow-time** ``t``: sinusoidal embedding (continuous, MDM-style) pushed
  through a small MLP; the result is broadcast to every frame and
  **concatenated** to the flattened pose features. This tells the model how
  noisy the input is.
* **Frame position** within the 80-frame clip: additive sinusoidal positional
  embedding indexed by frame index. This lets the transformer learn temporal
  relationships.

Architecture (MDM reference, ICLR 2023; Tevet et al.), resized for CARE-PD:
``d_model=256, nhead=4, num_layers=4, ff_dim=1024, dropout=0.2,
time_emb_dim=128`` — see the plan for rationale.

The forward signature includes an optional ``cond`` argument so class
conditioning can be added later without a refactor. For now it is ignored.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

# Human3.6M 17-joint left/right swap used for bilateral augmentation.
# Index: 0 pelvis, 1-3 right_hip/knee/ankle, 4-6 left_hip/knee/ankle,
#        7-10 spine/neck/nose/head, 11-13 left_shoulder/elbow/wrist,
#        14-16 right_shoulder/elbow/wrist.
H36M_17J_MIRROR_PERM = (
    0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13,
)


# ---------------------------------------------------------------------------
# Sinusoidal embeddings
# ---------------------------------------------------------------------------

def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Continuous sinusoidal embedding as used in diffusion/flow papers.

    ``values`` is a 1-D tensor (any shape is flattened to 1-D then reshaped
    back). The output has shape ``values.shape + (dim,)``. ``dim`` must be
    even (an explicit assertion is raised if not).
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_embedding expects even dim, got {dim}.")
    orig_shape = values.shape
    v = values.reshape(-1).to(torch.float32)  # (N,)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=v.device)
        / half
    )  # (half,)
    args = v[:, None] * freqs[None, :]  # (N, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, dim)
    return emb.reshape(*orig_shape, dim)


class FramePositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over frame index.

    Registered as a buffer so it moves with ``.to(device)`` and is not a
    learnable parameter.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
        self.max_len = max_len

    def forward(self, T: int) -> torch.Tensor:
        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds FramePositionalEncoding.max_len={self.max_len}"
            )
        return self.pe[:T]


class FlowTimeMLP(nn.Module):
    """Linear -> SiLU -> Linear on the sinusoidal time embedding.

    Matches MDM's ``TimestepEmbedder`` structure, but takes a continuous
    ``t`` (flow time in [0, 1]) instead of an integer diffusion step. The
    sinusoidal wavelength is shifted so that t ~ 1 doesn't collapse to a
    near-constant embedding.
    """

    def __init__(self, sinusoid_dim: int, out_dim: int, time_scale: float = 1_000.0):
        super().__init__()
        self.sinusoid_dim = sinusoid_dim
        self.time_scale = time_scale
        self.mlp = nn.Sequential(
            nn.Linear(sinusoid_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        sin_t = sinusoidal_embedding(t * self.time_scale, self.sinusoid_dim)
        return self.mlp(sin_t)


# ---------------------------------------------------------------------------
# Velocity network
# ---------------------------------------------------------------------------

class VelocityNet(nn.Module):
    """MDM-style encoder-only transformer that outputs per-frame velocity.

    Parameters
    ----------
    n_joints, n_coords:
        Skeleton geometry (default 17, 3).
    d_model:
        Transformer hidden size.
    nhead:
        Number of attention heads.
    num_layers:
        Transformer encoder layer count.
    ff_dim:
        Feed-forward dim inside each transformer layer.
    dropout:
        Dropout on attention and FFN.
    time_emb_dim:
        Sinusoidal + MLP dim for the flow time embedding. This dim is
        concatenated to the flattened pose features per frame before the
        input projection.
    max_len:
        Maximum clip length supported by the positional encoding buffer.
    """

    def __init__(
        self,
        n_joints: int = 17,
        n_coords: int = 3,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.2,
        time_emb_dim: int = 128,
        max_len: int = 512,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.n_coords = n_coords
        self.d_model = d_model
        self.time_emb_dim = time_emb_dim
        input_dim = n_joints * n_coords  # e.g. 51

        # Flow-time embedding (continuous t in [0, 1])
        self.flow_time_mlp = FlowTimeMLP(
            sinusoid_dim=time_emb_dim, out_dim=time_emb_dim
        )

        # Per-frame input projection: concat(pose_flat, t_tok) -> d_model
        self.in_proj = nn.Linear(input_dim + time_emb_dim, d_model)

        # Frame positional encoding (added to token embeddings)
        self.frame_pos = FramePositionalEncoding(d_model, max_len=max_len)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Readout: d_model -> n_joints * n_coords, then reshape
        self.out_proj = nn.Linear(d_model, input_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Truncated-normal weight init consistent with ACTOR/MDM conventions.
        # Note: we intentionally do NOT zero the output projection; doing so
        # kills the gradient through the encoder on the first step (since
        # d loss / d encoder_out factors through W_out which would be 0).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,  # reserved for class conditioning
    ) -> torch.Tensor:
        """Predict the velocity ``v_theta(x_t, t)``.

        Parameters
        ----------
        x_t : ``(B, T, J, C)`` tensor of noised poses.
        t : ``(B,)`` tensor of flow times in [0, 1].
        mask : optional ``(B, T)`` bool tensor with ``True`` on real frames
               and ``False`` on padded frames. ``src_key_padding_mask`` for
               the transformer is built as ``~mask``.
        cond : unused; accepted for forward compatibility with class
               conditioning.

        Returns
        -------
        ``(B, T, J, C)`` tensor of predicted velocities.
        """
        if cond is not None:  # intentionally unused right now
            pass
        B, T, J, C = x_t.shape
        if J != self.n_joints or C != self.n_coords:
            raise ValueError(
                f"Expected (B, T, {self.n_joints}, {self.n_coords}), got {tuple(x_t.shape)}"
            )
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"t must be (B,), got {tuple(t.shape)}")

        x_flat = x_t.reshape(B, T, J * C)                       # (B, T, 51)
        t_emb = self.flow_time_mlp(t)                           # (B, time_emb_dim)
        t_tok = t_emb.unsqueeze(1).expand(B, T, self.time_emb_dim)  # (B, T, 128)
        tok = torch.cat([x_flat, t_tok], dim=-1)                # (B, T, 51+128)
        tok = self.in_proj(tok)                                 # (B, T, d_model)
        tok = tok + self.frame_pos(T).to(tok.dtype)             # add frame pos

        if mask is None:
            src_key_padding_mask = None
        else:
            if mask.shape != (B, T):
                raise ValueError(f"mask must be (B, T) = ({B}, {T}), got {tuple(mask.shape)}")
            # nn.Transformer expects True at positions that should be IGNORED.
            src_key_padding_mask = ~mask

        h = self.encoder(tok, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)
        v_flat = self.out_proj(h)                                           # (B, T, 51)
        v = v_flat.reshape(B, T, J, C)
        return v

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
