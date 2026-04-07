"""
Transformer encoder/decoder matching ACTOR (Petrovich et al., ICCV 2021).
Source reference: https://github.com/Mathux/ACTOR (MIT License).

Tensor layout matches ACTOR training: batch["x"] is (B, njoints, nfeats, nframes).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.motionclip.transformer import PositionalEncoding


class Encoder_TRANSFORMER(nn.Module):
    """ACTOR encoder: sequence of poses -> (mu, logvar) via query tokens."""

    def __init__(
        self,
        modeltype: str,
        njoints: int,
        nfeats: int,
        num_frames: int,
        num_classes: int,
        translation: bool,
        pose_rep: str,
        glob: bool,
        glob_rot,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        ablation: str | None = None,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats

        if self.ablation == "average_encoder":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
            self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

    def forward(self, batch: dict) -> dict:
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute(3, 0, 1, 2).reshape(nframes, bs, njoints * nfeats)
        x = self.skelEmbedding(x)

        if self.ablation == "average_encoder":
            x = self.sequence_pos_encoder(x)
            final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
            z = final.mean(axis=0)
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
            return {"mu": mu, "logvar": logvar}
        else:
            xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), dim=0)
            xseq = self.sequence_pos_encoder(xseq)
            muandsigmaMask = torch.ones((bs, 2), dtype=torch.bool, device=x.device)
            maskseq = torch.cat((muandsigmaMask, mask), dim=1)
            final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
            mu = final[0]
            logvar = final[1]
            # final[2:] are the T per-frame encoder representations.  These
            # carry rich per-frame context that is otherwise discarded.  The
            # decoder can use them as additional memory tokens so that its
            # cross-attention is frame-specific, avoiding the mean-pose collapse
            # that occurs with a single global z token.
            frame_tokens = final[2:]  # (T, B, latent_dim)

        return {"mu": mu, "logvar": logvar, "frame_tokens": frame_tokens}


class Decoder_TRANSFORMER(nn.Module):
    """ACTOR decoder: latent z + action y -> full sequence."""

    def __init__(
        self,
        modeltype: str,
        njoints: int,
        nfeats: int,
        num_frames: int,
        num_classes: int,
        translation: bool,
        pose_rep: str,
        glob: bool,
        glob_rot,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        ablation: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats

        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation,
        )
        self.seqTransDecoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_layers)
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, batch: dict) -> dict:
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        if self.ablation == "zandtime":
            raise NotImplementedError("zandtime ablation not wired in CARE-PD trainer.")
        if self.ablation == "concat_bias":
            z = torch.stack((z, self.actionBiases[y]), dim=0)
            memory = z
            memory_key_padding_mask = None
        else:
            z = z + self.actionBiases[y]
            frame_tokens = batch.get("frame_tokens", None)
            if frame_tokens is not None:
                # During reconstruction: prepend the global z token to the T
                # per-frame encoder representations.  Cross-attention can now
                # focus on frame_tokens[t] at position t, giving a per-frame
                # gradient that does not cancel over a gait cycle.  z still
                # provides global sequence conditioning via the first token.
                memory = torch.cat(
                    [z.unsqueeze(0), frame_tokens], dim=0
                )  # (T+1, B, latent_dim)
                global_mask = torch.ones(bs, 1, dtype=torch.bool, device=mask.device)
                memory_key_padding_mask = ~torch.cat([global_mask, mask], dim=1)
            else:
                # Generation mode (no encoder context): use z alone, matching
                # original ACTOR behaviour for sampling new sequences.
                memory = z.unsqueeze(0)  # (1, B, latent_dim)
                memory_key_padding_mask = None

        timequeries = torch.zeros(nframes, bs, latent_dim, device=memory.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(
            tgt=timequeries,
            memory=memory,
            tgt_key_padding_mask=~mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        output = output.permute(1, 2, 3, 0).contiguous()
        mask_exp = mask.unsqueeze(1).unsqueeze(1).expand_as(output)
        output = output * mask_exp.float()
        batch["output"] = output
        return batch
