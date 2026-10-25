"""Minimal CLIP-compatible VisionTransformer (plain torch/torchvision only).

Weight-compatible with open_clip ViT-B/16 (visual tower). Lives in the shared
core so the same code trains, evaluates, and ships inside both submission
zips (evaluation container has no open_clip).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """CLIP MHA: out_proj named 'out_proj' to match nn.MultiheadAttention."""

    def __init__(self, d: int, n: int):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.n = n
        self.qkv_split = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.in_proj(x).reshape(B, T, 3, self.n, D // self.n).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.c_fc = nn.Linear(d, h)
        self.c_proj = nn.Linear(h, d)

    def forward(self, x):
        # QuickGELU - matches OpenAI CLIP training-time activation exactly
        h = self.c_fc(x)
        return self.c_proj(h * torch.sigmoid(1.702 * h))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d: int, n: int, h: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = MultiheadAttention(d, n)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = MLP(d, h)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    """open_clip ViT-B/16 visual tower clone (global-pool image branch)."""

    def __init__(self, embed_dim: int = 768, layers: int = 12, heads: int = 12,
                 patch_size: int = 16, image_size: int = 224, width: int = 768,
                 output_dim: int = 512):
        super().__init__()
        self.input_resolution = image_size
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(width, heads, width * 4)
                                     for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        # open_clip names the proj 'conv1'.. 'ln_1'.. and the head 'proj'
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                       # B, width, 14, 14
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # B, 196, width
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)          # B, 197, width
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NOTE: blocks are batch-first (B,T,D); no open_clip-style (T,B,D) permute here
        for blk in self.blocks:
            x = blk(x)
        # OpenAI CLIP ViT: ln_post over the full sequence, then CLASS-TOKEN pool
        # (pool_type='tok', final_ln_after_pool=False in open_clip)
        x = self.ln_post(x)
        pooled = x[:, 0]
        if self.proj is not None:
            pooled = pooled @ self.proj
        return pooled


class PARModel(nn.Module):
    """CLIP visual tower + linear multi-label head for 40 attributes."""

    def __init__(self, vit: VisionTransformer, num_attributes: int = 40):
        super().__init__()
        self.visual = vit
        self.head = nn.Linear(vit.proj.shape[1], num_attributes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.visual(x)
        return self.head(feats)


def load_open_clip_visual(state: dict) -> dict:
    """Map an open_clip ViT-B/16 state dict onto VisionTransformer keys."""
    out = {}
    for k, v in state.items():
        if not k.startswith("visual."):
            continue
        nk = k[len("visual."):]
        # open_clip: visual.transformer.resblocks.<i>.* -> blocks.<i>.*
        nk = nk.replace("transformer.resblocks.", "blocks.")
        # open_clip attn: in_proj_weight/in_proj_bias + out_proj.* -> ours
        nk = nk.replace("attn.in_proj_weight", "attn.in_proj.weight")
        nk = nk.replace("attn.in_proj_bias", "attn.in_proj.bias")
        out[nk] = v
    return out
