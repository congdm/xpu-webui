# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from:
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_z_image.py
# Changes vs. upstream:
#   - Removed diffusers/accelerate/ConfigMixin dependencies (plain nn.Module)
#   - Replaced torch.amp.autocast("cuda") with dtype-cast helpers for XPU compatibility
#   - Replaced diffusers Attention class with self-contained ZImageAttention
#   - Replaced dispatch_attention_fn with F.scaled_dot_product_attention
#   - Removed gradient-checkpointing wiring (can be re-added if needed)

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# ── Constants (must match model config) ─────────────────────────────────────
ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
X_PAD_DIM = 64


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class Transformer2DModelOutput:
    sample: object  # list[torch.Tensor]


class RMSNorm(nn.Module):
    """Root Mean Square normalisation (bias-free)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm * self.weight.float()).to(orig_dtype)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings; computed in float32 for precision."""
    x_f32 = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_f32)
    freqs_cis = freqs_cis.unsqueeze(2)  # [B, S, 1, D/2]
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.to(x.dtype)


# ── Sub-modules ───────────────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → MLP projection."""

    def __init__(self, out_size: int, mid_size: int | None = None, frequency_embedding_size: int = 256):
        super().__init__()
        mid_size = mid_size or out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # Always in float32 to preserve numerical precision (XPU-safe: no autocast needed)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageAttention(nn.Module):
    """
    Self-attention with optional QK-RMSNorm and RoPE.

    Parameter names are kept identical to diffusers' Attention class so that
    checkpoint weights from Comfy-Org/z_image_turbo load without remapping:
      to_q / to_k / to_v / to_out.0 / norm_q / norm_k
    """

    def __init__(self, query_dim: int, dim_head: int, heads: int, qk_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(query_dim, query_dim, bias=False)
        self.to_v = nn.Linear(query_dim, query_dim, bias=False)
        # ModuleList wrapping matches diffusers' Attention.to_out naming (to_out.0.weight)
        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim, bias=False)])
        if qk_norm:
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(hidden_states).unflatten(-1, (self.heads, self.dim_head))
        k = self.to_k(hidden_states).unflatten(-1, (self.heads, self.dim_head))
        v = self.to_v(hidden_states).unflatten(-1, (self.heads, self.dim_head))

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = _apply_rotary_emb(q, freqs_cis)
            k = _apply_rotary_emb(k, freqs_cis)

        dtype = q.dtype
        # [B, S, H, Dh] → [B, H, S, Dh]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attention_mask is not None and attention_mask.ndim == 2:
            # [B, S] → [B, 1, 1, S] (broadcast over heads and query positions)
            attention_mask = attention_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).flatten(2).to(dtype)  # [B, S, H*Dh]
        return self.to_out[0](out)


def _select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    mask = noise_mask.unsqueeze(-1)  # [B, S, 1]
    return torch.where(
        mask == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


class ZImageTransformerBlock(nn.Module):
    """Single transformer block with adaLN modulation."""

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        self.attention = ZImageAttention(
            query_dim=dim,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
        )
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim * 8 / 3))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            seq_len = x.shape[1]

            if noise_mask is not None:
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)
                scale_msa_n, gate_msa_n, scale_mlp_n, gate_mlp_n = mod_noisy.chunk(4, dim=1)
                scale_msa_c, gate_msa_c, scale_mlp_c, gate_mlp_c = mod_clean.chunk(4, dim=1)

                gate_msa_n, gate_mlp_n = gate_msa_n.tanh(), gate_mlp_n.tanh()
                gate_msa_c, gate_mlp_c = gate_msa_c.tanh(), gate_mlp_c.tanh()

                scale_msa = _select_per_token(1.0 + scale_msa_n, 1.0 + scale_msa_c, noise_mask, seq_len)
                scale_mlp = _select_per_token(1.0 + scale_mlp_n, 1.0 + scale_mlp_c, noise_mask, seq_len)
                gate_msa = _select_per_token(gate_msa_n, gate_msa_c, noise_mask, seq_len)
                gate_mlp = _select_per_token(gate_mlp_n, gate_mlp_c, noise_mask, seq_len)
            else:
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    """Final adaLN + linear projection before unpatchify."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        c_noisy: torch.Tensor | None = None,
        c_clean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        if noise_mask is not None:
            scale = _select_per_token(
                1.0 + self.adaLN_modulation(c_noisy),
                1.0 + self.adaLN_modulation(c_clean),
                noise_mask,
                seq_len,
            )
        else:
            assert c is not None
            scale = (1.0 + self.adaLN_modulation(c)).unsqueeze(1)
        x = self.norm_final(x) * scale
        return self.linear(x)


class RopeEmbedder:
    """Rotary position embedder with multi-axis support."""

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self._cache: list[torch.Tensor] | None = None

    @staticmethod
    def _precompute(dim: list[int], end: list[int], theta: float = 256.0) -> list[torch.Tensor]:
        result = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            t = torch.arange(e, dtype=torch.float64)
            freqs = torch.outer(t, freqs).float()
            result.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return result

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if self._cache is None:
            self._cache = self._precompute(self.axes_dims, self.axes_lens, self.theta)
        cache = [c.to(device) for c in self._cache]
        return torch.cat([cache[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


# ── Main model ────────────────────────────────────────────────────────────────

class ZImageTransformer2DModel(nn.Module):
    """
    Z-Image-Turbo transformer.

    Ported from diffusers' ZImageTransformer2DModel. Weight parameter names are
    preserved exactly so that checkpoints (including ComfyUI single-file format)
    load without key remapping.
    """

    def __init__(
        self,
        all_patch_size: tuple[int, ...] = (2,),
        all_f_patch_size: tuple[int, ...] = (1,),
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        siglip_feat_dim: int | None = None,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: list[int] = (32, 48, 48),
        axes_lens: list[int] = (1024, 512, 512),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder: dict[str, nn.Linear] = {}
        all_final_layer: dict[str, FinalLayer] = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            key = f"{patch_size}-{f_patch_size}"
            all_x_embedder[key] = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_final_layer[key] = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [ZImageTransformerBlock(1000 + i, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
             for i in range(n_refiner_layers)]
        )
        self.context_refiner = nn.ModuleList(
            [ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
             for i in range(n_refiner_layers)]
        )

        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=norm_eps),
                nn.Linear(siglip_feat_dim, dim, bias=True),
            )
            self.siglip_refiner = nn.ModuleList(
                [ZImageTransformerBlock(2000 + i, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
                 for i in range(n_refiner_layers)]
            )
            self.siglip_pad_token = nn.Parameter(torch.empty((1, dim)))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
             for i in range(n_layers)]
        )

        assert (dim // n_heads) == sum(axes_dims), "head_dim must equal sum(axes_dims)"
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    # ── Patchify helpers ──────────────────────────────────────────────────────

    def _patchify_image(self, image: torch.Tensor, patch_size: int, f_patch_size: int):
        pH = pW = patch_size
        pF = f_patch_size
        C, F, H, W = image.size()
        Ft, Ht, Wt = F // pF, H // pH, W // pW
        image = image.view(C, Ft, pF, Ht, pH, Wt, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(Ft * Ht * Wt, pF * pH * pW * C)
        return image, (F, H, W), (Ft, Ht, Wt)

    @staticmethod
    def _coord_grid(size, start=None, device=None):
        if start is None:
            start = [0] * len(size)
        axes = [torch.arange(s0, s0 + span, dtype=torch.int32, device=device) for s0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def _pad_with_ids(self, feat, pos_grid_size, pos_start, device, noise_mask_val=None):
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF
        total_len = ori_len + pad_len

        ori_pos = self._coord_grid(pos_grid_size, pos_start, device).flatten(0, 2)
        if pad_len > 0:
            pad_pos = self._coord_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(pad_len, 1)
            pos_ids = torch.cat([ori_pos, pad_pos], dim=0)
            feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat([
                torch.zeros(ori_len, dtype=torch.bool, device=device),
                torch.ones(pad_len, dtype=torch.bool, device=device),
            ])
        else:
            pos_ids = ori_pos
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        nm = [noise_mask_val] * total_len if noise_mask_val is not None else None
        return feat, pos_ids, pad_mask, total_len, nm

    def _patchify_and_embed(self, all_image, all_cap_feats, patch_size, f_patch_size):
        device = all_image[0].device
        all_img_out, all_img_size, all_img_pos, all_img_pad = [], [], [], []
        all_cap_out, all_cap_pos, all_cap_pad = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats):
            cap_out, cap_pos, cap_pad, cap_len, _ = self._pad_with_ids(
                cap_feat, (len(cap_feat) + (-len(cap_feat)) % SEQ_MULTI_OF, 1, 1), (1, 0, 0), device
            )
            all_cap_out.append(cap_out)
            all_cap_pos.append(cap_pos)
            all_cap_pad.append(cap_pad)

            patches, size, (Ft, Ht, Wt) = self._patchify_image(image, patch_size, f_patch_size)
            img_out, img_pos, img_pad, _, _ = self._pad_with_ids(
                patches, (Ft, Ht, Wt), (cap_len + 1, 0, 0), device
            )
            all_img_out.append(img_out)
            all_img_size.append(size)
            all_img_pos.append(img_pos)
            all_img_pad.append(img_pad)

        return all_img_out, all_cap_out, all_img_size, all_img_pos, all_cap_pos, all_img_pad, all_cap_pad

    def _prepare_sequence(self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None):
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)

        # Replace padding positions with pad_token
        feats_cat = torch.cat(feats, dim=0)
        mask = torch.cat(inner_pad_mask).unsqueeze(-1)
        feats_cat = torch.where(mask, pad_token, feats_cat)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        # RoPE embeddings
        freqs_cis = list(
            self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0)
        )

        # Batch padding
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        # Attention mask
        if all(s == max_seqlen for s in item_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((len(feats), max_seqlen), dtype=torch.bool, device=device)
            for i, s in enumerate(item_seqlens):
                attn_mask[i, :s] = 1

        # Noise mask tensor
        nm_tensor = None
        if noise_mask is not None:
            nm_tensor = pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, : feats.shape[1]]

        return feats, freqs_cis, attn_mask, item_seqlens, nm_tensor

    def _build_unified_sequence(self, x, x_freqs, x_seqlens, cap, cap_freqs, cap_seqlens, device):
        """Basic mode: unified = [x, cap] per batch item."""
        bsz = len(x_seqlens)
        unified, unified_freqs = [], []
        for i in range(bsz):
            xl, cl = x_seqlens[i], cap_seqlens[i]
            unified.append(torch.cat([x[i][:xl], cap[i][:cl]]))
            unified_freqs.append(torch.cat([x_freqs[i][:xl], cap_freqs[i][:cl]]))

        unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
        max_seqlen = max(unified_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        if all(s == max_seqlen for s in unified_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
            for i, s in enumerate(unified_seqlens):
                attn_mask[i, :s] = 1

        return unified, unified_freqs, attn_mask

    # ── Unpatchify ────────────────────────────────────────────────────────────

    def unpatchify(self, x: list[torch.Tensor], size: list[tuple], patch_size: int, f_patch_size: int):
        pH = pW = patch_size
        pF = f_patch_size
        for i in range(len(x)):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        return_dict: bool = True,
        patch_size: int = 2,
        f_patch_size: int = 1,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
    ):
        """
        Args:
            x:          list[Tensor[C, F, H, W]] — one latent per batch item.
            t:          Tensor[B] — normalised timesteps in [0, 1].
            cap_feats:  list[Tensor[S, cap_feat_dim]] — text embeddings.
        """
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        device = x[0].device
        key = f"{patch_size}-{f_patch_size}"

        # Timestep embedding (basic mode: single adaln per sample)
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])

        # Patchify
        x, cap_feats, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = (
            self._patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
        )

        # Image tokens: embed → refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[key](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _, _ = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, device=device
        )
        for layer in self.noise_refiner:
            x = layer(x, x_mask, x_freqs, adaln_input)

        # Caption tokens: embed → refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, device=device
        )
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs)

        # Unified sequence [x, cap]
        unified, unified_freqs, unified_mask = self._build_unified_sequence(
            x, x_freqs, x_seqlens, cap_feats, cap_freqs, cap_seqlens, device
        )

        # Main transformer layers
        for idx, layer in enumerate(self.layers):
            unified = layer(unified, unified_mask, unified_freqs, adaln_input)
            if controlnet_block_samples is not None and idx in controlnet_block_samples:
                unified = unified + controlnet_block_samples[idx]

        # Final projection
        unified = self.all_final_layer[key](unified, c=adaln_input)

        # Unpatchify
        x = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size)

        if return_dict:
            return Transformer2DModelOutput(sample=x)
        return (x,)
