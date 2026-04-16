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
# Lightweight attention-dispatch layer.
#
# Inspired by diffusers' attention_dispatch.py:
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py
#
# Key design decisions vs. upstream:
#   - No dependency on diffusers utilities or optional packages.
#   - Backends are plain callables; the registry is a dict keyed by AttentionBackend.
#   - Tensor layout convention matches diffusers: inputs are [B, S, H, Dh].
#     The "native" (SDPA) backend permutes to [B, H, S, Dh] internally, matching
#     PyTorch's F.scaled_dot_product_attention expectation.
#   - New backends (flash-attn, xformers, Intel XeTLA, …) can be added by calling
#     register_attention_backend() and passing the backend name to dispatch_attention_fn.

from __future__ import annotations

import math
import os
import contextlib
import threading
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn.functional as F


# ── Backend names ─────────────────────────────────────────────────────────────

class AttentionBackend(str, Enum):
    """Named attention backends.

    "native"  – PyTorch F.scaled_dot_product_attention (default, always available).
    Additional backends (e.g. "flash", "xformers") can be registered at runtime
    via :func:`register_attention_backend`.
    """
    NATIVE = "native"
    CHUNKED = "chunked"


# ── Registry ──────────────────────────────────────────────────────────────────

# Maps AttentionBackend → callable(query, key, value, attn_mask, dropout_p, is_causal, scale, **kw)
# All callables receive tensors in [B, S, H, Dh] layout and must return [B, S, H, Dh].
_BACKENDS: dict[str, Callable] = {}

# Thread-local active backend (set by the attention_backend() context manager).
_local = threading.local()

# Module-level default (can be changed with set_attention_backend()).
_DEFAULT_BACKEND: str = (
    AttentionBackend.CHUNKED.value
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else AttentionBackend.NATIVE.value
)

# Query-chunk size used by the built-in chunked backend.
_CHUNK_SIZE_DEFAULT = int(os.environ.get("ZIMAGE_ATTN_CHUNK_SIZE", "256"))


def _backend_key(name: str | AttentionBackend) -> str:
    """Normalise a backend name to the bare string value."""
    return name.value if isinstance(name, AttentionBackend) else str(name)


def register_attention_backend(name: str | AttentionBackend, fn: Callable) -> None:
    """Register a custom attention backend.

    Args:
        name: A string or :class:`AttentionBackend` value that identifies the backend.
        fn:   A callable with signature::

                  fn(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, **kwargs) -> Tensor

              Tensors are in **[B, S, H, Dh]** layout.  The callable must return
              a tensor in the same layout.

    Example — registering a flash-attention backend::

        import flash_attn

        def flash_backend(query, key, value, attn_mask=None,
                          dropout_p=0.0, is_causal=False, scale=None, **kw):
            # flash_attn expects [B, S, H, Dh] — no permute needed.
            return flash_attn.flash_attn_func(query, key, value,
                                              dropout_p=dropout_p,
                                              softmax_scale=scale,
                                              causal=is_causal)

        register_attention_backend("flash", flash_backend)
    """
    _BACKENDS[_backend_key(name)] = fn


def set_attention_backend(name: str | AttentionBackend) -> None:
    """Set the module-level default attention backend."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = _backend_key(name)


def set_attention_chunk_size(chunk_size: int) -> None:
    """Set the module-level default query chunk size for the chunked backend."""
    global _CHUNK_SIZE_DEFAULT
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    _CHUNK_SIZE_DEFAULT = int(chunk_size)


@contextlib.contextmanager
def attention_backend(name: str | AttentionBackend = AttentionBackend.NATIVE):
    """Context manager that temporarily switches the active attention backend.

    Example::

        with attention_backend("flash"):
            image = pipe(prompt, ...)
    """
    prev = getattr(_local, "backend", None)
    _local.backend = _backend_key(name)
    try:
        yield
    finally:
        _local.backend = prev


@contextlib.contextmanager
def attention_chunk_size(chunk_size: int):
    """Temporarily override the active query chunk size for chunked attention."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    prev = getattr(_local, "chunk_size", None)
    _local.chunk_size = int(chunk_size)
    try:
        yield
    finally:
        _local.chunk_size = prev


def _get_active_backend() -> str:
    """Return the currently active backend name."""
    return getattr(_local, "backend", None) or _DEFAULT_BACKEND


def _get_active_chunk_size() -> int:
    """Return the currently active query chunk size."""
    return getattr(_local, "chunk_size", None) or _CHUNK_SIZE_DEFAULT


# ── Built-in backends ─────────────────────────────────────────────────────────

def _native_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """PyTorch native SDPA backend.

    Accepts [B, S, H, Dh] — permutes to [B, H, S, Dh] for SDPA, then permutes back.
    """
    q = query.permute(0, 2, 1, 3)   # [B, H, S, Dh]
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return out.permute(0, 2, 1, 3)   # [B, S, H, Dh]


def _chunked_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Memory-bounded SDPA by slicing the query sequence into smaller chunks.

    Inputs and output use [B, S, H, Dh] layout.
    """
    if is_causal:
        raise NotImplementedError("Chunked backend currently supports non-causal attention only.")

    q = query.permute(0, 2, 1, 3)  # [B, H, Sq, Dh]
    k = key.permute(0, 2, 1, 3)    # [B, H, Sk, Dh]
    v = value.permute(0, 2, 1, 3)  # [B, H, Sk, Dh]

    _, _, seq_len_q, head_dim = q.shape
    scale_val = float(scale) if scale is not None else (1.0 / math.sqrt(head_dim))

    chunk_size = int(kwargs.get("chunk_size", _get_active_chunk_size()))
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    out = torch.empty_like(q)

    for start in range(0, seq_len_q, chunk_size):
        end = min(start + chunk_size, seq_len_q)
        q_chunk = q[:, :, start:end, :]  # [B, H, C, Dh]

        # Handle masks that carry a query dimension by slicing it per chunk.
        mask_chunk = attn_mask
        if mask_chunk is not None and mask_chunk.ndim >= 4 and mask_chunk.shape[-2] == seq_len_q:
            mask_chunk = mask_chunk[..., start:end, :]

        try:
            # Keep chunked memory behavior while using optimized SDPA kernels.
            out[:, :, start:end, :] = F.scaled_dot_product_attention(
                q_chunk,
                k,
                v,
                attn_mask=mask_chunk,
                dropout_p=dropout_p,
                is_causal=False,
                scale=scale,
            )
        except RuntimeError:
            # Fallback for runtimes that cannot handle SDPA in this configuration.
            k_t = k.transpose(-2, -1)
            attn_scores = torch.matmul(q_chunk, k_t) * scale_val  # [B, H, C, Sk]
            if mask_chunk is not None:
                if mask_chunk.dtype == torch.bool:
                    attn_scores = attn_scores.masked_fill(~mask_chunk, torch.finfo(attn_scores.dtype).min)
                else:
                    attn_scores = attn_scores + mask_chunk

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_chunk.dtype)
            if dropout_p:
                attn_probs = F.dropout(attn_probs, p=dropout_p, training=False)
            out[:, :, start:end, :] = torch.matmul(attn_probs, v)

    return out.permute(0, 2, 1, 3)  # [B, S, H, Dh]


register_attention_backend(AttentionBackend.NATIVE, _native_backend)
register_attention_backend(AttentionBackend.CHUNKED, _chunked_backend)


# ── Public dispatch function ──────────────────────────────────────────────────

def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    *,
    backend: str | AttentionBackend | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch a scaled dot-product attention call to the active backend.

    Tensor layout convention (matches diffusers upstream):
        - query / key / value: **[B, S, H, Dh]**  (batch, sequence, heads, head-dim)
        - attn_mask:           **[B, 1, S_q, S_kv]** or broadcastable equivalent
        - return:              **[B, S, H, Dh]**

    Args:
        query:      Query tensor [B, S, H, Dh].
        key:        Key tensor   [B, S, H, Dh].
        value:      Value tensor [B, S, H, Dh].
        attn_mask:  Optional attention mask (additive float or boolean).
        dropout_p:  Dropout probability (only respected by some backends).
        is_causal:  Whether to apply causal masking.
        scale:      Optional query scale override; defaults to ``1/sqrt(Dh)``.
        backend:    Backend name override.  If ``None``, the context-manager /
                    module-level default is used (initially ``"native"``).
        **kwargs:   Extra keyword arguments forwarded to the backend callable.
    """
    name = _backend_key(backend) if backend is not None else _get_active_backend()
    fn = _BACKENDS.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown attention backend '{name}'. "
            f"Available backends: {list(_BACKENDS)}. "
            "Register a new one with register_attention_backend()."
        )
    return fn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        **kwargs,
    )
