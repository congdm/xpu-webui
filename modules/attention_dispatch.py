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


# ── Registry ──────────────────────────────────────────────────────────────────

# Maps AttentionBackend → callable(query, key, value, attn_mask, dropout_p, is_causal, scale, **kw)
# All callables receive tensors in [B, S, H, Dh] layout and must return [B, S, H, Dh].
_BACKENDS: dict[str, Callable] = {}

# Thread-local active backend (set by the attention_backend() context manager).
_local = threading.local()

# Module-level default (can be changed with set_attention_backend()).
_DEFAULT_BACKEND: str = AttentionBackend.NATIVE.value


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


def _get_active_backend() -> str:
    """Return the currently active backend name."""
    return getattr(_local, "backend", None) or _DEFAULT_BACKEND


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


register_attention_backend(AttentionBackend.NATIVE, _native_backend)


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
