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
# Simplified port of diffusers' FlowMatchEulerDiscreteScheduler:
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py

import math

import torch


_SIGMA_MIN_CLAMP = 1e-9  # guard against division by zero in exponential shift


class FlowMatchEulerDiscreteScheduler:
    """
    Euler scheduler for flow-matching diffusion models (e.g. Z-Image-Turbo).

    Key differences from a plain DDPM scheduler:
    - Timesteps run from σ=1 (full noise) to σ=0 (clean).
    - Euler step: x_{t-1} = x_t + (σ_{t-1} - σ_t) * model_output
    - An optional resolution-based exponential shift (µ) is applied so that
      larger images are denoised with a flatter noise schedule.
    """

    order = 1  # single-step (Euler)

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_min: float = 0.0

        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._begin_index: int = 0
        self._step_index: int | None = None

    # ── Config dict (consumed by pipeline's calculate_shift helper) ───────────
    @property
    def config(self) -> dict:
        return {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }

    # ── Sigma schedule ────────────────────────────────────────────────────────

    @staticmethod
    def _linear_sigmas(n: int) -> torch.Tensor:
        """Uniform σ schedule from 1 → 0 (n+1 values including terminal 0)."""
        return torch.linspace(1.0, 0.0, n + 1)

    @staticmethod
    def _apply_exp_shift(sigmas: torch.Tensor, mu: float) -> torch.Tensor:
        """
        Exponential (resolution-aware) shift used by Flux / Z-Image.

        σ' = exp(μ) / (exp(μ) + (1/σ - 1))   — same as diffusers upstream.
        """
        exp_mu = math.exp(mu)
        # Guard against division by zero at σ=0
        denom = exp_mu + (1.0 / sigmas.clamp(min=_SIGMA_MIN_CLAMP) - 1.0)
        return (exp_mu / denom).clamp(0.0, 1.0)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
    ):
        if sigmas is not None:
            sigma_tensor = torch.tensor(sigmas, dtype=torch.float32)
            # Append terminal σ=0 if not already present
            if sigma_tensor[-1] != 0.0:
                sigma_tensor = torch.cat([sigma_tensor, torch.zeros(1)])
        else:
            assert num_inference_steps is not None
            sigma_tensor = self._linear_sigmas(num_inference_steps)

        if mu is not None:
            sigma_tensor = self._apply_exp_shift(sigma_tensor, mu)

        # Clamp σ_min
        sigma_tensor = sigma_tensor.clamp(min=self.sigma_min)

        self.sigmas = sigma_tensor.to(device or "cpu")
        # Timesteps are the σ values scaled to [0, num_train_timesteps]
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).to(device or "cpu")
        self._step_index = None

    # ── Step ─────────────────────────────────────────────────────────────────

    def _sigma_at(self, timestep: torch.Tensor) -> torch.Tensor:
        """Look up σ for the given (scalar) timestep."""
        # Find the index where self.timesteps matches; fall back to linear search
        idx = (self.timesteps - timestep.item()).abs().argmin().item()
        return self.sigmas[idx]

    def _sigma_next_at(self, timestep: torch.Tensor) -> torch.Tensor:
        idx = (self.timesteps - timestep.item()).abs().argmin().item()
        return self.sigmas[idx + 1]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        """
        Euler step: x_{t-1} = x_t + (σ_{t-1} - σ_t) * v_pred

        Args:
            model_output: velocity prediction from the denoiser.
            timestep:     current timestep (scalar tensor, 0–num_train_timesteps).
            sample:       current noisy latent x_t.
        """
        sigma = self._sigma_at(timestep)
        sigma_next = self._sigma_next_at(timestep)
        dt = sigma_next - sigma
        prev_sample = sample + model_output * dt

        if return_dict:
            return {"prev_sample": prev_sample}
        return (prev_sample,)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index
