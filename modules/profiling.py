from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


METRIC_TRANSFORMER_STEP = "transformer_step_total_ms"
METRIC_ATTN_BLOCK = "attention_block_total_ms"
METRIC_FFN_BLOCK = "ffn_total_ms"
METRIC_BLOCK_COPY = "block_copy_cpu_to_device_total_ms"


_ACTIVE_PROFILER: "GenerationProfiler | None" = None


def set_active_profiler(profiler: "GenerationProfiler | None") -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler


def get_active_profiler() -> "GenerationProfiler | None":
    return None  # PROFILING DISABLED – restore to: return _ACTIVE_PROFILER


def _device_type(device: str | torch.device | None) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


@dataclass
class GenerationProfiler:
    metadata: dict[str, Any] = field(default_factory=dict)
    _totals_ms: dict[str, float] = field(default_factory=dict)
    _step_ms: list[float] = field(default_factory=list)

    def add_ms(self, metric: str, elapsed_ms: float) -> None:
        self._totals_ms[metric] = self._totals_ms.get(metric, 0.0) + float(elapsed_ms)

    def record_transformer_step(self, elapsed_ms: float) -> None:
        value = float(elapsed_ms)
        self._step_ms.append(value)
        self.add_ms(METRIC_TRANSFORMER_STEP, value)

    def sync(self, device: str | torch.device | None) -> None:
        dtype = _device_type(device)
        if dtype == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            return
        if dtype == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()

    def start_timer(self, device: str | torch.device | None, *, synchronize: bool = False) -> float:
        if synchronize:
            self.sync(device)
        return time.perf_counter()

    def end_timer(
        self,
        metric: str,
        started: float,
        device: str | torch.device | None,
        *,
        synchronize: bool = False,
    ) -> None:
        if synchronize:
            self.sync(device)
        self.add_ms(metric, (time.perf_counter() - started) * 1000.0)

    def summary(self) -> dict[str, Any]:
        transformer_total = self._totals_ms.get(METRIC_TRANSFORMER_STEP, 0.0)
        step_count = len(self._step_ms)

        def _row(label: str, key: str) -> dict[str, float | str]:
            total_ms = self._totals_ms.get(key, 0.0)
            percent = (100.0 * total_ms / transformer_total) if transformer_total > 0 else 0.0
            avg_ms = (total_ms / step_count) if step_count > 0 else 0.0
            return {
                "component": label,
                "total_ms": total_ms,
                "percent_of_transformer": percent,
                "avg_ms_per_step": avg_ms,
            }

        return {
            "metadata": dict(self.metadata),
            "step_count": step_count,
            "step_transformer_ms": list(self._step_ms),
            "totals_ms": dict(self._totals_ms),
            "rows": [
                _row("Transformer total", METRIC_TRANSFORMER_STEP),
                _row("FFN total", METRIC_FFN_BLOCK),
                _row("Attention blocks total", METRIC_ATTN_BLOCK),
                _row("Block copy total (CPU->device)", METRIC_BLOCK_COPY),
            ],
        }
