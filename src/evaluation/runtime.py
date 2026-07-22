"""
Runtime and resource measurement helpers (CPU + optional CUDA).
"""

from __future__ import annotations

import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional


@dataclass
class RuntimeTracker:
    """Accumulate wall-clock timings and peak memory for one experiment."""

    times: Dict[str, float] = field(default_factory=dict)
    peak_cpu_ram_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    device_name: str = "cpu"
    number_of_trainable_parameters: Optional[int] = None

    def reset_gpu_peak(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        except Exception:
            pass

    def update_memory_peaks(self) -> None:
        try:
            import psutil

            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            if self.peak_cpu_ram_mb is None or rss_mb > self.peak_cpu_ram_mb:
                self.peak_cpu_ram_mb = float(rss_mb)
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self.peak_gpu_memory_mb = float(peak)
                self.device_name = torch.cuda.get_device_name(0)
            else:
                self.device_name = "cpu"
        except Exception:
            self.device_name = self.device_name or "cpu"

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        self.reset_gpu_peak() if name in ("training", "distillation") else None
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - t0)
            self.update_memory_peaks()

    def total_seconds(self) -> float:
        return float(sum(self.times.values()))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "training_time_seconds": self.times.get("training"),
            "distillation_time_seconds": self.times.get("distillation"),
            "sampling_time_seconds": self.times.get("sampling"),
            "evaluation_time_seconds": self.times.get("evaluation"),
            "total_runtime_seconds": self.total_seconds(),
            "peak_cpu_ram_mb": self.peak_cpu_ram_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "device_name": self.device_name,
            "number_of_trainable_parameters": self.number_of_trainable_parameters,
        }


def count_parameters(model) -> int:
    try:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        return 0


def environment_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_version": None,
        "git_commit": None,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = (
            torch.version.cuda if torch.cuda.is_available() else None
        )
    except Exception:
        pass
    try:
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
        )
        meta["git_commit"] = out.decode().strip()
    except Exception:
        pass
    return meta
