"""CUDA / GPU detection utilities for AutoTabML Studio."""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """Return True when a CUDA-capable GPU is reachable from the current Python runtime.

    Detection order:
    1. ``torch.cuda.is_available()`` — most reliable when PyTorch is installed.
    2. Probe for the NVIDIA driver shared library via ctypes as a lightweight fallback.
    """

    if _torch_cuda_available() is True:
        return True
    return _driver_probe()


@functools.lru_cache(maxsize=1)
def cuda_device_name() -> str | None:
    """Return the name of the first CUDA device, or None."""

    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    return None


def cuda_summary() -> dict[str, object]:
    """Return a lightweight summary dict suitable for diagnostics and logging."""

    available = is_cuda_available()
    device_name = cuda_device_name() if available else None
    device_count = _torch_device_count() if available else 0
    return {
        "cuda_available": available,
        "device_name": device_name,
        "device_count": device_count,
    }


def resolve_use_gpu(preference: bool | str) -> bool | str:
    """Resolve a user's GPU preference against actual hardware availability.

    Returns the validated ``use_gpu`` value to pass to PyCaret ``setup()``.
    When CUDA is unavailable, this gracefully degrades to ``False`` instead of
    letting PyCaret fail at setup time.
    """

    if preference is False:
        return False

    if preference == "force":
        return "force"

    if is_cuda_available():
        return preference

    logger.warning("GPU requested but CUDA is not available; falling back to CPU.")
    return False


# ---------------------------------------------------------------------------
# Internal probes
# ---------------------------------------------------------------------------


def _torch_cuda_available() -> bool | None:
    """Return True/False from torch, or None if torch is not installed."""

    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return None


def _torch_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _driver_probe() -> bool:
    """Lightweight probe for the NVIDIA driver library without importing torch.

    Loads the CUDA driver library and queries the actual device count via
    ``cuDeviceGetCount`` to avoid false positives on systems that have the
    driver DLL installed but no CUDA-capable GPU.
    """

    import ctypes
    import sys

    lib_names = (
        ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so", "libcuda.dylib"]
    )
    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue

        # cuInit(0) must be called before any other driver API function.
        try:
            result = lib.cuInit(0)
            if result != 0:
                continue
        except Exception:
            continue

        # Query actual device count to confirm at least one GPU is present.
        try:
            count = ctypes.c_int(0)
            result = lib.cuDeviceGetCount(ctypes.byref(count))
            if result == 0 and count.value > 0:
                return True
        except Exception:
            continue

    return False
