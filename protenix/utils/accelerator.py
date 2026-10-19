# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small CUDA/MUSA accelerator compatibility layer."""

from contextlib import nullcontext
from typing import Optional

import torch


def _try_enable_musa() -> None:
    if hasattr(torch, "musa"):
        return
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        pass


def accelerator_name() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"
    _try_enable_musa()
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    return None


def accelerator_api():
    name = accelerator_name()
    return getattr(torch, name) if name is not None else None


def accelerator_device(local_rank: int = 0) -> torch.device:
    name = accelerator_name()
    return torch.device(f"{name}:{local_rank}" if name is not None else "cpu")


def accelerator_device_count() -> int:
    api = accelerator_api()
    return api.device_count() if api is not None else 0


def accelerator_set_device(device: torch.device) -> None:
    api = accelerator_api()
    if api is not None:
        api.set_device(device)


def accelerator_empty_cache() -> None:
    api = accelerator_api()
    if api is not None:
        api.empty_cache()


def accelerator_manual_seed_all(seed: int) -> None:
    api = accelerator_api()
    if api is not None:
        api.manual_seed_all(seed)


def accelerator_autocast(dtype: torch.dtype, cache_enabled: bool = True):
    name = accelerator_name()
    if name is None:
        return nullcontext()
    return torch.autocast(
        device_type=name, dtype=dtype, cache_enabled=cache_enabled
    )


def distributed_backend() -> str:
    return "mccl" if accelerator_name() == "musa" else "nccl"


def visible_devices_env() -> str:
    if accelerator_name() == "musa":
        return "MTHREADS_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"
