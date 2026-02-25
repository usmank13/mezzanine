from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from .hash128 import TCHash128
from .rng import derive_seed_u64, rand_u64


@dataclass(frozen=True)
class TraceConfig:
    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 64
    num_samples: int = 16
    quant: Literal["fp16", "fp32", "int8"] = "fp16"
    include_coords: bool = True


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def sampled_gemm_trace(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    config: TraceConfig = TraceConfig(),
    hasher: Optional[TCHash128] = None,
) -> Tuple[bytes, List[Tuple[int, int, int]]]:
    """Hash a **sample** of intermediate GEMM tiles.

    This is a CPU reference for the protocol concept:
    - production kernels should update the hash inside the fused GEMM kernel
      while computing partial products.

    The sampling is deterministic from `sigma`.

    Returns (digest_bytes, sampled_coords).
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D tensors")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A is {A.shape}, B is {B.shape}")
    if any(x <= 0 for x in (config.tile_m, config.tile_n, config.tile_k)):
        raise ValueError("tile sizes must be positive")

    m_tiles = _ceil_div(int(M), int(config.tile_m))
    n_tiles = _ceil_div(int(N), int(config.tile_n))
    k_tiles = _ceil_div(int(K), int(config.tile_k))
    total_tiles = m_tiles * n_tiles * k_tiles
    if total_tiles == 0:
        raise ValueError("Empty matrices")

    h = hasher if hasher is not None else TCHash128(seed=sigma)

    seed_u64 = derive_seed_u64(sigma, domain="trace/samples")
    # Sample indices via uint64 stream.
    u = rand_u64((config.num_samples, 3), seed_u64=seed_u64)
    coords: List[Tuple[int, int, int]] = []
    for i in range(config.num_samples):
        mi = int(u[i, 0] % np.uint64(m_tiles))
        ni = int(u[i, 1] % np.uint64(n_tiles))
        ki = int(u[i, 2] % np.uint64(k_tiles))
        coords.append((mi, ni, ki))

    # Compute + hash sampled partial products.
    for (mi, ni, ki) in coords:
        m0 = mi * config.tile_m
        m1 = min((mi + 1) * config.tile_m, M)
        n0 = ni * config.tile_n
        n1 = min((ni + 1) * config.tile_n, N)
        k0 = ki * config.tile_k
        k1 = min((ki + 1) * config.tile_k, K)

        At = A[m0:m1, k0:k1].to(dtype=torch.float32)
        Bt = B[k0:k1, n0:n1].to(dtype=torch.float32)
        partial = At @ Bt  # [m, n]
        if config.include_coords:
            h.update_bytes(struct.pack("<III", mi, ni, ki))
            h.update_bytes(struct.pack("<III", m0, n0, k0))
        h.update_tensor(partial, quant=config.quant)

    return h.digest(), coords
