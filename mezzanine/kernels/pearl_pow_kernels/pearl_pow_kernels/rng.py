from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch


def _u64_from_bytes(b: bytes) -> int:
    return int.from_bytes(b[:8], byteorder="little", signed=False)


def derive_seed_u64(sigma: int | bytes | str, *, domain: str) -> int:
    """Derive a deterministic 64-bit seed from an arbitrary `sigma`.

    This is **not** a cryptographic KDF; it is a stable way to:
      - separate streams (domain separation),
      - keep unit tests deterministic.

    Parameters
    ----------
    sigma:
        A seed value (int/bytes/str).
    domain:
        A short domain string, e.g., "noiseA", "noiseB", "trace", "signs".

    Returns
    -------
    int
        Unsigned 64-bit integer in [0, 2^64).
    """
    if isinstance(sigma, int):
        sigma_b = sigma.to_bytes(32, "little", signed=False)
    elif isinstance(sigma, bytes):
        sigma_b = sigma
    elif isinstance(sigma, str):
        sigma_b = sigma.encode("utf-8")
    else:
        raise TypeError(f"Unsupported sigma type: {type(sigma)}")

    h = hashlib.blake2b(sigma_b + b"|" + domain.encode("utf-8"), digest_size=16).digest()
    return _u64_from_bytes(h)


def _splitmix64_uint64(x: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64.

    Returns uint64 array.
    """
    # All constants are uint64.
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = z ^ (z >> np.uint64(31))
    return z & np.uint64(0xFFFFFFFFFFFFFFFF)


def rand_u64(shape: Tuple[int, ...], *, seed_u64: int, nonce: int = 0) -> np.ndarray:
    """Deterministic uint64 RNG stream using SplitMix64(counter + seed)."""
    n = int(np.prod(shape))
    ctr = np.arange(n, dtype=np.uint64) + np.uint64(nonce)
    x = ctr + np.uint64(seed_u64)
    out = _splitmix64_uint64(x).reshape(shape)
    return out


def rand_uniform01(shape: Tuple[int, ...], *, seed_u64: int, nonce: int = 0) -> np.ndarray:
    """Deterministic U[0,1) floats (float64) from SplitMix64 output."""
    u = rand_u64(shape, seed_u64=seed_u64, nonce=nonce)
    # Use 53 high bits for IEEE754 mantissa.
    mant = (u >> np.uint64(11)).astype(np.uint64)
    return (mant.astype(np.float64)) * (1.0 / (1 << 53))


def rand_rademacher(shape: Tuple[int, ...], *, seed_u64: int, nonce: int = 0) -> np.ndarray:
    """Deterministic ±1 Rademacher."""
    u = rand_u64(shape, seed_u64=seed_u64, nonce=nonce)
    bit = (u & np.uint64(1)).astype(np.int8)
    return np.where(bit == 0, -1.0, 1.0).astype(np.float32)


def rand_normal(shape: Tuple[int, ...], *, seed_u64: int, nonce: int = 0) -> np.ndarray:
    """Deterministic standard normal N(0,1) using Box-Muller.

    Note: For production GPU kernels, use a counter-based RNG in-kernel.
    This is a CPU reference implementation intended for repeatable tests.
    """
    n = int(np.prod(shape))
    # Box-Muller needs pairs.
    m = (n + 1) // 2
    u1 = rand_uniform01((m,), seed_u64=seed_u64, nonce=nonce)
    u2 = rand_uniform01((m,), seed_u64=seed_u64, nonce=nonce + 10_000_000)
    u1 = np.clip(u1, 1e-12, 1.0)  # avoid log(0)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)
    z = np.stack([z0, z1], axis=1).reshape(-1)[:n].astype(np.float32)
    return z.reshape(shape)


def torch_noise(
    shape: Tuple[int, ...],
    *,
    sigma: int | bytes | str,
    domain: str,
    dist: Literal["normal", "rademacher", "uniform"] = "normal",
    scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic noise as a torch tensor (CPU reference).

    Parameters
    ----------
    shape:
        Output shape.
    sigma:
        Seed.
    domain:
        Domain separation label (e.g. "noiseA").
    dist:
        Distribution.
    scale:
        Multiply noise by this scale (std for normal).
    device:
        Target device. If CUDA is provided, data is generated on CPU then moved.
    dtype:
        Target dtype.

    Returns
    -------
    torch.Tensor
        Noise tensor.
    """
    seed_u64 = derive_seed_u64(sigma, domain=domain)
    if dist == "normal":
        arr = rand_normal(shape, seed_u64=seed_u64)
    elif dist == "rademacher":
        arr = rand_rademacher(shape, seed_u64=seed_u64)
    elif dist == "uniform":
        # Uniform in [-1, 1)
        u = rand_uniform01(shape, seed_u64=seed_u64)
        arr = (2.0 * u - 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown dist={dist}")

    t = torch.from_numpy(arr).to(dtype=torch.float32) * float(scale)
    if device is not None:
        t = t.to(device)
    return t.to(dtype=dtype)


def torch_sign_vector(
    length: int,
    *,
    sigma: int | bytes | str,
    domain: str = "signs",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic ±1 sign vector of size [length]."""
    seed_u64 = derive_seed_u64(sigma, domain=domain)
    s = rand_rademacher((length,), seed_u64=seed_u64).astype(np.float32)
    t = torch.from_numpy(s)
    if device is not None:
        t = t.to(device)
    return t.to(dtype=dtype)



def torch_uniform01(
    shape: Tuple[int, ...],
    *,
    sigma: int | bytes | str,
    domain: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic U[0,1) tensor."""
    seed_u64 = derive_seed_u64(sigma, domain=domain)
    arr = rand_uniform01(shape, seed_u64=seed_u64).astype(np.float32)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t.to(dtype=dtype)
