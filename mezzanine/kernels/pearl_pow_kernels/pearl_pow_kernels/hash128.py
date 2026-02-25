from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import torch

from .rng import derive_seed_u64, rand_u64

_MASK64 = (1 << 64) - 1


def _mix64(z: int) -> int:
    """SplitMix64 mixing function (scalar)."""
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return z


def _int8_matrix(L: int, out: int, *, seed_u64: int) -> np.ndarray:
    """Deterministically generate an int8 mixing matrix of shape [L, out]."""
    u = rand_u64((L, out), seed_u64=seed_u64)
    # Map to int8 in [-128, 127]
    w = ((u & np.uint64(0xFF)).astype(np.int16) - 128).astype(np.int8)
    return w


def _to_int8_bytes(data: bytes, *, block_size: int) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    b = (b - 128).astype(np.int8)  # center to [-128,127]
    if b.size < block_size:
        pad = np.zeros((block_size - b.size,), dtype=np.int8)
        b = np.concatenate([b, pad], axis=0)
    else:
        b = b[:block_size]
    return b


def tensor_to_bytes(t: torch.Tensor, *, quant: str = "fp16") -> bytes:
    """Canonical serialization for hashing.

    quant:
      - "fp16": cast to float16 then serialize
      - "fp32": cast to float32 then serialize
      - "int8": cast to int8 then serialize
    """
    if quant == "fp16":
        x = t.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy()
        return x.tobytes(order="C")
    if quant == "fp32":
        x = t.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
        return x.tobytes(order="C")
    if quant == "int8":
        x = t.detach().to(dtype=torch.int8, device="cpu").contiguous().numpy()
        return x.tobytes(order="C")
    raise ValueError(f"Unknown quant={quant}")


@dataclass
class TCHash128:
    """Incremental 128-bit hash with GEMM-friendly mixing.

    This is a **practical engineering primitive**: it is designed so the dominant
    mixing step can be implemented as small INT8 GEMMs (TensorCores on GPU),
    followed by cheap integer mixing.

    It is *not* a proven cryptographic construction. Unit tests check diffusion
    (avalanche) and collision sanity over small samples.
    """

    seed: int | bytes | str = 0
    block_size: int = 256
    out_words: int = 32

    def __post_init__(self) -> None:
        seed_u64 = derive_seed_u64(self.seed, domain="TCHash128/W")
        self._W = _int8_matrix(self.block_size, self.out_words, seed_u64=seed_u64).astype(np.int32)
        # Two-word state (128-bit)
        s0 = derive_seed_u64(self.seed, domain="TCHash128/s0")
        s1 = derive_seed_u64(self.seed, domain="TCHash128/s1")
        self._s0 = int(s0) & _MASK64
        self._s1 = int(s1) & _MASK64
        self._total_len = 0

    def update_bytes(self, data: bytes) -> "TCHash128":
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("update_bytes expects bytes-like")
        self._total_len += len(data)

        # Process in blocks.
        bs = self.block_size
        for off in range(0, len(data), bs):
            chunk = data[off : off + bs]
            v = _to_int8_bytes(chunk, block_size=bs).astype(np.int32)  # [bs]
            # GEMM-friendly: 1xbs @ bsxout -> 1xout
            y = v @ self._W  # [out_words] int32
            # Nonlinear mixing into 128-bit state.
            for i, yi in enumerate(y.tolist()):
                # Map int32 -> uint64 with sign extension handled.
                u = (int(yi) & 0xFFFFFFFF)  # keep low 32 bits
                self._s0 = _mix64(self._s0 ^ (u * 0x9E3779B97F4A7C15 & _MASK64))
                self._s1 = _mix64((self._s1 + u * 0xBF58476D1CE4E5B9) & _MASK64)

        # Length finalize-like.
        self._s0 = _mix64(self._s0 ^ (self._total_len & _MASK64))
        self._s1 = _mix64(self._s1 + (self._total_len & _MASK64))
        return self

    def update_tensor(self, t: torch.Tensor, *, quant: str = "fp16") -> "TCHash128":
        return self.update_bytes(tensor_to_bytes(t, quant=quant))

    def digest(self) -> bytes:
        # Finalize with another mixing round
        s0 = _mix64(self._s0 ^ 0xA5A5A5A5A5A5A5A5)
        s1 = _mix64(self._s1 ^ 0x5A5A5A5A5A5A5A5A)
        return struct.pack("<QQ", s0 & _MASK64, s1 & _MASK64)

    def hexdigest(self) -> str:
        return self.digest().hex()


def tc_hash128(data: bytes, *, seed: int | bytes | str = 0) -> bytes:
    """Convenience one-shot 128-bit hash."""
    return TCHash128(seed=seed).update_bytes(data).digest()
