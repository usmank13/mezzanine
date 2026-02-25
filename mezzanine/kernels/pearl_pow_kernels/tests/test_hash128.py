import os
import random

import numpy as np
import torch

from pearl_pow_kernels.hash128 import TCHash128, tc_hash128


def _hamming_bytes(a: bytes, b: bytes) -> int:
    assert len(a) == len(b)
    x = int.from_bytes(a, "little") ^ int.from_bytes(b, "little")
    return x.bit_count()


def test_hash_deterministic():
    data = b"hello world" * 10
    h1 = tc_hash128(data, seed=123)
    h2 = tc_hash128(data, seed=123)
    assert h1 == h2
    h3 = tc_hash128(data, seed=124)
    assert h1 != h3


def test_hash_avalanche_single_bit_flip():
    rng = random.Random(0)
    data = bytearray(rng.randbytes(256))
    h1 = tc_hash128(bytes(data), seed=0)
    # Flip one bit
    data[17] ^= 0x01
    h2 = tc_hash128(bytes(data), seed=0)
    d = _hamming_bytes(h1, h2)
    # Expect ~64 bit flips out of 128; allow slack.
    assert d >= 40


def test_hash_no_collisions_small_sample():
    rng = random.Random(1)
    seen = set()
    for i in range(500):
        msg = rng.randbytes(64)
        h = tc_hash128(msg, seed=999).hex()
        assert h not in seen
        seen.add(h)
