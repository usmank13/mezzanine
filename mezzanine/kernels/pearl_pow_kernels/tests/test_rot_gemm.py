import torch

from pearl_pow_kernels.rot_gemm import RotGemmConfig, rot_encode, rot_gemm
from pearl_pow_kernels.trace import TraceConfig


def test_rot_gemm_matches_plain_matmul():
    torch.manual_seed(0)
    M, K, N = 32, 64, 24  # K must be power-of-two
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    C_ref = A @ B
    C, digest, meta = rot_gemm(A, B, sigma=123, config=RotGemmConfig(stages=2, trace=TraceConfig(num_samples=8)))

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    assert isinstance(digest, (bytes, bytearray)) and len(digest) == 16
    assert meta["scheme"] == "rot_gemm"


def test_rot_transcript_deterministic_and_sensitive():
    torch.manual_seed(0)
    M, K, N = 16, 32, 16
    A = torch.randn(M, K)
    B = torch.randn(K, N)

    cfg = RotGemmConfig(stages=2, trace=TraceConfig(num_samples=8, tile_k=8))
    _, d1, _ = rot_gemm(A, B, sigma=777, config=cfg)
    _, d2, _ = rot_gemm(A, B, sigma=777, config=cfg)
    assert d1 == d2

    # Different sigma should change the transcript
    _, d3, _ = rot_gemm(A, B, sigma=778, config=cfg)
    assert d1 != d3

    # Slight input change should change the transcript (make/break)
    A2 = A.clone()
    A2[0, 0] += 1e-3
    _, d4, _ = rot_gemm(A2, B, sigma=777, config=cfg)
    assert d1 != d4


def test_rot_encode_scrambles_all_ones_after_two_stages():
    # Make/break test: with >=2 stages, all-ones input should not stay sparse after encode.
    M, K, N = 4, 64, 4
    A = torch.ones(M, K)
    B = torch.ones(K, N)

    A_rot, B_rot = rot_encode(A, B, sigma=42, stages=2)

    # The encoded A should have many distinct values / signs (not just 1 nonzero column).
    # We check that at least 25% of entries are non-zero and variance is non-trivial.
    nz = (A_rot.abs() > 1e-6).float().mean().item()
    assert nz > 0.25
    assert A_rot.var().item() > 1e-4
