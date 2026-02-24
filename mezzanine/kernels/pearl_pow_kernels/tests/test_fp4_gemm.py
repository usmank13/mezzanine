import torch

from pearl_pow_kernels.fp4_gemm import FP4ScaleHashGemmConfig, fp4_scale_hash_gemm


def test_fp4_gemm_reproducible_and_reasonable_error():
    torch.manual_seed(0)
    M, K, N = 16, 64, 16
    # Keep small magnitude so FP4 isn't totally saturated.
    A = torch.randn(M, K) * 0.1
    B = torch.randn(K, N) * 0.1
    C_ref = (A @ B).to(torch.float32)

    cfg = FP4ScaleHashGemmConfig(group_size=32, scale_jitter=0.0)
    C, digest, _ = fp4_scale_hash_gemm(A, B, sigma=1, config=cfg)

    assert isinstance(digest, (bytes, bytearray)) and len(digest) == 16
    # FP4 is coarse; allow a loose tolerance but still bounded.
    rel = (C - C_ref).abs().mean() / (C_ref.abs().mean() + 1e-6)
    assert rel.item() < 0.35


def test_fp4_scale_jitter_changes_transcript_but_not_too_much():
    torch.manual_seed(0)
    M, K, N = 8, 64, 8
    A = torch.randn(M, K) * 0.1
    B = torch.randn(K, N) * 0.1

    cfg0 = FP4ScaleHashGemmConfig(group_size=32, scale_jitter=0.0)
    C0, d0, _ = fp4_scale_hash_gemm(A, B, sigma=7, config=cfg0)

    cfgj = FP4ScaleHashGemmConfig(group_size=32, scale_jitter=0.02)
    C1, d1, _ = fp4_scale_hash_gemm(A, B, sigma=7, config=cfgj)
    C2, d2, _ = fp4_scale_hash_gemm(A, B, sigma=8, config=cfgj)

    # Same sigma => deterministic
    C1b, d1b, _ = fp4_scale_hash_gemm(A, B, sigma=7, config=cfgj)
    torch.testing.assert_close(C1, C1b, rtol=0, atol=0)
    assert d1 == d1b

    # Jitter should affect transcript and output
    assert d0 != d1
    assert d1 != d2

    # But output drift vs no-jitter should be limited (bounded accuracy loss proxy).
    drift = (C1 - C0).abs().mean().item()
    base = C0.abs().mean().item() + 1e-6
    assert drift / base < 0.5
