import numpy as np
import torch

from pearl_pow_kernels.qnoise_gemm import QNoiseGemmConfig, qnoise_gemm
from pearl_pow_kernels.quant import dequantize_int8, quantize_int8_unbiased
from pearl_pow_kernels.trace import TraceConfig


def test_qnoise_deterministic_output_and_transcript():
    torch.manual_seed(0)
    A = torch.randn(16, 32)
    B = torch.randn(32, 8)

    cfg = QNoiseGemmConfig(noise_dist="normal", noise_scale=0.02, quant="float8")
    C1, d1, _ = qnoise_gemm(A, B, sigma=999, config=cfg)
    C2, d2, _ = qnoise_gemm(A, B, sigma=999, config=cfg)
    torch.testing.assert_close(C1, C2, rtol=0, atol=0)
    assert d1 == d2

    C3, d3, _ = qnoise_gemm(A, B, sigma=1000, config=cfg)
    assert d1 != d3
    # Different sigma should generally change the output due to noise
    assert not torch.allclose(C1, C3)


def test_int8_unbiased_quantization_is_unbiased_in_mean():
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float32) * 0.1  # keep in range
    # Fix scale by calling once (scale is deterministic for x)
    q0 = quantize_int8_unbiased(x, sigma=0, domain="unbiased/test")
    scale = q0.scale

    ys = []
    for s in range(200):
        q = quantize_int8_unbiased(x, sigma=s, domain="unbiased/test", scale=scale)
        y = dequantize_int8(q.q, q.scale)
        ys.append(y)
    mean = torch.stack(ys, dim=0).mean(dim=0)
    # Mean error should be small compared to scale.
    err = (mean - x).abs().mean().item()
    assert err < float(scale.mean()) * 0.2 + 1e-4


def test_qnoise_transcript_changes_with_input_make_break():
    torch.manual_seed(0)
    A = torch.randn(8, 16)
    B = torch.randn(16, 8)
    cfg = QNoiseGemmConfig(noise_scale=0.05, quant="int8", int8_unbiased=True, trace=TraceConfig(tile_m=8, tile_n=8, tile_k=16, num_samples=1))

    _, d1, _ = qnoise_gemm(A, B, sigma=123, config=cfg)
    A2 = A.clone()
    A2[0, 0] += 1.0
    _, d2, _ = qnoise_gemm(A2, B, sigma=123, config=cfg)
    assert d1 != d2


def test_qnoise_int8_per_axis_scale_runs():
    torch.manual_seed(0)
    A = torch.randn(8, 16)
    B = torch.randn(16, 7)
    for axis in (0, 1):
        cfg = QNoiseGemmConfig(
            noise_dist="normal",
            noise_scale=0.0,
            quant="int8",
            int8_axis=axis,
            int8_unbiased=False,
            trace=TraceConfig(num_samples=0),
        )
        C, d, _ = qnoise_gemm(A, B, sigma=123, config=cfg)
        assert C.shape == (8, 7)
        assert isinstance(d, (bytes, bytearray))
