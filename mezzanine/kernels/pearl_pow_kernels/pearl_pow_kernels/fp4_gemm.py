
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .fp4 import fp4_quantize_groupwise
from .trace import TraceConfig, sampled_gemm_trace

# Optional Triton fused GEMM+hash
try:  # pragma: no cover
    from .gemm_triton_fused import gemm_with_fused_trace, triton_available
except Exception:  # pragma: no cover
    gemm_with_fused_trace = None
    def triton_available() -> bool:
        return False


@dataclass(frozen=True)
class FP4ScaleHashGemmConfig:
    group_size: int = 32
    # Relative scale jitter amplitude (0.0 disables).
    scale_jitter: float = 0.01
    out_dtype: torch.dtype = torch.float32
    trace: TraceConfig = TraceConfig(quant="fp16")


def fp4_scale_hash_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    config: FP4ScaleHashGemmConfig = FP4ScaleHashGemmConfig(),
) -> Tuple[torch.Tensor, bytes, dict]:
    """FP4 group-quant GEMM with seeded scale jitter + trace hash.

    CUDA fast path:
      - still does FP4 quant/dequant in PyTorch (portable)
      - but uses Triton TensorCore GEMM with fused epilogue hash for A_q @ B_q

    CPU path:
      - torch matmul + CPU trace reference.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D tensors")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A is {A.shape}, B is {B.shape}")

    Qa = fp4_quantize_groupwise(
        A,
        group_size=config.group_size,
        axis=-1,
        sigma=sigma,
        domain="fp4/A",
        scale_jitter=config.scale_jitter,
    )
    Qb = fp4_quantize_groupwise(
        B,
        group_size=config.group_size,
        axis=0,
        sigma=sigma,
        domain="fp4/B",
        scale_jitter=config.scale_jitter,
    )

    A_q = Qa.dequant  # float16
    B_q = Qb.dequant  # float16

    use_fast = (
        triton_available()
        and A_q.device.type == "cuda"
        and B_q.device.type == "cuda"
        and (not A_q.requires_grad)
        and (not B_q.requires_grad)
        and gemm_with_fused_trace is not None
    )

    if use_fast:
        C, digest, coords, meta2 = gemm_with_fused_trace(
            A_q, B_q,
            sigma=sigma,
            trace=config.trace,
            out_dtype=config.out_dtype,
            permute_ktiles=False,
            noise_scale=0.0,
        )
        meta = {
            "scheme": "fp4_scale_hash_gemm",
            "group_size": int(config.group_size),
            "scale_jitter": float(config.scale_jitter),
            "trace_coords": coords,
            **meta2,
        }
        return C, digest, meta

    # Reference
    C = (A_q @ B_q).to(config.out_dtype)
    transcript, coords = sampled_gemm_trace(A_q, B_q, sigma=sigma, config=config.trace)
    meta = {
        "scheme": "fp4_scale_hash_gemm",
        "group_size": int(config.group_size),
        "scale_jitter": float(config.scale_jitter),
        "trace_coords": coords,
        "backend": "torch+cpu_trace",
    }
    return C, transcript, meta
