# pearl-pow-kernels

A **reference / integration-ready** codebase implementing five kernel-level developments inspired by the
**Pearl Polymath Project** open problems:

1. **RotGEMM**: random-rotation (randomized Hadamard) MatMul + trace hashing  
2. **QNoiseGEMM**: add-noise → quantize (FP8/INT8) → GEMM + trace hashing  
3. **FP4ScaleHashGEMM**: FP4 group-quantization with **seeded scale-jitter** + trace hashing  
4. **TrainPOW-GEMM**: training/finetuning wrappers with unbiased quantization options (stochastic rounding)  
5. **TC-Hash Epilogue**: a “TensorCore-friendly” incremental 128-bit hash primitive + a proof-of-inference activation sampler

> Note: This repo is CPU-only in this environment, but the APIs are written so the compute kernels
> can be swapped for Triton/CUDA kernels. Unit tests validate **functional** and **make/break**
> properties (determinism, invariances, unbiasedness, avalanche), not full cryptographic security.

## Quickstart

```bash
pip install -e .
pytest
```

## Design goals

- Deterministic, replayable perturbations from a seed `sigma`
- Clean separation between:
  - (a) **encoding** (noise/rotation/scale-jitter),
  - (b) **compute** (GEMM),
  - (c) **trace hashing** (sampled intermediate tiles)
- “Make/break” tests: tests that fail if the critical protocol property is broken.
