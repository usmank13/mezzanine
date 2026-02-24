import torch

from pearl_pow_kernels.rot_gemm import RotGemmConfig, rot_gemm
from pearl_pow_kernels.trace import TraceConfig
from pearl_pow_kernels.train_pow import PowLinear, TrainPowConfig, train_pow_gemm


def test_rot_scheme_gradients_match_plain_matmul():
    torch.manual_seed(0)
    M, K, N = 8, 32, 6
    A = torch.randn(M, K, dtype=torch.float32, requires_grad=True)
    B = torch.randn(K, N, dtype=torch.float32, requires_grad=True)

    C_rot, _, _ = rot_gemm(A, B, sigma=123, config=RotGemmConfig(stages=2, trace=TraceConfig(num_samples=4, tile_k=8)))
    loss_rot = C_rot.sum()
    loss_rot.backward()
    gA_rot = A.grad.detach().clone()
    gB_rot = B.grad.detach().clone()

    A2 = A.detach().clone().requires_grad_(True)
    B2 = B.detach().clone().requires_grad_(True)
    C_ref = A2 @ B2
    loss_ref = C_ref.sum()
    loss_ref.backward()
    gA_ref = A2.grad.detach().clone()
    gB_ref = B2.grad.detach().clone()

    torch.testing.assert_close(gA_rot, gA_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gB_rot, gB_ref, rtol=1e-4, atol=1e-4)


def test_qnoise_int8_unbiased_gradient_matches_in_mean():
    torch.manual_seed(0)
    M, K, N = 8, 16, 7
    A_base = torch.randn(M, K, dtype=torch.float32)
    B_base = torch.randn(K, N, dtype=torch.float32) * 0.2  # keep in range
    G = torch.randn(M, N, dtype=torch.float32)

    # Baseline grad wrt A for loss = sum(C * G)
    A_ref = A_base.clone().requires_grad_(True)
    B_ref = B_base.clone().requires_grad_(False)
    loss_ref = ((A_ref @ B_ref) * G).sum()
    loss_ref.backward()
    gA_ref = A_ref.grad.detach()

    cfg = TrainPowConfig(scheme="qnoise", qnoise_quant="int8", int8_unbiased=True, noise_scale=0.0)

    grads = []
    for s in range(200):
        A = A_base.clone().requires_grad_(True)
        B = B_base.clone().requires_grad_(False)
        C, _ = train_pow_gemm(A, B, sigma=s, cfg=cfg, return_transcript=False)
        loss = (C * G).sum()
        loss.backward()
        grads.append(A.grad.detach())
    gA_mean = torch.stack(grads, dim=0).mean(dim=0)

    # Mean should be close to baseline (unbiasedness proxy).
    torch.testing.assert_close(gA_mean, gA_ref, rtol=0.15, atol=0.02)


def test_powlinear_can_fit_tiny_linear_regression():
    torch.manual_seed(0)
    B = 256
    in_f, out_f = 16, 4
    X = torch.randn(B, in_f)
    W_true = torch.randn(in_f, out_f) * 0.5
    y = X @ W_true + 0.01 * torch.randn(B, out_f)

    cfg = TrainPowConfig(scheme="qnoise", qnoise_quant="int8", int8_unbiased=True, noise_scale=0.0)
    layer = PowLinear(in_f, out_f, bias=False, cfg=cfg)

    opt = torch.optim.SGD(layer.parameters(), lr=0.8)

    def mse(pred, target):
        return ((pred - target) ** 2).mean()

    with torch.no_grad():
        loss0 = mse(layer(X, sigma=0), y).item()

    for step in range(80):
        opt.zero_grad(set_to_none=True)
        pred = layer(X, sigma=step)  # vary sigma like a block-seed schedule
        loss = mse(pred, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        loss1 = mse(layer(X, sigma=999), y).item()

    assert loss1 < loss0 * 0.6


def test_train_pow_qnoise_int8_per_axis_scale_backward_runs():
    torch.manual_seed(0)
    M, K, N = 8, 16, 7
    A = torch.randn(M, K, dtype=torch.float32, requires_grad=True)
    B = torch.randn(K, N, dtype=torch.float32, requires_grad=True)
    cfg = TrainPowConfig(scheme="qnoise", qnoise_quant="int8", int8_axis=0, int8_unbiased=False, noise_scale=0.0)

    C, _ = train_pow_gemm(A, B, sigma=123, cfg=cfg, return_transcript=False)
    C.sum().backward()

    assert A.grad is not None and A.grad.shape == A.shape
    assert B.grad is not None and B.grad.shape == B.shape
