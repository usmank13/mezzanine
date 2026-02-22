from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MLPRegressorConfig:
    in_dim: int
    out_dim: int
    hidden: int = 512
    depth: int = 2
    dropout: float = 0.0


def _build_mlp(cfg: MLPRegressorConfig):
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    d = int(cfg.in_dim)
    for _ in range(int(cfg.depth)):
        layers.append(nn.Linear(d, int(cfg.hidden)))
        layers.append(nn.GELU())
        if float(cfg.dropout) > 0:
            layers.append(nn.Dropout(float(cfg.dropout)))
        d = int(cfg.hidden)
    layers.append(nn.Linear(d, int(cfg.out_dim)))
    return nn.Sequential(*layers)


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: MLPRegressorConfig,
    steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    wd: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an MLP regressor with MSE loss."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    model = _build_mlp(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))

    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)

    loss_fn = torch.nn.MSELoss()
    for _ in range(int(steps)):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_val = model(Xva.to(device)).cpu().numpy()
    metrics = {
        "mse_val": float(np.mean((pred_val - y_val) ** 2)),
    }
    return model, metrics


def train_regressor_distill(
    X_train: np.ndarray,
    y_soft_train: np.ndarray,
    y_hard_train: np.ndarray,
    X_val: np.ndarray,
    y_soft_val: np.ndarray,
    y_hard_val: np.ndarray,
    *,
    cfg: MLPRegressorConfig,
    steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    wd: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    hard_label_weight: float = 0.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an MLP regressor on soft labels with an optional hard-label anchor.

    Loss:
      (1 - w) * MSE(pred, y_soft) + w * MSE(pred, y_hard), with w=hard_label_weight.
    """

    if not (0.0 <= float(hard_label_weight) <= 1.0):
        raise ValueError("hard_label_weight must be in [0, 1]")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ysoft_tr = torch.tensor(y_soft_train, dtype=torch.float32)
    yhard_tr = torch.tensor(y_hard_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    ysoft_va = torch.tensor(y_soft_val, dtype=torch.float32)
    yhard_va = torch.tensor(y_hard_val, dtype=torch.float32)

    model = _build_mlp(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))

    ds = TensorDataset(Xtr, ysoft_tr, yhard_tr)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)

    loss_fn = torch.nn.MSELoss()
    w = float(hard_label_weight)

    for _ in range(int(steps)):
        try:
            xb, ysb, yhb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, ysb, yhb = next(it)
        xb = xb.to(device)
        ysb = ysb.to(device)
        yhb = yhb.to(device)
        pred = model(xb)
        loss_soft = loss_fn(pred, ysb)
        if w > 0.0:
            loss_hard = loss_fn(pred, yhb)
            loss = (1.0 - w) * loss_soft + w * loss_hard
        else:
            loss = loss_soft
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_val = model(Xva.to(device)).cpu().numpy()

    metrics = {
        "mse_val": float(np.mean((pred_val - y_hard_val) ** 2)),
        "mse_val_soft": float(np.mean((pred_val - y_soft_val) ** 2)),
    }
    return model, metrics


def predict(model: Any, X: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    import torch

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(Xt).detach().cpu().numpy()
    return y


def warrant_gap_regression(pred_views: np.ndarray) -> Dict[str, float]:
    """Compute a simple warrant gap proxy for regression.

    Args:
      pred_views: array [K, N, out_dim]

    Returns:
      dict with:
        - gap_mse: E[ ||p - E[p]||^2 ] averaged over views & examples
        - gap_l2:  E[ ||p - E[p]|| ]
    """

    if pred_views.ndim != 3:
        raise ValueError(f"pred_views must be [K,N,D], got {pred_views.shape}")
    mu = pred_views.mean(axis=0, keepdims=True)
    diff = pred_views - mu
    gap_mse = float(np.mean(diff**2))
    gap_l2 = float(np.mean(np.linalg.norm(diff, axis=-1)))
    return {"gap_mse": gap_mse, "gap_l2": gap_l2}


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def warrant_gap_l2_from_views(Y_views: np.ndarray) -> Dict[str, float]:
    """Regression analogue of warrant gap: L2 instability across symmetry views.

    Args:
        Y_views: array shaped [N, K, D] (N examples, K views, D dims)
    Returns:
        dict with:
          - mean_l2_to_mean: mean over (N,K) of ||y_{n,k} - mean_k y_{n,k}||_2
          - mean_pairwise_l2: mean over N of ||y_{n,0} - y_{n,1}||_2 (if K>=2)
    """

    if Y_views.ndim != 3:
        raise ValueError(f"Expected Y_views with shape [N,K,D], got {Y_views.shape}")
    _, K, _ = Y_views.shape

    Y_bar = Y_views.mean(axis=1, keepdims=True)  # [N,1,D]
    diff = Y_views - Y_bar
    l2 = np.sqrt(np.sum(diff * diff, axis=-1))  # [N,K]
    mean_l2_to_mean = float(l2.mean())

    if K >= 2:
        d01 = Y_views[:, 0, :] - Y_views[:, 1, :]
        mean_pairwise = float(np.sqrt(np.sum(d01 * d01, axis=-1)).mean())
    else:
        mean_pairwise = 0.0

    return {"mean_l2_to_mean": mean_l2_to_mean, "mean_pairwise_l2": mean_pairwise}


@dataclass
class MLPRegHeadConfig:
    in_dim: int
    out_dim: int
    hidden: int = 256
    depth: int = 3
    dropout: float = 0.0
    residual: bool = True


def train_soft_regression_head(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    steps: int = 6000,
    batch_size: int = 8192,
    lr: float = 2e-4,
    wd: float = 1e-4,
    grad_clip: float = 1.0,
    eval_every: int = 200,
    seed: int = 0,
    device: str = "cuda",
    cfg: Optional[MLPRegHeadConfig] = None,
) -> Tuple[Any, Dict[str, float], MLPRegHeadConfig]:
    """Train a regression head on frozen features with MSE loss.

    This utility is used by recipes that treat the teacher target as a regression
    vector (e.g. distilling ensemble means).
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)

    in_dim = int(X_train.shape[1])
    out_dim = int(Y_train.shape[1])
    if cfg is None:
        cfg = MLPRegHeadConfig(in_dim=in_dim, out_dim=out_dim)

    class Head(nn.Module):
        def __init__(self, cfg: MLPRegHeadConfig):
            super().__init__()
            self.cfg = cfg
            layers: list[nn.Module] = []
            d = int(cfg.in_dim)
            for _ in range(max(0, int(cfg.depth) - 1)):
                layers.append(nn.Linear(d, int(cfg.hidden)))
                layers.append(nn.GELU())
                if float(cfg.dropout) > 0:
                    layers.append(nn.Dropout(float(cfg.dropout)))
                d = int(cfg.hidden)
            self.body = nn.Sequential(*layers) if layers else None
            self.out = nn.Linear(d if layers else int(cfg.in_dim), int(cfg.out_dim))
            if bool(cfg.residual) and int(cfg.in_dim) == int(cfg.out_dim):
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.body(x) if self.body is not None else x
            y = self.out(h)
            if bool(self.cfg.residual) and int(self.cfg.in_dim) == int(self.cfg.out_dim):
                return x + y
            return y

    model = Head(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    if len(dl) == 0:
        # Avoid an empty loader when the dataset is tiny (e.g. in smoke tests).
        dl = DataLoader(ds, batch_size=max(1, min(int(batch_size), len(ds))), shuffle=True, drop_last=False)

    have_val = X_val is not None and Y_val is not None and len(X_val) > 0
    if have_val:
        Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
        Yv = torch.from_numpy(np.asarray(Y_val, dtype=np.float32)).to(device)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    it = iter(dl)
    for step in range(1, int(steps) + 1):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)

        xb = xb.to(device)
        yb = yb.to(device)

        loss = F.mse_loss(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()

        if have_val and (step % int(eval_every) == 0 or step == int(steps)):
            with torch.no_grad():
                v = float(F.mse_loss(model(Xv), Yv).item())
            if v < best_val:
                best_val = v
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if have_val:
        model.load_state_dict(best_state)

    return model, {"val_mse": float(best_val) if have_val else float("nan")}, cfg


def predict_regression(
    model: Any,
    X: np.ndarray,
    *,
    device: str = "cuda",
    chunk: int = 65536,
) -> np.ndarray:
    import torch

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be [N,D], got {X.shape}")

    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, int(X.shape[0]), int(chunk)):
            xb = torch.from_numpy(X[i : i + int(chunk)]).to(device=device, dtype=torch.float32)
            yb = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
            outs.append(yb)
    if not outs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outs, axis=0)
