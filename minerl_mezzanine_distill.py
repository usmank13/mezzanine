#!/usr/bin/env python3
"""minerl_mezzanine_distill.py

VPT teacher + Mezzanine-style nuisance-marginalized distillation for MineRL.

What this script gives you:
  1) A frozen VPT encoder (teacher) producing hidden-state latents.
  2) A feed-forward student that predicts an H-step future latent trajectory in one pass:
        (current_latent, action_seq[H]) -> future_latents[H]
  3) Nuisance marginalization while distilling:
        - VPT dropout variation (teacher in train() mode at extraction time)
        - Observation perturbations (crop/flip/brightness/noise)
        - Action jitter (swap to a neighboring camera bin in VPT's ac minerl_mezzanine_distill.pytion space)
  4) A MAKE/BREAK report that checks:
        - fidelity to teacher mean future
        - stability across nuisances
        - speedup vs teacher recurrent rollout

CLI:
  setup-vpt   Download VPT .model + .weights
  distill     Extract VPT latents, distill student
  eval        MAKE/BREAK report
  plan        Online planning (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# -----------------------------
# MineRL dataset compatibility
# -----------------------------

def _load_minerl_data() -> Any:
    try:
        # Prefer user's compat loader (Zenodo-format MineRL recordings).
        import minerl_data_compat as minerl_data  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import minerl_data_compat. Put minerl_data_compat.py next to this script."
        ) from e
    return minerl_data


# -----------------------------
# Optional: Mezzanine helpers
# -----------------------------


def _maybe_add_mezzanine_to_path() -> None:
    """Best-effort: if a 'mezzanine' package isn't importable, look for a sibling repo."""
    try:
        import mezzanine  # noqa: F401
        return
    except Exception:
        pass

    # Common layouts: repo root contains mezzanine/ package.
    here = Path(__file__).resolve().parent
    candidates = [here, here / "mezzanine_repo", here / "mezzanine_merged_repo", here / ".." / "mezzanine"]
    for c in candidates:
        if (c / "mezzanine").exists():
            sys.path.insert(0, str(c))
            return


_maybe_add_mezzanine_to_path()

try:
    from mezzanine.pipelines.latent_dynamics import (  # type: ignore
        LatentDynamicsTrainConfig,
        eval_latent_dynamics,
        train_latent_dynamics,
    )
except Exception:
    LatentDynamicsTrainConfig = None  # type: ignore
    train_latent_dynamics = None  # type: ignore
    eval_latent_dynamics = None  # type: ignore


# -----------------------------
# VPT import helpers
# -----------------------------


def _maybe_add_vpt_to_path() -> Optional[Path]:
    env = os.environ.get("VPT_REPO")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent
    candidates += [
        here / "Video-Pre-Training",
        here / "vpt",
        here.parent / "Video-Pre-Training",
        Path("/home/azureuser/Video-Pre-Training"),
    ]
    for c in candidates:
        if c.exists() and (c / "agent.py").exists():
            sys.path.insert(0, str(c))
            return c
    return None


VPT_REPO = _maybe_add_vpt_to_path()

try:
    from agent import ACTION_TRANSFORMER_KWARGS as VPT_ACTION_TRANSFORMER_KWARGS  # type: ignore
    from lib.action_mapping import CameraHierarchicalMapping  # type: ignore
    from lib.actions import ActionTransformer, Buttons  # type: ignore
    from lib.policy import MinecraftAgentPolicy  # type: ignore
    from lib.tree_util import tree_map  # type: ignore
except Exception:
    VPT_ACTION_TRANSFORMER_KWARGS = None  # type: ignore
    CameraHierarchicalMapping = None  # type: ignore
    ActionTransformer = None  # type: ignore
    MinecraftAgentPolicy = None  # type: ignore
    Buttons = None  # type: ignore
    tree_map = None  # type: ignore


# -----------------------------
# Repro / utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_torch_uint8(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.uint8))


def obs_to_float(obs_uint8: torch.Tensor) -> torch.Tensor:
    # (B,T,H,W,C)
    return obs_uint8.float() / 255.0


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


def chunked(it: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(it), n):
        yield it[i : i + n]


def _repeat_tree(tree: Any, repeats: int) -> Any:
    if tree is None:
        return None
    if isinstance(tree, (list, tuple)):
        return type(tree)(_repeat_tree(t, repeats) for t in tree)
    if isinstance(tree, dict):
        return {k: _repeat_tree(v, repeats) for k, v in tree.items()}
    if torch.is_tensor(tree):
        return tree.repeat_interleave(repeats, dim=0)
    return tree


# -----------------------------
# VPT download helpers
# -----------------------------


VPT_MODEL_URLS = {
    "1x": "https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model",
    "2x": "https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model",
    "3x": "https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.model",
}

VPT_MODEL_FILES = {
    "1x": "foundation-model-1x.model",
    "2x": "2x.model",
    "3x": "foundation-model-3x.model",
}

VPT_WEIGHT_URLS = {
    "1x": "https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights",
    "2x": "https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-2x.weights",
    "2x-diamond": "https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-foundation-2x.weights",
    "3x": "https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.weights",
}

VPT_WEIGHT_FILES = {
    "1x": "foundation-model-1x.weights",
    "2x": "foundation-model-2x.weights",
    "2x-diamond": "rl-from-foundation-2x.weights",
    "3x": "foundation-model-3x.weights",
}


def _download_file(url: str, out_path: Path, *, overwrite: bool = False) -> None:
    if out_path.exists() and not overwrite:
        print(f"[setup-vpt] exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup-vpt] downloading: {url} -> {out_path}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", "0"))
        with open(out_path, "wb") as f, tqdm(
            total=total if total > 0 else None, unit="B", unit_scale=True, desc=out_path.name
        ) as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                if total > 0:
                    pbar.update(len(chunk))


def setup_vpt_assets(out_dir: str, variant: str, model_url: Optional[str], weights_url: Optional[str], overwrite: bool) -> Tuple[Path, Path]:
    if variant not in VPT_MODEL_URLS:
        raise ValueError(f"Unknown variant: {variant}")
    out_dir_p = Path(out_dir)
    model_name = VPT_MODEL_FILES[variant]
    weights_name = VPT_WEIGHT_FILES[variant]
    model_url = model_url or VPT_MODEL_URLS[variant]
    weights_url = weights_url or VPT_WEIGHT_URLS[variant]

    model_path = out_dir_p / model_name
    weights_path = out_dir_p / weights_name
    _download_file(model_url, model_path, overwrite=overwrite)
    _download_file(weights_url, weights_path, overwrite=overwrite)
    return model_path, weights_path


def resolve_vpt_paths(vpt_dir: str, variant: str, vpt_model: Optional[str], vpt_weights: Optional[str]) -> Tuple[str, str]:
    if vpt_model and vpt_weights:
        return vpt_model, vpt_weights
    if variant not in VPT_MODEL_FILES:
        raise ValueError(f"Unknown VPT variant: {variant}")
    model_path = Path(vpt_dir) / VPT_MODEL_FILES[variant]
    weights_path = Path(vpt_dir) / VPT_WEIGHT_FILES[variant]
    if not model_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            "VPT assets not found. Run 'setup-vpt' or pass --vpt-model/--vpt-weights."
        )
    return str(model_path), str(weights_path)


# -----------------------------
# VPT teacher
# -----------------------------


@dataclass
class VPTState:
    latent: torch.Tensor
    state: Any
    last_obs: torch.Tensor

    def feat(self) -> torch.Tensor:
        return self.latent


class RewardHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 3:
            feats = feats.reshape(-1, feats.shape[-1])
        return self.net(feats).squeeze(-1)


class VPTTeacher(nn.Module):
    def __init__(self, model_path: str, weights_path: str, device: str):
        super().__init__()
        if MinecraftAgentPolicy is None or ActionTransformer is None or CameraHierarchicalMapping is None:
            raise RuntimeError("VPT repo not available. Set VPT_REPO or clone Video-Pre-Training.")

        self.device = torch.device(device)

        with open(model_path, "rb") as f:
            agent_parameters = pickle.load(f)
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        from gym3.types import DictType  # imported lazily to avoid gym3 at module import

        action_space = DictType(**action_space)

        self.policy = MinecraftAgentPolicy(action_space=action_space, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs).to(self.device)
        self.policy.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        self.latent_dim = self.policy.net.output_latent_size()

        transformer_kwargs = VPT_ACTION_TRANSFORMER_KWARGS or {
            "camera_binsize": 2,
            "camera_maxval": 10,
            "camera_mu": 10,
            "camera_quantization_scheme": "mu_law",
        }
        self.action_transformer = ActionTransformer(**transformer_kwargs)
        self.n_camera_bins = int((transformer_kwargs["camera_maxval"] / transformer_kwargs["camera_binsize"]) * 2 + 1)
        self.num_actions = self.n_camera_bins * self.n_camera_bins

        self.action_emb = nn.Embedding(self.num_actions, self.latent_dim).to(self.device)
        self.reward_head = RewardHead(self.latent_dim).to(self.device)

    def _set_dropout(self, enable: bool) -> None:
        if enable:
            self.policy.train()
        else:
            self.policy.eval()

    def _resize_obs(self, obs_uint8: torch.Tensor, target_hw: int = 128) -> torch.Tensor:
        # obs_uint8: (B,T,H,W,C)
        b, t, h, w, c = obs_uint8.shape
        obs_f = obs_uint8.float().permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        obs_r = F.interpolate(obs_f, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
        obs_r = obs_r.reshape(b, t, c, target_hw, target_hw).permute(0, 1, 3, 4, 2)
        return obs_r

    def _forward_seq(self, obs_seq: torch.Tensor, state_in: Any, first: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        obs = {"img": obs_seq}
        (pi_latent, _), state_out = self.policy.net(obs, state_in, context={"first": first})
        return pi_latent, state_out

    @torch.no_grad()
    def infer_posterior(self, obs_uint8: torch.Tensor, act_idx: torch.Tensor, *, sample: bool = True) -> VPTState:
        # act_idx unused; kept for interface compatibility
        self._set_dropout(sample)
        obs = self._resize_obs(obs_uint8.to(self.device))
        b, t = obs.shape[:2]
        first = torch.zeros((b, t), device=self.device, dtype=torch.bool)
        first[:, 0] = True
        state_in = self.policy.initial_state(b)
        pi_latent, state_out = self._forward_seq(obs, state_in, first)
        latent = pi_latent[:, -1]
        last_obs = obs[:, -1:, ...]
        return VPTState(latent=latent, state=state_out, last_obs=last_obs)

    @torch.no_grad()
    def imagine_rollout(self, start: VPTState, future_act: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        # VPT is observation-conditioned; we roll forward using the last observation.
        self._set_dropout(sample)
        b, h = future_act.shape
        obs_rep = start.last_obs.repeat(1, h, 1, 1, 1)
        first = torch.zeros((b, h), device=self.device, dtype=torch.bool)
        pi_latent, _state_out = self._forward_seq(obs_rep, start.state, first)
        return pi_latent

    def repeat_state(self, start: VPTState, repeats: int) -> VPTState:
        if tree_map is not None:
            state_rep = tree_map(lambda x: x.repeat_interleave(repeats, dim=0) if torch.is_tensor(x) else x, start.state)
        else:
            state_rep = _repeat_tree(start.state, repeats)
        return VPTState(
            latent=start.latent.repeat_interleave(repeats, dim=0),
            state=state_rep,
            last_obs=start.last_obs.repeat_interleave(repeats, dim=0),
        )


# -----------------------------
# Nuisances
# -----------------------------


class VPTActionJitter:
    def __init__(self, n_bins: int, prob: float = 0.25):
        self.n_bins = n_bins
        self.prob = prob

    def jitter(self, act_seq: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = act_seq.copy()
        for i in range(len(out)):
            if rng.random() < self.prob:
                x, y = divmod(int(out[i]), self.n_bins)
                neighbors = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.n_bins and 0 <= ny < self.n_bins:
                            neighbors.append(nx * self.n_bins + ny)
                if neighbors:
                    out[i] = int(rng.choice(neighbors))
        return out


def perturb_obs(
    obs_uint8: np.ndarray,
    *,
    seed: int,
    crop_frac: float = 0.90,
    hflip: bool = True,
    brightness: float = 0.15,
    gaussian_noise: float = 0.02,
) -> np.ndarray:
    """Perturb a single observation (uint8 HWC)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    img = obs_uint8
    x = Image.fromarray(img)
    w, h = x.size
    frac = float(crop_frac)
    cw, ch = int(w * frac), int(h * frac)
    if cw > 0 and ch > 0 and (cw != w or ch != h):
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        x = x.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h))
    if hflip and bool(rng.integers(0, 2)):
        x = x.transpose(Image.FLIP_LEFT_RIGHT)

    arr = np.asarray(x).astype(np.float32)
    # Brightness jitter (multiplicative)
    if brightness > 0:
        scale = float(rng.uniform(1.0 - brightness, 1.0 + brightness))
        arr *= scale
    # Additive gaussian noise
    if gaussian_noise > 0:
        arr += rng.normal(0.0, gaussian_noise * 255.0, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


# -----------------------------
# Action mapping helpers
# -----------------------------


def action_to_vpt_index(action: Dict[str, Any], action_transformer: ActionTransformer, n_bins: int) -> int:
    cam = action.get("camera", np.array([0.0, 0.0], dtype=np.float32))
    cam = np.asarray(cam, dtype=np.float32)
    cam_bin = action_transformer.discretize_camera(cam)
    cam_bin = np.clip(cam_bin, 0, n_bins - 1)
    return int(cam_bin[0]) * n_bins + int(cam_bin[1])


def vpt_index_to_env_action(idx: int, action_transformer: ActionTransformer, n_bins: int) -> Dict[str, Any]:
    if Buttons is None:
        raise RuntimeError("VPT Buttons enum not available; check VPT repo.")
    x = int(idx) // n_bins
    y = int(idx) % n_bins
    buttons = np.zeros((len(Buttons.ALL),), dtype=np.int64)
    camera = np.array([x, y], dtype=np.int64)
    return action_transformer.numpy_to_dict({"buttons": buttons, "camera": camera})


# -----------------------------
# Student: feed-forward trajectory predictor
# -----------------------------


@dataclass
class StudentConfig:
    horizon: int = 8
    action_emb_dim: int = 64
    hidden: int = 2048
    depth: int = 4
    lr: float = 2e-4
    wd: float = 1e-6
    steps: int = 40_000
    batch_size: int = 64
    variants: int = 8
    beta_var: float = 0.25
    reward_weight: float = 0.1
    action_jitter_prob: float = 0.25
    seed: int = 0
    log_every: int = 100
    save_every: int = 2000


class StudentFFTrajectory(nn.Module):
    """(latent0, action_seq[H]) -> latent_seq[H] in one forward pass (no recurrent unroll)."""

    def __init__(self, latent_dim: int, num_actions: int, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_emb = nn.Embedding(num_actions, cfg.action_emb_dim)
        in_dim = latent_dim + cfg.horizon * cfg.action_emb_dim

        layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(cfg.depth):
            layers += [nn.Linear(cur, cfg.hidden), nn.GELU()]
            cur = cfg.hidden
        layers.append(nn.Linear(cur, cfg.horizon * latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, latent0: torch.Tensor, future_act: torch.Tensor) -> torch.Tensor:
        # latent0: (B, D), future_act: (B, H)
        a = self.action_emb(future_act)  # (B,H,A)
        x = torch.cat([latent0, a.reshape(a.shape[0], -1)], dim=-1)
        y = self.net(x)
        return y.view(-1, self.cfg.horizon, self.latent_dim)


# -----------------------------
# Streaming window iterator over MineRL
# -----------------------------


def iterate_windows(
    data: Any,
    traj_names: List[str],
    action_transformer: ActionTransformer,
    n_bins: int,
    *,
    context: int,
    horizon: int,
    max_windows: int,
    seed: int,
    shuffle_traj: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (obs_seq, act_seq, rew_seq) windows.

    obs_seq: (T+1,64,64,3) where T=context+horizon
    act_seq: (T,) discrete indices (camera bins flattened)
    rew_seq: (T,) rewards aligned with act/transition obs[t]->obs[t+1]
    """
    rng = np.random.default_rng(seed)
    names = list(traj_names)
    if shuffle_traj:
        rng.shuffle(names)
    produced = 0
    for tn in names:
        try:
            frames: List[np.ndarray] = []
            actions: List[int] = []
            rewards: List[float] = []
            last_next: Optional[np.ndarray] = None
            for obs, act, rew, next_obs, done in data.load_data(tn):
                idx = action_to_vpt_index(act, action_transformer, n_bins)
                frames.append(obs["pov"].astype(np.uint8))
                actions.append(int(idx))
                rewards.append(float(rew))
                last_next = next_obs.get("pov", None)
            # Add final next frame if possible (fallback to last frame).
            if len(frames) > 0:
                if last_next is not None:
                    frames.append(np.asarray(last_next, dtype=np.uint8))
                else:
                    frames.append(frames[-1].copy())
            T = context + horizon
            if len(actions) < T or len(frames) < T + 1:
                continue
            frames_np = np.stack(frames, axis=0)
            actions_np = np.array(actions, dtype=np.int64)
            rewards_np = np.array(rewards, dtype=np.float32)
            # Sample start indices to avoid massive overlap.
            starts = list(range(0, len(actions) - T, max(1, T // 2)))
            rng.shuffle(starts)
            for s in starts:
                obs_seq = frames_np[s : s + T + 1]
                act_seq = actions_np[s : s + T]
                rew_seq = rewards_np[s : s + T]
                yield obs_seq, act_seq, rew_seq
                produced += 1
                if produced >= max_windows:
                    return
        except Exception:
            continue


def batch_windows(
    window_it: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    buf_obs, buf_act, buf_rew = [], [], []
    for obs, act, rew in window_it:
        buf_obs.append(obs)
        buf_act.append(act)
        buf_rew.append(rew)
        if len(buf_obs) >= batch_size:
            yield (
                np.stack(buf_obs, axis=0),
                np.stack(buf_act, axis=0),
                np.stack(buf_rew, axis=0),
            )
            buf_obs, buf_act, buf_rew = [], [], []


# -----------------------------
# Distillation with nuisance marginalization
# -----------------------------


@dataclass
class DistillDataConfig:
    context: int = 8
    horizon: int = 8
    max_windows: int = 120_000
    holdout_windows: int = 5_000
    seed: int = 0


def distill_student(
    *,
    data_dir: str,
    vpt_model: str,
    vpt_weights: str,
    out_path: str,
    device: str,
    student_cfg: StudentConfig,
    data_cfg: DistillDataConfig,
    env_id: Optional[str] = None,
) -> None:
    set_seed(student_cfg.seed)
    minerl_data = _load_minerl_data()
    data = minerl_data.make(env_id or "MineRLObtainDiamond-v0", data_dir=data_dir)
    traj_names = list(data.get_trajectory_names())

    teacher = VPTTeacher(vpt_model, vpt_weights, device=device)
    latent_dim = teacher.latent_dim
    if student_cfg.horizon != data_cfg.horizon:
        raise ValueError("student_cfg.horizon must match data_cfg.horizon")

    student = StudentFFTrajectory(latent_dim=latent_dim, num_actions=teacher.num_actions, cfg=student_cfg).to(device)
    reward_head = teacher.reward_head
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(reward_head.parameters()),
        lr=student_cfg.lr,
        weight_decay=student_cfg.wd,
    )

    jitter = VPTActionJitter(teacher.n_camera_bins, prob=student_cfg.action_jitter_prob)
    device_t = torch.device(device)

    # Streaming windows
    T = data_cfg.context + data_cfg.horizon
    win_it = iterate_windows(
        data,
        traj_names,
        teacher.action_transformer,
        teacher.n_camera_bins,
        context=data_cfg.context,
        horizon=data_cfg.horizon,
        max_windows=data_cfg.max_windows,
        seed=data_cfg.seed,
    )
    batch_it = batch_windows(win_it, student_cfg.batch_size)

    student.train()
    reward_head.train()
    pbar = tqdm(total=student_cfg.steps, desc="distill")
    for step in range(1, student_cfg.steps + 1):
        try:
            obs_seq, act_seq, rew_seq = next(batch_it)
        except StopIteration:
            win_it = iterate_windows(
                data,
                traj_names,
                teacher.action_transformer,
                teacher.n_camera_bins,
                context=data_cfg.context,
                horizon=data_cfg.horizon,
                max_windows=data_cfg.max_windows,
                seed=data_cfg.seed + step,
            )
            batch_it = batch_windows(win_it, student_cfg.batch_size)
            obs_seq, act_seq, rew_seq = next(batch_it)

        # torch tensors
        obs_u = to_torch_uint8(obs_seq).to(device_t)  # (B,T+1,64,64,3)
        act = torch.from_numpy(act_seq.astype(np.int64)).to(device_t)  # (B,T)
        rew = torch.from_numpy(rew_seq.astype(np.float32)).to(device_t)  # (B,T)

        b = obs_u.shape[0]
        v = student_cfg.variants
        h = data_cfg.horizon
        c = data_cfg.context

        # Split into history and future.
        # history: obs[0..C], act[0..C-1]
        # future actions: act[C..C+H-1]
        obs_hist = obs_u[:, : c + 1].contiguous()
        act_hist = act[:, :c].contiguous()
        act_fut = act[:, c : c + h].contiguous()
        rew_fut = rew[:, c : c + h].contiguous()

        # Create V nuisance variants.
        # We'll perturb only the final history observation obs_hist[:,C].
        obs_hist_np = obs_hist.detach().cpu().numpy()  # uint8
        act_fut_np = act_fut.detach().cpu().numpy()

        obs_var_list = []
        act_fut_list = []
        for v_i in range(v):
            # Observation perturbation
            seeds = (step * 1000 + v_i * 17 + np.arange(b) * 131).astype(np.int64)
            obs_pert = obs_hist_np.copy()
            for i in range(b):
                obs_pert[i, c] = perturb_obs(obs_pert[i, c], seed=int(seeds[i]))
            # Action jitter (neighbor swap)
            a_pert = act_fut_np.copy()
            for i in range(b):
                a_pert[i] = jitter.jitter(a_pert[i], seed=int(seeds[i] + 999))
            obs_var_list.append(torch.from_numpy(obs_pert).to(device_t))
            act_fut_list.append(torch.from_numpy(a_pert).to(device_t))

        obs_var = torch.stack(obs_var_list, dim=1)  # (B,V,C+1,64,64,3)
        act_fut_var = torch.stack(act_fut_list, dim=1)  # (B,V,H)

        # Infer posterior state for each variant (includes dropout variation).
        with torch.no_grad():
            cur_states: List[VPTState] = []
            fut_feats: List[torch.Tensor] = []
            for v_i in range(v):
                st = teacher.infer_posterior(obs_var[:, v_i], act_hist, sample=True)
                cur_states.append(st)
                feats = teacher.imagine_rollout(st, act_fut_var[:, v_i], sample=True)  # (B,H,D)
                fut_feats.append(feats)
            cur_feat = torch.stack([s.feat() for s in cur_states], dim=1)  # (B,V,D)
            fut_feats = torch.stack(fut_feats, dim=1)  # (B,V,H,D)
            tgt = fut_feats.mean(dim=1)  # (B,H,D)

        # Student predictions for each variant input.
        pred = []
        for v_i in range(v):
            pred.append(student(cur_feat[:, v_i], act_fut_var[:, v_i]))
        pred = torch.stack(pred, dim=1)  # (B,V,H,D)

        # Loss = fidelity to target mean + invariance (variance across nuisances)
        loss_fit = F.mse_loss(pred, tgt[:, None].expand_as(pred), reduction="mean")
        pred_mean = pred.mean(dim=1, keepdim=True)
        loss_var = (pred - pred_mean).pow(2).mean()

        # Reward head loss (teacher features to observed rewards)
        reward_pred = reward_head(tgt).view(b, h)
        loss_rew = F.mse_loss(reward_pred, rew_fut, reduction="mean")

        loss = loss_fit + student_cfg.beta_var * loss_var + student_cfg.reward_weight * loss_rew

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % student_cfg.log_every == 0:
            with torch.no_grad():
                # quick diagnostics on this batch
                cos = cosine_sim(pred_mean.squeeze(1)[:, -1], tgt[:, -1]).mean()
            pbar.set_postfix(
                {
                    "loss": float(loss.detach().cpu()),
                    "fit": float(loss_fit.detach().cpu()),
                    "var": float(loss_var.detach().cpu()),
                    "rew": float(loss_rew.detach().cpu()),
                    "cos@H": float(cos.detach().cpu()),
                }
            )

        if step % student_cfg.save_every == 0 or step == student_cfg.steps:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            torch.save(
                {
                    "student_cfg": student_cfg.__dict__,
                    "data_cfg": data_cfg.__dict__,
                    "num_actions": teacher.num_actions,
                    "latent_dim": latent_dim,
                    "model": student.state_dict(),
                    "reward_head": reward_head.state_dict(),
                },
                out_path,
            )

        pbar.update(1)
    pbar.close()
    print(f"[distill] wrote: {out_path}")


def load_student(student_path: str, device: str) -> Tuple[StudentFFTrajectory, Optional[RewardHead], StudentConfig]:
    ckpt = torch.load(student_path, map_location=device)
    cfg = StudentConfig(**ckpt["student_cfg"])
    model = StudentFFTrajectory(latent_dim=int(ckpt["latent_dim"]), num_actions=int(ckpt["num_actions"]), cfg=cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    reward_head = None
    if "reward_head" in ckpt:
        reward_head = RewardHead(int(ckpt["latent_dim"])).to(device)
        reward_head.load_state_dict(ckpt["reward_head"], strict=True)
        reward_head.eval()

    return model, reward_head, cfg


# -----------------------------
# Evaluation + MAKE/BREAK
# -----------------------------


@dataclass
class EvalConfig:
    context: int = 8
    horizon: int = 8
    windows: int = 2000
    variants_teacher: int = 16
    variants_student: int = 16
    batch_size: int = 64
    action_jitter_prob: float = 0.25
    seed: int = 0

    # Make/break thresholds
    min_cos: float = 0.90
    min_var_reduction: float = 10.0
    min_speedup: float = 3.0


@torch.no_grad()
def eval_make_break(
    *,
    data_dir: str,
    vpt_model: str,
    vpt_weights: str,
    student_path: str,
    device: str,
    cfg: EvalConfig,
    env_id: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(cfg.seed)
    minerl_data = _load_minerl_data()
    data = minerl_data.make(env_id or "MineRLObtainDiamond-v0", data_dir=data_dir)
    traj_names = list(data.get_trajectory_names())

    teacher = VPTTeacher(vpt_model, vpt_weights, device=device)
    student, reward_head, student_cfg = load_student(student_path, device=device)
    if reward_head is not None:
        teacher.reward_head = reward_head
    if cfg.horizon != student_cfg.horizon:
        raise ValueError("eval horizon must match student horizon")

    jitter = VPTActionJitter(teacher.n_camera_bins, prob=cfg.action_jitter_prob)
    device_t = torch.device(device)

    win_it = iterate_windows(
        data,
        traj_names,
        teacher.action_transformer,
        teacher.n_camera_bins,
        context=cfg.context,
        horizon=cfg.horizon,
        max_windows=cfg.windows,
        seed=cfg.seed,
        shuffle_traj=True,
    )
    batch_it = batch_windows(win_it, cfg.batch_size)

    var_teacher_list = []
    var_student_list = []
    cos_list = []
    t_teacher = 0.0
    t_student = 0.0

    processed = 0
    for obs_seq, act_seq, _rew_seq in batch_it:
        obs_u = to_torch_uint8(obs_seq).to(device_t)  # (B,T+1,64,64,3)
        act = torch.from_numpy(act_seq.astype(np.int64)).to(device_t)  # (B,T)

        b = obs_u.shape[0]
        h = cfg.horizon
        c = cfg.context

        obs_hist = obs_u[:, : c + 1].contiguous()
        act_hist = act[:, :c].contiguous()
        act_fut = act[:, c : c + h].contiguous()

        # Teacher rollouts (dropout variants)
        st = teacher.infer_posterior(obs_hist, act_hist, sample=False)
        feats = []
        for _k in range(cfg.variants_teacher):
            feats.append(teacher.imagine_rollout(st, act_fut, sample=True))
        feats = torch.stack(feats, dim=1)  # (B,K,H,D)
        teacher_mean = feats.mean(dim=1)  # (B,H,D)
        teacher_var = feats.var(dim=1).mean(dim=(1, 2))  # (B,)

        # Benchmark teacher recurrent unroll vs student single MLP pass.
        k = cfg.variants_teacher
        start_rep = teacher.repeat_state(st, k)
        act_rep = act_fut.repeat_interleave(k, dim=0)
        t0 = time.perf_counter()
        _ = teacher.imagine_rollout(start_rep, act_rep, sample=True)
        t_teacher += time.perf_counter() - t0

        # Student predictions with nuisances
        obs_hist_np = obs_hist.detach().cpu().numpy()
        act_fut_np = act_fut.detach().cpu().numpy()

        obs_var_list = []
        act_fut_list = []
        for v_i in range(cfg.variants_student):
            seeds = (processed * 1000 + v_i * 17 + np.arange(b) * 131).astype(np.int64)
            obs_pert = obs_hist_np.copy()
            for i in range(b):
                obs_pert[i, c] = perturb_obs(obs_pert[i, c], seed=int(seeds[i]))
            a_pert = act_fut_np.copy()
            for i in range(b):
                a_pert[i] = jitter.jitter(a_pert[i], seed=int(seeds[i] + 999))
            obs_var_list.append(torch.from_numpy(obs_pert).to(device_t))
            act_fut_list.append(torch.from_numpy(a_pert).to(device_t))

        obs_var = torch.stack(obs_var_list, dim=1)  # (B,V,C+1,64,64,3)
        act_fut_var = torch.stack(act_fut_list, dim=1)  # (B,V,H)

        cur_states = []
        for v_i in range(cfg.variants_student):
            st_v = teacher.infer_posterior(obs_var[:, v_i], act_hist, sample=True)
            cur_states.append(st_v.feat())
        cur_feat = torch.stack(cur_states, dim=1)  # (B,V,D)

        t0 = time.perf_counter()
        pred = []
        for v_i in range(cfg.variants_student):
            pred.append(student(cur_feat[:, v_i], act_fut_var[:, v_i]))
        pred = torch.stack(pred, dim=1)  # (B,V,H,D)
        t_student += time.perf_counter() - t0

        student_mean = pred.mean(dim=1)
        student_var = pred.var(dim=1).mean(dim=(1, 2))

        # Fidelity (cosine) to teacher mean at final step.
        cos = cosine_sim(student_mean[:, -1], teacher_mean[:, -1]).detach().cpu().numpy()

        var_teacher_list.append(teacher_var.detach().cpu().numpy())
        var_student_list.append(student_var.detach().cpu().numpy())
        cos_list.append(cos)

        processed += b
        if processed >= cfg.windows:
            break

    vt_all = np.concatenate(var_teacher_list) if var_teacher_list else np.zeros((0,))
    vs_all = np.concatenate(var_student_list) if var_student_list else np.zeros((0,))
    cos_all = np.concatenate(cos_list) if cos_list else np.zeros((0,))

    var_teacher = float(np.mean(vt_all)) if vt_all.size else 0.0
    var_student = float(np.mean(vs_all)) if vs_all.size else 0.0
    var_reduction = float(var_teacher / max(1e-9, var_student)) if vs_all.size else 0.0
    cos_mean = float(np.mean(cos_all)) if cos_all.size else 0.0
    speedup = float((t_teacher / max(1e-9, t_student))) if t_student > 0 else 0.0

    # Optional: Mezzanine latent dynamics retrieval metrics for extra signal.
    mezz_metrics = None
    if train_latent_dynamics is not None and eval_latent_dynamics is not None:
        zt_arr = []
        ztp_arr = []
        a_arr = []
        sample_count = 0
        for obs_seq, act_seq, _rew_seq in batch_it:
            obs_u = to_torch_uint8(obs_seq).to(device_t)
            act = torch.from_numpy(act_seq.astype(np.int64)).to(device_t)

            obs_hist = obs_u[:, : cfg.context + 1].contiguous()
            act_hist = act[:, : cfg.context].contiguous()
            act_fut = act[:, cfg.context : cfg.context + cfg.horizon].contiguous()

            st = teacher.infer_posterior(obs_hist, act_hist, sample=False)
            ztp = teacher.imagine_rollout(st, act_fut, sample=False).detach().cpu().numpy()[:, -1]
            aemb = teacher.action_emb(act_fut).detach().cpu().numpy().reshape(act_fut.shape[0], -1)

            zt_arr.append(st.feat().detach().cpu().numpy())
            ztp_arr.append(ztp)
            a_arr.append(aemb)

            sample_count += obs_seq.shape[0]
            if sample_count >= min(2000, cfg.windows):
                break

        if zt_arr:
            zt_arr = np.concatenate(zt_arr)
            ztp_arr = np.concatenate(ztp_arr)
            a_arr = np.concatenate(a_arr)

            ld_cfg = LatentDynamicsTrainConfig()  # type: ignore
            models = train_latent_dynamics(zt_arr, ztp_arr, a_arr, ld_cfg, device=str(device_t))
            mezz_metrics = eval_latent_dynamics(models, zt_arr, ztp_arr, a_arr, device=str(device_t), seed=0)

    rep = {
        "cos_mean": cos_mean,
        "teacher_var_mean": var_teacher,
        "student_var_mean": var_student,
        "var_reduction": var_reduction,
        "teacher_seconds": t_teacher,
        "student_seconds": t_student,
        "speedup": speedup,
        "make": bool(cos_mean >= cfg.min_cos and var_reduction >= cfg.min_var_reduction and speedup >= cfg.min_speedup),
        "thresholds": {
            "min_cos": cfg.min_cos,
            "min_var_reduction": cfg.min_var_reduction,
            "min_speedup": cfg.min_speedup,
        },
        "mezzanine_latent_dynamics": mezz_metrics,
    }
    return rep


# -----------------------------
# Planning demo
# -----------------------------


@dataclass
class PlanConfig:
    context: int = 8
    horizon: int = 8
    candidates: int = 1024
    gamma: float = 0.99
    max_env_steps: int = 6000
    mode: str = "student"  # or "teacher" (slow baseline)
    seed: int = 0


def _make_env(env_id: str):
    # MineRL typically uses gym; newer stacks may use gymnasium.
    try:
        import gym  # type: ignore
        try:
            import minerl  # type: ignore  # registers MineRL envs with gym
        except Exception:
            pass
        return gym.make(env_id)
    except Exception:
        import gymnasium as gym  # type: ignore

        return gym.make(env_id)


@torch.no_grad()
def plan_episode(
    *,
    env_id: Optional[str],
    vpt_model: str,
    vpt_weights: str,
    student_path: Optional[str],
    device: str,
    cfg: PlanConfig,
) -> Dict[str, Any]:
    """Random-shooting planner.

    This is an end-to-end demo showing how the distilled student replaces teacher rollouts.
    It is not tuned for SOTA MineRL results out-of-the-box.
    """
    set_seed(cfg.seed)
    teacher = VPTTeacher(vpt_model, vpt_weights, device=device)
    student, reward_head, student_cfg = load_student(student_path, device=device) if student_path else (None, None, None)
    if reward_head is not None:
        teacher.reward_head = reward_head
    if cfg.mode == "student" and student is None:
        raise ValueError("mode=student requires --student")
    if student is not None and cfg.horizon != student_cfg.horizon:
        raise ValueError("planner horizon must match student horizon")

    env = _make_env(env_id or "MineRLObtainDiamond-v0")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Buffers holding the last (context+1) observations and last context actions.
    from collections import deque

    obs_buf: deque = deque(maxlen=cfg.context + 1)
    act_buf: deque = deque(maxlen=cfg.context)
    obs_buf.append(np.asarray(obs["pov"], dtype=np.uint8))

    total_reward = 0.0
    steps = 0
    device_t = torch.device(device)

    for _t in range(cfg.max_env_steps):
        # Pad buffers if needed.
        while len(obs_buf) < cfg.context + 1:
            obs_buf.appendleft(obs_buf[0])
        while len(act_buf) < cfg.context:
            act_buf.appendleft(0)

        obs_hist = np.stack(list(obs_buf), axis=0)[None]  # (1,C+1,64,64,3)
        act_hist = np.array(list(act_buf), dtype=np.int64)[None]  # (1,C)

        obs_t = to_torch_uint8(obs_hist).to(device_t)
        act_t = torch.from_numpy(act_hist).to(device_t)
        st = teacher.infer_posterior(obs_t, act_t, sample=False)

        # Sample candidate action sequences.
        cand = cfg.candidates
        a = teacher.num_actions
        act_cand = torch.randint(0, a, size=(cand, cfg.horizon), device=device_t)

        if cfg.mode == "teacher":
            start_rep = teacher.repeat_state(st, cand)
            feats = teacher.imagine_rollout(start_rep, act_cand, sample=True)  # (cand,H,D)
        else:
            latent_rep = st.feat().repeat(cand, 1)
            assert student is not None
            feats = student(latent_rep, act_cand)  # (cand,H,D)

        # Reward prediction on imagined features.
        r = teacher.reward_head(feats.reshape(-1, feats.shape[-1])).view(cand, cfg.horizon)
        discounts = (cfg.gamma ** torch.arange(cfg.horizon, device=device_t).float())[None, :]
        ret = (r * discounts).sum(dim=1)

        best = int(torch.argmax(ret).item())
        a0 = int(act_cand[best, 0].item())
        action = vpt_index_to_env_action(a0, teacher.action_transformer, teacher.n_camera_bins)

        step_out = env.step(action)
        if len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)

        total_reward += float(reward)
        steps += 1

        # Update buffers for next posterior inference.
        obs_buf.append(np.asarray(obs["pov"], dtype=np.uint8))
        act_buf.append(a0)

        if done:
            break

    env.close()
    return {"return": total_reward, "steps": steps, "mode": cfg.mode}


# -----------------------------
# CLI
# -----------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MineRL VPT -> Mezzanine distillation")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup-vpt", help="Download VPT model + weights")
    p_setup.add_argument("--out-dir", default="vpt")
    p_setup.add_argument("--variant", choices=["1x", "2x", "3x"], default="2x")
    p_setup.add_argument("--model-url", default=None)
    p_setup.add_argument("--weights-url", default=None)
    p_setup.add_argument("--overwrite", action="store_true")

    p_ds = sub.add_parser("distill", help="Distill feed-forward student with nuisance marginalization")
    p_ds.add_argument("--data-dir", required=True)
    p_ds.add_argument("--vpt-dir", default="vpt")
    p_ds.add_argument("--vpt-variant", choices=["1x", "2x", "3x"], default="2x")
    p_ds.add_argument("--vpt-model", default=None)
    p_ds.add_argument("--vpt-weights", default=None)
    p_ds.add_argument("--out", default="student.pt")
    p_ds.add_argument("--env", default=None)
    p_ds.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_ds.add_argument("--context", type=int, default=8)
    p_ds.add_argument("--horizon", type=int, default=8)
    p_ds.add_argument("--steps", type=int, default=40_000)
    p_ds.add_argument("--batch", type=int, default=64)
    p_ds.add_argument("--variants", type=int, default=8)
    p_ds.add_argument("--max-windows", type=int, default=120_000)
    p_ds.add_argument("--seed", type=int, default=0)

    p_ev = sub.add_parser("eval", help="Evaluate student and print MAKE/BREAK")
    p_ev.add_argument("--data-dir", required=True)
    p_ev.add_argument("--vpt-dir", default="vpt")
    p_ev.add_argument("--vpt-variant", choices=["1x", "2x", "3x"], default="2x")
    p_ev.add_argument("--vpt-model", default=None)
    p_ev.add_argument("--vpt-weights", default=None)
    p_ev.add_argument("--student", required=True)
    p_ev.add_argument("--env", default=None)
    p_ev.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_ev.add_argument("--context", type=int, default=8)
    p_ev.add_argument("--horizon", type=int, default=8)
    p_ev.add_argument("--windows", type=int, default=2000)
    p_ev.add_argument("--batch", type=int, default=64)
    p_ev.add_argument("--teacher-samples", type=int, default=16)
    p_ev.add_argument("--student-variants", type=int, default=16)
    p_ev.add_argument("--seed", type=int, default=0)
    p_ev.add_argument("--min-cos", type=float, default=0.90)
    p_ev.add_argument("--min-var-reduction", type=float, default=10.0)
    p_ev.add_argument("--min-speedup", type=float, default=3.0)
    p_ev.add_argument("--json-out", default=None)

    p_pl = sub.add_parser("plan", help="Online planner demo (requires MineRL env installed)")
    p_pl.add_argument("--env", default=None)
    p_pl.add_argument("--vpt-dir", default="vpt")
    p_pl.add_argument("--vpt-variant", choices=["1x", "2x", "3x"], default="2x")
    p_pl.add_argument("--vpt-model", default=None)
    p_pl.add_argument("--vpt-weights", default=None)
    p_pl.add_argument("--student", default=None)
    p_pl.add_argument("--mode", choices=["student", "teacher"], default="student")
    p_pl.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_pl.add_argument("--context", type=int, default=8)
    p_pl.add_argument("--horizon", type=int, default=8)
    p_pl.add_argument("--candidates", type=int, default=1024)
    p_pl.add_argument("--gamma", type=float, default=0.99)
    p_pl.add_argument("--max_env_steps", type=int, default=6000)
    p_pl.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.cmd == "setup-vpt":
        model_path, weights_path = setup_vpt_assets(
            args.out_dir,
            args.variant,
            model_url=args.model_url,
            weights_url=args.weights_url,
            overwrite=args.overwrite,
        )
        print(f"[setup-vpt] model: {model_path}")
        print(f"[setup-vpt] weights: {weights_path}")
        return

    if args.cmd == "distill":
        vpt_model, vpt_weights = resolve_vpt_paths(args.vpt_dir, args.vpt_variant, args.vpt_model, args.vpt_weights)
        student_cfg = StudentConfig(
            horizon=args.horizon,
            steps=args.steps,
            batch_size=args.batch,
            variants=args.variants,
            seed=args.seed,
        )
        data_cfg = DistillDataConfig(
            context=args.context,
            horizon=args.horizon,
            max_windows=args.max_windows,
            seed=args.seed,
        )
        distill_student(
            data_dir=args.data_dir,
            vpt_model=vpt_model,
            vpt_weights=vpt_weights,
            out_path=args.out,
            device=args.device,
            student_cfg=student_cfg,
            data_cfg=data_cfg,
            env_id=args.env,
        )
        return

    if args.cmd == "eval":
        vpt_model, vpt_weights = resolve_vpt_paths(args.vpt_dir, args.vpt_variant, args.vpt_model, args.vpt_weights)
        cfg = EvalConfig(
            context=args.context,
            horizon=args.horizon,
            windows=args.windows,
            variants_teacher=args.teacher_samples,
            variants_student=args.student_variants,
            batch_size=args.batch,
            seed=args.seed,
            min_cos=args.min_cos,
            min_var_reduction=args.min_var_reduction,
            min_speedup=args.min_speedup,
        )
        rep = eval_make_break(
            data_dir=args.data_dir,
            vpt_model=vpt_model,
            vpt_weights=vpt_weights,
            student_path=args.student,
            device=args.device,
            cfg=cfg,
            env_id=args.env,
        )
        print(json.dumps(rep, indent=2))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(rep, f, indent=2)
            print(f"[eval] wrote json report: {args.json_out}")
        return

    if args.cmd == "plan":
        vpt_model, vpt_weights = resolve_vpt_paths(args.vpt_dir, args.vpt_variant, args.vpt_model, args.vpt_weights)
        cfg = PlanConfig(
            context=args.context,
            horizon=args.horizon,
            candidates=args.candidates,
            gamma=args.gamma,
            max_env_steps=args.max_env_steps,
            mode=args.mode,
            seed=args.seed,
        )
        out = plan_episode(
            env_id=args.env,
            vpt_model=vpt_model,
            vpt_weights=vpt_weights,
            student_path=args.student,
            device=args.device,
            cfg=cfg,
        )
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
