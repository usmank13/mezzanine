from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import torch

from ..core.config import deep_update, load_config
from ..encoders.hf_vision import HFVisionEncoder, HFVisionEncoderConfig
from ..pipelines.latent_dynamics import (
    LatentDynamicsTrainConfig,
    train_latent_dynamics,
    eval_latent_dynamics,
    eval_goal_action_retrieval,
    cem_plan_one_step,
)
from ..utils.lerobot_data import build_pairs, resolve_camera_action_keys
from ..viz.latent_dynamics import save_diagnostics, save_montage
from ..worlds.lerobot import LeRobotAdapter, LeRobotAdapterConfig
from .recipe_base import Recipe


class LeRobotLatentDynamicsRecipe(Recipe):
    NAME = "lerobot_latent_dynamics"
    DESCRIPTION = (
        "LeRobot: distill action-conditioned latent dynamics in frozen vision latents, "
        "with action-shuffle counterfactual + lightweight goal-conditioned action retrieval planning."
    )

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # Dataset
        p.add_argument("--repo_id", type=str, default="lerobot/pusht_image")
        p.add_argument("--train_split", type=str, default="train")
        p.add_argument("--test_split", type=str, default="test")
        p.add_argument("--camera_key", type=str, default="observation.image")
        p.add_argument("--action_key", type=str, default="action")
        p.add_argument("--n_train", type=int, default=4000)
        p.add_argument("--n_test", type=int, default=2000)
        p.add_argument(
            "--delta_steps", type=int, default=1, help="Step offset Δ between (t, t+Δ)."
        )
        p.add_argument(
            "--per_episode_samples",
            type=int,
            default=1,
            help="If the dataset is episode-format, sample this many pairs per episode row.",
        )

        # Encoder
        p.add_argument("--ijepa_model", type=str, default="facebook/ijepa_vith14_1k")
        p.add_argument("--encode_bs", type=int, default=32)
        p.add_argument("--no_fp16", action="store_true")
        p.add_argument(
            "--embed_mode",
            type=str,
            default="mean_std",
            choices=["cls", "mean", "mean_std"],
        )
        p.add_argument("--embed_layer", type=int, default=-4)

        # Training
        p.add_argument("--train_steps", type=int, default=800)
        p.add_argument("--train_bs", type=int, default=256)
        p.add_argument("--train_lr", type=float, default=1e-3)
        p.add_argument("--hidden", type=int, default=1024)
        p.add_argument("--depth", type=int, default=2)

        # Planning (downstream use)
        p.add_argument(
            "--do_planning",
            action="store_true",
            help="Run lightweight goal-conditioned action retrieval evaluation.",
        )
        p.add_argument("--plan_candidates", type=int, default=32)
        p.add_argument("--plan_eval", type=int, default=512)

        # V-JEPA-style 1-step CEM planning (offline proxy)
        p.add_argument(
            "--do_cem",
            action="store_true",
            help="Run V-JEPA-style CEM planning objective for 1-step goals (offline proxy).",
        )
        p.add_argument("--cem_eval", type=int, default=128)
        p.add_argument("--cem_samples", type=int, default=256)
        p.add_argument("--cem_elite", type=int, default=16)
        p.add_argument("--cem_iters", type=int, default=4)
        p.add_argument("--cem_sigma", type=float, default=1.0)
        p.add_argument(
            "--cem_l1_bound",
            type=float,
            default=None,
            help="Optional L1-ball bound for actions. If omitted, uses a quantile of train action L1 norms.",
        )
        p.add_argument("--cem_l1_quantile", type=float, default=0.95)

        # Output
        p.add_argument("--out_dir", type=str, default=str(self.out_dir))

        args = p.parse_args(argv)

        # Apply config file defaults BEFORE seeding/logger/cache creation
        file_cfg = load_config(getattr(args, "config", None))
        merged_cfg = deep_update(file_cfg, self.config)
        self.apply_config_defaults(p, args, merged_cfg)

        ctx = self.build_context(args)
        out_dir = ctx.out_dir
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- World adapter ---
        w_cfg = LeRobotAdapterConfig(
            repo_id=str(args.repo_id),
            train_split=str(args.train_split),
            test_split=str(args.test_split),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
            camera_key=str(args.camera_key),
            action_key=str(args.action_key),
        )
        world = LeRobotAdapter(w_cfg)
        data = world.load()
        world_fp = world.fingerprint()

        train_ds = data["train_ds"]
        test_ds = data["test_ds"]
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]

        # Resolve camera/action keys against the dataset's actual structure.
        # LeRobot datasets may use either nested dicts (observation->images->...) or
        # flattened dotted columns ("observation.images.cam_high").
        ex0 = train_ds[int(train_idx[0])]
        cam_key, act_key, key_meta = resolve_camera_action_keys(
            ex0,
            camera_key=str(args.camera_key),
            action_key=str(args.action_key),
        )

        if cam_key != str(args.camera_key) or act_key != str(args.action_key):
            print(
                f"[lerobot_latent_dynamics] Resolved keys: camera_key={cam_key!r} action_key={act_key!r} "
                f"(requested camera_key={str(args.camera_key)!r} action_key={str(args.action_key)!r})"
            )

        # --- Build (t, t+Δ) pairs ---
        # We create explicit sample dicts to keep the downstream pipeline identical
        # to I-PHYRE latent dynamics (same make/break condition).
        train_pairs = build_pairs(
            train_ds,
            train_idx,
            camera_key=cam_key,
            action_key=act_key,
            delta_steps=int(args.delta_steps),
            per_episode_samples=int(args.per_episode_samples),
            seed=int(args.seed),
            max_pairs=int(args.n_train),
        )
        test_pairs = build_pairs(
            test_ds,
            test_idx,
            camera_key=cam_key,
            action_key=act_key,
            delta_steps=int(args.delta_steps),
            per_episode_samples=int(args.per_episode_samples),
            seed=int(args.seed) + 1,
            max_pairs=int(args.n_test),
        )

        if len(train_pairs) == 0 or len(test_pairs) == 0:
            raise RuntimeError(
                "No usable (t, t+Δ) pairs were built. "
                "Check camera_key/action_key, delta_steps, and dataset structure."
            )

        imgs_train_t = [ex["img_t"] for ex in train_pairs]
        imgs_train_tp = [ex["img_tp"] for ex in train_pairs]
        a_train = np.stack([ex["action_feat"] for ex in train_pairs], axis=0).astype(
            np.float32
        )

        imgs_test_t = [ex["img_t"] for ex in test_pairs]
        imgs_test_tp = [ex["img_tp"] for ex in test_pairs]
        a_test = np.stack([ex["action_feat"] for ex in test_pairs], axis=0).astype(
            np.float32
        )

        # --- Encoder ---
        enc_cfg = HFVisionEncoderConfig(
            model_name=str(args.ijepa_model),
            batch_size=int(args.encode_bs),
            fp16=not bool(args.no_fp16),
            embed_mode=str(args.embed_mode),  # type: ignore[arg-type]
            embed_layer=int(args.embed_layer),
        )
        enc = HFVisionEncoder(enc_cfg, device=device)
        enc_fp = enc.fingerprint()

        # --- Cache-aware encoding ---
        extra_common = {
            "repo_id": str(args.repo_id),
            "camera_key": str(cam_key),
            "action_key": str(act_key),
            "delta_steps": int(args.delta_steps),
            "embed_mode": str(args.embed_mode),
            "embed_layer": int(args.embed_layer),
        }

        if ctx.cache is not None:
            z_train_t = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="train",
                    tag="z_t",
                    extra=extra_common,
                ),
                lambda: enc.encode(imgs_train_t),
                meta={"n": len(imgs_train_t), "world_fp": world_fp, "enc_fp": enc_fp},
            )
            z_train_tp = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="train",
                    tag="z_tp",
                    extra=extra_common,
                ),
                lambda: enc.encode(imgs_train_tp),
                meta={"n": len(imgs_train_tp), "world_fp": world_fp, "enc_fp": enc_fp},
            )
            z_test_t = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="test",
                    tag="z_t",
                    extra=extra_common,
                ),
                lambda: enc.encode(imgs_test_t),
                meta={"n": len(imgs_test_t), "world_fp": world_fp, "enc_fp": enc_fp},
            )
            z_test_tp = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="test",
                    tag="z_tp",
                    extra=extra_common,
                ),
                lambda: enc.encode(imgs_test_tp),
                meta={"n": len(imgs_test_tp), "world_fp": world_fp, "enc_fp": enc_fp},
            )
        else:
            z_train_t = enc.encode(imgs_train_t)
            z_train_tp = enc.encode(imgs_train_tp)
            z_test_t = enc.encode(imgs_test_t)
            z_test_tp = enc.encode(imgs_test_tp)

        # --- Predictors + evaluation ---
        cfg = LatentDynamicsTrainConfig(
            steps=int(args.train_steps),
            batch=int(args.train_bs),
            lr=float(args.train_lr),
            hidden=int(args.hidden),
            depth=int(args.depth),
            seed=int(args.seed),
        )
        train_out = train_latent_dynamics(
            z_train_t, z_train_tp, a_train, cfg, device=device
        )
        ev = eval_latent_dynamics(
            train_out, z_test_t, z_test_tp, a_test, device=device, seed=int(args.seed)
        )

        # --- Downstream: goal-conditioned action retrieval planning ---
        planning: Dict[str, Any] = {}
        if bool(args.do_planning):
            planning["action_retrieval"] = eval_goal_action_retrieval(
                train_out,
                z_test_t,
                z_test_tp,
                a_test,
                device=device,
                seed=int(args.seed),
                n_candidates=int(args.plan_candidates),
                n_eval=int(args.plan_eval),
            )

        # --- V-JEPA-like CEM objective (offline proxy) ---
        if bool(args.do_cem):
            # L1-ball bound: either explicit, or derived from action magnitudes.
            if args.cem_l1_bound is None:
                l1 = np.sum(np.abs(a_train), axis=1)
                bound = float(np.quantile(l1, float(args.cem_l1_quantile)))
            else:
                bound = float(args.cem_l1_bound)

            rng = np.random.default_rng(int(args.seed))
            n_eval = min(int(args.cem_eval), len(z_test_t))
            idx = rng.choice(len(z_test_t), size=n_eval, replace=False)

            energies_gt: List[float] = []
            energies_rand: List[float] = []
            energies_cem: List[float] = []

            m_a = train_out["model_action"]
            for ii in idx:
                i = int(ii)
                z0 = z_test_t[i]
                zg = z_test_tp[i]
                a_gt = a_test[i]

                # Ground-truth energy under the learned predictor
                with torch.no_grad():
                    z0_t = torch.tensor(
                        z0.reshape(1, -1), dtype=torch.float32, device=device
                    )
                    a_gt_t = torch.tensor(
                        a_gt.reshape(1, -1), dtype=torch.float32, device=device
                    )
                    pred_gt = torch.nn.functional.normalize(
                        m_a(torch.cat([z0_t, a_gt_t], dim=-1)), dim=-1
                    )
                    zg_t = torch.tensor(
                        zg.reshape(1, -1), dtype=torch.float32, device=device
                    )
                    e_gt = torch.mean(torch.abs(pred_gt - zg_t)).item()
                energies_gt.append(float(e_gt))

                # Random action baseline (sample from empirical action distribution)
                j = int(rng.integers(0, len(a_train)))
                a_r = a_train[j]
                with torch.no_grad():
                    a_r_t = torch.tensor(
                        a_r.reshape(1, -1), dtype=torch.float32, device=device
                    )
                    pred_r = torch.nn.functional.normalize(
                        m_a(torch.cat([z0_t, a_r_t], dim=-1)), dim=-1
                    )
                    e_r = torch.mean(torch.abs(pred_r - zg_t)).item()
                energies_rand.append(float(e_r))

                # CEM optimized action
                _, e_cem = cem_plan_one_step(
                    m_a,
                    z0,
                    zg,
                    act_dim=a_gt.shape[0],
                    device=device,
                    seed=int(args.seed) + i,
                    n_samples=int(args.cem_samples),
                    n_elite=int(args.cem_elite),
                    n_iters=int(args.cem_iters),
                    sigma_init=float(args.cem_sigma),
                    l1_bound=bound,
                )
                energies_cem.append(float(e_cem))

            planning["vjepa_cem_1step"] = {
                "n_eval": int(n_eval),
                "l1_bound": float(bound),
                "energy_mean": {
                    "gt": float(np.mean(energies_gt)) if energies_gt else 0.0,
                    "random": float(np.mean(energies_rand)) if energies_rand else 0.0,
                    "cem": float(np.mean(energies_cem)) if energies_cem else 0.0,
                },
                "energy_median": {
                    "gt": float(np.median(energies_gt)) if energies_gt else 0.0,
                    "random": float(np.median(energies_rand)) if energies_rand else 0.0,
                    "cem": float(np.median(energies_cem)) if energies_cem else 0.0,
                },
            }

        # --- Visuals ---
        # (Montage is useful as a sanity check across morphologies/tasks.)
        save_montage(test_pairs, out_dir / "montage.png", n_rows=8, thumb=192)
        save_diagnostics(ev, out_dir / "diagnostics.png")

        # --- Results ---
        def _flatten(prefix: str, obj: Any, out: Dict[str, float]) -> None:
            if isinstance(obj, dict):
                for kk, vv in obj.items():
                    _flatten(prefix + str(kk) + "/", vv, out)
            else:
                try:
                    out[prefix[:-1]] = float(obj)
                except Exception:
                    pass

        result: Dict[str, Any] = {
            "exp": "lerobot_latent_dynamics",
            "device": device,
            "world": {
                "adapter": "lerobot",
                "fingerprint": world_fp,
                "config": w_cfg.__dict__,
                "meta": {
                    **data.get("meta", {}),
                    "resolved_keys": key_meta,
                },
            },
            "encoder": {
                "name": "hf_vision",
                "fingerprint": enc_fp,
                "config": enc_cfg.__dict__,
            },
            "metrics": ev["metrics"],
            "deltas": ev["deltas"],
            # IMPORTANT: keep make/break identical to other latent-dynamics experiments.
            "make_break": ev["make_break"],
            "planning": planning,
            "artifacts": {
                "results_json": str(out_dir / "results.json"),
                "diagnostics_png": str(out_dir / "diagnostics.png"),
                "montage_png": str(out_dir / "montage.png"),
            },
        }

        (out_dir / "results.json").write_text(json.dumps(result, indent=2))

        # Optional logging hooks
        try:
            flat: Dict[str, float] = {}
            _flatten("metrics/", result["metrics"], flat)
            _flatten("deltas/", result["deltas"], flat)
            _flatten("planning/", result.get("planning", {}), flat)
            ctx.logger.log_metrics(flat)
            ctx.logger.log_text(
                "make_break/verdict", str(result["make_break"].get("verdict", ""))
            )
            ctx.logger.log_artifact(out_dir / "results.json", name="results.json")
            ctx.logger.log_artifact(out_dir / "diagnostics.png", name="diagnostics.png")
            ctx.logger.log_artifact(out_dir / "montage.png", name="montage.png")
        finally:
            try:
                ctx.logger.close()
            except Exception:
                pass

        return result
