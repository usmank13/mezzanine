from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..pipelines.audio_distill import build_views, encode_examples
from ..pipelines.text_distill import (
    MLPHeadConfig,
    train_hard_label_head,
    train_soft_label_head,
    predict_proba,
    accuracy,
    warrant_gap_from_views,
)
from ..recipes.recipe_base import Recipe
from ..symmetries.audio_playback import AudioPlaybackConfig, AudioPlaybackSymmetry
from ..utils.audio_features import AudioFeatureConfig
from ..worlds.audio_folder import AudioFolderAdapter, AudioFolderAdapterConfig
from ..worlds.esc50 import Esc50Adapter, Esc50AdapterConfig
from ..worlds.urbansound8k import UrbanSound8KAdapter, UrbanSound8KAdapterConfig


class AudioWarrantMixRecipe(Recipe):
    NAME = "audio_warrant_mix"
    DESCRIPTION = "Audio folder: measure playback nuisance gap and distill a symmetry-marginalized student head."

    @staticmethod
    def _parse_int_list(raw: str | None) -> List[int] | None:
        if not raw:
            return None
        out = [int(x) for x in str(raw).split(",") if str(x).strip()]
        return out or None

    @staticmethod
    def _parse_str_list(raw: str | None) -> List[str] | None:
        if not raw:
            return None
        out = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
        return out or None

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # Dataset
        p.add_argument("--audio_dir", type=str, default=None)
        p.add_argument("--pattern", type=str, default="*.wav")
        p.add_argument("--recursive", action="store_true")
        p.add_argument("--sr", type=int, default=48000)
        p.add_argument("--clip_seconds", type=float, default=2.0)
        p.add_argument("--mono", action="store_true")
        p.add_argument("--train_fraction", type=float, default=0.8)
        p.add_argument("--label_from_subdir", action="store_true")
        p.add_argument("--esc50_csv", type=str, default=None)
        p.add_argument("--esc50_audio_dir", type=str, default=None)
        p.add_argument("--esc50_test_folds", type=str, default=None)
        p.add_argument("--esc50_classes", type=str, default=None)
        p.add_argument("--urbansound_csv", type=str, default=None)
        p.add_argument("--urbansound_audio_root", type=str, default=None)
        p.add_argument("--urbansound_test_folds", type=str, default=None)
        p.add_argument("--urbansound_classes", type=str, default=None)

        # Symmetry
        p.add_argument("--k_train", type=int, default=4)
        p.add_argument("--k_test", type=int, default=8)
        p.add_argument("--gain_db_min", type=float, default=AudioPlaybackConfig.gain_db_min)
        p.add_argument("--gain_db_max", type=float, default=AudioPlaybackConfig.gain_db_max)
        p.add_argument("--noise_db_min", type=float, default=AudioPlaybackConfig.noise_db_min)
        p.add_argument("--noise_db_max", type=float, default=AudioPlaybackConfig.noise_db_max)
        p.add_argument("--mono_prob", type=float, default=AudioPlaybackConfig.mono_prob)
        p.add_argument("--lowpass_hz_min", type=float, default=AudioPlaybackConfig.lowpass_hz_min)
        p.add_argument("--lowpass_hz_max", type=float, default=AudioPlaybackConfig.lowpass_hz_max)
        p.add_argument("--time_stretch_min", type=float, default=AudioPlaybackConfig.time_stretch_min)
        p.add_argument("--time_stretch_max", type=float, default=AudioPlaybackConfig.time_stretch_max)

        # Feature config
        p.add_argument("--n_bands", type=int, default=4)
        p.add_argument("--rolloff_frac", type=float, default=0.85)

        # Head
        p.add_argument("--hidden", type=int, default=64)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=200)
        p.add_argument("--student_steps", type=int, default=200)
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--device", type=str, default="cpu")

        args = p.parse_args(argv)
        file_cfg = {}
        self.apply_config_defaults(p, args, file_cfg)

        ctx = self.build_context(args)

        if args.esc50_csv and args.urbansound_csv:
            raise ValueError("Choose only one of --esc50_csv or --urbansound_csv.")

        if args.esc50_csv:
            if not args.esc50_audio_dir:
                raise ValueError("--esc50_audio_dir is required with --esc50_csv.")
            adapter_cfg = Esc50AdapterConfig(
                csv_path=str(args.esc50_csv),
                audio_dir=str(args.esc50_audio_dir),
                sr=int(args.sr),
                clip_seconds=float(args.clip_seconds),
                mono=bool(args.mono),
                seed=int(args.seed),
                train_fraction=float(args.train_fraction),
                test_folds=self._parse_int_list(args.esc50_test_folds),
                classes=self._parse_str_list(args.esc50_classes),
                include_audio=True,
            )
            adapter = Esc50Adapter(adapter_cfg)
        elif args.urbansound_csv:
            if not args.urbansound_audio_root:
                raise ValueError("--urbansound_audio_root is required with --urbansound_csv.")
            adapter_cfg = UrbanSound8KAdapterConfig(
                csv_path=str(args.urbansound_csv),
                audio_root=str(args.urbansound_audio_root),
                sr=int(args.sr),
                clip_seconds=float(args.clip_seconds),
                mono=bool(args.mono),
                seed=int(args.seed),
                train_fraction=float(args.train_fraction),
                test_folds=self._parse_int_list(args.urbansound_test_folds),
                classes=self._parse_str_list(args.urbansound_classes),
                include_audio=True,
            )
            adapter = UrbanSound8KAdapter(adapter_cfg)
        else:
            if not args.audio_dir:
                raise ValueError("--audio_dir is required when not using --esc50_csv or --urbansound_csv.")
            adapter_cfg = AudioFolderAdapterConfig(
                audio_dir=args.audio_dir,
                pattern=args.pattern,
                recursive=bool(args.recursive),
                sr=int(args.sr),
                clip_seconds=float(args.clip_seconds),
                mono=bool(args.mono),
                train_fraction=float(args.train_fraction),
                include_audio=True,
                label_from_subdir=bool(args.label_from_subdir),
                seed=int(args.seed),
            )
            adapter = AudioFolderAdapter(adapter_cfg)
        world = adapter.load()

        train = world["train"]
        test = world["test"]
        if not train or not test:
            raise ValueError("Need non-empty train and test splits.")
        if "label" not in train[0]:
            raise ValueError("Labels missing; use --label_from_subdir or provide labeled data.")

        y_train = np.array([int(ex["label"]) for ex in train], dtype=np.int64)
        y_test = np.array([int(ex["label"]) for ex in test], dtype=np.int64)
        num_classes = int(max(y_train.max(), y_test.max()) + 1)

        feat_cfg = AudioFeatureConfig(sr=int(args.sr), n_bands=int(args.n_bands), rolloff_frac=float(args.rolloff_frac))

        # Canonical embeddings
        Z_train = encode_examples(train, feat_cfg)
        Z_test = encode_examples(test, feat_cfg)

        # Deterministic train/val split
        n = Z_train.shape[0]
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        Z_tr, y_tr = Z_train[idx_tr], y_train[idx_tr]
        Z_val, y_val = Z_train[idx_val], y_train[idx_val]

        head_cfg = MLPHeadConfig(
            in_dim=int(Z_train.shape[1]),
            num_classes=num_classes,
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        base_head, base_metrics = train_hard_label_head(
            Z_tr, y_tr, Z_val, y_val,
            cfg=head_cfg,
            steps=int(args.base_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=str(args.device),
            seed=int(args.seed),
        )

        # Symmetry views
        symmetry_cfg = AudioPlaybackConfig(
            sr=int(args.sr),
            gain_db_min=float(args.gain_db_min),
            gain_db_max=float(args.gain_db_max),
            noise_db_min=float(args.noise_db_min),
            noise_db_max=float(args.noise_db_max),
            mono_prob=float(args.mono_prob),
            lowpass_hz_min=float(args.lowpass_hz_min),
            lowpass_hz_max=float(args.lowpass_hz_max),
            time_stretch_min=float(args.time_stretch_min),
            time_stretch_max=float(args.time_stretch_max),
        )
        symmetry = AudioPlaybackSymmetry(symmetry_cfg)
        views_train = build_views(train, symmetry=symmetry, seed=int(args.seed), k=int(args.k_train))
        views_test = build_views(test, symmetry=symmetry, seed=int(args.seed) + 1, k=int(args.k_test))

        # Teacher probs (train)
        P_views_train = []
        for v in views_train:
            Z_v = encode_examples(v, feat_cfg)
            P_views_train.append(predict_proba(base_head, Z_v, device=str(args.device)))
        P_views_train = np.stack(P_views_train, axis=1)  # [N,K,C]
        P_teacher = P_views_train.mean(axis=1)

        # Base metrics
        P_base_test = predict_proba(base_head, Z_test, device=str(args.device))
        base_acc = accuracy(P_base_test, y_test)
        P_views_test = []
        for v in views_test:
            Z_v = encode_examples(v, feat_cfg)
            P_views_test.append(predict_proba(base_head, Z_v, device=str(args.device)))
        P_views_test = np.stack(P_views_test, axis=1)
        base_gap = warrant_gap_from_views(P_views_test)
        base_view_acc = float(np.mean([accuracy(P_views_test[:, i, :], y_test) for i in range(P_views_test.shape[1])]))

        # Student
        student_head, student_metrics = train_soft_label_head(
            Z_tr, P_teacher[idx_tr], Z_val, y_val,
            cfg=head_cfg,
            steps=int(args.student_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=str(args.device),
            seed=int(args.seed),
        )

        P_student_test = predict_proba(student_head, Z_test, device=str(args.device))
        student_acc = accuracy(P_student_test, y_test)

        P_views_test_student = []
        for v in views_test:
            Z_v = encode_examples(v, feat_cfg)
            P_views_test_student.append(predict_proba(student_head, Z_v, device=str(args.device)))
        P_views_test_student = np.stack(P_views_test_student, axis=1)
        student_gap = warrant_gap_from_views(P_views_test_student)
        student_view_acc = float(np.mean([accuracy(P_views_test_student[:, i, :], y_test) for i in range(P_views_test_student.shape[1])]))

        gap_reduced = student_gap["mean_tv_to_mean"] <= base_gap["mean_tv_to_mean"] * 0.9
        acc_ok = student_acc >= base_acc - 0.05

        results = {
            "adapter": adapter.NAME,
            "world_fingerprint": adapter.fingerprint(),
            "feature_cfg": asdict(feat_cfg),
            "symmetry": symmetry.NAME,
            "metrics": {
                "base": {"acc": base_acc, "view_acc": base_view_acc, **base_gap, **base_metrics},
                "student": {"acc": student_acc, "view_acc": student_view_acc, **student_gap, **student_metrics},
            },
            "make_break": {
                "gap_reduced": bool(gap_reduced),
                "acc_ok": bool(acc_ok),
                "verdict": bool(gap_reduced and acc_ok),
            },
        }

        out_path = Path(ctx.out_dir) / "results.json"
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        return results
