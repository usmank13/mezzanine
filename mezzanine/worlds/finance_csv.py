from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.cache import hash_dict
from ..utils.market_features import ReturnContextConfig, x_from_context
from .base import WorldAdapter


@dataclass
class FinanceCSVTapeAdapterConfig:
    """CSV -> return-tape supervised dataset.

    Expected CSV columns:
      - a price column (default: "close")
      - optionally a timestamp column (default: "timestamp")

    This adapter produces examples suitable for symmetry distillation under
    "bar alignment" style perturbations. Each example includes:
      - x: canonical feature vector (trend + return window)
      - label: binary label for the next return sign
      - r_context: return context buffer of length lookback + 2*max_offset
      - trend: slow feature computed from a longer trailing window (stable across small offsets)
    """

    path: str

    close_col: str = "close"
    timestamp_col: Optional[str] = "timestamp"

    # Optional filtering (e.g., multi-symbol CSVs)
    symbol_col: Optional[str] = None
    symbol: Optional[str] = None

    # Parsing
    delimiter: str = ","
    has_header: bool = True
    sort_by_timestamp: bool = False

    # Example construction
    lookback: int = 32
    max_offset: int = 1
    trend_lookback: int = 128
    label_horizon: int = 1
    stride: int = 1

    # Split sizes (chronological: first n_train, last n_test)
    n_train: int = 5000
    n_test: int = 2000
    gap: int = 0  # number of anchors to skip between train and test

    # Returns
    return_type: str = "log"  # "log" or "pct"

    seed: int = 0  # kept for fingerprinting / reproducibility


def _find_col(name_to_idx: Dict[str, int], want: Optional[str], *, kind: str) -> Optional[int]:
    if want is None:
        return None
    if want not in name_to_idx:
        raise KeyError(f"Missing {kind} column {want!r}. Available: {sorted(name_to_idx.keys())}")
    return int(name_to_idx[want])


def _parse_rows(
    path: Path,
    *,
    delimiter: str,
    has_header: bool,
    close_col: str,
    timestamp_col: Optional[str],
    symbol_col: Optional[str],
    symbol: Optional[str],
    sort_by_timestamp: bool,
) -> tuple[Optional[Sequence[Any]], np.ndarray]:
    """Return (timestamps, close_prices). timestamps may be None."""
    with path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        header: Optional[list[str]] = None
        if has_header:
            try:
                header = next(reader)
            except StopIteration as e:
                raise ValueError(f"CSV is empty: {path}") from e

        if header is None:
            # No header: interpret columns by index (close_col/timestamp_col must be ints-as-strings).
            try:
                idx_close = int(close_col)
            except Exception as e:
                raise ValueError("has_header=false requires close_col to be an integer column index") from e
            idx_ts = int(timestamp_col) if timestamp_col is not None else None
            idx_sym = int(symbol_col) if symbol_col is not None else None
        else:
            name_to_idx = {str(k): i for i, k in enumerate(header)}
            idx_close = _find_col(name_to_idx, close_col, kind="close")
            assert idx_close is not None
            idx_ts = _find_col(name_to_idx, timestamp_col, kind="timestamp")
            idx_sym = _find_col(name_to_idx, symbol_col, kind="symbol")

        ts_list: list[str] = []
        close_list: list[float] = []

        for row in reader:
            if not row:
                continue
            if idx_sym is not None and symbol is not None:
                if idx_sym >= len(row):
                    continue
                if str(row[idx_sym]) != str(symbol):
                    continue

            if idx_close >= len(row):
                continue
            if idx_ts is not None and idx_ts >= len(row):
                # Keep close/timestamp alignment: if timestamp_col is requested, require it per-row.
                continue
            try:
                c = float(row[idx_close])
            except Exception:
                continue
            close_list.append(c)

            if idx_ts is not None:
                ts_list.append(str(row[idx_ts]))

        if not close_list:
            raise ValueError(f"No rows parsed from {path} (after filtering).")

        close = np.asarray(close_list, dtype=np.float64)

        timestamps: Optional[Sequence[Any]]
        if idx_ts is None:
            timestamps = None
        else:
            timestamps = ts_list

        if sort_by_timestamp and timestamps is not None:
            # Sort by float timestamps if possible, else by string.
            float_ok = True
            ts_float: list[float] = []
            for t in timestamps:
                try:
                    ts_float.append(float(t))
                except Exception:
                    float_ok = False
                    break
            order = np.argsort(np.asarray(ts_float if float_ok else list(timestamps)))
            close = close[order]
            timestamps = [timestamps[int(i)] for i in order.tolist()]

        return timestamps, close


def _compute_returns(close: np.ndarray, *, return_type: str) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    if close.ndim != 1:
        close = close.reshape(-1)
    if close.shape[0] < 3:
        raise ValueError("Need at least 3 prices to compute returns + labels.")
    if str(return_type) == "log":
        if np.any(close <= 0):
            raise ValueError("Log returns require all close prices > 0.")
        r = np.log(close[1:] / close[:-1])
    elif str(return_type) == "pct":
        r = (close[1:] - close[:-1]) / np.clip(close[:-1], 1e-12, None)
    else:
        raise ValueError(f"Unknown return_type={return_type!r} (expected 'log' or 'pct').")
    return r.astype(np.float32, copy=False)


class FinanceCSVTapeAdapter(WorldAdapter):
    NAME = "finance_csv"
    DESCRIPTION = "CSV market tape adapter (close->returns->window features) for bar-offset symmetry distillation."

    def __init__(self, cfg: FinanceCSVTapeAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def load(self) -> Dict[str, Any]:
        path = Path(self.cfg.path).expanduser()
        if not path.exists():
            raise FileNotFoundError(str(path))

        timestamps, close = _parse_rows(
            path,
            delimiter=str(self.cfg.delimiter),
            has_header=bool(self.cfg.has_header),
            close_col=str(self.cfg.close_col),
            timestamp_col=self.cfg.timestamp_col,
            symbol_col=self.cfg.symbol_col,
            symbol=self.cfg.symbol,
            sort_by_timestamp=bool(self.cfg.sort_by_timestamp),
        )
        r = _compute_returns(close, return_type=str(self.cfg.return_type))

        L = int(self.cfg.lookback)
        O = int(self.cfg.max_offset)
        H = int(self.cfg.label_horizon)
        stride = max(1, int(self.cfg.stride))
        gap = max(0, int(self.cfg.gap))

        if L <= 1:
            raise ValueError("lookback must be >= 2")
        if O < 0:
            raise ValueError("max_offset must be >= 0")
        if H <= 0:
            raise ValueError("label_horizon must be >= 1")

        # Anchor i is a return index. We predict r[i+H] from window ending at r[i].
        i_min = (L - 1) + O
        i_max = (r.shape[0] - 1 - H) - O
        if i_max < i_min:
            raise ValueError("Not enough data for requested lookback/max_offset/label_horizon.")

        anchors = np.arange(i_min, i_max + 1, stride, dtype=np.int64)
        need = int(self.cfg.n_train) + gap + int(self.cfg.n_test)
        if anchors.shape[0] < need:
            raise ValueError(f"Not enough anchors ({anchors.shape[0]}) for n_train+gap+n_test={need}.")

        train_anchors = anchors[: int(self.cfg.n_train)]
        test_anchors = anchors[int(self.cfg.n_train) + gap : int(self.cfg.n_train) + gap + int(self.cfg.n_test)]

        trend_L = max(2, int(self.cfg.trend_lookback))
        ctx_cfg = ReturnContextConfig(max_offset=O)

        def _make_example(i: int) -> Dict[str, Any]:
            # r_context covers r[i-(L-1)-O ... i+O] inclusive
            start = i - (L - 1) - O
            end = i + O
            r_context = r[start : end + 1].astype(np.float32, copy=False)
            if r_context.shape[0] != L + 2 * O:
                raise AssertionError("bad r_context length")
            # Slow trend feature from a longer trailing window (stable under small offsets).
            t_start = max(0, i - (trend_L - 1))
            trend = float(r[t_start : i + 1].sum())

            y_ret = float(r[i + H])
            label = int(y_ret > 0)

            x = x_from_context(r_context, trend=trend, offset=0, cfg=ctx_cfg)

            ex: Dict[str, Any] = {
                "i": int(i),
                "x": x.astype(np.float32, copy=False),
                "label": label,
                "fwd_return": y_ret,
                "r_context": r_context,
                "trend": float(trend),
            }
            if timestamps is not None:
                # timestamps are for the *price* series; return index i corresponds to price index i+1.
                ex["timestamp"] = str(timestamps[int(i + 1)])
            return ex

        train = [_make_example(int(i)) for i in train_anchors.tolist()]
        test = [_make_example(int(i)) for i in test_anchors.tolist()]

        meta: Dict[str, Any] = {
            "path": str(path),
            "n_rows": int(close.shape[0]),
            "return_type": str(self.cfg.return_type),
            "lookback": L,
            "max_offset": O,
            "trend_lookback": trend_L,
            "label_horizon": H,
            "stride": stride,
            "n_train": len(train),
            "n_test": len(test),
            "gap": gap,
            "seed": int(self.cfg.seed),
            "columns": {
                "timestamp_col": self.cfg.timestamp_col,
                "close_col": self.cfg.close_col,
                "symbol_col": self.cfg.symbol_col,
                "symbol": self.cfg.symbol,
            },
        }
        return {"train": train, "test": test, "meta": meta}


# Register
from ..registry import ADAPTERS

ADAPTERS.register("finance_csv")(FinanceCSVTapeAdapter)
