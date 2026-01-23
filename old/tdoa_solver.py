from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class TdoaSolveResult:
    position: np.ndarray
    residual_rms: float
    iterations: int
    success: bool


@dataclass
class AnchorFrame:
    seq: int
    tx_ts: int
    rx_ts: int


class TdoaFrameMatcher:
    def __init__(
        self,
        tick_frequency_hz: float,
        antenna_delay_ticks: int = 16436,
        clock_correction: float = 1.0,
        history_size: int = 8,
    ) -> None:
        self._tick_frequency_hz = tick_frequency_hz
        self._antenna_delay_ticks = antenna_delay_ticks
        self._clock_correction = clock_correction
        self._clock_correction_by_anchor: Dict[int, float] = {}
        self._history_size = max(1, history_size)
        self._frames: Dict[int, Deque[AnchorFrame]] = {}
        self._clock_correction_spec_min = 1.0 - 100e-6 * 2
        self._clock_correction_spec_max = 1.0 + 100e-6 * 2
        self._clock_correction_filter = 0.1

    def _push_frame(self, anchor_id: int, frame: AnchorFrame) -> None:
        history = self._frames.get(anchor_id)
        if history is None:
            history = deque(maxlen=self._history_size)
            self._frames[anchor_id] = history
        history.appendleft(frame)

    def _find_frame(self, anchor_id: int, seq: int) -> tuple[AnchorFrame | None, int | None]:
        history = self._frames.get(anchor_id)
        if not history:
            return None, None
        for age, frame in enumerate(history):
            if frame.seq == seq:
                return frame, age
        return None, None

    def _update_clock_correction(self, anchor_id: int, frame: AnchorFrame) -> float | None:
        history = self._frames.get(anchor_id)
        if not history or len(history) < 2:
            return None
        prev = history[1]
        tick_count_remote = self._wrapped_delta(frame.tx_ts, prev.tx_ts, 2**32)
        tick_count_local = self._wrapped_delta(frame.rx_ts, prev.rx_ts, 2**40)
        if tick_count_remote == 0:
            return None
        candidate = float(tick_count_local) / float(tick_count_remote)
        if not (self._clock_correction_spec_min <= candidate <= self._clock_correction_spec_max):
            return None
        current = self._clock_correction_by_anchor.get(anchor_id, candidate)
        filtered = current * (1.0 - self._clock_correction_filter) + candidate * self._clock_correction_filter
        self._clock_correction_by_anchor[anchor_id] = filtered
        return filtered

    @staticmethod
    def _signed_delta(value: int, reference: int, modulus: int) -> int:
        delta = (value - reference) % modulus
        if delta > modulus // 2:
            delta -= modulus
        return delta

    @staticmethod
    def _wrapped_delta(value: int, reference: int, modulus: int) -> int:
        return (value - reference) % modulus

    def _compute(self, parsed_frame: dict) -> tuple[List[Tuple[int, float]], dict]:
        src = parsed_frame["src"]
        seq = parsed_frame["seq"]
        tx_ts = parsed_frame["tx_ts"]
        rx_ts = parsed_frame["rx_ts"]
        new_frame = AnchorFrame(seq=seq, tx_ts=tx_ts, rx_ts=rx_ts)
        self._push_frame(src, new_frame)
        estimated_clock_correction = self._update_clock_correction(src, new_frame)

        debug_info = {
            "src": src,
            "seq": seq,
            "tx_ts": tx_ts,
            "rx_ts": rx_ts,
            "tick_frequency_hz": self._tick_frequency_hz,
            "antenna_delay_ticks": self._antenna_delay_ticks,
            "clock_correction": self._clock_correction,
            "clock_correction_estimate": estimated_clock_correction,
            "remotes": [],
        }

        measurements: List[Tuple[int, float]] = []
        for remote in parsed_frame.get("remotes", []):
            remote_id = remote["id"]
            remote_seq = remote["seq"]
            dist_ticks = int(remote.get("dist_ticks", 0))
            remote_rx_ts = int(remote.get("rx_ts", 0))
            debug_remote = {
                "id": remote_id,
                "seq": remote_seq,
                "dist_ticks": dist_ticks,
                "remote_rx_ts": remote_rx_ts,
                "status": "unknown",
            }

            if dist_ticks <= self._antenna_delay_ticks:
                debug_remote["status"] = "dist_too_small"
                debug_info["remotes"].append(debug_remote)
                continue

            remote_frame, history_age = self._find_frame(remote_id, remote_seq)
            if not remote_frame:
                debug_remote["status"] = "no_remote_frame"
                debug_info["remotes"].append(debug_remote)
                continue
            if history_age is not None and history_age > 0:
                debug_remote["matched_from_history"] = True
                debug_remote["history_age"] = history_age

            tof_ticks = dist_ticks - self._antenna_delay_ticks
            clock_correction = self._clock_correction_by_anchor.get(src, self._clock_correction)
            delta_tx = self._wrapped_delta(tx_ts, remote_rx_ts, 2**32)
            delta_tx = int(round(delta_tx * clock_correction))
            delta_rx = self._wrapped_delta(rx_ts, remote_frame.rx_ts, 2**40)
            tdoa_ticks = delta_rx - (tof_ticks + delta_tx)
            distance_diff = SPEED_OF_LIGHT * tdoa_ticks / self._tick_frequency_hz
            measurements.append((remote_id, float(distance_diff)))
            debug_remote.update(
                {
                    "status": "ok",
                    "clock_correction_used": clock_correction,
                    "tof_ticks": tof_ticks,
                    "delta_tx": delta_tx,
                    "delta_rx": delta_rx,
                    "tdoa_ticks": tdoa_ticks,
                    "distance_diff": float(distance_diff),
                }
            )
            debug_info["remotes"].append(debug_remote)

        return measurements, debug_info

    def update(self, parsed_frame: dict) -> List[Tuple[int, float]]:
        measurements, _ = self._compute(parsed_frame)
        return measurements

    def update_with_debug(self, parsed_frame: dict) -> tuple[List[Tuple[int, float]], dict]:
        return self._compute(parsed_frame)


def load_anchors(path: str | Path) -> Dict[int, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    anchors: Dict[int, np.ndarray] = {}

    if isinstance(data, dict) and "anchors" in data:
        raw = data["anchors"]
        if isinstance(raw, dict):
            for key, value in raw.items():
                anchors[int(key)] = np.asarray(value, dtype=float)
        elif isinstance(raw, list):
            for item in raw:
                anchors[int(item["id"])] = np.asarray(item["pos"], dtype=float)
    elif isinstance(data, list):
        for item in data:
            anchors[int(item["id"])] = np.asarray(item["pos"], dtype=float)
    else:
        raise ValueError("Unsupported anchors JSON format")

    for anchor_id, pos in anchors.items():
        if pos.shape != (3,):
            raise ValueError(f"Anchor {anchor_id} position must have 3 elements")

    return anchors


def build_tdoa_measurements(parsed_frame: dict, tick_frequency_hz: float) -> List[Tuple[int, float]]:
    measurements: List[Tuple[int, float]] = []
    for remote in parsed_frame.get("remotes", []):
        dist_ticks = remote.get("dist_ticks")
        if not dist_ticks:
            continue
        distance_diff = dist_ticks / tick_frequency_hz * SPEED_OF_LIGHT
        measurements.append((remote["id"], float(distance_diff)))
    return measurements


def solve_tdoa_position(
    anchors: Dict[int, np.ndarray],
    reference_id: int,
    measurements: Iterable[Tuple[int, float]],
    initial_position: np.ndarray | None = None,
    fixed_z: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> TdoaSolveResult:
    if reference_id not in anchors:
        raise ValueError("Reference anchor is missing")

    pairs = [(anchor_id, diff) for anchor_id, diff in measurements if anchor_id in anchors]
    min_required = 2 if fixed_z is not None else 3
    if len(pairs) < min_required:
        if fixed_z is not None:
            raise ValueError("Need at least 2 TDoA measurements with known anchors for fixed-z mode")
        raise ValueError("Need at least 3 TDoA measurements with known anchors")

    anchor_positions = np.stack([anchors[anchor_id] for anchor_id, _ in pairs], axis=0)
    diffs = np.asarray([diff for _, diff in pairs], dtype=float)
    ref_pos = anchors[reference_id]

    if initial_position is None:
        initial_position = np.mean(np.vstack([ref_pos, anchor_positions]), axis=0)

    if fixed_z is None:
        x = np.asarray(initial_position, dtype=float)
        if x.shape != (3,):
            x = np.asarray(initial_position[:3], dtype=float)
        param_dim = 3
    else:
        x_init = np.asarray(initial_position, dtype=float)
        if x_init.shape == (3,):
            x = x_init[:2]
        else:
            x = x_init[:2]
        param_dim = 2
    success = False

    for iteration in range(1, max_iter + 1):
        if fixed_z is None:
            position = x
        else:
            position = np.array([x[0], x[1], fixed_z], dtype=float)

        vec_ref = position - ref_pos
        dist_ref = np.linalg.norm(vec_ref)
        if dist_ref < 1e-9:
            dist_ref = 1e-9

        vec_other = position - anchor_positions
        dist_other = np.linalg.norm(vec_other, axis=1)
        dist_other = np.where(dist_other < 1e-9, 1e-9, dist_other)

        residuals = (dist_ref - dist_other) - diffs
        jacobian_full = vec_ref / dist_ref - (vec_other / dist_other[:, None])
        jacobian = jacobian_full[:, :param_dim]

        lhs = jacobian.T @ jacobian + damping * np.eye(param_dim)
        rhs = -jacobian.T @ residuals

        step = np.linalg.solve(lhs, rhs)
        x = x + step

        if np.linalg.norm(step) < tol:
            success = True
            break

    if fixed_z is None:
        final_position = x
    else:
        final_position = np.array([x[0], x[1], fixed_z], dtype=float)

    vec_ref = final_position - ref_pos
    dist_ref = np.linalg.norm(vec_ref)
    if dist_ref < 1e-9:
        dist_ref = 1e-9
    vec_other = final_position - anchor_positions
    dist_other = np.linalg.norm(vec_other, axis=1)
    dist_other = np.where(dist_other < 1e-9, 1e-9, dist_other)
    final_residuals = (dist_ref - dist_other) - diffs
    residual_rms = float(np.sqrt(np.mean(final_residuals**2)))
    return TdoaSolveResult(position=final_position, residual_rms=residual_rms, iterations=iteration, success=success)
