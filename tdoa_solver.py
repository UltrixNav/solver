from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple, Mapping

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
    tx_ts: int  # 32-bit (anchor clock domain)
    rx_ts: int  # 32-bit (tag clock domain; user reports LSBs dropped)


class TdoaFrameMatcher:
    """Matches TDoA3 packets and computes TDoA distance-difference measurements.

    This is a Python adaptation of the Crazyflie TDoA3 math:

    - Maintain per-anchor history of (seq, tx_ts, rx_ts)
    - Maintain per-anchor clock correction estimate (tag_clock / anchor_clock)
    - For each remote report in a packet, find the corresponding remote anchor frame
      with matching seq number.
    - Compute TDoA in tag clock ticks and convert to meters.

    Notes about fields (based on Bitcraze TDoA3 packet):
    - For an incoming packet from anchor An:
        txAn_in_cl_An = packet.tx_ts
        rxAn_by_T_in_cl_T = packet.rx_ts
    - For a remote anchor Ar referenced inside An's packet:
        rxAr_by_An_in_cl_An = remote.rx_ts
        tof_Ar_to_An_in_cl_An = remote.dist_ticks (in anchor clock ticks)

    The core equation (ticks in tag clock domain):

    tdoa_ticks = truncate(rxAn_by_T - rxAr_by_T) - (tof_Ar_to_An + truncate(txAn - rxAr_by_An)) * clock_correction

    where clock_correction ~ (tag_ticks / anchor_ticks) for anchor An.
    """

    def __init__(
        self,
        tick_frequency_hz: float,
        antenna_delay_ticks: int = 16436,
        clock_correction: float = 1.0,
        history_size: int = 8,
        # Conservative bounds for DW1000 drift (100 ppm) but doubled for margin.
        clock_correction_ppm: float = 200.0,
        clock_correction_filter: float = 0.1,
        max_abs_distance_diff_m: float = 1000.0,
        subtract_antenna_delay: bool = True,
        tdoa_sign: float = 1.0,
        # If True, and dist_ticks looks already compensated, we'll skip antenna-delay subtraction.
        auto_antenna_delay: bool = True,
        # Basic sanity gates (ticks) to avoid blowing up the solver on bad matches.
        min_tof_ticks: int = 0,
        max_tof_ticks: int = 65535,
        # Optional: use anchor positions to reject impossible distance-differences.
        anchors: Mapping[int, np.ndarray] | None = None,
        distance_diff_margin_m: float = 0.25,
        max_remote_history_age: int = 1,
    ) -> None:
        self._tick_frequency_hz = float(tick_frequency_hz)
        self._antenna_delay_ticks = int(antenna_delay_ticks)
        self._clock_correction_default = float(clock_correction)
        self._history_size = max(1, int(history_size))

        self._frames: Dict[int, Deque[AnchorFrame]] = {}
        self._clock_correction_by_anchor: Dict[int, float] = {}

        corr = clock_correction_ppm * 1e-6
        self._clock_correction_spec_min = 1.0 - corr
        self._clock_correction_spec_max = 1.0 + corr
        self._clock_correction_filter = float(clock_correction_filter)


        # Your packets use 32-bit RX timestamps (with some LSBs dropped).
        # Keep everything in the same 32-bit space to avoid false huge deltas.
        self._truncate_mask = (1 << 32) - 1

        # Hard outlier rejection. If your anchor constellation is larger, increase this.
        self._max_abs_distance_diff_m = float(max_abs_distance_diff_m)
        self._subtract_antenna_delay = bool(subtract_antenna_delay)
        self._tdoa_sign = float(tdoa_sign)
        self._auto_antenna_delay = bool(auto_antenna_delay)
        self._min_tof_ticks = int(min_tof_ticks)
        self._max_tof_ticks = int(max_tof_ticks)
        self._anchors = dict(anchors) if anchors is not None else None
        self._distance_diff_margin_m = float(distance_diff_margin_m)
        self._max_remote_history_age = int(max_remote_history_age)

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

        # Remote (anchor) timestamps are 32-bit
        tick_count_remote = self._wrapped_delta(frame.tx_ts, prev.tx_ts, 2**32)
        # Local (tag) RX timestamps are also 32-bit in your stream
        tick_count_local = self._wrapped_delta(frame.rx_ts, prev.rx_ts, 2**32)

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
    def _wrapped_delta(value: int, reference: int, modulus: int) -> int:
        return (int(value) - int(reference)) % int(modulus)

    def _truncate_to_anchor_ts(self, value: int) -> int:
        # Equivalent to firmware bitmap truncation: keep lower 32 bits.
        return int(value) & self._truncate_mask

    def _compute(self, parsed_frame) -> tuple[List[Tuple[int, float]], dict]:
        # parsed_frame is expected to be an AnchorReport from data_classes.py
        src = int(parsed_frame.src)
        seq = int(parsed_frame.seq) & 0x7F
        tx_ts = int(parsed_frame.tx_ts)  # uint32
        rx_ts = int(parsed_frame.rx_ts) & 0xFFFFFFFF  # uint32

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
            "clock_correction_estimate": estimated_clock_correction,
            "remotes": [],
        }

        measurements: List[Tuple[int, float]] = []

        ref_pos = None
        if self._anchors is not None:
            ref_pos = self._anchors.get(src)

        # For each remote anchor Ar listed by current anchor An
        for remote in getattr(parsed_frame, "remotes", []) or []:
            remote_id = int(remote.id)
            remote_seq = int(remote.seq) & 0x7F
            rxAr_by_An_in_cl_An = int(remote.rx_ts)  # uint32 in anchor clock domain
            dist_ticks = int(remote.dist_ticks)  # uint16; used as ToF in anchor clock domain

            debug_remote = {
                "id": remote_id,
                "seq": remote_seq,
                "remote_rx_ts": rxAr_by_An_in_cl_An,
                "dist_ticks": dist_ticks,
                "status": "unknown",
            }

            # In Bitcraze, distance field is ToF in ticks; subtract antenna delay.
            # Your firmware may already pre-compensate; but your config exposes ANTENNA_DELAY_TICKS,
            # so we follow the Crazyflie convention.

            # If dist_ticks is raw ToF including antenna delay, subtracting is correct.
            # If firmware already subtracted it, then dist_ticks can be smaller than ANTENNA_DELAY_TICKS.
            # With auto_antenna_delay enabled we handle both without changing settings.
            if self._subtract_antenna_delay:
                if self._auto_antenna_delay and dist_ticks <= self._antenna_delay_ticks:
                    # Looks like it's already compensated.
                    tof_Ar_to_An_in_cl_An = dist_ticks
                    debug_remote["antenna_delay_mode"] = "skipped(auto)"
                else:
                    if dist_ticks <= self._antenna_delay_ticks:
                        debug_remote["status"] = "dist_too_small"
                        debug_info["remotes"].append(debug_remote)
                        continue
                    tof_Ar_to_An_in_cl_An = dist_ticks - self._antenna_delay_ticks
                    debug_remote["antenna_delay_mode"] = "subtracted"
            else:
                tof_Ar_to_An_in_cl_An = dist_ticks
                debug_remote["antenna_delay_mode"] = "disabled"

            if not (self._min_tof_ticks <= tof_Ar_to_An_in_cl_An <= self._max_tof_ticks):
                debug_remote["status"] = "tof_out_of_range"
                debug_info["remotes"].append(debug_remote)
                continue

            # Find the remote anchor frame (Ar packet) by seq
            remote_frame, history_age = self._find_frame(remote_id, remote_seq)
            if remote_frame is None:
                debug_remote["status"] = "no_remote_frame"
                debug_info["remotes"].append(debug_remote)
                continue

            if history_age is not None and history_age > self._max_remote_history_age:
                debug_remote["status"] = "stale_remote_frame"
                debug_remote["history_age"] = history_age
                debug_info["remotes"].append(debug_remote)
                continue

            # Apply clock correction for current anchor An
            clock_correction = self._clock_correction_by_anchor.get(src, self._clock_correction_default)

            # delta_txAr_to_txAn_in_cl_An = tof + truncate(txAn - rxAr_by_An)
            # Note: txAn and rxAr_by_An are 32-bit in anchor clock domain.
            delta_tx = self._truncate_to_anchor_ts(tx_ts - rxAr_by_An_in_cl_An)
            delta_txAr_to_txAn_in_cl_An = int(tof_Ar_to_An_in_cl_An + delta_tx)

            # timeDiffOfArrival_in_cl_T = trunc(rxAn_by_T - rxAr_by_T) - delta_txAr_to_txAn_in_cl_An * clockCorrection
            delta_rx = self._truncate_to_anchor_ts(rx_ts - (remote_frame.rx_ts & 0xFFFFFFFF))

            tdoa_ticks = int(self._tdoa_sign * (delta_rx - (delta_txAr_to_txAn_in_cl_An * clock_correction)))
            distance_diff = SPEED_OF_LIGHT * float(tdoa_ticks) / self._tick_frequency_hz

            # Geometry gate: | ||p-Ar|| - ||p-Aref|| | <= ||Ar - Aref|| for any point p.
            if ref_pos is not None and self._anchors is not None:
                ar_pos = self._anchors.get(remote_id)
                if ar_pos is not None:
                    max_possible = float(np.linalg.norm(ar_pos - ref_pos)) + self._distance_diff_margin_m
                    if abs(distance_diff) > max_possible:
                        debug_remote.update(
                            {
                                "status": "outlier_geometry",
                                "distance_diff_m": float(distance_diff),
                                "max_possible_m": float(max_possible),
                                "tdoa_ticks": int(tdoa_ticks),
                            }
                        )
                        debug_info["remotes"].append(debug_remote)
                        continue

            # Reject obvious mismatches (wrong seq match, wrap issues, etc)
            if abs(distance_diff) > self._max_abs_distance_diff_m:
                debug_remote.update(
                    {
                        "status": "outlier_distance_diff",
                        "clock_correction_used": float(clock_correction),
                        "tof_ticks_used": int(tof_Ar_to_An_in_cl_An),
                        "delta_tx_ticks_trunc": int(delta_tx),
                        "delta_txAr_to_txAn": int(delta_txAr_to_txAn_in_cl_An),
                        "delta_rx_T_ticks": int(delta_rx),
                        "tdoa_ticks": int(tdoa_ticks),
                        "distance_diff_m": float(distance_diff),
                    }
                )
                debug_info["remotes"].append(debug_remote)
                continue

            measurements.append((remote_id, float(distance_diff)))

            debug_remote.update(
                {
                    "status": "ok",
                    "matched_from_history": bool(history_age and history_age > 0),
                    "history_age": history_age,
                    "clock_correction_used": float(clock_correction),
                    "tof_ticks_used": int(tof_Ar_to_An_in_cl_An),
                    "delta_tx_ticks_trunc": int(delta_tx),
                    "delta_txAr_to_txAn": int(delta_txAr_to_txAn_in_cl_An),
                    "delta_rx_T_ticks": int(delta_rx),
                    "tdoa_ticks": int(tdoa_ticks),
                    "distance_diff_m": float(distance_diff),
                }
            )
            debug_info["remotes"].append(debug_remote)

        return measurements, debug_info

    def update(self, parsed_frame) -> List[Tuple[int, float]]:
        measurements, _ = self._compute(parsed_frame)
        return measurements

    def update_with_debug(self, parsed_frame) -> tuple[List[Tuple[int, float]], dict]:
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


def solve_tdoa_position(
    anchors: Dict[int, np.ndarray],
    reference_id: int,
    measurements: Iterable[Tuple[int, float]],
    initial_position: np.ndarray | None = None,
    fixed_z: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    damping: float = 1e-3,
    # Robust outlier filtering in the solver space
    enable_outlier_filter: bool = True,
    max_outlier_passes: int = 2,
    outlier_k: float = 6.0,
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
        x = x_init[:2]
        param_dim = 2

    def _solve_once(anchor_positions_in: np.ndarray, diffs_in: np.ndarray) -> tuple[np.ndarray, float, int, bool]:
        """Run Gauss-Newton once for the provided measurement set."""
        nonlocal initial_position

        if initial_position is None:
            initial_position = np.mean(np.vstack([ref_pos, anchor_positions_in]), axis=0)

        if fixed_z is None:
            x_loc = np.asarray(initial_position, dtype=float)
            if x_loc.shape != (3,):
                x_loc = np.asarray(initial_position[:3], dtype=float)
            param_dim = 3
        else:
            x_init = np.asarray(initial_position, dtype=float)
            x_loc = x_init[:2]
            param_dim = 2

        success_loc = False
        iteration_loc = 0

        for iteration_loc in range(1, max_iter + 1):
            if fixed_z is None:
                position = x_loc
            else:
                position = np.array([x_loc[0], x_loc[1], fixed_z], dtype=float)

            vec_ref = position - ref_pos
            dist_ref = np.linalg.norm(vec_ref)
            if dist_ref < 1e-9:
                dist_ref = 1e-9

            vec_other = position - anchor_positions_in
            dist_other = np.linalg.norm(vec_other, axis=1)
            dist_other = np.where(dist_other < 1e-9, 1e-9, dist_other)

            residuals = (dist_ref - dist_other) - diffs_in
            jacobian_full = vec_ref / dist_ref - (vec_other / dist_other[:, None])
            jacobian = jacobian_full[:, :param_dim]

            lhs = jacobian.T @ jacobian + damping * np.eye(param_dim)
            rhs = -jacobian.T @ residuals

            step = np.linalg.solve(lhs, rhs)
            x_loc = x_loc + step

            if np.linalg.norm(step) < tol:
                success_loc = True
                break

        if fixed_z is None:
            final_position = x_loc
        else:
            final_position = np.array([x_loc[0], x_loc[1], fixed_z], dtype=float)

        # final rms
        vec_ref = final_position - ref_pos
        dist_ref = np.linalg.norm(vec_ref)
        if dist_ref < 1e-9:
            dist_ref = 1e-9

        vec_other = final_position - anchor_positions_in
        dist_other = np.linalg.norm(vec_other, axis=1)
        dist_other = np.where(dist_other < 1e-9, 1e-9, dist_other)
        final_residuals = (dist_ref - dist_other) - diffs_in
        residual_rms = float(np.sqrt(np.mean(final_residuals**2)))
        return final_position, residual_rms, iteration_loc, success_loc

    # Optional robust outlier filtering loop
    keep = np.ones(len(diffs), dtype=bool)
    if enable_outlier_filter and len(diffs) > min_required:
        for _ in range(max_outlier_passes):
            pos_try, _, _, _ = _solve_once(anchor_positions[keep], diffs[keep])

            # compute residuals at pos_try
            vec_ref = pos_try - ref_pos
            dist_ref = np.linalg.norm(vec_ref)
            if dist_ref < 1e-9:
                dist_ref = 1e-9
            vec_other = pos_try - anchor_positions[keep]
            dist_other = np.linalg.norm(vec_other, axis=1)
            dist_other = np.where(dist_other < 1e-9, 1e-9, dist_other)
            res = (dist_ref - dist_other) - diffs[keep]

            med = float(np.median(res))
            mad = float(np.median(np.abs(res - med)))
            # If MAD is zero, residuals are already perfectly consistent.
            if mad < 1e-9:
                break

            thresh = outlier_k * 1.4826 * mad
            bad_local = np.abs(res - med) > thresh
            if not np.any(bad_local):
                break

            # Drop the worst offender only (stable)
            idx_local = int(np.argmax(np.abs(res - med)))
            # Map local index back to global keep mask index
            keep_indices = np.flatnonzero(keep)
            keep[keep_indices[idx_local]] = False

            if int(np.sum(keep)) <= min_required:
                # Can't drop more
                break

    final_position, residual_rms, iteration, success = _solve_once(anchor_positions[keep], diffs[keep])

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

    return TdoaSolveResult(position=final_position, residual_rms=residual_rms, iterations=iteration, success=success)
