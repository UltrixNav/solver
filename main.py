from __future__ import annotations

import json
import struct
import time
from pathlib import Path
import socket
from data_classes import *
import serial
from typing import Dict

from tdoa_solver import SPEED_OF_LIGHT, TdoaFrameMatcher, load_anchors, solve_tdoa_position

# -----------------------------
# User settings (edit here only)
# -----------------------------
PORT = "COM5"
BAUD_RATE = 115200
ANCHORS_JSON = "anchors.example.json"

ENABLE_SOLVE = True
FIXED_Z = 1.87  # set to None for full 3D solve
TICK_FREQUENCY_HZ = 63_897_600_000.0
PRINT_RATE_HZ = 2.0  # 0 disables throttling
ANTENNA_DELAY_TICKS = 16436
CLOCK_CORRECTION = 1.0
MATCH_HISTORY_SIZE = 8

# Reference anchor selection
# Set to an anchor id (e.g. 1) to lock reference to that anchor; set to None to use the incoming frame's src.
FIXED_REF_ID: int | None = 3

# Debug knobs
TDOA_SIGN = 1.0  # set to -1.0 to test sign convention
SUBTRACT_ANTENNA_DELAY = False
PRINT_MEASUREMENTS = True

# UDP telemetry
ENABLE_UDP = True
UDP_HOST = "255.255.255.255"  # broadcast by default; set to e.g. "127.0.0.1" for local
UDP_PORT = 7777
UDP_RATE_HZ = 20.0

# Output position filtering (simple, practical)
ENABLE_POS_FILTER = False
POS_FILTER_WINDOW_S = 0.5
POS_JUMP_MAX_M = 3.0  # reject if jump > this within POS_FILTER_WINDOW_S

START_BYTES = b"\xAA\x55"
POSITION_BLOCK_TYPE = 0xA1


def _make_udp_socket() -> socket.socket:
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	# allow broadcast
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.setblocking(False)
	return s


def _udp_send(sock: socket.socket, host: str, port: int, payload: dict) -> None:
	try:
		data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
		sock.sendto(data, (host, int(port)))
	except OSError:
		# ignore send errors (no listener, network down, etc)
		pass


def _throttled(now: float, last: float, rate_hz: float) -> bool:
	if rate_hz <= 0:
		return True
	period = 1.0 / float(rate_hz)
	return (now - last) >= period


def iter_frames(ser: serial.Serial):
	buffer = bytearray()
	while True:
		chunk = ser.read(256)
		if chunk:
			buffer.extend(chunk)
		else:
			time.sleep(0.001)

		while True:
			start_idx = buffer.find(START_BYTES)
			if start_idx == -1:
				if len(buffer) > 2:
					del buffer[:-2]
				break
			if start_idx > 0:
				del buffer[:start_idx]

			if len(buffer) < 4:
				break

			payload_len = struct.unpack_from("<H", buffer, 2)[0]
			frame_len = 2 + 2 + payload_len
			if len(buffer) < frame_len:
				break

			frame = bytes(buffer[:frame_len])
			del buffer[:frame_len]
			yield frame


def parse_frame(frame: bytes):
	payload_len = struct.unpack_from("<H", frame, 2)[0]
	payload = memoryview(frame)[4 : 4 + payload_len]

	if len(payload) < 1 + 1 + 1 + 4 + 8:
		return None

	offset = 0
	src = payload[offset]
	offset += 1
	seq = payload[offset]
	offset += 1
	remote_count = payload[offset]
	offset += 1
	tx_ts = struct.unpack_from("<I", payload, offset)[0]
	offset += 4
	rx_ts = struct.unpack_from("<Q", payload, offset)[0]
	offset += 8

	remotes = []
	for _ in range(remote_count):
		if offset + 8 > len(payload):
			break
		anchor_id = payload[offset]
		seq_nr = payload[offset + 1]
		rx_anchor_ts = struct.unpack_from("<I", payload, offset + 2)[0]
		dist_ticks = struct.unpack_from("<H", payload, offset + 6)[0]
		remotes.append(
			RemoteAnchorReport(
				id=anchor_id,
				seq=seq_nr,
				rx_ts=rx_anchor_ts,
				dist_ticks=dist_ticks,
			)
		)
		offset += 8

	position = None
	if offset + 1 + 12 <= len(payload) and payload[offset] == POSITION_BLOCK_TYPE:
		position = struct.unpack_from("<fff", payload, offset + 1)

	return AnchorReport(
		src=src,
		seq=seq,
		tx_ts=tx_ts,
		rx_ts=rx_ts,
		remote_count=remote_count,
		remotes=remotes,
		position=position,
	)

def main() -> int:
	anchors = load_anchors(Path(ANCHORS_JSON))

	udp_sock = None
	last_udp = 0.0
	if ENABLE_UDP:
		udp_sock = _make_udp_socket()

	matcher = TdoaFrameMatcher(
		tick_frequency_hz=TICK_FREQUENCY_HZ,
		antenna_delay_ticks=ANTENNA_DELAY_TICKS,
		clock_correction=CLOCK_CORRECTION,
		history_size=MATCH_HISTORY_SIZE,
		tdoa_sign=TDOA_SIGN,
		subtract_antenna_delay=SUBTRACT_ANTENNA_DELAY,
		anchors=anchors,
	)

	last_print = 0.0
	last_solution = None
	last_debug = None
	traj_xy: list[tuple[float, float]] = []
	TRAJ_MAX_POINTS = 200
	last_initial_position = None

	# Filter state
	pos_hist: list[tuple[float, float, float]] = []  # (t, x, y)
	filtered_pos_xy: tuple[float, float] | None = None

	with serial.Serial(PORT, BAUD_RATE, timeout=0.1) as ser:
		for frame in iter_frames(ser):
			parsed = parse_frame(frame)
			if parsed is None:
				continue

			# Always feed frames into the matcher so per-anchor history stays fresh.
			measurements, dbg = matcher.update_with_debug(parsed)
			last_debug = dbg

			effective_ref = int(parsed.src)
			frame_is_ref = True
			if FIXED_REF_ID is not None:
				effective_ref = int(FIXED_REF_ID)
				frame_is_ref = int(parsed.src) == effective_ref

			# Try to solve (only on reference frames when ref is fixed)
			min_meas = 2 if FIXED_Z is not None else 3
			if frame_is_ref and ENABLE_SOLVE and len(measurements) >= min_meas:
				try:
					# Warm-start to reduce hyperbola-branch jumps, especially while moving.
					initial_guess = None
					if ENABLE_POS_FILTER and filtered_pos_xy is not None:
						initial_guess = [filtered_pos_xy[0], filtered_pos_xy[1], FIXED_Z if FIXED_Z is not None else 0.0]
					elif last_solution is not None:
						p = last_solution.position
						initial_guess = [float(p[0]), float(p[1]), float(p[2])]
					elif last_initial_position is not None:
						initial_guess = last_initial_position

					res = solve_tdoa_position(
						anchors=anchors,
						reference_id=effective_ref,
						measurements=measurements,
						initial_position=initial_guess,
						fixed_z=FIXED_Z,
						max_iter=100,
						damping=1e-2,
					)
					last_solution = res
					last_initial_position = [float(res.position[0]), float(res.position[1]), float(res.position[2])]

					raw_xy = (float(res.position[0]), float(res.position[1]))
					tnow = time.time()

					# Sliding-window filter + jump rejection
					if ENABLE_POS_FILTER:
						# Drop old samples
						pos_hist.append((tnow, raw_xy[0], raw_xy[1]))
						cut = tnow - float(POS_FILTER_WINDOW_S)
						pos_hist = [p for p in pos_hist if p[0] >= cut]

						# Jump gate from last filtered position
						if filtered_pos_xy is not None:
							dx = raw_xy[0] - filtered_pos_xy[0]
							dy = raw_xy[1] - filtered_pos_xy[1]
							if (dx * dx + dy * dy) ** 0.5 > float(POS_JUMP_MAX_M):
								# Reject this update; do not add to trajectory
								raised = True
							else:
								raised = False
						else:
							raised = False

						if not raised:
							x_mean = sum(p[1] for p in pos_hist) / len(pos_hist)
							y_mean = sum(p[2] for p in pos_hist) / len(pos_hist)
							filtered_pos_xy = (float(x_mean), float(y_mean))
					else:
						filtered_pos_xy = raw_xy

					# Keep a short trajectory (filtered if enabled)
					try:
						xy_for_traj = filtered_pos_xy if (ENABLE_POS_FILTER and filtered_pos_xy is not None) else raw_xy
						traj_xy.append((float(xy_for_traj[0]), float(xy_for_traj[1])))
						if len(traj_xy) > TRAJ_MAX_POINTS:
							traj_xy = traj_xy[-TRAJ_MAX_POINTS:]
					except Exception:
						pass
				except ValueError:
					# Not enough measurements / unknown anchors yet
					pass

			now = time.time()
			if frame_is_ref and udp_sock is not None and _throttled(now, last_udp, UDP_RATE_HZ):
				last_udp = now
				pos_out = None
				ok = False
				rms = None
				it = None
				if last_solution is not None:
					ok = bool(last_solution.success)
					rms = float(last_solution.residual_rms)
					it = int(last_solution.iterations)
					p = last_solution.position
					pos_out = [float(p[0]), float(p[1]), float(p[2])]

				_udp_send(
					udp_sock,
					UDP_HOST,
					UDP_PORT,
					{
						"t": now,
						"ref": int(effective_ref),
						"src": int(parsed.src),
						"seq": int(parsed.seq),
						"anchors": {str(aid): [float(p[0]), float(p[1]), float(p[2])] for aid, p in anchors.items()},
						"remote_count": int(len(parsed.remotes) if parsed.remotes else 0),
						"meas": [{"id": int(rid), "dd": float(dd)} for rid, dd in measurements],
						"pos": pos_out,
						"ok": ok,
						"rms": rms,
						"it": it,
						"clockCorr": dbg.get("clock_correction_estimate"),
					},
				)

			if frame_is_ref and _throttled(now, last_print, PRINT_RATE_HZ):
				last_print = now
				if PRINT_MEASUREMENTS and measurements:
					meas_str = ", ".join([f"{rid}:{dd:+.3f}m" for rid, dd in measurements])
				else:
					meas_str = ""
				if last_solution is not None:
					pos_val = last_solution.position
					if ENABLE_POS_FILTER and filtered_pos_xy is not None:
						x_out, y_out = filtered_pos_xy
						z_out = float(pos_val[2])
					else:
						x_out, y_out, z_out = float(pos_val[0]), float(pos_val[1]), float(pos_val[2])
					print(
						f"ref={effective_ref:3d} seq={parsed.seq:3d} rem={len(parsed.remotes):2d} "
						f"meas={len(measurements):2d} "
						f"pos=({x_out: .3f}, {y_out: .3f}, {z_out: .3f}) "
						f"rms={last_solution.residual_rms:.3f} it={last_solution.iterations} ok={last_solution.success} "
						f"meas=[{meas_str}]"
					)
				else:
					print(
						f"ref={effective_ref:3d} seq={parsed.seq:3d} rem={len(parsed.remotes):2d} meas={len(measurements):2d} "
						f"clockCorr={dbg.get('clock_correction_estimate')} meas=[{meas_str}]"
					)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

