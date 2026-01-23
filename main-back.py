from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from data_classes import *
import serial

from tdoa_solver import SPEED_OF_LIGHT, TdoaFrameMatcher, load_anchors, solve_tdoa_position

# -----------------------------
# User settings (edit here only)
# -----------------------------
PORT = "COM5"
BAUD_RATE = 115200
ANCHORS_JSON = "anchors.example.json"

ENABLE_SOLVE = True
FIXED_Z = 1.0  # set to None for full 3D solve
TICK_FREQUENCY_HZ = 63_897_600_000.0
PRINT_RATE_HZ = 2.0  # 0 disables throttling
ANTENNA_DELAY_TICKS = 16436
CLOCK_CORRECTION = 1.0
MATCH_HISTORY_SIZE = 8

START_BYTES = b"\xAA\x55"
POSITION_BLOCK_TYPE = 0xA1


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

prev_frames = []
matched_frames = []
def main() -> int:
	global prev_frames
	with serial.Serial(PORT, BAUD_RATE, timeout=0.1) as ser:
		for frame in iter_frames(ser):
			parsed = parse_frame(frame)
			if parsed is None:
				continue
			# save last 100 frames to prev frames
			prev_frames.append(parsed)
			if len(prev_frames) > 100:
				prev_frames.pop(0)
			# if prev_frames has frames with same src and seq as in remotes, print them
			remotes_matched = []
			for remote in parsed.remotes:
				for prev in prev_frames:
					if prev.src == remote.id and prev.seq == remote.seq:
						remotes_matched.append(prev)
			if len(remotes_matched) >= 2:
				matched_frames.append((parsed, remotes_matched))
				
				# break
			print(matched_frames[0] if matched_frames else None)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

({'src': 1, 'seq': 70, 'tx_ts': 869094912, 'rx_ts': 797654863818, 'remote_count': 2, 'remotes': [{'id': 3, 'seq': 116, 'rx_ts': 3525157670, 'dist_ticks': 16952}, {'id': 2, 'seq': 39, 'rx_ts': 290235772, 'dist_ticks': 16983}], 'position': (0.0, 0.0, 0.0)}, [{'src': 3, 'seq': 116, 'tx_ts': 603825664, 'rx_ts': 796015927155, 'remote_count': 2, 'remotes': [{'id': 1, 'seq': 68, 'rx_ts': 70442233, 'dist_ticks': 16956}, {'id': 2, 'seq': 37, 'rx_ts': 2955823877, 'dist_ticks': 17020}], 'position': (0.0, 0.0, 0.0)}, {'src': 2, 'seq': 39, 'tx_ts': 733267456, 'rx_ts': 797075982310, 'remote_count': 2, 'remotes': [{'id': 1, 'seq': 67, 'rx_ts': 2412552371, 'dist_ticks': 17003}, {'id': 3, 'seq': 116, 'rx_ts': 3968202633, 'dist_ticks': 17029}], 'position': (0.0, 0.0, 0.0)}])