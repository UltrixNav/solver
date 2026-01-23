import argparse
import json
import struct
import time

import serial

from tdoa_solver import TdoaFrameMatcher, load_anchors, solve_tdoa_position


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
	# frame: [AA55][len u16][payload]
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
			{
				"id": anchor_id,
				"seq": seq_nr,
				"rx_ts": rx_anchor_ts,
				"dist_ticks": dist_ticks,
			}
		)
		offset += 8

	position = None
	if offset + 1 + 12 <= len(payload) and payload[offset] == POSITION_BLOCK_TYPE:
		position = struct.unpack_from("<fff", payload, offset + 1)

	return {
		"src": src,
		"seq": seq,
		"tx_ts": tx_ts,
		"rx_ts": rx_ts,
		"remote_count": remote_count,
		"remotes": remotes,
		"position": position,
	}


def main() -> int:
	parser = argparse.ArgumentParser(description="Read TDoA3 binary frames from UART")
	parser.add_argument("--port", default="COM5", help="Serial port (default: COM5)")
	parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
	parser.add_argument("--anchors", help="Path to anchors JSON file")
	parser.add_argument("--solve", action="store_true", help="Estimate position from TDoA data")
	parser.add_argument(
		"--fixed-z",
		type=float,
		default=None,
		help="Fix Z coordinate for 2D solving (requires only 3 anchors)",
	)
	parser.add_argument(
		"--tick-freq",
		type=float,
		default=63_897_600_000.0,
		help="DW1000 tick frequency in Hz (default: 63.8976e9)",
	)
	parser.add_argument(
		"--antenna-delay",
		type=int,
		default=16436,
		help="Antenna delay in DW1000 ticks (default: 16436)",
	)
	parser.add_argument(
		"--clock-correction",
		type=float,
		default=1.0,
		help="Clock correction multiplier for remote anchors (default: 1.0)",
	)
	parser.add_argument(
		"--print-rate",
		type=float,
		default=2.0,
		help="Max JSON prints per second (default: 2.0, 0 = no limit)",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Include debug fields for TDoA matching and solver inputs",
	)
	args = parser.parse_args()

	anchors = None
	last_position = {}
	matcher = None
	if args.solve:
		if not args.anchors:
			raise SystemExit("--anchors is required when --solve is enabled")
		anchors = load_anchors(args.anchors)
		matcher = TdoaFrameMatcher(
			tick_frequency_hz=args.tick_freq,
			antenna_delay_ticks=args.antenna_delay,
			clock_correction=args.clock_correction,
		)

	last_print_time = 0.0
	min_print_interval = 0.0 if args.print_rate <= 0 else 1.0 / args.print_rate

	with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
		for frame in iter_frames(ser):
			parsed = parse_frame(frame)
			if parsed is None:
				continue

			if args.solve and anchors and matcher:
				reference_id = parsed["src"]
				if args.debug:
					measurements, debug_info = matcher.update_with_debug(parsed)
					parsed["debug"] = debug_info
				else:
					measurements = matcher.update(parsed)
				min_required = 2 if args.fixed_z is not None else 3
				if reference_id in anchors and len(measurements) >= min_required:
					initial = last_position.get(reference_id)
					try:
						result = solve_tdoa_position(
							anchors,
							reference_id,
							measurements,
							initial_position=initial,
							fixed_z=args.fixed_z,
						)
						parsed["est_position"] = [round(v, 6) for v in result.position.tolist()]
						parsed["est_rms"] = round(result.residual_rms, 6)
						parsed["est_iters"] = result.iterations
						parsed["est_ok"] = result.success
						last_position[reference_id] = result.position
					except ValueError:
						parsed["est_position"] = None
				else:
					parsed["est_position"] = None

			now = time.time()
			if min_print_interval == 0.0 or (now - last_print_time) >= min_print_interval:
				print(json.dumps(parsed, separators=(",", ":")))
				last_print_time = now

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
