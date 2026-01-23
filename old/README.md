# TDoA3 Solver

Python utility to read TDoA3 frames from a DW1000 UART stream and optionally estimate position from TDoA distance differences.

## Anchor file format

Create a JSON file (for example, `anchors.json`):

```json
{
  "anchors": {
    "1": [0.0, 0.0, 0.0],
    "2": [2.0, 0.0, 0.0],
    "3": [0.0, 2.0, 0.0],
    "4": [2.0, 2.0, 1.5]
  }
}
```

The solver uses anchor `src` in each frame as the reference anchor, and each remote entry provides the distance difference (in DW1000 ticks) relative to that reference.

## Usage

```bash
python main.py --port COM5 --baud 115200
```

With position solving:

```bash
python main.py --port COM5 --baud 115200 --solve --anchors anchors.json
```

Fixed-Z (2D) solving with 3 anchors:

```bash
python main.py --port COM5 --baud 115200 --solve --anchors anchors.json --fixed-z 0.5
```

Optional flags:
- `--tick-freq` to override the DW1000 tick frequency (Hz).
- `--fixed-z` to fix the Z coordinate and run a 2D solve (needs only 3 anchors).
- `--print-rate` to limit output frequency (prints per second).
- `--debug` to include matching internals in the JSON output.
- `--antenna-delay` to subtract antenna delay ticks when computing anchor-to-anchor time of flight.
- `--clock-correction` to apply a fixed clock correction (default 1.0).

## Output

Each UART frame is printed as JSON. When position solving is enabled, the output includes:
- `est_position`: estimated $[x, y, z]$ in meters
- `est_rms`: residual RMS in meters
- `est_iters`: number of solver iterations
- `est_ok`: whether the optimizer converged
