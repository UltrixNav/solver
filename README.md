# TDoA3 Solver (minimal)

Minimal Python utility to read TDoA3 frames from UART and estimate the tag position.

## Anchor file format

Create a JSON file like `anchors.example.json`:

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

## Usage

Run the script (defaults to COM5):

```bash
python main.py
```

Edit the constants at the top of `main.py` to change the port, baud rate, or solver options.
