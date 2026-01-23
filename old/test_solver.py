from __future__ import annotations

import numpy as np

from tdoa_solver import solve_tdoa_position


def main() -> int:
    anchors = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([0.0, 3.0, 0.0]),
        4: np.array([0.0, 0.0, 3.0]),
    }
    true_pos = np.array([1.0, 1.2, 0.8])
    ref_id = 1

    measurements = []
    for anchor_id, pos in anchors.items():
        if anchor_id == ref_id:
            continue
        diff = np.linalg.norm(true_pos - anchors[ref_id]) - np.linalg.norm(true_pos - pos)
        measurements.append((anchor_id, diff))

    result = solve_tdoa_position(anchors, ref_id, measurements)
    error = np.linalg.norm(result.position - true_pos)

    print("Estimated:", result.position)
    print("True:", true_pos)
    print("Error:", error)

    if error > 1e-3:
        raise SystemExit("Solver error too large")

    anchors_2d = {
        1: np.array([0.0, 0.0, 0.5]),
        2: np.array([3.0, 0.0, 0.5]),
        3: np.array([0.0, 3.0, 0.5]),
    }
    true_pos_2d = np.array([1.2, 1.1, 0.5])
    measurements_2d = []
    for anchor_id, pos in anchors_2d.items():
        if anchor_id == ref_id:
            continue
        diff = np.linalg.norm(true_pos_2d - anchors_2d[ref_id]) - np.linalg.norm(true_pos_2d - pos)
        measurements_2d.append((anchor_id, diff))

    result_2d = solve_tdoa_position(
        anchors_2d,
        ref_id,
        measurements_2d,
        fixed_z=0.5,
    )
    error_2d = np.linalg.norm(result_2d.position - true_pos_2d)
    print("Estimated 2D:", result_2d.position)
    print("True 2D:", true_pos_2d)
    print("Error 2D:", error_2d)
    if error_2d > 1e-3:
        raise SystemExit("2D solver error too large")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
