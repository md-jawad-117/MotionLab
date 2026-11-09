import numpy as np


def angle(a, b, c):
    """Interior angle in degrees; invalid/degenerate geometry stays missing."""
    points = np.asarray([a, b, c], dtype=float)
    if not np.isfinite(points).all():
        return None
    u, v = points[0] - points[1], points[2] - points[1]
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-10:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1))))
