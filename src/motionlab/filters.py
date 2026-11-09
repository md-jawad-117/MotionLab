import math


class CausalEMA:
    def __init__(self, time_constant_s=0.15, reset_gap_s=0.3):
        self.tau, self.gap = time_constant_s, reset_gap_s
        self.value = self.time = None

    def update(self, value, timestamp):
        if value is None or not math.isfinite(value):
            return None  # no interpolation or stale displayed angles
        if self.time is None or timestamp - self.time > self.gap:
            self.value = value
        else:
            dt = max(0, timestamp - self.time)
            alpha = 1 if self.tau == 0 else -math.expm1(-dt / self.tau)
            self.value += alpha * (value - self.value)
        self.time = timestamp
        return self.value
