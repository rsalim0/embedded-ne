"""Convert speaker tracking error into motor commands.

Pipeline role:  recognized-face-bbox  ->  horizontal error  ->  command token.

Commands published to the ESP8266 (command_topic):
    LEFT:<deg>   RIGHT:<deg>   CENTER   STOP   SCAN

Human-readable statuses (status_topic / logs):
    MOVED_LEFT   MOVED_RIGHT   CENTERED   STOPPED   OUT_OF_FRAME   SEARCHING

Sign convention (default, invert with tracking.invert_direction):
    speaker to the RIGHT of frame center  -> camera rotates RIGHT  -> "RIGHT"
    speaker to the LEFT  of frame center  -> camera rotates LEFT   -> "LEFT"
Whether that re-centers the speaker depends on how the servo horn faces the
scene; flip `invert_direction` once during bring-up if motion runs the wrong way.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Command:
    token: str       # payload sent to ESP, e.g. "RIGHT:4", "STOP", "SCAN"
    status: str      # human status for logs/HUD, e.g. "MOVED_RIGHT"
    error_norm: float  # smoothed normalized error in [-1, 1] (0 when no target)


class Tracker:
    def __init__(self, cfg):
        t = cfg["tracking"]
        self.deadband = float(t["deadband_frac"])
        self.max_step = int(t["max_step_deg"])
        self.min_step = int(t["min_step_deg"])
        self.smoothing = float(t["smoothing"])      # 0..1, higher = smoother/slower
        self.invert = bool(t["invert_direction"])
        self.lost_grace = int(t["lost_grace_frames"])
        self.scan_after = int(t["scan_after_frames"])

        self._ema = 0.0
        self._lost = 0

    def reset(self):
        self._ema = 0.0
        self._lost = 0

    def update(self, face_center_x: float | None, frame_width: int) -> Command:
        """Advance the state machine one frame and return the command to publish."""
        if face_center_x is None:
            return self._on_lost()

        self._lost = 0
        half = frame_width / 2.0
        raw = (face_center_x - half) / half            # -1 (left) .. +1 (right)
        raw = max(-1.0, min(1.0, raw))
        # Exponential moving average to suppress jitter.
        self._ema = self.smoothing * self._ema + (1.0 - self.smoothing) * raw
        err = self._ema
        if self.invert:
            err = -err

        if abs(err) <= self.deadband:
            return Command("CENTER", "CENTERED", self._ema)

        # Proportional step (deg), scaled by how far outside the deadband we are.
        span = max(1e-6, 1.0 - self.deadband)
        mag = (abs(err) - self.deadband) / span        # 0..1
        step = int(round(self.min_step + mag * (self.max_step - self.min_step)))
        step = max(self.min_step, min(self.max_step, step))

        if err > 0:
            return Command(f"RIGHT:{step}", "MOVED_RIGHT", self._ema)
        return Command(f"LEFT:{step}", "MOVED_LEFT", self._ema)

    def _on_lost(self) -> Command:
        self._lost += 1
        # Decay the remembered error so we don't lurch on re-acquire.
        self._ema *= 0.8
        if self._lost <= self.lost_grace:
            return Command("STOP", "STOPPED", self._ema)
        if self._lost >= self.scan_after:
            return Command("SCAN", "SEARCHING", self._ema)
        # Between grace and scan thresholds: hold position, flag loss.
        return Command("STOP", "OUT_OF_FRAME", self._ema)
