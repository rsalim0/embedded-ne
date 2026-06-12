"""Robust camera opening — handles backend quirks (DSHOW vs MSMF) and rotation.

Some USB cameras (e.g. the external FalconEye here) only enumerate under the
Media Foundation (MSMF) backend, while the built-in webcam works under DirectShow
(DSHOW). This helper tries the configured backend first, then falls back, and
verifies it can actually grab a frame before returning the capture.
"""
from __future__ import annotations

import cv2

_BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "any": cv2.CAP_ANY,
}

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def open_capture(index: int, width: int, height: int, backend: str = "msmf"):
    """Open camera `index`, trying `backend` first then falling back.

    Returns an opened cv2.VideoCapture that has produced at least one frame,
    or raises SystemExit with a clear message.
    """
    order = [backend] + [b for b in ("msmf", "dshow", "any") if b != backend]
    for name in order:
        cap = cv2.VideoCapture(index, _BACKENDS.get(name, cv2.CAP_ANY))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[camera] opened index {index} via {name.upper()} "
                          f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
                    return cap
        cap.release()
    raise SystemExit(
        f"[camera] could not open camera index {index} on any backend. "
        f"Try a different camera.index in config.json or --camera N."
    )


def apply_orientation(frame, rotate: int = 0, flip_horizontal: bool = False):
    """Rotate (0/90/180/270) then optionally mirror a frame."""
    if rotate in _ROTATIONS:
        frame = cv2.rotate(frame, _ROTATIONS[rotate])
    if flip_horizontal:
        frame = cv2.flip(frame, 1)
    return frame
