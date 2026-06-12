"""Speaker enrollment: capture face samples and store one reusable profile.

Usage:
    python enroll.py                 # auto-capture ~20 samples, then save
    python enroll.py --name alice --samples 25
    python enroll.py --camera 1

Controls (in the preview window):
    SPACE  capture the current face manually
    A      toggle auto-capture on/off (on by default)
    R      reset / discard collected samples
    ENTER  finish early and save what's collected (need >= 5)
    ESC/Q  abort without saving

The profile is the L2-normalized mean of all sample embeddings, saved to
config recognition.profile_path (default data/speaker_profile.json).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.camera import apply_orientation, open_capture
from src.config import Config
from src.recognizer import build_recognizer


def largest_face(faces):
    faces = [f for f in faces if f.embedding is not None]
    return max(faces, key=lambda f: f.area) if faces else None


def main():
    ap = argparse.ArgumentParser(description="Enroll the single authorized speaker.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=None, help="speaker id label")
    ap.add_argument("--samples", type=int, default=20, help="target sample count (10-30)")
    ap.add_argument("--camera", type=int, default=None, help="override camera index")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    name = args.name or cfg.get("recognition.speaker_name", "speaker")
    target = max(10, min(30, args.samples))
    cam_index = args.camera if args.camera is not None else cfg.get("camera.index", 0)
    flip = bool(cfg.get("camera.flip_horizontal", True))

    print("[enroll] loading recognizer (first run downloads the model)...")
    rec = build_recognizer(
        det_size=cfg.get("recognition.det_size", 640),
        model_pack=cfg.get("recognition.model_pack", "buffalo_s"),
    )

    cap = open_capture(
        cam_index,
        cfg.get("camera.width", 1280),
        cfg.get("camera.height", 720),
        backend=cfg.get("camera.backend", "msmf"),
    )
    rotate = int(cfg.get("camera.rotate", 0))

    embeddings: list[np.ndarray] = []
    auto = True
    last_capture = 0.0
    capture_interval = 0.35  # seconds between auto-captures
    print(f"[enroll] enrolling '{name}', target {target} samples. SPACE=capture, "
          f"A=auto, R=reset, ENTER=save, ESC=abort")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = apply_orientation(frame, rotate, flip)
        faces = rec.detect(frame)
        face = largest_face(faces)
        single = face is not None and len(faces) == 1

        do_capture = False
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            do_capture = face is not None
        elif key in (ord("a"), ord("A")):
            auto = not auto
        elif key in (ord("r"), ord("R")):
            embeddings.clear()
        elif key in (13, 10):  # ENTER
            break
        elif key in (27, ord("q"), ord("Q")):
            print("[enroll] aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return

        now = time.time()
        if auto and single and (now - last_capture) >= capture_interval:
            do_capture = True
        if do_capture and face is not None and face.embedding is not None:
            embeddings.append(face.embedding.astype(np.float32))
            last_capture = now

        # --- HUD ---
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            color = (0, 200, 0) if f is face else (120, 120, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        msg = f"Samples: {len(embeddings)}/{target}   auto:{'ON' if auto else 'off'}"
        cv2.putText(frame, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        hint = "Hold steady, vary angle/expression slightly. ENTER to save."
        cv2.putText(frame, hint, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        if face is not None and len(faces) > 1:
            cv2.putText(frame, "Multiple faces - only ONE should be visible while enrolling",
                        (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        cv2.imshow("BENAX Enrollment", frame)

        if len(embeddings) >= target:
            print("[enroll] reached target sample count.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 5:
        raise SystemExit(f"[enroll] only {len(embeddings)} samples — need >= 5. Not saved.")

    arr = np.stack(embeddings)
    mean = arr.mean(axis=0)
    mean /= (np.linalg.norm(mean) + 1e-9)

    # Self-consistency: mean cosine similarity of samples to the template.
    sims = arr @ mean / (np.linalg.norm(arr, axis=1) + 1e-9)
    quality = float(sims.mean())

    profile_path = cfg.abspath(cfg.get("recognition.profile_path", "data/speaker_profile.json"))
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "speaker_id": name,
        "backend": rec.name,
        "dim": int(mean.shape[0]),
        "num_samples": len(embeddings),
        "quality_mean_cosine": round(quality, 4),
        "created": datetime.now().isoformat(timespec="seconds"),
        "embedding": mean.astype(float).tolist(),
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"[enroll] saved profile -> {profile_path}")
    print(f"[enroll] samples={len(embeddings)} quality(mean cos)={quality:.3f} "
          f"backend={rec.name}")
    if quality < 0.5:
        print("[enroll] WARNING: low self-consistency — re-enroll with steadier, "
              "better-lit captures for a stronger lock.")


if __name__ == "__main__":
    main()
