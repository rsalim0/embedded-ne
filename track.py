"""BENAX single-speaker tracking pipeline.

    capture -> detect faces -> recognize enrolled speaker (ignore others)
            -> compute horizontal error -> command -> MQTT publish -> log

Usage:
    python track.py                      # uses config.json + data/speaker_profile.json
    python track.py --no-mqtt            # vision only, no broker needed (dry run)
    python track.py --camera 1

Controls:  ESC / Q to quit.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import cv2
import numpy as np

from src.camera import apply_orientation, open_capture
from src.config import Config
from src.logger import EvidenceLogger
from src.mqtt_client import MqttPublisher
from src.recognizer import build_recognizer, cosine_similarity
from src.tracker import Tracker


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    emb = np.asarray(prof["embedding"], dtype=np.float32)
    emb /= (np.linalg.norm(emb) + 1e-9)
    return prof["speaker_id"], emb


def best_match(faces, template, threshold):
    """Return (face, confidence) for the enrolled speaker, ignoring all others."""
    best, best_sim = None, -1.0
    for f in faces:
        if f.embedding is None:
            continue
        sim = cosine_similarity(f.embedding, template)
        if sim > best_sim:
            best, best_sim = f, sim
    if best is not None and best_sim >= threshold:
        return best, best_sim
    return None, best_sim


def draw_hud(frame, faces, speaker, conf, cmd, recognized, mqtt_ok, fps):
    h, w = frame.shape[:2]
    cx = w // 2
    cv2.line(frame, (cx, 0), (cx, h), (60, 60, 60), 1)
    # Deadband band is drawn by caller via cmd.error only; keep simple here.
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        is_speaker = f is speaker
        color = (0, 220, 0) if is_speaker else (130, 130, 130)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_speaker else 1)
        label = "SPEAKER" if is_speaker else "ignored"
        cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    bar = [
        f"{'LOCKED' if recognized else 'SEARCHING'}",
        f"conf={conf:.2f}",
        f"status={cmd.status}",
        f"cmd={cmd.token}",
        f"mqtt={'OK' if mqtt_ok else 'OFF'}",
        f"{fps:4.1f}fps",
    ]
    col = (0, 220, 0) if recognized else (0, 165, 255)
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, "   ".join(bar), (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def _save_orientation(cfg, rotate, flip):
    """Persist the live-chosen rotate/flip back into config.json."""
    try:
        with open(cfg.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("camera", {})["rotate"] = int(rotate)
        data["camera"]["flip_horizontal"] = bool(flip)
        with open(cfg.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"[track] could not save orientation: {e}")


def main():
    ap = argparse.ArgumentParser(description="Single-speaker tracking + MQTT motor control.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--no-mqtt", action="store_true", help="run vision only, skip broker")
    ap.add_argument("--threshold", type=float, default=None, help="override similarity threshold")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    threshold = args.threshold if args.threshold is not None else cfg.get(
        "recognition.similarity_threshold", 0.40)
    cam_index = args.camera if args.camera is not None else cfg.get("camera.index", 0)
    flip = bool(cfg.get("camera.flip_horizontal", True))
    publish_hz = float(cfg.get("tracking.publish_hz", 12))
    publish_dt = 1.0 / max(1.0, publish_hz)

    profile_path = cfg.abspath(cfg.get("recognition.profile_path", "data/speaker_profile.json"))
    if not profile_path.exists():
        raise SystemExit(f"[track] no speaker profile at {profile_path}. Run enroll.py first.")
    speaker_id, template = load_profile(profile_path)
    print(f"[track] loaded speaker '{speaker_id}' (threshold {threshold:.2f})")

    rec = build_recognizer(
        det_size=cfg.get("recognition.det_size", 640),
        model_pack=cfg.get("recognition.model_pack", "buffalo_s"),
    )
    tracker = Tracker(cfg)
    logger = EvidenceLogger(cfg.abspath(cfg.get("logging.dir", "logs")))
    print(f"[track] logging to {logger.csv_path}")

    pub = None
    if not args.no_mqtt:
        pub = MqttPublisher(cfg)
        if not pub.connect(timeout=5.0):
            print("[track] WARNING: MQTT not connected yet — will keep retrying in background.")

    cap = open_capture(
        cam_index,
        cfg.get("camera.width", 1280),
        cfg.get("camera.height", 720),
        backend=cfg.get("camera.backend", "msmf"),
    )
    rotate = int(cfg.get("camera.rotate", 0))

    last_pub = 0.0
    last_cmd_token = None
    t_prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = apply_orientation(frame, rotate, flip)
            h, w = frame.shape[:2]

            faces = rec.detect(frame)
            speaker, conf = best_match(faces, template, threshold)
            recognized = speaker is not None

            face_cx = speaker.center[0] if recognized else None
            cmd = tracker.update(face_cx, w)

            now = time.time()
            # Publish at a fixed rate, and immediately whenever the token changes.
            if now - last_pub >= publish_dt or cmd.token != last_cmd_token:
                if pub is not None:
                    pub.publish_command(cmd.token)
                    pub.publish_status(json.dumps({
                        "speaker_id": speaker_id,
                        "recognized": recognized,
                        "confidence": round(conf, 3),
                        "status": cmd.status,
                        "command": cmd.token,
                        "error_norm": round(cmd.error_norm, 3),
                        "ts": datetime.now().isoformat(timespec="milliseconds"),
                    }))
                logger.log(
                    speaker_id=speaker_id,
                    recognized=recognized,
                    confidence=conf,
                    num_faces=len(faces),
                    error_norm=cmd.error_norm,
                    status=cmd.status,
                    command=cmd.token,
                )
                last_pub = now
                last_cmd_token = cmd.token

            # FPS (EMA)
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            mqtt_ok = pub.connected if pub is not None else False
            draw_hud(frame, faces, speaker, conf, cmd, recognized, mqtt_ok, fps)
            cv2.putText(frame, f"rotate={rotate}  [r]otate  [f]lip:{int(flip)}  [s]ave  [q]uit",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("BENAX Speaker Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("r"), ord("R")):
                rotate = (rotate + 90) % 360
                print(f"[track] rotate -> {rotate}")
            elif key in (ord("f"), ord("F")):
                flip = not flip
                print(f"[track] flip -> {flip}")
            elif key in (ord("s"), ord("S")):
                _save_orientation(cfg, rotate, flip)
                print(f"[track] saved rotate={rotate} flip={flip} to config.json")
    finally:
        if pub is not None:
            pub.publish_command("STOP")
            time.sleep(0.1)
            pub.close()
        logger.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[track] session ended. {logger.count} records -> {logger.csv_path}")


if __name__ == "__main__":
    main()
