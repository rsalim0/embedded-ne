"""BENAX web dashboard — live camera feed + recognition status + servo + logs.

Runs the SAME pipeline as track.py (capture -> recognize -> track -> MQTT -> log)
but renders to a browser UI instead of an OpenCV window. Open http://localhost:5000.

Usage:
    python dashboard.py                 # uses config.json + data/speaker_profile.json
    python dashboard.py --no-mqtt       # vision only (no broker/servo)
    python dashboard.py --port 8000

Run this INSTEAD of track.py (they both need the camera). ESC the browser tab to stop,
then Ctrl+C the server.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify, render_template

from src.camera import apply_orientation, open_capture
from src.config import Config
from src.logger import EvidenceLogger
from src.mqtt_client import MqttPublisher
from src.recognizer import build_recognizer, cosine_similarity
from src.tracker import Tracker


# --------------------------------------------------------------------------- #
# Shared state between the vision worker and the Flask routes
# --------------------------------------------------------------------------- #
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.data = {
            "recognized": False,
            "speaker_id": "-",
            "confidence": 0.0,
            "status": "STARTING",
            "command": "-",
            "error_norm": 0.0,
            "num_faces": 0,
            "fps": 0.0,
            "mqtt": False,
            "esp_angle": None,
            "esp_mode": "-",
            "threshold": 0.4,
            "broker": "-",
            "running": True,
        }
        self.logs = deque(maxlen=60)

    def update(self, **kw):
        with self.lock:
            self.data.update(kw)

    def set_jpeg(self, b):
        with self.lock:
            self.jpeg = b

    def add_log(self, row):
        with self.lock:
            self.logs.appendleft(row)

    def snapshot(self):
        with self.lock:
            return dict(self.data), list(self.logs)


shared = Shared()


def best_match(faces, template, threshold):
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


def annotate(frame, faces, speaker, conf, status, recognized):
    h, w = frame.shape[:2]
    cx = w // 2
    cv2.line(frame, (cx, 0), (cx, h), (70, 70, 70), 1)
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        is_sp = f is speaker
        color = (0, 220, 0) if is_sp else (140, 140, 140)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_sp else 1)
        cv2.putText(frame, "SPEAKER" if is_sp else "ignored",
                    (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if is_sp else 1)
    tag = f"{'LOCKED' if recognized else 'SEARCHING'}  conf={conf:.2f}  {status}"
    col = (0, 220, 0) if recognized else (0, 165, 255)
    cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(frame, tag, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return frame


def vision_worker(cfg, use_mqtt):
    threshold = float(cfg.get("recognition.similarity_threshold", 0.40))
    cam_index = cfg.get("camera.index", 0)
    rotate = int(cfg.get("camera.rotate", 0))
    flip = bool(cfg.get("camera.flip_horizontal", False))
    publish_dt = 1.0 / max(1.0, float(cfg.get("tracking.publish_hz", 12)))

    profile_path = cfg.abspath(cfg.get("recognition.profile_path", "data/speaker_profile.json"))
    if not profile_path.exists():
        shared.update(status="NO PROFILE - run enroll.py", running=False)
        return
    with open(profile_path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    speaker_id = prof["speaker_id"]
    template = np.asarray(prof["embedding"], dtype=np.float32)
    template /= (np.linalg.norm(template) + 1e-9)

    rec = build_recognizer(det_size=cfg.get("recognition.det_size", 640),
                           model_pack=cfg.get("recognition.model_pack", "buffalo_s"))
    tracker = Tracker(cfg)
    logger = EvidenceLogger(cfg.abspath(cfg.get("logging.dir", "logs")))

    pub = None
    sub = None
    broker = f"{cfg.get('mqtt.host')}:{cfg.get('mqtt.port')}"
    if use_mqtt:
        pub = MqttPublisher(cfg)
        pub.connect(timeout=4.0)
        # Separate subscriber for ESP status (servo angle / mode).
        sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="benax-dash-sub")

        def on_msg(c, u, m):
            try:
                d = json.loads(m.payload.decode())
                shared.update(esp_angle=d.get("angle"), esp_mode=d.get("mode", "-"))
            except Exception:
                pass

        def on_con(c, u, fl, rc, p=None):
            c.subscribe(cfg.get("mqtt.status_topic", "benax/camera/status"))
        sub.on_message = on_msg
        sub.on_connect = on_con
        try:
            sub.connect(cfg.get("mqtt.host"), int(cfg.get("mqtt.port")), 30)
            sub.loop_start()
        except Exception:
            sub = None

    cap = open_capture(cam_index, cfg.get("camera.width", 1280), cfg.get("camera.height", 720),
                       backend=cfg.get("camera.backend", "msmf"))
    shared.update(threshold=threshold, broker=broker, status="RUNNING")

    last_pub, last_token, t_prev, fps = 0.0, None, time.time(), 0.0
    try:
        while shared.data.get("running", True):
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
            if now - last_pub >= publish_dt or cmd.token != last_token:
                if pub is not None:
                    pub.publish_command(cmd.token)
                logger.log(speaker_id=speaker_id, recognized=recognized, confidence=conf,
                           num_faces=len(faces), error_norm=cmd.error_norm,
                           status=cmd.status, command=cmd.token)
                shared.add_log({
                    "t": datetime.now().strftime("%H:%M:%S"),
                    "status": cmd.status, "command": cmd.token,
                    "confidence": round(conf, 3), "faces": len(faces),
                    "recognized": recognized,
                })
                last_pub, last_token = now, cmd.token

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            annotate(frame, faces, speaker, conf, cmd.status, recognized)
            okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if okj:
                shared.set_jpeg(buf.tobytes())
            shared.update(recognized=recognized, speaker_id=speaker_id,
                          confidence=round(conf, 3), status=cmd.status, command=cmd.token,
                          error_norm=round(cmd.error_norm, 3), num_faces=len(faces),
                          fps=round(fps, 1), mqtt=(pub.connected if pub else False))
    finally:
        if pub is not None:
            pub.publish_command("STOP")
            pub.close()
        if sub is not None:
            sub.loop_stop()
        logger.close()
        cap.release()


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    data, logs = shared.snapshot()
    return jsonify({"state": data, "logs": logs})


@app.route("/video")
def video():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with shared.lock:
                jpg = shared.jpeg
            if jpg is not None:
                yield boundary + jpg + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    ap = argparse.ArgumentParser(description="BENAX tracking web dashboard.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    worker = threading.Thread(target=vision_worker, args=(cfg, not args.no_mqtt), daemon=True)
    worker.start()

    print(f"[dashboard] open  http://localhost:{args.port}  in your browser")
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
