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
            "rotate": 0,
            "flip": False,
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


def vision_worker(cfg, use_mqtt, on_ready=None):
    threshold = float(cfg.get("recognition.similarity_threshold", 0.40))
    cam_index = cfg.get("camera.index", 0)
    publish_dt = 1.0 / max(1.0, float(cfg.get("tracking.publish_hz", 12)))
    # rotate/flip live-adjustable via the dashboard buttons (stored in shared).
    shared.update(rotate=int(cfg.get("camera.rotate", 0)),
                  flip=bool(cfg.get("camera.flip_horizontal", False)))

    profile_path = cfg.abspath(cfg.get("recognition.profile_path", "data/speaker_profile.json"))
    if not profile_path.exists():
        shared.update(status="NO PROFILE - run enroll.py", running=False)
        if on_ready:
            on_ready()
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

    # Open the camera BEFORE starting any background threads (Flask, paho loop):
    # on Windows the MSMF/DirectShow COM init can hang if other threads are
    # already running when the capture is created.
    cap = open_capture(cam_index, cfg.get("camera.width", 1280), cfg.get("camera.height", 720),
                       backend=cfg.get("camera.backend", "msmf"))

    # Camera is up — now it's safe to start the HTTP server thread.
    if on_ready:
        on_ready()

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

    shared.update(threshold=threshold, broker=broker, status="RUNNING")

    # Detection is the expensive step (ArcFace on CPU). Run it every Nth frame and
    # reuse the last result in between; capture + annotate + JPEG stream still run
    # every frame, so the displayed feed stays smooth instead of stalling at the
    # detector's rate. detect_every=1 restores the old detect-every-frame behaviour.
    detect_every = max(1, int(cfg.get("recognition.detect_every", 2)))
    last_pub, last_token, t_prev, fps = 0.0, None, time.time(), 0.0
    frame_i = 0
    faces, speaker, conf, recognized, cmd = [], None, 0.0, False, None
    try:
        while shared.data.get("running", True):
            ok, frame = cap.read()
            if not ok:
                continue
            frame = apply_orientation(frame, shared.data["rotate"], shared.data["flip"])
            h, w = frame.shape[:2]
            now = time.time()

            if frame_i % detect_every == 0:
                faces = rec.detect(frame)
                speaker, conf = best_match(faces, template, threshold)
                recognized = speaker is not None
                face_cx = speaker.center[0] if recognized else None
                cmd = tracker.update(face_cx, w)

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
            frame_i += 1

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
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
CFG = None  # set in main(); used by /api/save


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/rotate", methods=["POST"])
def api_rotate():
    with shared.lock:
        shared.data["rotate"] = (shared.data["rotate"] + 90) % 360
        r = shared.data["rotate"]
    return jsonify(ok=True, rotate=r)


@app.route("/api/flip", methods=["POST"])
def api_flip():
    with shared.lock:
        shared.data["flip"] = not shared.data["flip"]
        f = shared.data["flip"]
    return jsonify(ok=True, flip=f)


@app.route("/api/save", methods=["POST"])
def api_save():
    with shared.lock:
        r, f = shared.data["rotate"], shared.data["flip"]
    try:
        with open(CFG.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("camera", {})["rotate"] = int(r)
        data["camera"]["flip_horizontal"] = bool(f)
        with open(CFG.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return jsonify(ok=True, rotate=r, flip=f)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 500


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
    ap.add_argument("--camera", type=int, default=None, help="override camera index")
    args = ap.parse_args()

    global CFG
    cfg = Config.load(args.config)
    CFG = cfg
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera

    # The camera/vision loop runs on the MAIN thread (Windows MSMF/DirectShow
    # capture uses COM and hangs in worker threads). Flask runs in a daemon
    # thread, but only AFTER the camera is open (on_ready), so no background
    # thread is alive while the capture is being created.
    started = {"v": False}

    def start_server():
        if started["v"]:
            return
        started["v"] = True
        threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=args.port, threaded=True,
                                   debug=False, use_reloader=False),
            daemon=True).start()
        print(f"[dashboard] open  http://localhost:{args.port}  in your browser")

    try:
        vision_worker(cfg, not args.no_mqtt, on_ready=start_server)
    except KeyboardInterrupt:
        shared.update(running=False)
        print("\n[dashboard] stopped.")
        return
    except SystemExit as e:
        # e.g. camera busy/unavailable — keep serving so the UI shows the error.
        shared.update(status=f"CAMERA ERROR: {e}", running=False)
        print(f"[dashboard] {e}")
    # Keep the HTTP server alive after the vision loop ends so the page (or the
    # error status) stays reachable until the user stops it.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[dashboard] stopped.")


if __name__ == "__main__":
    main()
