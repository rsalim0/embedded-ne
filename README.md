# BENAX — AI-Powered Single-Speaker Face Recognition & Camera Tracking

Locks onto **one pre-enrolled speaker**, ignores every other face in frame, and
rotates a camera horizontally to keep that speaker centered. A PC runs the AI
pipeline and publishes motor commands over **MQTT/Wi-Fi**; an **ESP8266** drives
the **servo**.

```
USB Camera → [PC: detect → recognize speaker → track → command] → MQTT → ESP8266 → Servo → camera pans
```

See **`docs/pipeline.md`** (Recognize→Track→Command flowchart) and
**`docs/wiring_power.md`** (wiring, power architecture, safety).

---

## 1. Repository layout

```
embedded/
├─ config.json                 # all tunables: MQTT, camera, thresholds, tracking
├─ enroll.py                   # (a) speaker enrollment → data/speaker_profile.json
├─ track.py                    # (b–e) recognize → track → command → MQTT → log (OpenCV window)
├─ dashboard.py                # same pipeline as track.py, served as a live web UI (Flask)
├─ templates/dashboard.html    # the dashboard front-end
├─ servo_test.py               # bring-up: drive the servo over MQTT, no camera needed
├─ requirements.txt
├─ src/
│  ├─ config.py                # config loader
│  ├─ recognizer.py            # face detect + ArcFace embedding (insightface | fallback)
│  ├─ tracker.py               # error → LEFT/RIGHT/CENTER/STOP/SCAN + scan state machine
│  ├─ mqtt_client.py           # paho-mqtt publisher
│  └─ logger.py                # CSV + JSONL evidence logger
├─ firmware/esp8266_servo/
│  └─ esp8266_servo.ino        # MQTT subscriber → servo on D4/GPIO2
├─ tools/
│  ├─ setup_broker.ps1         # configure + run local Mosquitto for LAN access
│  └─ download_model.py        # fetch ArcFace ONNX (fallback recognizer)
├─ docs/  (pipeline.md, wiring_power.md)
├─ data/  (speaker_profile.json — created by enroll.py)
└─ logs/  (session_*.csv / .jsonl — created by track.py)
```

---

## 2. One-time setup

### 2.1 Python deps
Already present on this machine: OpenCV, NumPy, MediaPipe, onnxruntime, paho-mqtt.
The strong recognizer is `insightface`:

```powershell
python -m pip install -r requirements.txt
```

**Recognition models (offline, already in `models/`).** The recognizer uses two
local model files and needs **no download**:

- `models/face_landmarker.task` — MediaPipe Tasks FaceLandmarker (detect + 478 landmarks)
- `models/embedder_arcface.onnx` — ArcFace embedder (512-d, NHWC input)

`build_recognizer()` prefers this offline path automatically. It falls back to
**insightface** (`config.json → recognition.model_pack`, default `buffalo_s`)
only if those local files are missing. On networks that reset large downloads,
fetch an insightface pack resumably with:

```powershell
python tools\fetch_pack.py buffalo_s    # resumable; extracts into the insightface cache
```

### 2.4 Hardware bring-up (before the full pipeline)
With the broker running and the ESP8266 flashed + wired, validate the servo alone:

```powershell
python servo_test.py                  # LEFT/CENTER/RIGHT/SCAN motion sequence
python servo_test.py --interactive    # type commands by hand
```

### 2.2 MQTT broker (local Mosquitto)
The PC and the ESP8266 talk through a Mosquitto broker on this PC
(`192.168.1.100:1884`). 192.168.1.100 is the PC's Ethernet LAN IP (the stable
interface — the Wi-Fi IP 192.168.1.200 is currently disconnected; both are on the
same 192.168.1.x subnet, so the ESP8266 on Wi-Fi still reaches it via the router).
Port 1884 is used instead of the usual 1883 because the winget-installed Mosquitto
Windows *service* already holds 1883 but only on localhost; our LAN-facing broker
runs on 1884 bound to 0.0.0.0. Default Mosquitto only listens on localhost, so use
the helper:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_broker.ps1
```

It writes `tools\mosquitto.conf` (`listener 1884 0.0.0.0`, `allow_anonymous true`),
opens TCP 1884, and starts the broker in the foreground (Ctrl+C to stop).

### 2.3 ESP8266 firmware
Edit the config block at the top of `firmware/esp8266_servo/esp8266_servo.ino`:
`WIFI_SSID`, `WIFI_PASS`, and confirm `MQTT_HOST = "192.168.1.100"`.
Flash with arduino-cli (installed) or the Arduino IDE:

```powershell
arduino-cli config init
arduino-cli config add board_manager.additional_urls http://arduino.esp8266.com/stable/package_esp8266com_index.json
arduino-cli core update-index
arduino-cli core install esp8266:esp8266
arduino-cli lib install "PubSubClient"
# plug the board in, find the COM port:
arduino-cli board list
arduino-cli compile  --fqbn esp8266:esp8266:nodemcuv2 firmware\esp8266_servo
arduino-cli upload -p COM5 --fqbn esp8266:esp8266:nodemcuv2 firmware\esp8266_servo
```
Wire the servo per `docs/wiring_power.md` (signal→D4/GPIO2, V+→VIN/external 5V,
GND→common ground).

---

## 3. Running the system

**Terminal 1 — broker**
```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_broker.ps1
```

**Step (a) — enroll the speaker** (only the speaker should be in frame):
```powershell
python enroll.py --name speaker --samples 20
```
Vary angle/expression slightly; it saves `data/speaker_profile.json`
(L2-normalized mean of 10–30 ArcFace embeddings) and reports a quality score.

**Steps (b–e) — track:**
```powershell
python track.py
```
On-screen HUD shows LOCKED/SEARCHING, confidence, status, the published command,
MQTT state and FPS. The recognized speaker is boxed **green**; all other faces are
boxed grey and labelled *ignored*. ESC/Q to quit.

Vision-only dry run (no broker/board needed):
```powershell
python track.py --no-mqtt
```

### Web dashboard (Flask) — alternative to the OpenCV window

`dashboard.py` runs the **same** capture → recognize → track → command → MQTT → log
pipeline as `track.py`, but renders to a browser instead of an OpenCV window: live
MJPEG feed, recognition/servo/log panels, and buttons to rotate/flip the camera and
save the orientation back to `config.json`. Run it **instead of** `track.py` (both
need the camera).

```powershell
python dashboard.py                 # then open http://localhost:5000
python dashboard.py --no-mqtt       # vision only, no broker/board
python dashboard.py --port 8000     # serve on a different port
python dashboard.py --camera 1      # override config.json camera.index
```

It serves on `0.0.0.0`, so other devices on the LAN can watch at
`http://<PC-LAN-IP>:5000`. Stop it with **Ctrl+C** in the terminal.

> **Smoothness:** the feed streams every captured frame while face detection (the
> expensive step) runs every Nth frame — set by `recognition.detect_every` in
> `config.json` (default `2`). Raise it (e.g. `3`–`4`) for a smoother feed on slow
> CPUs; set `1` to detect on every frame.

---

## 4. Quick MQTT verification (no camera)

With the broker running, in another terminal:
```powershell
# watch what the PC publishes / the ESP receives
mosquitto_sub -h 192.168.1.100 -p 1884 -t "benax/camera/#" -v
# manually drive the servo to prove the board works
mosquitto_pub -h 192.168.1.100 -p 1884 -t "benax/camera/command" -m "RIGHT:5"
mosquitto_pub -h 192.168.1.100 -p 1884 -t "benax/camera/command" -m "SCAN"
mosquitto_pub -h 192.168.1.100 -p 1884 -t "benax/camera/command" -m "CENTER"
```
The ESP also publishes a heartbeat (`{"angle":..,"mode":..}`) to
`benax/camera/status`.

---

## 5. Evidence logging

`track.py` writes one record per published decision to
`logs/session_<timestamp>.csv` **and** `.jsonl`:

| column | meaning |
|--------|---------|
| `timestamp_iso`, `epoch` | when |
| `speaker_id` | enrolled speaker label |
| `recognized` | 1 if the enrolled speaker was locked this frame |
| `confidence` | cosine similarity to the stored template |
| `num_faces` | total faces detected (proof others were present + ignored) |
| `error_norm` | smoothed horizontal error (−1 left … +1 right) |
| `status` | MOVED_LEFT/RIGHT, CENTERED, STOPPED, OUT_OF_FRAME, SEARCHING |
| `command` | exact MQTT payload sent to the ESP (LEFT:n/RIGHT:n/CENTER/STOP/SCAN) |

These logs are the operational evidence for the demonstration scenarios below.

---

## 6. Demonstration scenarios (what to show the assessor)

1. **Multiple faces** — bring co-presenters into frame; only the enrolled speaker
   is boxed green and tracked; others are *ignored* (visible in HUD + `num_faces` > 1
   while commands still follow the speaker).
2. **Occlusion / re-acquire** — block the speaker briefly → `STOPPED`; longer →
   `SEARCHING` (`SCAN`, ESP sweeps). Reveal the **same** speaker → lock resumes.
3. **Lateral movement** — speaker walks across frame → `MOVED_LEFT`/`MOVED_RIGHT`
   with proportional step; camera pans to re-center; `CENTERED` in the deadband.

---

## 7. Tuning (`config.json`)

| key | effect |
|-----|--------|
| `recognition.similarity_threshold` | higher = stricter lock (fewer false accepts). 0.40 default; raise if other people get accepted, lower if the real speaker drops out. |
| `recognition.detect_every` | run detection every Nth frame (dashboard feed smoothness). 2 default; raise for a smoother feed on slow CPUs, 1 = every frame. |
| `tracking.deadband_frac` | central dead-zone (no motion) as a fraction of half-width. |
| `tracking.max_step_deg` / `min_step_deg` | per-command servo step range. |
| `tracking.smoothing` | 0–1 jitter filter; higher = smoother but slower. |
| `tracking.invert_direction` | flip if the camera pans the wrong way. |
| `tracking.lost_grace_frames` / `scan_after_frames` | how long to hold before searching. |
| `camera.flip_horizontal` | mirror preview (selfie view). |

---

## 8. Troubleshooting

- **Camera won't open** → try `--camera 1`; close other apps using the webcam.
- **Speaker never locks** → re-run `enroll.py` with better lighting; lower the
  threshold a little; check enrollment quality score.
- **Other people get accepted** → raise `similarity_threshold`.
- **ESP not connecting** → confirm broker started with `setup_broker.ps1` (0.0.0.0
  listener + firewall), and the ESP `MQTT_HOST` matches the PC LAN IP; check Serial @115200.
- **Camera pans the wrong way** → set `invert_direction: true`.
- **insightface missing** → fallback runs automatically; run `tools\download_model.py`.
