# CLAUDE.md — BENAX Single-Speaker Face-Tracking System

Guidance for Claude Code working in this repo. Read this first.

## What this project is

An AI-powered camera that locks onto **one pre-enrolled speaker**, ignores all
other faces, and rotates horizontally to keep that speaker centered.

- **PC (Python)** runs the vision pipeline: capture → detect faces → recognize the
  enrolled speaker (ArcFace embeddings) → compute horizontal error → emit a motor
  command → publish over MQTT → log evidence.
- **MQTT broker (Mosquitto)** on the PC relays commands over Wi-Fi/LAN.
- **ESP8266** subscribes to commands and drives a **servo** (camera pan).

Pipeline diagram: `docs/pipeline.md`. Wiring/power/safety: `docs/wiring_power.md`.

## Repository map

```
config.json              # ALL tunables (MQTT host/port, camera, thresholds, tracking)
enroll.py                # (a) enroll speaker -> data/speaker_profile.json
track.py                 # (b-e) main pipeline: recognize -> track -> command -> MQTT -> log
servo_test.py            # bring-up: drive servo over MQTT, no camera/model needed
src/
  config.py              # dotted-path config loader, ROOT-relative paths
  recognizer.py          # face detect + ArcFace embed; 3 backends (see below)
  tracker.py             # error -> LEFT/RIGHT/CENTER/STOP/SCAN + occlusion state machine
  mqtt_client.py         # paho-mqtt v2 publisher
  logger.py              # CSV + JSONL evidence logger (one row per published decision)
firmware/esp8266_servo/esp8266_servo.ino   # MQTT subscriber -> servo
tools/
  setup_broker.ps1       # write mosquitto.conf + firewall + run broker (foreground)
  mosquitto.conf         # listener 1884 0.0.0.0 ; allow_anonymous true
  fetch_pack.py          # RESUMABLE insightface model-pack downloader (network resets here)
  download_model.py      # fallback single-file ArcFace downloader
models/                  # face_landmarker.task + embedder_arcface.onnx  (NOT in git — add manually)
docs/                    # pipeline.md, wiring_power.md
data/                    # speaker_profile.json (created by enroll.py; gitignored)
logs/                    # session_*.csv / .jsonl (created by track.py; gitignored)
```

## How to run (in order)

```powershell
# 1. Broker (own terminal; keep open)
mosquitto -v -c tools\mosquitto.conf        # or: tools\setup_broker.ps1
# 2. Enroll the speaker (only that person in frame)
python enroll.py --name speaker --samples 20
# 3. Track
python track.py                             # ESC/Q to quit;  --no-mqtt for vision-only
# Servo bring-up without camera:
python servo_test.py                        # or --interactive
```

## Recognizer backends (`src/recognizer.py`)

`build_recognizer()` picks the first that works, in this order:
1. **`LandmarkerArcFaceBackend`** — PREFERRED, fully offline. MediaPipe Tasks
   FaceLandmarker (`models/face_landmarker.task`) for detect+landmarks +
   `_ArcFaceEmbedder` (`models/embedder_arcface.onnx`). The ONNX is **NHWC**
   (`[N,112,112,3]`) — `_ArcFaceEmbedder` auto-detects NHWC vs NCHW. Preprocess =
   align to the standard ArcFace 5-point template, `(rgb-127.5)/127.5`, L2-normalize.
2. **`InsightFaceBackend`** — needs an insightface model pack cached; downloads are
   UNRELIABLE on this network (resets), so this is a fallback only.
3. **`MeshArcFaceBackend`** — legacy `mp.solutions.face_mesh` + ArcFace ONNX.

Recognition = cosine similarity of each face's embedding vs the enrolled template;
accept the best match if `>= recognition.similarity_threshold` (default 0.40),
ignore everyone else.

## Command protocol (PC -> ESP, topic `benax/camera/command`)

`LEFT:<n>` `RIGHT:<n>` `CENTER` `STOP` `SCAN` — `<n>` is a proportional degree step.
Statuses logged/published to `benax/camera/status`: `MOVED_LEFT` `MOVED_RIGHT`
`CENTERED` `STOPPED` `OUT_OF_FRAME` `SEARCHING`.

## Environment facts that bit us (IMPORTANT — see also memory/)

- **This network forcibly resets large downloads** (WinError 10054 / aborted TCP).
  Anything >~15 MB from GitHub/CDN needs a resumable, retrying download:
  - insightface packs → `python tools/fetch_pack.py buffalo_s`
  - arduino-cli ESP8266 core → loop `arduino-cli core install esp8266:esp8266` until exit 0.
- **MQTT broker = `192.168.1.100:1884`** (NOT the usual 1883). The winget Mosquitto
  *Windows service* squats on 1883 localhost-only and can't be changed without admin,
  so our LAN broker runs on **1884** bound `0.0.0.0`. Host `192.168.1.100` is the PC's
  **Ethernet** IP; the Wi-Fi IP (192.168.1.200) drops when Wi-Fi disconnects — both on
  the same `192.168.1.x` subnet. **If you move to another PC, update `config.json`
  `mqtt.host` and the firmware `MQTT_HOST` to that PC's LAN IP, and `mqtt.port` if 1883
  is free there.**
- **No MSVC compiler** on the original PC — packages needing source builds fail.
- **ESP8266**: servo signal on **D4 (GPIO2)**, powered from **VIN** (small servo) or
  external 5V; **common ground** required. Board enumerates as a **CP210x** serial port
  (was COM8). Flash FQBN `esp8266:esp8266:nodemcuv2`. arduino-cli is at
  `C:\Program Files\Arduino CLI\arduino-cli.exe` (not on PATH).

## Conventions

- All tunables live in `config.json`; read via `src/config.py` `Config.load()` and
  `cfg.get("dotted.path", default)`. Don't hard-code hosts/ports/thresholds.
- Paths are resolved relative to repo ROOT via `cfg.abspath(...)`.
- Keep the ESP firmware dependency-light (no ArduinoJson); status strings are hand-built.
- Camera index in `config.json` (`camera.index`); override with `--camera N`.

## Moving to a new PC — checklist

1. `git clone` the repo.
2. **Add the models manually** — `models/` is NOT in git. Copy these two files into
   `models/` on the new PC (carry them on a USB stick, or from the original PC):
   - `models/face_landmarker.task`  (MediaPipe Tasks FaceLandmarker; also downloadable
     from Google: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`)
   - `models/embedder_arcface.onnx`  (ArcFace embedder, NHWC, 512-d — copy from the
     original PC; no canonical public URL)
   Without these, `build_recognizer()` falls back to insightface, whose download is
   unreliable on this network.
3. `pip install -r requirements.txt`.
4. Set `config.json` `mqtt.host` = new PC's LAN IP; set firmware `MQTT_HOST` to match.
5. Start broker; re-run `python enroll.py` (speaker profile is per-person, gitignored).
6. Flash firmware (update `WIFI_SSID`/`WIFI_PASS`), then `python track.py`.
