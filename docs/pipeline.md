# Recognize → Track → Command Pipeline

## Block diagram

```
┌──────────────────────────────────────── PC (Python AI pipeline) ─────────────────────────────────────────┐
│                                                                                                           │
│   USB Camera ──► Frame Capture ──► Face Detection ──► For each face: ArcFace Embedding                     │
│   (OpenCV)        (OpenCV)         (InsightFace/        │                                                  │
│                                     MediaPipe)          ▼                                                  │
│                                              Cosine similarity vs ENROLLED TEMPLATE                        │
│                                                         │                                                  │
│                                          ┌──────────────┴───────────────┐                                 │
│                                          ▼                              ▼                                  │
│                                  sim ≥ threshold?                 all other faces                          │
│                                  pick best match                  ── IGNORED ──                            │
│                                          │                                                                 │
│                          ┌───────────────┴───────────────┐                                                │
│                          ▼ YES (LOCKED)                   ▼ NO (lost)                                      │
│              horizontal error =                  state machine:                                            │
│              face_cx − frame_cx                  grace → STOP                                              │
│                          │                       longer  → SCAN                                            │
│                          ▼                                │                                                │
│              EMA smoothing + deadband                     │                                                │
│                          │                                │                                                │
│                          ▼                                ▼                                                │
│              LEFT:n / RIGHT:n / CENTER          STOP / SCAN                                                │
│                          └───────────────┬───────────────┘                                                │
│                                          ▼                                                                 │
│                          MQTT publish  +  CSV/JSONL evidence log                                          │
│                          (paho-mqtt)      (speaker id, confidence, ts, command)                            │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                            │  Wi-Fi / MQTT  (topic: benax/camera/command)
                                            ▼
┌──────────────────────────────── Mosquitto Broker (PC, 192.168.1.100:1884) ───────────────────────────────┐
└──────────────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                            │  Wi-Fi
                                            ▼
┌──────────────────────────────────────── ESP8266 (subscriber) ────────────────────────────────────────────┐
│   Parse command ──► update target angle ──► ease current→target (smooth) ──► Servo signal (D4/GPIO2)       │
│                     SCAN = autonomous sweep                          publishes heartbeat → status topic    │
└───────────────────────────────────────────────────────────────────────┬───────────────────────────────────┘
                                                                         ▼
                                                              Servo rotates camera horizontally
```

## Command protocol (PC → ESP8266, topic `benax/camera/command`)

| Payload      | Meaning                          | Servo action                          |
|--------------|----------------------------------|---------------------------------------|
| `RIGHT:<n>`  | speaker right of center          | target angle += n° (n = 1..10)        |
| `LEFT:<n>`   | speaker left of center           | target angle −= n°                    |
| `CENTER`     | within deadband                  | move toward 90°                       |
| `STOP`       | hold (brief loss / centered)     | hold current angle                    |
| `SCAN`       | speaker lost beyond grace        | autonomous sweep 30°↔150° until next cmd |

Step size `n` is **proportional** to how far the speaker is outside the deadband,
so corrections are large when far off-center and gentle when nearly centered.

## Status vocabulary (logged + published to `benax/camera/status`)

`MOVED_LEFT` · `MOVED_RIGHT` · `CENTERED` · `STOPPED` · `OUT_OF_FRAME` · `SEARCHING`

## Robustness / re-acquisition

1. Speaker recognized every frame by ArcFace cosine similarity ≥ threshold.
2. Brief occlusion (≤ `lost_grace_frames`) → `STOP`, camera holds position.
3. Longer loss (≥ `scan_after_frames`) → `SCAN`, ESP sweeps to search.
4. When the **same enrolled** speaker re-appears and clears threshold → lock
   resumes; smoothed error is decayed during loss so re-acquire is not jerky.
