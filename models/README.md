# models/ — add these two files manually (not tracked in git)

The recognizer's preferred offline backend needs:

| File | What | Source |
|------|------|--------|
| `face_landmarker.task` | MediaPipe Tasks FaceLandmarker (detect + 478 landmarks) | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| `embedder_arcface.onnx` | ArcFace embedder, NHWC `[N,112,112,3]` → 512-d | copy from the original PC (no canonical public URL) |

Drop both files in this folder. Without them, `build_recognizer()` falls back to
insightface (whose model download is unreliable on this network).
