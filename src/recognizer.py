"""Face detection + ArcFace embedding for single-speaker recognition.

Two interchangeable backends behind one interface:
  * InsightFaceBackend  - primary. Uses insightface FaceAnalysis (buffalo_l).
  * MeshArcFaceBackend  - fallback. MediaPipe FaceMesh for detection+alignment,
                          ArcFace ONNX (located from the insightface cache or
                          models/w600k_r50.onnx) for the embedding.

Both return a list of `Face` objects with an L2-normalized 512-d embedding so the
rest of the pipeline (cosine similarity vs the enrolled template) is backend-agnostic.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Standard ArcFace 5-point destination template for a 112x112 aligned crop,
# ordered [left_eye, right_eye, nose, left_mouth, right_mouth] in image coords.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass
class Face:
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2)
    embedding: np.ndarray | None         # L2-normalized 512-d, or None
    det_score: float
    kps: np.ndarray | None = None        # 5x2 keypoints if available

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors (robust if not)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# Primary backend: insightface
# --------------------------------------------------------------------------- #
class InsightFaceBackend:
    name = "insightface"

    def __init__(self, det_size: int = 640, model_pack: str = "buffalo_s"):
        from insightface.app import FaceAnalysis  # imported lazily
        self.name = f"insightface:{model_pack}"
        self.app = FaceAnalysis(
            name=model_pack, providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        faces = []
        for f in self.app.get(frame_bgr):
            x1, y1, x2, y2 = f.bbox.astype(int)
            faces.append(
                Face(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    embedding=np.asarray(f.normed_embedding, dtype=np.float32),
                    det_score=float(f.det_score),
                    kps=np.asarray(f.kps, dtype=np.float32) if f.kps is not None else None,
                )
            )
        return faces


# --------------------------------------------------------------------------- #
# Fallback backend: MediaPipe FaceMesh + ArcFace ONNX
# --------------------------------------------------------------------------- #
def _find_arcface_onnx() -> str | None:
    """Locate w600k_r50.onnx in the project or the insightface model cache."""
    candidates = []
    env = os.environ.get("ARCFACE_ONNX")
    if env:
        candidates.append(env)
    root = Path(__file__).resolve().parent.parent
    # Locally provided ArcFace embedders (preferred), then insightface cache.
    candidates.append(str(root / "models" / "embedder_arcface.onnx"))
    candidates.append(str(root / "models" / "w600k_r50.onnx"))
    candidates += glob.glob(str(root / "models" / "*arcface*.onnx"))
    home = Path.home() / ".insightface" / "models"
    # Recognition models across packs: w600k_r50 (buffalo_l), w600k_mbf (buffalo_s).
    for pat in ("w600k_r50.onnx", "w600k_mbf.onnx", "*w600k*.onnx", "*r50*.onnx", "*mbf*.onnx"):
        candidates += glob.glob(str(home / "**" / pat), recursive=True)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _find_landmarker_task() -> str | None:
    """Locate the MediaPipe Tasks FaceLandmarker bundle (.task)."""
    root = Path(__file__).resolve().parent.parent
    env = os.environ.get("FACE_LANDMARKER_TASK")
    for c in ([env] if env else []) + [
        str(root / "models" / "face_landmarker.task"),
        *glob.glob(str(root / "models" / "*landmark*.task")),
    ]:
        if c and os.path.isfile(c):
            return c
    return None


class _ArcFaceEmbedder:
    """ArcFace ONNX wrapper that auto-adapts to NHWC or NCHW input layout."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self._inp = inp.name
        self._out = self.session.get_outputs()[0].name
        shape = inp.shape  # e.g. [N,112,112,3] (NHWC) or [N,3,112,112] (NCHW)
        self.nhwc = len(shape) == 4 and shape[-1] == 3
        self.path = onnx_path

    def embed(self, frame_bgr: np.ndarray, src5: np.ndarray) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(src5, _ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            M = cv2.getAffineTransform(src5[:3], _ARCFACE_DST[:3])
        aligned = cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0.0)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - 127.5) / 127.5            # standard ArcFace normalization
        blob = norm[None, ...] if self.nhwc else norm.transpose(2, 0, 1)[None, ...]
        emb = self.session.run([self._out], {self._inp: blob})[0].ravel()
        n = np.linalg.norm(emb)
        return (emb / n).astype(np.float32) if n else emb.astype(np.float32)


# Shared 5-point extraction from the 468/478-landmark FaceMesh/FaceLandmarker topology.
_IRIS_R, _IRIS_L = 468, 473   # iris centers (refine/Tasks landmarks)
_NOSE = 1
_MOUTH_A, _MOUTH_B = 61, 291  # mouth corners


def _five_points(pts: np.ndarray) -> np.ndarray:
    """Build the ordered [left_eye,right_eye,nose,left_mouth,right_mouth] template."""
    eye_a, eye_b = pts[_IRIS_R], pts[_IRIS_L]
    mouth_a, mouth_b = pts[_MOUTH_A], pts[_MOUTH_B]
    nose = pts[_NOSE]
    left_eye, right_eye = sorted([eye_a, eye_b], key=lambda p: p[0])
    left_m, right_m = sorted([mouth_a, mouth_b], key=lambda p: p[0])
    return np.array([left_eye, right_eye, nose, left_m, right_m], dtype=np.float32)


class LandmarkerArcFaceBackend:
    """MediaPipe Tasks FaceLandmarker (detect+align) + ArcFace ONNX (embed).

    Uses the locally provided models/face_landmarker.task and
    models/embedder_arcface.onnx — fully offline, multi-face capable.
    """

    name = "facelandmarker+arcface"

    def __init__(self, max_faces: int = 5):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        task_path = _find_landmarker_task()
        onnx_path = _find_arcface_onnx()
        if task_path is None or onnx_path is None:
            raise FileNotFoundError(
                "Need models/face_landmarker.task and an ArcFace .onnx "
                f"(task={task_path}, onnx={onnx_path})."
            )
        self.embedder = _ArcFaceEmbedder(onnx_path)
        self.name = f"facelandmarker+arcface({Path(onnx_path).name})"
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        # Keep a handle to mp.Image constructor.
        import mediapipe as mp
        self._mp = mp

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_image)
        faces: list[Face] = []
        if not res.face_landmarks:
            return faces
        for lms in res.face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            src5 = _five_points(pts)
            try:
                emb = self.embedder.embed(frame_bgr, src5)
            except Exception:
                emb = None
            faces.append(Face(bbox=(int(x1), int(y1), int(x2), int(y2)),
                              embedding=emb, det_score=1.0, kps=src5))
        return faces


class MeshArcFaceBackend:
    """Legacy MediaPipe solutions FaceMesh + ArcFace ONNX (no .task needed)."""

    name = "facemesh+arcface"

    def __init__(self, max_faces: int = 5):
        import mediapipe as mp
        onnx_path = _find_arcface_onnx()
        if onnx_path is None:
            raise FileNotFoundError("ArcFace ONNX model not found in models/ or insightface cache.")
        self.embedder = _ArcFaceEmbedder(onnx_path)
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=max_faces,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        faces: list[Face] = []
        if not res.multi_face_landmarks:
            return faces
        for lms in res.multi_face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark], dtype=np.float32)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            src5 = _five_points(pts)
            try:
                emb = self.embedder.embed(frame_bgr, src5)
            except Exception:
                emb = None
            faces.append(Face(bbox=(int(x1), int(y1), int(x2), int(y2)),
                              embedding=emb, det_score=1.0, kps=src5))
        return faces


def build_recognizer(det_size: int = 640, model_pack: str = "buffalo_s"):
    """Return the best available backend.

    Preference order:
      1. Local FaceLandmarker + ArcFace ONNX (offline, uses provided models/)
      2. insightface (requires the model pack to be downloaded/cached)
      3. Legacy FaceMesh + ArcFace ONNX
    """
    # 1. Local provided models — preferred (no network).
    if _find_landmarker_task() and _find_arcface_onnx():
        try:
            backend = LandmarkerArcFaceBackend()
            print(f"[recognizer] using backend: {backend.name}")
            return backend
        except Exception as e:  # noqa: BLE001
            print(f"[recognizer] local FaceLandmarker backend failed ({e}); trying insightface.")

    # 2. insightface (may need to download the pack).
    try:
        backend = InsightFaceBackend(det_size=det_size, model_pack=model_pack)
        print(f"[recognizer] using backend: {backend.name}")
        return backend
    except Exception as e:  # noqa: BLE001
        print(f"[recognizer] insightface unavailable ({e}); trying legacy fallback.")

    # 3. Legacy FaceMesh + ArcFace ONNX.
    backend = MeshArcFaceBackend()
    print(f"[recognizer] using backend: {backend.name}")
    return backend
