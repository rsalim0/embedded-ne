"""
download_model.py

Ensure the ArcFace ONNX model ``w600k_r50.onnx`` is available for the
recognizer's fallback path.

This script is only needed when the ``insightface`` Python package itself
failed to install (in which case its bundled model auto-download will not
run). It uses the standard library only so it can run in a minimal
environment.

Behavior:
  1. If the model already exists at the target location, or anywhere under
     ``~/.insightface/models/**``, copy/confirm it into ``models/`` and exit.
  2. Otherwise download the InsightFace ``buffalo_l`` model pack zip from the
     official GitHub release, extract just ``w600k_r50.onnx`` into ``models/``,
     and clean up the temporary zip.

Pure standard library (urllib, zipfile, shutil, glob, pathlib). No third-party
dependencies.
"""

import glob
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "w600k_r50.onnx"

# Project root is the parent of this tools/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_PATH = MODELS_DIR / MODEL_NAME

# Official InsightFace release containing the buffalo_l model pack.
BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/"
    "v0.7/buffalo_l.zip"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_existing_model() -> Path | None:
    """Return the path to an existing w600k_r50.onnx, or None.

    Checks the target location first, then anywhere under the InsightFace
    cache directory (``~/.insightface/models/**``).
    """
    if TARGET_PATH.exists():
        return TARGET_PATH

    cache_root = Path.home() / ".insightface" / "models"
    if cache_root.exists():
        # Recursive glob for the model file anywhere under the cache.
        pattern = str(cache_root / "**" / MODEL_NAME)
        for match in glob.glob(pattern, recursive=True):
            candidate = Path(match)
            if candidate.is_file():
                return candidate

    return None


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple percent-based download progress printed to one line."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\r  Downloading buffalo_l.zip: {percent:3d}% "
            f"({mb_done:6.1f} / {mb_total:6.1f} MiB)"
        )
    else:
        # Unknown total size: just show bytes downloaded.
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading buffalo_l.zip: {mb_done:6.1f} MiB")
    sys.stdout.flush()


def download_and_extract() -> int:
    """Download buffalo_l.zip and extract w600k_r50.onnx into models/.

    Returns a process exit code (0 on success, non-zero on failure).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download to a temporary file so a partial/failed download never leaves a
    # corrupt zip behind in the project tree.
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip", prefix="buffalo_l_")
    tmp_zip = Path(tmp_name)
    # We only needed the path; close the OS-level handle so urllib can write.
    import os
    os.close(tmp_fd)

    try:
        print(f"Model not found locally. Downloading from:\n  {BUFFALO_L_URL}")
        try:
            urllib.request.urlretrieve(
                BUFFALO_L_URL, tmp_zip, reporthook=_progress_hook
            )
            print()  # newline after the in-place progress line
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print()  # finish the progress line cleanly
            print(f"ERROR: failed to download the model pack: {exc}")
            print(
                "Note: this model is only needed if the 'insightface' package "
                "failed to install.\n"
                "      Check your network connection, or reinstall insightface "
                "with:  pip install insightface"
            )
            return 1

        # Extract just the one model file we need.
        print(f"Extracting {MODEL_NAME} ...")
        try:
            with zipfile.ZipFile(tmp_zip) as zf:
                member = _locate_member(zf, MODEL_NAME)
                if member is None:
                    print(
                        f"ERROR: {MODEL_NAME} was not found inside the "
                        "downloaded zip. The release contents may have changed."
                    )
                    return 1
                # Stream the member out to the target path (flatten any
                # internal directory structure).
                with zf.open(member) as src, open(TARGET_PATH, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile as exc:
            print(f"ERROR: the downloaded file is not a valid zip: {exc}")
            return 1

        print(f"Done. Model saved to:\n  {TARGET_PATH}")
        return 0

    finally:
        # Always clean up the temp zip, even on error.
        try:
            if tmp_zip.exists():
                tmp_zip.unlink()
        except OSError:
            pass


def _locate_member(zf: zipfile.ZipFile, filename: str) -> str | None:
    """Find the zip member whose basename matches *filename* (case-insensitive)."""
    target = filename.lower()
    for name in zf.namelist():
        if name.lower().rsplit("/", 1)[-1] == target:
            return name
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    existing = find_existing_model()
    if existing is not None:
        if existing == TARGET_PATH:
            print(f"Model already present at:\n  {TARGET_PATH}")
        else:
            # Found in the insightface cache: copy it into models/.
            print(f"Found existing model in cache:\n  {existing}")
            shutil.copy2(existing, TARGET_PATH)
            print(f"Copied to:\n  {TARGET_PATH}")
        return 0

    return download_and_extract()


if __name__ == "__main__":
    raise SystemExit(main())
