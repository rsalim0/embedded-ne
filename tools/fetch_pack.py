"""Resumable, retrying downloader for an InsightFace model pack.

This network forcibly resets long downloads (WinError 10054), and insightface's
built-in one-shot downloader cannot resume — so it never finishes. This script
downloads the pack zip with HTTP Range resume + unlimited retries, then extracts
it into the insightface cache (~/.insightface/models/<pack>/) so FaceAnalysis
loads it offline with no further download.

Usage:
    python tools/fetch_pack.py                # buffalo_s (default)
    python tools/fetch_pack.py buffalo_l
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://github.com/deepinsight/insightface/releases/download/v0.7"
CACHE = Path.home() / ".insightface" / "models"
CHUNK = 256 * 1024
MAX_RETRIES = 200
RETRY_SLEEP = 2.0


def download_resumable(url: str, dest: Path) -> None:
    """Download `url` to `dest`, resuming from whatever is already on disk."""
    # Total size via a ranged probe.
    probe = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(probe, timeout=30) as r:
        cr = r.headers.get("Content-Range", "")
        total = int(cr.split("/")[-1]) if "/" in cr else None

    attempt = 0
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have >= total:
            print(f"\n[fetch] download complete ({have} bytes).")
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "ab") as f:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    have += len(block)
                    if total:
                        pct = have * 100 // total
                        sys.stdout.write(
                            f"\r[fetch] {pct:3d}%  {have/1e6:7.1f} / {total/1e6:7.1f} MB"
                            f"  (retries: {attempt})")
                        sys.stdout.flush()
            # Reached EOF cleanly — loop re-checks completeness.
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            sys.stdout.write(f"\r[fetch] reset at {have/1e6:.1f} MB; "
                             f"retry {attempt}/{MAX_RETRIES} ... ")
            sys.stdout.flush()
            time.sleep(RETRY_SLEEP)


def extract_pack(zip_path: Path, pack_dir: Path) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            base = os.path.basename(member)
            if not base:
                continue
            with zf.open(member) as src, open(pack_dir / base, "wb") as dst:
                dst.write(src.read())
    onnx = sorted(p.name for p in pack_dir.glob("*.onnx"))
    print(f"[fetch] extracted -> {pack_dir}")
    print(f"[fetch] onnx models: {onnx}")


def main() -> int:
    pack = sys.argv[1] if len(sys.argv) > 1 else "buffalo_s"
    url = f"{BASE}/{pack}.zip"
    pack_dir = CACHE / pack
    rec_models = list(pack_dir.glob("*w600k*.onnx")) if pack_dir.exists() else []
    if rec_models:
        print(f"[fetch] {pack} already present at {pack_dir} ({[p.name for p in rec_models]})")
        return 0

    CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE / f"{pack}.zip"
    print(f"[fetch] downloading {url}\n[fetch] -> {zip_path} (resumable)")
    download_resumable(url, zip_path)
    print("[fetch] extracting ...")
    extract_pack(zip_path, pack_dir)
    try:
        zip_path.unlink()
    except OSError:
        pass
    print(f"[fetch] DONE. insightface can now load '{pack}' offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
