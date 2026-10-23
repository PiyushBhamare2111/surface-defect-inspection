"""Download and verify the NEU-CLS steel surface defect dataset into data/.

Source: Figshare mirror of NEU-CLS (CC BY 4.0), no login needed
  https://doi.org/10.6084/m9.figshare.28903550.v1
Original: K. Song and Y. Yan, Northeastern University (China), 2013.

The archive is treated as untrusted data: it is checked against a known MD5,
and only image files are read out of it (by basename, never by the archive's
own paths), so nothing in the zip can write outside data/NEU-CLS or execute.

Usage:  python scripts/download_data.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from defect_inspect.config import CLASSES, DATA_DIR, IMAGE_DIR  # noqa: E402

URL = "https://ndownloader.figshare.com/files/54094775"
MD5 = "19af5a1250d8ab0cb0828d31ee2b349d"
ZIP_PATH = DATA_DIR / "raw" / "NEU-CLS.zip"
PER_CLASS = 300


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> None:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force and md5sum(ZIP_PATH) == MD5:
        print(f"Archive already present and verified: {ZIP_PATH}")
        return
    print(f"Downloading {URL} ...")
    with requests.get(URL, stream=True, timeout=60,
                      headers={"User-Agent": "surface-defect-inspection/1.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = ZIP_PATH.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:5.1f} / {total/1e6:.1f} MB", end="")
        print()
    got = md5sum(tmp)
    if got != MD5:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"MD5 mismatch (got {got}, expected {MD5}) - download corrupted or changed.")
    tmp.replace(ZIP_PATH)
    print("MD5 verified.")


def extract() -> None:
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    for c in CLASSES:
        (IMAGE_DIR / c).mkdir(parents=True)
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name  # basename only: no path traversal
            if info.is_dir() or not name.lower().endswith(".jpg"):
                continue  # skip label .txt files and anything else
            prefix = name.rsplit("_", 1)[0]
            if prefix not in CLASSES:
                print(f"  skipping unexpected file {info.filename!r}")
                continue
            dest = IMAGE_DIR / prefix / name
            if dest.exists():
                raise SystemExit(f"Duplicate image name in archive: {name}")
            data = zf.read(info)
            with Image.open(io.BytesIO(data)) as im:  # validate it is a real image
                im.verify()
            dest.write_bytes(data)
            counts[prefix] += 1
    print("Images per class:", dict(counts))
    bad = {c: counts[c] for c in CLASSES if counts[c] != PER_CLASS}
    if bad:
        raise SystemExit(f"Unexpected class counts {bad}; expected {PER_CLASS} each.")
    with Image.open(next((IMAGE_DIR / CLASSES[0]).iterdir())) as im:
        print(f"Example image: mode={im.mode} size={im.size}")
    print(f"OK - {sum(counts.values())} images in {IMAGE_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the archive exists")
    args = ap.parse_args()
    download(args.force)
    extract()


if __name__ == "__main__":
    main()
