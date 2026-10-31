"""iOS capture formats read from the original file, for Media/<type> tags.

iOS keeps these markers in the file itself, which is how Photos recognises a portrait or spatial
photo downloaded back from Immich. Immich's own exif table stores none of them, so the original
is read with exiftool. Markers checked against the iOS smart albums on 31/10/2026:

  Portrait   depth or disparity auxiliary image, and no stereo pair   80/80
  Spatial    HEIF 'ster' stereo group (photo), SpatialFormat key (video)   80/80
  Cinematic  Cinematic-video key   2/2
  Bursts     BurstUUID in the Apple maker note   2/2
  RAW        ProRAW capture type or a camera RAW file type   16/17
  Slo-mo     frame rate of 100 or more   4/44

ponytail: Slo-mo misses clips stored at 30 fps with the slow section kept elsewhere, read the
.AAE edit or the timed-metadata track if those matter.
"""

from __future__ import annotations

import json
import logging
import subprocess

LOG = logging.getLogger("immich-declutter")

PORTRAIT = "portrait"
SPATIAL = "spatial"
CINEMATIC = "cinematic"
SLOMO = "slomo"
BURSTS = "bursts"
RAW = "raw"
TYPES = (PORTRAIT, SPATIAL, CINEMATIC, SLOMO, BURSTS, RAW)
DEFAULT_TAGS = {
    PORTRAIT: "Portrait",
    SPATIAL: "Spatial",
    CINEMATIC: "Cinematic",
    SLOMO: "Slo-mo",
    BURSTS: "Bursts",
    RAW: "RAW",
}

RAW_EXTENSIONS = (".dng", ".raf", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2")
# -G4 keeps repeated tags apart (Copy1:AuxiliaryImageType, ...); -u exposes the 'ster' box.
EXIFTOOL = [
    "exiftool", "-j", "-n", "-a", "-u", "-G4", "-fast",
    "-AuxiliaryImageType", "-Unknown_ster", "-SpatialFormat-version", "-Cinematic-video",
    "-VideoFrameRate", "-BurstUUID", "-ImageCaptureType", "-FileType",
]


def needs_exiftool(asset: dict) -> bool:
    """Only Apple captures carry the markers; RAW is known from the file name."""
    return ((asset.get("exifInfo") or {}).get("make") or "") == "Apple"


def detect(meta: dict) -> set[str]:
    values: dict[str, list] = {}
    for key, value in meta.items():
        values.setdefault(key.rsplit(":", 1)[-1], []).append(value)
    aux = {str(v).rsplit(":", 1)[-1] for v in values.get("AuxiliaryImageType", [])}
    found = set()
    if "Unknown_ster" in values or "SpatialFormat-version" in values:
        found.add(SPATIAL)
    elif aux & {"depth", "disparity"}:
        found.add(PORTRAIT)
    if "Cinematic-video" in values:
        found.add(CINEMATIC)
    rates = [v for v in values.get("VideoFrameRate", []) if isinstance(v, (int, float))]
    if rates and max(rates) >= 100:
        found.add(SLOMO)
    if any(values.get("BurstUUID", [])):
        found.add(BURSTS)
    if 1 in values.get("ImageCaptureType", []):  # ProRAW
        found.add(RAW)
    return found


def from_name(asset: dict) -> set[str]:
    return {RAW} if (asset.get("originalFileName") or "").lower().endswith(RAW_EXTENSIONS) else set()


def read(paths: list[str], timeout: int = 600) -> dict[str, dict]:
    """exiftool over a batch of originals; missing or unreadable files are left out."""
    if not paths:
        return {}
    proc = subprocess.run(
        [*EXIFTOOL, "-@", "-"], input="\n".join(paths), capture_output=True, text=True, timeout=timeout
    )
    if not proc.stdout.strip():
        LOG.warning("exiftool returned nothing for %d files: %s", len(paths), proc.stderr.strip()[:200])
        return {}
    return {m["SourceFile"]: m for m in json.loads(proc.stdout)}
