"""iOS capture formats read from the original file, for Media/<type> tags.

iOS keeps these markers in the file itself, which is how Photos recognises a portrait or spatial
photo downloaded back from Immich. Immich's own exif table stores none of them, so the original
is read with exiftool. Markers checked against the iOS smart albums on 31/10/2026:

  Portrait   depth or disparity image and processing flag 0x40, no stereo pair   74 vs 80 in album
  Spatial    HEIF 'ster' stereo group (photo), SpatialFormat key (video)   80/80
  Cinematic  Cinematic-video key, intent not 0   2/2, and 1/1 test clip
  Bursts     BurstUUID in the Apple maker note   2/2
  RAW        a RAW file extension; ProRAW is saved as DNG   16/17
  Slo-mo     frame rate of 100 or more   4/44
  UniFi      UniFi Protect export file name, no camera make   46/46 known exports

Since the iPhone 15 the camera saves depth data whenever a person or pet is in frame, and the
iPhone 18 Pro Max writes a Cinematic-video key with intent 0 on every video. Checked 01/11/2026
against a Photo-mode and a Portrait-mode shot of one subject: only Portrait sets bit 0x40 of the
Apple ImageProcessingFlags (12,645 depth photos, 74 with the bit). The maker note's
ImageCaptureType 1, which exiftool names ProRAW, is also set on ordinary JPG and HEIC photos from
phones that cannot shoot ProRAW (iPhone XS, 7 Plus), so RAW comes from the file name alone.

ponytail: UniFi matches the export file name only, read the "UniFi" watermark (bottom right) from
a video frame if renamed exports turn up.

ponytail: Slo-mo misses clips stored at 30 fps with the slow section kept elsewhere, read the
.AAE edit or the timed-metadata track if those matter.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess

LOG = logging.getLogger("immich-declutter")

PORTRAIT = "portrait"
SPATIAL = "spatial"
CINEMATIC = "cinematic"
SLOMO = "slomo"
BURSTS = "bursts"
RAW = "raw"
UNIFI = "unifi"
TYPES = (PORTRAIT, SPATIAL, CINEMATIC, SLOMO, BURSTS, RAW, UNIFI)
# Bump when a detection rule changes; assets tagged with RECHECK types are read again once.
RULES_VERSION = 3
RECHECK = {PORTRAIT, CINEMATIC, RAW}
PORTRAIT_FLAG = 0x40
DEFAULT_TAGS = {
    PORTRAIT: "Portrait",
    SPATIAL: "Spatial",
    CINEMATIC: "Cinematic",
    SLOMO: "Slo-mo",
    BURSTS: "Bursts",
    RAW: "RAW",
    UNIFI: "UniFi",
}
# Tags that live outside MEDIA_TAG_PARENT, as a full path
FULL_PATHS = {UNIFI: "Cameras/UniFi"}

# UniFi Protect exports: "unifi_180524_2014_84.0s.mp4" or
# "Front Door 11-22-2025, 14.55.00 GMT - 11-22-2025, 15.55.00 GMT.mp4"
UNIFI_NAME = re.compile(
    r"^unifi_\d{6}_\d{4}_|\d{1,2}-\d{1,2}-\d{4}, \d{2}\.\d{2}\.\d{2}.* - \d{1,2}-\d{1,2}-\d{4}, \d{2}\.\d{2}\.\d{2}", re.I
)
RAW_EXTENSIONS = (".dng", ".raf", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2")
# -G4 keeps repeated tags apart (Copy1:AuxiliaryImageType, ...); -u exposes the 'ster' box.
EXIFTOOL = [
    "exiftool", "-j", "-n", "-a", "-u", "-G4", "-fast",
    "-AuxiliaryImageType", "-Unknown_ster", "-SpatialFormat-version", "-Cinematic-video",
    "-Cinematic-video-intent", "-ImageProcessingFlags", "-VideoFrameRate", "-BurstUUID", "-FileType",
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
    elif aux & {"depth", "disparity"} and any(
        isinstance(v, int) and v & PORTRAIT_FLAG for v in values.get("ImageProcessingFlags", [])
    ):
        found.add(PORTRAIT)
    # iPhone 15 Pro Max clips carry no intent key; the 18 Pro Max sets 0 on ordinary video
    if "Cinematic-video" in values and 0 not in values.get("Cinematic-video-intent", []):
        found.add(CINEMATIC)
    rates = [v for v in values.get("VideoFrameRate", []) if isinstance(v, (int, float))]
    if rates and max(rates) >= 100:
        found.add(SLOMO)
    if any(values.get("BurstUUID", [])):
        found.add(BURSTS)
    return found


def from_name(asset: dict) -> set[str]:
    name = asset.get("originalFileName") or ""
    found = {RAW} if name.lower().endswith(RAW_EXTENSIONS) else set()
    if asset.get("type") == "VIDEO" and not (asset.get("exifInfo") or {}).get("make") and UNIFI_NAME.search(name):
        found.add(UNIFI)
    return found


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
