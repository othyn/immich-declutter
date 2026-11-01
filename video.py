"""Size and length bands for timeline videos, as tags to prune by hand.

Video is 87% of the library this was written for (3,301 of 3,796 GB on 01/11/2026), spread across
every size: no single band holds most of it. Each unprotected video gets one size tag and one length
tag, so a review can start from any slice. Protected videos get none: favourites, anything in an
album, anything showing a named person, and anything under a KEEP_TAGS tag (iOS formats, the Fuji).

The tags are recomputed every run and reconciled both ways. Favouriting a video or adding it to an
album takes it out of review on the next run; removing a tag by hand does not, it comes back.
Live Photo motion clips are hidden assets and never listed.
"""

from __future__ import annotations

import math
from datetime import datetime

SIZE_BANDS = (
    (20e6, "Size/1 Under 20 MB"),
    (50e6, "Size/2 20-50 MB"),
    (100e6, "Size/3 50-100 MB"),
    (200e6, "Size/4 100-200 MB"),
    (500e6, "Size/5 200-500 MB"),
    (1e9, "Size/6 500 MB-1 GB"),
    (2e9, "Size/7 1-2 GB"),
    (math.inf, "Size/8 Over 2 GB"),
)
# Seconds. Under 3 s is a pocket or accidental clip, tagged on its own.
LENGTH_BANDS = (
    (3, "Accidental"),
    (10, "Length/1 3-10 s"),
    (30, "Length/2 10-30 s"),
    (60, "Length/3 30 s-1 min"),
    (180, "Length/4 1-3 min"),
    (300, "Length/5 3-5 min"),
    (900, "Length/6 5-15 min"),
    (math.inf, "Length/7 Over 15 min"),
)
BACK_TO_BACK = "Back to back"
SCREEN_RECORDINGS = "Screen recordings"
SCREEN_RECORDING_PREFIXES = ("rpreplay", "screenrecording")
TAGS = tuple(t for _, t in SIZE_BANDS) + tuple(t for _, t in LENGTH_BANDS) + (BACK_TO_BACK, SCREEN_RECORDINGS)


def band(value: float, bands: tuple[tuple[float, str], ...]) -> str:
    return next(name for limit, name in bands if value < limit)


def taken(asset: dict) -> datetime:
    return datetime.fromisoformat(asset["fileCreatedAt"].replace("Z", "+00:00"))


def back_to_back(videos: list[dict], minutes: float) -> set[str]:
    """Videos shot within `minutes` of the previous or next video. Protected ones count as neighbours."""
    ordered = sorted(videos, key=taken)
    out = set()
    for a, b in zip(ordered, ordered[1:]):
        if (taken(b) - taken(a)).total_seconds() < minutes * 60:
            out |= {a["id"], b["id"]}
    return out


def protected(asset: dict, in_albums: set[str], kept: set[str]) -> bool:
    return (
        asset.get("isFavorite", False)
        or asset["id"] in in_albums
        or asset["id"] in kept
        or any(p.get("name") for p in asset.get("people") or [])
    )


def desired(videos: list[dict], in_albums: set[str], kept: set[str], minutes: float) -> dict[str, set[str]]:
    """Tag (relative to the parent) -> asset ids that should carry it."""
    out: dict[str, set[str]] = {t: set() for t in TAGS}
    runs = back_to_back(videos, minutes)
    for a in videos:
        if protected(a, in_albums, kept):
            continue
        size = (a.get("exifInfo") or {}).get("fileSizeInByte") or 0
        out[band(size, SIZE_BANDS)].add(a["id"])
        # duration is milliseconds in Immich 3.x
        out[band((a.get("duration") or 0) / 1000, LENGTH_BANDS)].add(a["id"])
        if a["id"] in runs:
            out[BACK_TO_BACK].add(a["id"])
        if (a.get("originalFileName") or "").lower().startswith(SCREEN_RECORDING_PREFIXES):
            out[SCREEN_RECORDINGS].add(a["id"])
    return out
