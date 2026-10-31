"""Minimal Immich API client (Immich 3.x)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import requests


class NotFound(Exception):
    pass


# What a restricted API key needs; "all" covers everything.
REQUIRED_PERMISSIONS = ("asset.read", "asset.view", "tag.create", "tag.asset")


class Immich:
    def __init__(self, url: str, api_key: str, timeout: int = 120) -> None:
        self.base = url.rstrip("/") + "/api"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})

    def _req(self, method: str, path: str, **kw):
        r = self.session.request(method, self.base + path, timeout=self.timeout, **kw)
        # Immich 3.x answers 400 "Not found or no asset.read access" for a deleted asset.
        # That is only trustworthy after check_access() (otherwise it may be a permission problem).
        if r.status_code == 404 or (r.status_code == 400 and r.text.startswith('{"message":"Not found')):
            raise NotFound(path)
        r.raise_for_status()
        return r

    def ping(self) -> dict:
        return self._req("GET", "/server/version").json()

    def check_access(self) -> None:
        """Verify the API key's permissions, then try them on one existing asset (search, asset, thumbnail, ocr)."""
        granted = set(self._req("GET", "/api-keys/me").json().get("permissions", []))
        missing = [p for p in REQUIRED_PERMISSIONS if p not in granted]
        if "all" not in granted and missing:
            raise PermissionError(f"API key lacks permission {', '.join(missing)}")
        # A day old, so its thumbnail exists: a fresh upload 404s on /thumbnail and reads as no access.
        day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        body = {"size": 1, "page": 1, "type": "IMAGE", "createdBefore": day_ago}
        items = self._req("POST", "/search/metadata", json=body).json()["assets"]["items"]
        if not items:
            return
        asset_id = items[0]["id"]
        for what, call in (
            ("asset.read", lambda: self.asset(asset_id)),
            ("asset.view", lambda: self.thumbnail(asset_id)),
            ("ocr", lambda: self.ocr(asset_id)),
        ):
            try:
                call()
            except NotFound:
                raise PermissionError(f"API key lacks permission {what} (tested on asset {asset_id})") from None

    def search(self, **filters) -> Iterator[dict]:
        """All assets matching the metadata filters (paginated)."""
        page = 1
        while page:
            body = {"size": 1000, "page": page, "withExif": True, **filters}
            data = self._req("POST", "/search/metadata", json=body).json()["assets"]
            yield from data["items"]
            page = int(data["nextPage"]) if data.get("nextPage") else None

    def asset(self, asset_id: str) -> dict:
        return self._req("GET", f"/assets/{asset_id}").json()

    def thumbnail(self, asset_id: str) -> tuple[bytes, str]:
        r = self._req("GET", f"/assets/{asset_id}/thumbnail", params={"size": "thumbnail"})
        return r.content, r.headers.get("content-type", "image/webp")

    def ocr(self, asset_id: str) -> list[dict]:
        data = self._req("GET", f"/assets/{asset_id}/ocr").json()
        return data if isinstance(data, list) else []

    def upsert_tag(self, value: str) -> str:
        """Create the tag (and its parents, from a "Parent/Child" path) if missing; returns the leaf id."""
        tags = self._req("PUT", "/tags", json={"tags": [value]}).json()
        return next(t["id"] for t in tags if t["value"] == value)

    def tag_asset_ids(self, tag_id: str) -> set[str]:
        # Search without a visibility filter covers timeline and archive (everything but hidden).
        return {a["id"] for a in self.search(tagIds=[tag_id], withExif=False)}

    def tag_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        for i in range(0, len(asset_ids), 500):
            self._req("PUT", f"/tags/{tag_id}/assets", json={"ids": asset_ids[i : i + 500]})

    def album_asset_ids(self, album_id: str) -> set[str]:
        """Only used to migrate the upstream review albums to tags (needs album.read)."""
        return {a["id"] for a in self.search(albumIds=[album_id], withExif=False)}
