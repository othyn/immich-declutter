import numpy as np

import declutter
import rules
from declutter import Config, Declutter
from immich import NotFound
from state import State


class FakeImmich:
    def __init__(self, assets):
        self.assets = {a["id"]: a for a in assets}
        self.tag_members: dict[str, set[str]] = {}
        self.album_members: dict[str, set[str]] = {}

    def ping(self):
        return {}

    def check_access(self):
        pass

    def upsert_tag(self, value):
        self.tag_members.setdefault(value, set())
        return value

    def tag_asset_ids(self, tag_id):
        return set(self.tag_members[tag_id])

    def tag_assets(self, tag_id, ids):
        self.tag_members[tag_id] |= set(ids)

    def untag_assets(self, tag_id, ids):
        self.tag_members[tag_id] -= set(ids)

    def album_ids(self):
        return list(self.album_members)

    def album_asset_ids(self, album_id):
        return set(self.album_members[album_id])

    def search(self, **filters):
        out = list(self.assets.values())
        if "tagIds" in filters:
            out = [a for a in out if a["id"] in self.tag_members[filters["tagIds"][0]]]
        if "type" in filters:
            out = [a for a in out if a.get("type") == filters["type"]]
        return out

    def asset(self, asset_id):
        if asset_id not in self.assets:
            raise NotFound(asset_id)
        return self.assets[asset_id]

    def thumbnail(self, asset_id):
        return asset_id.encode(), "image/webp"

    def ocr(self, asset_id):
        return []


class FakeClip:
    """Thumbnail bytes = asset id; ids starting with 'meme' score as memes."""

    def load_prompts(self):
        pass

    def image(self, data, mime):
        return np.array([1.0 if data.startswith(b"meme") else 0.0])

    def scores(self, emb):
        return {"meme": float(emb[0])}


def asset(id_, make=None):
    return {"id": id_, "type": "IMAGE", "width": 1200, "height": 1600, "visibility": "timeline",
            "exifInfo": {"make": make}, "originalPath": f"/x/{id_}.jpg"}


def make(tmp_path, monkeypatch, assets):
    for k, v in {"IMMICH_URL": "http://i", "IMMICH_API_KEY": "k", "ML_URL": "http://m",
                 "FULL_SCAN": "true", "MIN_AGE_HOURS": "0", "THREADS": "1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(declutter, "image_stats", lambda data: {})  # fake thumbnails are not images
    monkeypatch.setattr(declutter.media, "read", lambda paths: {p: {} for p in paths})
    cfg = Config.from_env()
    immich = FakeImmich(assets)
    return Declutter(cfg, immich, FakeClip(), State(str(tmp_path / "s.db"))), immich, cfg


def test_full_cycle(tmp_path, monkeypatch):
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("meme1"), asset("meme2"), asset("photo"),
                                                  asset("meme3", make="Apple")])
    r = d.run()
    tag = cfg.tags[rules.MEMES]
    assert tag == "Utilities/Memes & Web Images"
    assert r.new[rules.MEMES] == 2 and immich.tag_members[tag] == {"meme1", "meme2"}

    # user keeps meme1 (removes the tag) and deletes meme2
    immich.tag_members[tag] = set()
    del immich.assets["meme2"]
    r = d.run()
    assert (r.kept, r.deleted, r.new[rules.MEMES]) == (1, 1, 0)
    assert immich.tag_members[tag] == set()  # meme1 is never proposed again


def test_favourites_are_never_proposed(tmp_path, monkeypatch):
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("meme1") | {"isFavorite": True}])
    assert d.run().new[rules.MEMES] == 0


def test_migrates_upstream_albums_to_tags(tmp_path, monkeypatch):
    """Assets still in an upstream review album get tagged; ones the user removed count as kept."""
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("meme1"), asset("meme2")])
    d.state.set("meme1", "proposed", rules.MEMES)
    d.state.set("meme2", "proposed", rules.MEMES)
    d.state.set_meta(f"album:{rules.MEMES}", "old-album")
    immich.album_members["old-album"] = {"meme1"}
    r = d.run()
    assert immich.tag_members[cfg.tags[rules.MEMES]] == {"meme1"}
    assert r.kept == 1 and d.state.get_meta(f"album:{rules.MEMES}") is None


def test_tags_written_before_state_commit(tmp_path, monkeypatch):
    """A crash after the tag flush must not turn proposals into 'kept' on the next run."""
    many = [asset(f"meme{i}") for i in range(600)]
    d, immich, cfg = make(tmp_path, monkeypatch, many)
    calls = []
    orig = immich.tag_assets
    immich.tag_assets = lambda tid, ids: (calls.append(len(ids)), orig(tid, ids))
    d.run()
    assert calls == [500, 100]


def test_failed_tag_flush_rolls_back_state(tmp_path, monkeypatch):
    """A failing tag write must not leave 'proposed' rows behind for the next run to commit."""
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("meme1"), asset("photo")])
    real_add = immich.tag_assets

    def forbidden(tid, ids):
        raise RuntimeError("403 Forbidden")

    immich.tag_assets = forbidden
    try:
        d.run()
    except RuntimeError:
        pass
    else:
        raise AssertionError("run should fail")

    immich.tag_assets = real_add
    r = d.run()
    assert (r.kept, r.new[rules.MEMES]) == (0, 1)
    assert immich.tag_members[cfg.tags[rules.MEMES]] == {"meme1"}


def test_new_categories_re_evaluate_none_assets(tmp_path, monkeypatch):
    """An upstream db marked 'photo' as 'none'; enabling new categories must look at it again, once."""
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("photo")])
    d.state.set("photo", "none")
    d.state.commit()
    assert d.run().scanned == 1
    assert d.run().scanned == 0


def test_media_tags_read_once(tmp_path, monkeypatch):
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("pic", make="Apple"), asset("cam") | {"originalFileName": "D.RAF"},
                                                  asset("gone", make="Apple")])
    reads = []

    def fake_read(paths):
        reads.extend(paths)
        return {p: {":BurstUUID": "x"} for p in paths if "gone" not in p}

    monkeypatch.setattr(declutter.media, "read", fake_read)
    r = d.run()
    assert immich.tag_members["Media/Bursts"] == {"pic"} and immich.tag_members["Media/RAW"] == {"cam"}
    assert r.media["bursts"] == 1 and "/x/cam.jpg" not in reads  # RAW from the name, no exiftool
    reads.clear()
    d.run()
    assert reads == ["/x/gone.jpg"]  # unreadable last time, retried; the rest is not read again


def test_media_retry_survives_cutoff(tmp_path, monkeypatch):
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("late", make="Apple")])
    monkeypatch.setattr(declutter.media, "read", lambda paths: {})
    d.run()
    immich.search = lambda **f: []  # with a cutoff, the old asset is no longer listed
    monkeypatch.setattr(declutter.media, "read", lambda paths: {p: {":BurstUUID": "x"} for p in paths})
    d.run()
    assert immich.tag_members["Media/Bursts"] == {"late"}


def test_media_rule_change_untags_old_matches(tmp_path, monkeypatch):
    d, immich, cfg = make(tmp_path, monkeypatch, [asset("auto"), asset("real")])
    portrait = cfg.media_tags["portrait"]
    immich.tag_members[portrait] = {"auto", "real"}
    for a in ("auto", "real"):
        d.state.set_media(a, {"portrait"})
    d.state.set_meta("media_cutoff", "2026-01-01T00:00:00Z")
    monkeypatch.setattr(declutter.media, "read", lambda paths: {p: {} for p in paths})
    monkeypatch.setattr(declutter.media, "needs_exiftool", lambda a: True)
    monkeypatch.setattr(declutter.media, "detect", lambda m: set())
    monkeypatch.setattr(declutter.media, "from_name", lambda a: {"portrait"} if a["id"] == "real" else set())
    d.run()
    assert immich.tag_members[portrait] == {"real"}
    assert d.state.get_meta("media_rules") == str(declutter.media.RULES_VERSION)


def test_video_tags_reconcile(tmp_path, monkeypatch):
    def vid(id_, mb, s, at, **kw):
        return {"id": id_, "type": "VIDEO", "visibility": "timeline", "duration": s * 1000,
                "fileCreatedAt": f"2026-10-01T12:{at:02d}:00Z", "originalFileName": f"{id_}.MOV",
                "exifInfo": {"make": None, "fileSizeInByte": mb * 1e6}, "originalPath": f"/x/{id_}.mov", **kw}

    vids = [vid("pocket", 1, 1, 0), vid("burst", 30, 20, 1), vid("fav", 600, 400, 30, isFavorite=True),
            vid("bob", 30, 20, 40, people=[{"name": "Bob"}]), vid("alone", 2500, 1000, 50)]
    d, immich, cfg = make(tmp_path, monkeypatch, vids)
    p = cfg.video_tag_parent
    d.run()
    members = {t.removeprefix(p + "/"): ids for t, ids in immich.tag_members.items() if t.startswith(p)}
    assert members["Accidental"] == {"pocket"}
    assert members["Back to back"] == {"pocket", "burst"}
    assert members["Size/2 20-50 MB"] == {"burst"}
    assert members["Size/8 Over 2 GB"] == members["Length/7 Over 15 min"] == {"alone"}
    assert not any("fav" in ids or "bob" in ids for ids in members.values())
    immich.assets["alone"]["isFavorite"] = True
    d.run()
    assert not immich.tag_members[f"{p}/Size/8 Over 2 GB"]
