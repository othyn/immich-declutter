"""immich-declutter: periodically tags clutter candidates (memes, TikToks, screenshots, receipts,
YouTube grabs, photos of screens, documents, blurry and low-quality shots, messaging-app media)
with one Immich tag per category under a parent such as "Utilities". It never deletes anything.

Usage:
  python declutter.py                    # scheduler (daily at RUN_AT)
  python declutter.py --once             # a single run, now
  python declutter.py --once --dry-run
  python declutter.py --mark-kept ids.txt   # never propose these asset ids (one per line)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

import rules
from immich import Immich, NotFound
from ml import Clip, image_stats
from state import State

LOG = logging.getLogger("immich-declutter")
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
VISIBLE = ("timeline", "archive")

# Tags keep albums for curation. Removing a tag from an asset marks it kept: it is never proposed again.
DEFAULT_TAGS = {
    rules.MEMES: "Memes & Web Images",
    rules.TIKTOK: "TikTok & Reels",
    rules.SCREENSHOTS: "Screenshots",
    rules.RECEIPTS: "Receipts & Barcodes",
    rules.YOUTUBE: "YouTube",
    rules.SCREENS: "Photos of Screens",
    rules.DOCUMENTS: "Documents",
    rules.BLURRY: "Blurry",
    rules.LOWQUALITY: "Low Quality",
    rules.MESSAGING: "Messaging",
}


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        sys.exit(f"Environment variable {name} is missing")
    return value


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        sys.exit(f"{name} must be an integer")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        sys.exit(f"{name} must be a number")


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    immich_url: str
    immich_api_key: str
    ml_url: str
    clip_model: str
    data_dir: str
    timezone: str
    run_at: str
    run_on_start: bool
    full_scan: bool
    dry_run: bool
    min_age_hours: int
    max_assets: int
    threads: int
    max_failures: int
    categories: set[str]
    tags: dict[str, str]
    thresholds: rules.Thresholds
    public_url: str
    uptime_kuma_push_url: str
    pushover_token: str
    pushover_user: str

    @classmethod
    def from_env(cls) -> Config:
        cats = {c.strip().lower() for c in os.getenv("CATEGORIES", ",".join(rules.CATEGORIES)).split(",") if c.strip()}
        unknown = cats - set(rules.CATEGORIES)
        if unknown:
            sys.exit(f"Unknown CATEGORIES: {', '.join(sorted(unknown))} (choose from {', '.join(rules.CATEGORIES)})")
        run_at = os.getenv("RUN_AT", "03:30").strip()
        try:
            datetime.strptime(run_at, "%H:%M")
        except ValueError:
            sys.exit("RUN_AT must be HH:MM")
        defaults = rules.Thresholds()
        thresholds = rules.Thresholds(
            **{f.name: env_float(f"THRESHOLD_{f.name.upper()}", getattr(defaults, f.name)) for f in fields(defaults)}
        )
        return cls(
            immich_url=env_required("IMMICH_URL"),
            immich_api_key=env_required("IMMICH_API_KEY"),
            ml_url=env_required("ML_URL"),
            clip_model=os.getenv("CLIP_MODEL", "ViT-B-32__openai").strip(),
            data_dir=os.getenv("DATA_DIR", "/data"),
            timezone=os.getenv("TZ", "UTC"),
            run_at=run_at,
            run_on_start=env_bool("RUN_ON_START", False),
            full_scan=env_bool("FULL_SCAN", False),
            dry_run=env_bool("DRY_RUN", False),
            min_age_hours=env_int("MIN_AGE_HOURS", 3),
            max_assets=env_int("MAX_ASSETS", 0),
            threads=env_int("THREADS", 4, minimum=1),
            max_failures=env_int("MAX_FAILURES", 5, minimum=1),
            categories=cats,
            tags={
                c: f"{os.getenv('TAG_PARENT', 'Utilities').strip()}/{os.getenv(f'TAG_{c.upper()}', DEFAULT_TAGS[c]).strip()}"
                for c in rules.CATEGORIES
            },
            thresholds=thresholds,
            public_url=os.getenv("IMMICH_PUBLIC_URL", "").strip().rstrip("/"),
            uptime_kuma_push_url=os.getenv("UPTIME_KUMA_PUSH_URL", "").strip(),
            pushover_token=os.getenv("PUSHOVER_TOKEN", "").strip(),
            pushover_user=os.getenv("PUSHOVER_USER", "").strip(),
        )


class Notifier:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def pushover(self, title: str, message: str, url: str = "") -> None:
        if not (self.cfg.pushover_token and self.cfg.pushover_user):
            return
        data = {"token": self.cfg.pushover_token, "user": self.cfg.pushover_user, "title": title, "message": message}
        if url:
            data |= {"url": url, "url_title": "Open Immich tags"}
        try:
            requests.post(PUSHOVER_URL, data=data, timeout=15).raise_for_status()
        except requests.RequestException as exc:
            LOG.warning("Pushover failed: %s", exc)

    def heartbeat(self, ok: bool, message: str) -> None:
        if not self.cfg.uptime_kuma_push_url:
            return
        url = self.cfg.uptime_kuma_push_url.split("?", 1)[0]
        try:
            requests.get(url, params={"status": "up" if ok else "down", "msg": message[:200]}, timeout=15)
        except requests.RequestException as exc:
            LOG.warning("Uptime Kuma push failed: %s", exc)


@dataclass
class RunResult:
    scanned: int = 0
    new: dict[str, int] = field(default_factory=dict)
    kept: int = 0
    deleted: int = 0
    failed: int = 0

    def summary(self, tags: dict[str, str]) -> str:
        parts = [f"{tags[c]}: {n}" for c, n in self.new.items() if n]
        return (
            f"{self.scanned} evaluated, {sum(self.new.values())} new candidates"
            + (f" ({'; '.join(parts)})" if parts else "")
            + f", {self.kept} kept, {self.deleted} deleted since last run, {self.failed} failed"
        )


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class Declutter:
    def __init__(self, cfg: Config, immich: Immich, clip: Clip, state: State) -> None:
        self.cfg = cfg
        self.immich = immich
        self.clip = clip
        self.state = state

    # --- tags ---

    def ensure_tags(self) -> dict[str, str]:
        if self.cfg.dry_run:
            for cat in sorted(self.cfg.categories):
                LOG.info("DRY_RUN: would ensure tag '%s'", self.cfg.tags[cat])
            return {}
        return {cat: self.immich.upsert_tag(self.cfg.tags[cat]) for cat in sorted(self.cfg.categories)}

    def migrate_albums(self, tag_ids: dict[str, str]) -> None:
        """One-off: tag what upstream put in its review albums, so reconcile() keeps treating
        removals as 'kept'. The emptied albums are left for the user to delete."""
        for cat, tag_id in tag_ids.items():
            album_id = self.state.get_meta(f"album:{cat}")
            if not album_id:
                continue
            ids = sorted(self.immich.album_asset_ids(album_id))
            self.immich.tag_assets(tag_id, ids)
            self.state.delete_meta(f"album:{cat}")
            LOG.info("Migrated %d assets from album %s to tag '%s'", len(ids), album_id, self.cfg.tags[cat])

    def reconcile(self, tag_ids: dict[str, str], result: RunResult) -> None:
        """What the user untagged since the last run: kept or deleted."""
        proposed = self.state.proposed()
        if not proposed or not tag_ids:
            return
        members = {cat: self.immich.tag_asset_ids(tid) for cat, tid in tag_ids.items()}
        for asset_id, cat in proposed.items():
            if cat not in members or asset_id in members[cat]:
                continue
            try:
                gone = self.immich.asset(asset_id).get("isTrashed", False)
            except NotFound:
                gone = True
            if gone:
                result.deleted += 1
            else:
                result.kept += 1
            self.state.set(asset_id, "deleted" if gone else "kept")
        self.state.commit()

    # --- candidates ---

    def candidates(self, until: datetime) -> list[dict]:
        cutoff = self.state.get_meta("cutoff")
        if self.cfg.full_scan:
            LOG.info("FULL_SCAN: whole library (up to %s)", iso(until))
            assets = list(self.immich.search(createdBefore=iso(until)))
        elif cutoff:
            LOG.info("New assets between %s and %s", cutoff, iso(until))
            assets = list(self.immich.search(createdAfter=cutoff, createdBefore=iso(until)))
        else:
            LOG.info("First run: only assets added from now on (set FULL_SCAN=true for the whole library)")
            assets = []

        seen = {a["id"] for a in assets}
        for asset_id in self.state.retry_ids(self.cfg.max_failures):
            if asset_id in seen:
                continue
            try:
                assets.append(self.immich.asset(asset_id))
            except NotFound:
                self.state.set(asset_id, "deleted")

        # Assets judged 'none' before a category existed may match it: re-evaluate them once.
        # A state db without the key comes from upstream, which knew the first four categories.
        stored = self.state.get_meta("categories")
        self.seen_categories = set(stored.split(",")) if stored else set(rules.CATEGORIES[:4])
        recheck = bool(self.cfg.categories - self.seen_categories)
        if recheck:
            LOG.info("New categories %s: re-evaluating assets judged 'none' before",
                     ", ".join(sorted(self.cfg.categories - self.seen_categories)))
        known = self.state.known(self.cfg.max_failures, include_none=not recheck)
        out = [
            a for a in assets
            if a["id"] not in known
            and a.get("visibility", "timeline") in VISIBLE
            and not a.get("isTrashed")
            and not a.get("isFavorite")
            and rules.needs_analysis(a, self.cfg.categories)
        ]
        if self.cfg.max_assets:
            out = out[: self.cfg.max_assets]
        return out

    def analyse(self, asset: dict) -> tuple[str | None, str]:
        data, mime = self.immich.thumbnail(asset["id"])
        scores = self.clip.scores(self.clip.image(data, mime))
        if rules.has_camera(asset) and asset.get("type") == "IMAGE":
            scores |= image_stats(data)
        need_ocr = rules.needs_ocr(asset, scores, self.cfg.thresholds)
        boxes = rules.boxes_from_api(self.immich.ocr(asset["id"])) if need_ocr else []
        return rules.classify(asset, scores, boxes, self.cfg.categories, self.cfg.thresholds)

    # --- run ---

    def run(self) -> RunResult:
        try:
            return self._run()
        except BaseException:
            # Drop uncommitted rows: 'proposed' assets whose tag flush failed would
            # otherwise be committed by the next run and mistaken for 'kept'.
            self.state.rollback()
            raise

    def _run(self) -> RunResult:
        result = RunResult(new={c: 0 for c in sorted(self.cfg.categories)})
        self.immich.ping()
        self.immich.check_access()
        tag_ids = self.ensure_tags()
        self.migrate_albums(tag_ids)
        self.reconcile(tag_ids, result)

        until = datetime.now(timezone.utc) - timedelta(hours=self.cfg.min_age_hours)
        assets = self.candidates(until)
        LOG.info("%d assets to evaluate", len(assets))
        if assets:
            self.clip.load_prompts()

        to_add: dict[str, list[str]] = {c: [] for c in self.cfg.categories}

        def flush() -> None:
            # Tags first, then state: a crash in between must never leave 'proposed'
            # assets untagged (the next run would mistake them for 'kept').
            for cat, ids in to_add.items():
                if ids and not self.cfg.dry_run:
                    self.immich.tag_assets(tag_ids[cat], ids)
                ids.clear()
            self.state.commit()

        def work(asset: dict):
            try:
                return asset, *self.analyse(asset), None
            except NotFound:
                return asset, None, "", "gone"
            except Exception as exc:  # network/ML error: retry next run
                return asset, None, "", f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(self.cfg.threads) as pool:
            for n, (asset, cat, reason, error) in enumerate(pool.map(work, assets), 1):
                result.scanned += 1
                if error == "gone":
                    if not self.cfg.dry_run:
                        self.state.set(asset["id"], "deleted")
                elif error:
                    result.failed += 1
                    LOG.warning("%s: %s", asset.get("originalFileName"), error)
                    if not self.cfg.dry_run:
                        self.state.fail(asset["id"], error)
                elif cat:
                    to_add[cat].append(asset["id"])
                    result.new[cat] += 1
                    LOG.info("Candidate %-11s %-28s %s", cat, reason, asset.get("originalPath"))
                    if not self.cfg.dry_run:
                        self.state.set(asset["id"], "proposed", cat, reason)
                elif not self.cfg.dry_run:
                    self.state.set(asset["id"], "none")
                if n % 500 == 0:
                    flush()
                    LOG.info("... %d/%d evaluated", n, len(assets))

        flush()
        if not self.cfg.dry_run:
            self.state.set_meta("categories", ",".join(sorted(self.cfg.categories | self.seen_categories)))
            self.state.set_meta("cutoff", iso(until))
            self.state.set_meta("last_run", iso(datetime.now(timezone.utc)))
        self.state.commit()
        return result


def next_run(now: datetime, run_at: str) -> datetime:
    hh, mm = map(int, run_at.split(":"))
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="run once now, then exit")
    parser.add_argument("--dry-run", action="store_true", help="write nothing, only log")
    parser.add_argument("--mark-kept", metavar="FILE", help="never propose the asset ids in this file")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = Config.from_env()
    if args.dry_run:
        cfg.dry_run = True
    os.makedirs(cfg.data_dir, exist_ok=True)
    state = State(os.path.join(cfg.data_dir, "declutter.db"))

    if args.mark_kept:
        with open(args.mark_kept) as fh:
            ids = [line.strip() for line in fh if line.strip()]
        for asset_id in ids:
            state.set(asset_id, "kept")
        state.commit()
        LOG.info("Marked %d assets as kept", len(ids))
        return

    declutter = Declutter(cfg, Immich(cfg.immich_url, cfg.immich_api_key), Clip(cfg.ml_url, cfg.clip_model), state)
    notifier = Notifier(cfg)
    tz = ZoneInfo(cfg.timezone)
    LOG.info(
        "immich-declutter started: categories %s, daily at %s %s%s%s",
        ", ".join(sorted(cfg.categories)), cfg.run_at, cfg.timezone,
        ", FULL_SCAN" if cfg.full_scan else "", ", DRY_RUN" if cfg.dry_run else "",
    )

    failing = False
    first = True
    while True:
        if not (first and (args.once or cfg.run_on_start)):
            wake = next_run(datetime.now(tz), cfg.run_at)
            LOG.info("Next run: %s", wake.strftime("%Y-%m-%d %H:%M"))
            while (remaining := (wake - datetime.now(tz)).total_seconds()) > 0:
                time.sleep(min(remaining, 300))
        first = False
        start = time.monotonic()
        try:
            result = declutter.run()
        except Exception as exc:
            LOG.exception("Run failed")
            notifier.heartbeat(False, str(exc))
            if not failing:
                notifier.pushover("immich-declutter: error", str(exc)[:500])
            failing = True
        else:
            summary = result.summary(cfg.tags)
            LOG.info("Done in %.0fs: %s", time.monotonic() - start, summary)
            notifier.heartbeat(True, summary)
            total = sum(result.new.values())
            if total and not cfg.dry_run:
                lines = [f"{cfg.tags[c]}: {n}" for c, n in result.new.items() if n]
                notifier.pushover(
                    f"immich-declutter: {total} new candidates",
                    "\n".join(lines),
                    f"{cfg.public_url}/tags" if cfg.public_url else "",
                )
            elif failing:
                notifier.pushover("immich-declutter: recovered", summary)
            failing = False
        if args.once:
            return


if __name__ == "__main__":
    main()
