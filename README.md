# immich-declutter

> Fork of [rvanderend/immich-declutter](https://github.com/rvanderend/immich-declutter). This fork writes Immich **tags** instead of albums, so albums stay for curated collections, and adds six categories. Favourites are never proposed. The new thresholds are first guesses on SigLIP2: dry-run each category before trusting it.

A headless companion container for [Immich](https://immich.app). Every night it looks at the photos and videos that were added to your library and tags likely clutter, one tag per category under a parent tag (`Utilities` by default), like the Utilities and Media Types lists in Apple Photos:

| Tag | What ends up there |
|---|---|
| Utilities/Memes & Web Images | memes, cartoons, quote images, pictures saved from the web (including stills with a TikTok watermark) |
| Utilities/Screenshots | screenshots of apps, chats, maps and websites; native PNG screenshots by size |
| Utilities/Receipts & Barcodes | till receipts, invoices, barcodes, QR codes, shipping labels, tickets |
| Utilities/Social Media | screenshots of YouTube and other video players, and videos with a TikTok/Reels/CapCut watermark |
| Utilities/Photos of Screens | camera photos of monitors, laptops, TVs and phones |
| Utilities/Documents | camera photos of letters, forms, pages and whiteboards |
| Utilities/Blurry | camera photos CLIP calls blurry *and* with low Laplacian variance on the thumbnail |
| Utilities/Low Quality | dark or flat (pocket) shots, and accidental photos of floors, ceilings and walls |
| Utilities/Messaging | media saved from WhatsApp, Telegram or Signal (by filename), when nothing else matched |

Each asset gets at most one tag. Archived assets are included; Immich's tag view shows them too.

**immich-declutter never deletes anything.** You review the tags in Immich:

- **Delete** what can go, with Immich's normal delete and trash.
- **Keep** a photo by removing its tag. immich-declutter remembers that and never proposes it again, not even during a full scan.

It also tags iOS capture formats under `Media`, read from the original file with exiftool. These are facts about the file, so they are tagged regardless of the clutter tags and never reconciled:

| Tag | Marker in the original |
|---|---|
| Media/Portrait | depth or disparity auxiliary image (and not a spatial photo) |
| Media/Spatial | HEIF stereo pair (`ster`) or the `SpatialFormat` video key |
| Media/Cinematic | `Cinematic-video` key |
| Media/Slo-mo | frame rate of 100 or more; clips iOS stores at 30 fps are missed |
| Media/Bursts | `BurstUUID` in the Apple maker note |
| Media/RAW | ProRAW capture type, or a camera RAW extension |

The container needs Immich's upload folder mounted read-only at the same path Immich uses (`/usr/src/app/upload`), because it reads `originalPath` directly. The first run reads every Apple original, after that only new uploads.

Upgrading from upstream: on the first run, everything still in the upstream review albums is tagged, so earlier keep decisions survive. Delete the emptied albums yourself afterwards. The migration needs the old `album.read` permission once.

## How it decides

The rules live in [`rules.py`](rules.py). They combine three signals that Immich already computes:

- **CLIP image understanding.** Calls Immich's own machine-learning service, so it uses the same model as Smart Search and needs no extra download. Each thumbnail is scored zero-shot against prompts such as *"an internet meme with text"* or *"a screenshot of an app"*, versus negatives like *"a family photo"* or *"a photo of a baby"*.
- **OCR.** Uses Immich's built-in text recognition to spot watermarks (TikTok, 9GAG, …), till-receipt vocabulary (*total*, *VAT*, *subtotal*, plus Dutch terms) and the classic meme layout: big text in a band at the top or bottom and nothing else.
- **Metadata.**
  - Memes, TikToks and screenshots carry no camera make in their EXIF, so photos *with* a camera make are only checked for receipts and barcodes. They also need real OCR text, because on its own CLIP's barcode prompt fires on things like airplanes.
  - Images saved from Facebook are exactly 790 px wide.
  - Small images (≤ 640 px) in a size no camera produces are almost always web images.

The default thresholds were calibrated on a real library of ~180k assets by checking random samples of each rule:
- CLIP meme score: ~85% of hits were right.
- 790 px images: ~90%.
- TikTok watermark: ~100%.
- Full-library run (~170k assets) after the rules were tightened: 9 receipt candidates among ~120k camera photos, all of them paperwork.

Your library will differ, so the thresholds can be tuned with `THRESHOLD_*` (see below). Run with `DRY_RUN=true` first to see what it would propose.

## Requirements

- Immich with **Smart Search (CLIP)** and **OCR** enabled. Tested with Immich 3.2.
- Network access to the Immich server **and** to the `immich-machine-learning` container (port 3003). Immich's default compose file doesn't publish that port, so the easiest way is to join Immich's Docker network (see `compose.yaml`).
- An Immich API key with these permissions: `asset.read`, `asset.view`, `tag.create`, `tag.asset` (plus `album.read` once, when migrating from upstream albums). The container checks them at the start of every run.

## Setup

```sh
mkdir immich-declutter && cd immich-declutter
curl -O https://raw.githubusercontent.com/rvanderend/immich-declutter/main/compose.yaml
curl -o .env https://raw.githubusercontent.com/rvanderend/immich-declutter/main/.env.example
# edit .env: at least IMMICH_API_KEY (and check IMMICH_URL / ML_URL / the network name in compose.yaml)
mkdir data && sudo chown 10001:10001 data
docker compose up -d
docker logs -f immich-declutter
```

Optionally, start with a dry run first:

```sh
docker compose run --rm immich-declutter python declutter.py --once --dry-run
```

## How it runs

- **Schedule:** once a day at `RUN_AT`.
- **What it evaluates:** everything added to Immich (`createdAt`) since the previous run that is at least `MIN_AGE_HOURS` old. That delay gives Immich time to finish thumbnails and OCR.
- **First run:** starts from "now". Set `FULL_SCAN=true` to evaluate your whole library instead. Assets that were already evaluated are skipped, so after the first full pass a full scan is cheap. Expect roughly 1–2 hours per 100k photos for that first pass, depending on your ML hardware.
- **State:** kept in `/data/declutter.db` (SQLite). For each asset it records one of: not a candidate, proposed, kept, deleted, or retry.
- **Failures:** assets that fail, for example because no thumbnail exists yet, are retried on later runs, up to `MAX_FAILURES` times.

### Manual commands

```sh
docker exec immich-declutter python declutter.py --once            # run now
docker exec immich-declutter python declutter.py --once --dry-run  # only log what would happen
docker exec immich-declutter python declutter.py --mark-kept /data/ids.txt  # never propose these asset ids
```

`--mark-kept` is handy if you already cleaned up by hand and want to protect the photos you decided to keep.

## Configuration

All settings are environment variables; see [`.env.example`](.env.example).

| Variable | Default | Description |
|---|---|---|
| `IMMICH_URL` | – | Immich server URL, e.g. `http://immich-server:2283` |
| `IMMICH_API_KEY` | – | API key (see permissions above) |
| `ML_URL` | – | Immich machine-learning URL, e.g. `http://immich-machine-learning:3003` |
| `CLIP_MODEL` | `ViT-B-32__openai` | must match the CLIP model configured in Immich |
| `TZ` | `UTC` | timezone for `RUN_AT` |
| `RUN_AT` | `03:30` | daily run time (HH:MM) |
| `RUN_ON_START` | `false` | also run immediately when the container starts |
| `CATEGORIES` | all | comma-separated: `memes,tiktok,screenshots,receipts,youtube,screens,documents,blurry,lowquality,messaging` |
| `MEDIA` | all | comma-separated: `portrait,spatial,cinematic,slomo,bursts,raw`, or `none` |
| `MEDIA_TAG_PARENT` | `Media` | parent tag for the media types; `MEDIA_TAG_<TYPE>` renames a child |
| `MEDIA_THREADS` | `4` | parallel exiftool processes, 100 files each |
| `FULL_SCAN` | `false` | evaluate the whole library instead of only new assets |
| `MIN_AGE_HOURS` | `3` | only evaluate assets at least this old |
| `TAG_PARENT` | `Utilities` | parent tag; each category is a child tag under it |
| `TAG_MEMES`, `TAG_TIKTOK`, `TAG_SCREENSHOTS`, `TAG_RECEIPTS`, `TAG_YOUTUBE`, `TAG_SCREENS`, `TAG_DOCUMENTS`, `TAG_BLURRY`, `TAG_LOWQUALITY`, `TAG_MESSAGING` | see table above | child tag names (no `/`). Renaming a tag in Immich makes the next run create a new one |
| `THRESHOLD_<NAME>` for every field of `Thresholds` in `rules.py` | see `rules.py` | CLIP score thresholds (0–1); `THRESHOLD_SHARPNESS`, `THRESHOLD_DARK` and `THRESHOLD_FLAT` are pixel statistics on the 0–255 greyscale thumbnail |
| `DRY_RUN` | `false` | write nothing (no tags, no state), only log |
| `THREADS` | `4` | parallel thumbnail/ML requests |
| `MAX_ASSETS` | `0` | limit per run (0 = unlimited), useful for testing |
| `IMMICH_PUBLIC_URL` | – | used for the link in notifications |
| `PUSHOVER_TOKEN`, `PUSHOVER_USER` | – | Pushover notification after a run with new candidates (and on errors) |
| `UPTIME_KUMA_PUSH_URL` | – | push monitor heartbeat after every run |
| `LOG_LEVEL` | `INFO` | |

## Notes

- **External libraries:** whether deleting in Immich also removes the file depends on whether Immich has write access to the library path. With a read-only mount, the file stays on disk.
- **Deleted vs. no access:** Immich answers *"Not found or no asset.read access"* for deleted assets. That's why the container verifies its permissions on an existing asset before every run, so a key with missing permissions can't be mistaken for "everything was deleted".

## Development

```sh
pip install -r requirements-dev.txt
pytest
```

## License

[MIT](LICENSE)
