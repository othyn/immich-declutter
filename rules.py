"""Classification rules: is a photo/video a deletion candidate, and for which tag?

Everything here is pure (no network) so it can be tested in isolation. The default
thresholds were calibrated on a real ~180k asset library (CLIP ViT-B-32__openai on
Immich thumbnails), measured as the share of hits in random samples:

- Memes: CLIP meme score >= 0.45 gave ~85% hits; a caption band at the top/bottom
  (OCR) combined with a CLIP caption score >= 0.5 ~40%; exactly 790 px (images saved
  from Facebook) ~90%; small (<= 640 px) images in a non-camera size ~85%.
- TikTok: videos only, OCR watermark (tiktok/reels/capcut/...) ~100%. (The 576x1024
  download size alone is not used: phones record personal videos in that size too.)
  Stills with a TikTok watermark go to memes.
- Screenshots: CLIP screenshot score >= 0.6.
- Receipts & barcodes: camera photos additionally need real OCR text (>= 8 words),
  because CLIP's barcode/receipt prompts also fire on e.g. airplanes and airports.

Fork additions (othyn/immich-declutter), thresholds are first guesses on SigLIP2 and
need a dry run per category before trusting them:

- YouTube: CLIP video-player prompts, or a weaker CLIP score plus OCR (subscribe/views).
- Native screenshots: PNG, no camera make, phone aspect ratio or a desktop resolution.
- Photos of screens, documents/whiteboards, blurry, low quality: camera photos only.
  Blurry needs CLIP agreement and low Laplacian variance, so plain skies are spared.
- Messaging: WhatsApp/Telegram/Signal filenames, as a fallback when nothing else matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MEMES = "memes"
TIKTOK = "tiktok"
SCREENSHOTS = "screenshots"
RECEIPTS = "receipts"
YOUTUBE = "youtube"
SCREENS = "screens"
DOCUMENTS = "documents"
BLURRY = "blurry"
LOWQUALITY = "lowquality"
MESSAGING = "messaging"
CATEGORIES = (MEMES, TIKTOK, SCREENSHOTS, RECEIPTS, YOUTUBE, SCREENS, DOCUMENTS, BLURRY, LOWQUALITY, MESSAGING)
CAMERA_CATEGORIES = {RECEIPTS, SCREENS, DOCUMENTS, BLURRY, LOWQUALITY}

# --- CLIP prompts. Each group gets its own softmax over positives + negatives. ---

GENERAL_MEME = [
    "a meme", "an internet meme with text", "a funny image macro with caption text",
    "a cartoon", "a comic strip", "a quote with text on a plain background",
    "a motivational quote image", "a funny picture from the internet",
    "a screenshot of a social media post", "a screenshot of a tweet",
    "a clip art illustration", "a cartoon emoji", "a demotivational poster",
]
GENERAL_SCREEN = [
    "a screenshot of a phone", "a screenshot of a chat conversation", "a screenshot of a website",
    "a screenshot of an app", "a screenshot of a map", "a screenshot of a text message",
    "a screenshot of a game",
]
GENERAL_NEG = [
    "a photo of a person", "a selfie", "a family photo", "a photo of children", "a photo of a baby",
    "a group photo of friends", "a landscape photo", "a photo of a city street", "a photo of food",
    "a photo of a pet", "a photo of a room interior", "a photo of a garden", "a photo of a car",
    "a photo of a document", "a photo of a handwritten note", "a photo of a drawing made by a child",
    "a photo of a computer screen", "a photo of a party", "a holiday photo", "a photo at the beach",
    "a photo of a birthday cake", "a photo of flowers", "a blurry photo", "a dark photo",
    "a selfie with a funny face filter", "a photo of a building", "a photo of a sports event",
    "a photo of a wedding", "a photo of a birthday card on a table", "a home video of kids playing",
    "a vertical phone video", "a photo of a concert", "a photo of a poster", "a flyer",
    "an invitation card", "a photo of a product", "a photo of a ticket", "a photo of a tv screen",
]

CAPTION_POS = [
    "a meme with a caption above a photo", "a photo with bold black text at the top",
    "a funny meme picture with text",
]
CAPTION_NEG = [
    "a photo of a product", "a photo of an object", "a photo of a person", "a screenshot", "a map",
    "a flyer", "a photo of a document", "a photo of a receipt",
]

RECEIPT_RECEIPT = [
    "a photo of a receipt", "a cash register receipt", "a long paper store receipt",
    "a pin payment receipt",
]
RECEIPT_BARCODE = [
    "a barcode", "a QR code", "a shipping label with a barcode", "a parcel label",
    "a ticket with a barcode", "a loyalty card with a barcode", "a screenshot of a QR code",
    "a screenshot of a barcode",
]
RECEIPT_NEG = [
    "a photo of a document", "a letter", "a handwritten note", "a screenshot of a website",
    "a screenshot of a chat", "a photo of a person", "a family photo", "a photo of a child",
    "a photo of a baby", "a selfie", "a landscape", "a photo of food", "a photo of a product",
    "a photo of an object", "a photo of a tool", "a photo of electronics", "a photo of clothing",
    "a photo of a room", "a photo of furniture", "a blurry photo", "a dark photo",
    "a photo of a floor", "a photo of a wall", "a photo of a car", "a photo of a pet",
    "a price tag in a store", "a photo of a book page", "a menu", "a map", "a meme",
    "a photo of a computer screen", "a photo of a sign", "a photo of a package box", "a calendar",
    "a spreadsheet", "a poster", "a flyer", "a greeting card", "a drawing", "a photo of an airplane",
    "a photo at an airport", "a photo inside an airplane", "a photo of a toy",
]

YOUTUBE_POS = [
    "a screenshot of a YouTube video", "a screenshot of a video player with playback controls",
    "a paused video with a progress bar", "a screenshot of a YouTube video with a title and view count",
    "a video thumbnail with large text",
]
YOUTUBE_NEG = GENERAL_SCREEN + GENERAL_MEME + [
    "a photo of a person", "a landscape photo", "a photo of food", "a photo of a room interior",
    "a photo of a tv screen", "a photo of a computer screen",
]

# Camera photos only: what kind of throwaway shot is it, if any.
CAMERA_BLURRY = ["a blurry photo", "an out of focus photo", "a photo with heavy motion blur", "a shaky blurred picture"]
CAMERA_JUNK = [
    "an accidental photo", "a photo taken inside a pocket", "a completely black photo", "a photo of the floor",
    "a photo of a ceiling", "a photo of a blank wall", "an overexposed white photo",
    "a photo with a finger covering the lens",
]
CAMERA_SCREEN = [
    "a photo of a computer screen", "a photo of a tv screen", "a photo of a laptop screen",
    "a photo of a monitor showing text", "a photo of a phone screen", "a photo of a screen with moire pattern",
]
CAMERA_DOCUMENT = [
    "a photo of a document", "a photo of a printed letter", "a photo of a whiteboard",
    "a photo of a page of text", "a photo of a form", "a photo of handwritten notes", "a photo of a book page",
]
CAMERA_NEG = [
    "a sharp photo of a person", "a selfie", "a family photo", "a group photo of friends", "a photo of a child",
    "a photo of a baby", "a landscape photo", "a photo of a sunset", "a photo of a mountain", "a photo at the beach",
    "a photo of a city street", "a photo of a building", "a photo of food", "a photo of a pet", "a photo of flowers",
    "a photo of a garden", "a photo of a room interior", "a photo of a car", "a photo of a concert",
    "a holiday photo", "a photo of a party", "a photo of a sports event", "a photo of a forest",
]

# name -> (positive groups, negatives). Every positive group yields its own score.
PROMPT_GROUPS = {
    "general": ([GENERAL_MEME, GENERAL_SCREEN], GENERAL_NEG),
    "caption": ([CAPTION_POS], CAPTION_NEG),
    "receipt": ([RECEIPT_RECEIPT, RECEIPT_BARCODE], RECEIPT_NEG),
    "youtube": ([YOUTUBE_POS], YOUTUBE_NEG),
    "camera": ([CAMERA_BLURRY, CAMERA_JUNK, CAMERA_SCREEN, CAMERA_DOCUMENT], CAMERA_NEG),
}
SCORE_NAMES = {
    "general": ("meme", "screen"),
    "caption": ("caption",),
    "receipt": ("receipt", "barcode"),
    "youtube": ("youtube",),
    "camera": ("blurry", "junk", "screen_photo", "document"),
}

# --- OCR keywords ---

CAMERA_MIN_OCR_WORDS = 8
TIKTOK_WORDS = ("tiktok", "reels", "capcut", "instagram", "snapchat")
# Images only; a TikTok watermark on a still means a saved frame, which is a meme (the TikTok tag is videos only)
MEME_SITE_WORDS = ("9gag", "imgflip", "dumpert", "geenstijl", "tiktok")
# English + Dutch till-receipt vocabulary (Immich's OCR reads both)
RECEIPT_WORDS = (
    "total", "subtotal", "receipt", "vat", "terminal", "transaction", "maestro", "mastercard",
    "totaal", "subtotaal", "btw", "kassabon", "bonnr", "pinnen", "betaald", "contant",
    "wisselgeld", "transactie", "kaartnr", "filiaal", "kassa",
)

YOUTUBE_WORDS = ("youtube", "subscribe", "subscribed", "views")
DOCUMENT_MIN_OCR_WORDS = 25
# Filenames of media saved from messaging apps (which strip the camera EXIF).
MESSAGING_PATTERNS = (
    (re.compile(r"^(IMG|VID|AUD|PTT)-\d{8}-WA\d+", re.I), "whatsapp"),
    (re.compile(r"^(photo|video)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", re.I), "telegram"),
    (re.compile(r"^signal-\d{4}-\d{2}-\d{2}", re.I), "signal"),
)
# Desktop/laptop panel resolutions (long, short); phones are caught by aspect ratio instead.
SCREEN_SIZES = {
    (1280, 720), (1280, 800), (1366, 768), (1440, 900), (1536, 864), (1600, 900), (1680, 1050), (1920, 1080),
    (1920, 1200), (2560, 1080), (2560, 1440), (2560, 1600), (2880, 1800), (3024, 1964), (3440, 1440),
    (3456, 2234), (3840, 2160), (5120, 2880),
}

# Common camera/webcam sizes; small images in any *other* size are almost always web images.
CAMERA_SMALL_SIZES = {
    (640, 480), (320, 240), (160, 120), (176, 144), (352, 288), (240, 240), (600, 600),
    (480, 360), (400, 300), (320, 320), (640, 360), (426, 320), (213, 160),
}


@dataclass(frozen=True)
class Thresholds:
    meme: float = 0.45
    meme_video: float = 0.6
    caption: float = 0.5
    screen: float = 0.6
    receipt_with_ocr: float = 0.3
    receipt: float = 0.85
    barcode: float = 0.75
    youtube: float = 0.6
    youtube_with_ocr: float = 0.3
    screen_photo: float = 0.6
    document: float = 0.85
    document_with_ocr: float = 0.4
    blurry: float = 0.5
    sharpness: float = 60.0
    junk: float = 0.6
    dark: float = 12.0
    flat: float = 6.0


@dataclass(frozen=True)
class Box:
    """OCR text box, coordinates normalised to 0..1."""

    top: float
    bottom: float
    text: str


def boxes_from_api(raw: list[dict]) -> list[Box]:
    return [
        Box(min(b["y1"], b["y2"]), max(b["y3"], b["y4"]), b.get("text") or "")
        for b in raw
        if b.get("isVisible", True)
    ]


def _tokens(boxes: list[Box]) -> list[str]:
    return re.findall(r"[a-z0-9]+", " ".join(b.text for b in boxes).lower())


def has_word(boxes: list[Box], words: tuple[str, ...]) -> str | None:
    """First keyword found. Short words (<5 chars) must match a whole token, longer ones may be part of one."""
    tokens = _tokens(boxes)
    joined = "".join(b.text for b in boxes).lower().replace(" ", "")
    for w in words:
        if len(w) < 5:
            if w in tokens:
                return w
        elif w in joined:
            return w
    return None


def caption_band(boxes: list[Box]) -> bool:
    """Large text only in a band at the top and/or bottom, hardly any other text (meme layout)."""
    big = [b for b in boxes if (b.bottom - b.top) >= 0.045 and len(b.text.strip()) >= 2]
    if not big:
        return False
    words = sum(len(b.text.split()) for b in big)
    top = [b for b in big if b.bottom <= 0.4]
    bottom = [b for b in big if b.top >= 0.6]
    other = len(boxes) - len(big)
    return words >= 4 and len(top) + len(bottom) == len(big) and other <= 2


def dims(asset: dict) -> tuple[int, int]:
    """(long side, short side)."""
    w, h = asset.get("width") or 0, asset.get("height") or 0
    return (max(w, h), min(w, h))


def is_web_size(asset: dict) -> str | None:
    long_, short = dims(asset)
    if not long_:
        return None
    if 790 in (long_, short):
        return "790px"
    if long_ <= 640 and (long_, short) not in CAMERA_SMALL_SIZES:
        return "small-web-size"
    return None


def has_camera(asset: dict) -> bool:
    return bool((asset.get("exifInfo") or {}).get("make"))


def is_native_screenshot(asset: dict) -> bool:
    if has_camera(asset) or not (asset.get("originalFileName") or "").lower().endswith(".png"):
        return False
    long_, short = dims(asset)
    return bool(short) and (long_ / short >= 2.0 or (long_, short) in SCREEN_SIZES)


def messaging_app(asset: dict) -> str | None:
    name = asset.get("originalFileName") or ""
    return next((app for pattern, app in MESSAGING_PATTERNS if pattern.match(name)), None)


def needs_ocr(asset: dict, scores: dict[str, float], t: Thresholds) -> bool:
    """OCR is only fetched where it can change the outcome (camera photos: only receipt/barcode/document suspects)."""
    if not has_camera(asset):
        return True
    return (
        scores.get("receipt", 0) >= t.receipt_with_ocr
        or scores.get("barcode", 0) >= t.barcode
        or scores.get("document", 0) >= t.document_with_ocr
    )


def needs_analysis(asset: dict, enabled: set[str]) -> bool:
    """Whether it is worth fetching thumbnail/CLIP/OCR at all."""
    if asset.get("type") == "VIDEO":
        return bool({TIKTOK, MESSAGING} & enabled) and not has_camera(asset)
    if has_camera(asset):
        return bool(CAMERA_CATEGORIES & enabled)
    return True


def camera_category(scores: dict[str, float], boxes: list[Box], enabled: set[str], t: Thresholds) -> tuple[str | None, str]:
    """Camera photos that are not receipts: throwaway shots, screens, documents. Stats come from ml.image_stats."""
    if LOWQUALITY in enabled:
        if scores.get("brightness", 255) < t.dark:
            return LOWQUALITY, f"dark={scores['brightness']:.0f}"
        if scores.get("contrast", 255) < t.flat:
            return LOWQUALITY, f"flat={scores['contrast']:.1f}"
        if scores.get("junk", 0) >= t.junk:
            return LOWQUALITY, f"junk={scores['junk']:.2f}"
    if BLURRY in enabled and scores.get("blurry", 0) >= t.blurry and scores.get("sharpness", 1e9) < t.sharpness:
        return BLURRY, f"blurry={scores['blurry']:.2f}+sharpness={scores['sharpness']:.0f}"
    if SCREENS in enabled and scores.get("screen_photo", 0) >= t.screen_photo:
        return SCREENS, f"screen_photo={scores['screen_photo']:.2f}"
    if DOCUMENTS in enabled:
        d = scores.get("document", 0)
        if d >= t.document_with_ocr and len(_tokens(boxes)) >= DOCUMENT_MIN_OCR_WORDS:
            return DOCUMENTS, f"document={d:.2f}+ocr"
        if d >= t.document:
            return DOCUMENTS, f"document={d:.2f}"
    return None, ""


def classify(
    asset: dict,
    scores: dict[str, float],
    boxes: list[Box],
    enabled: set[str],
    t: Thresholds = Thresholds(),
) -> tuple[str | None, str]:
    """Returns (category or None, reason)."""
    camera = has_camera(asset)

    if asset.get("type") == "VIDEO":
        if camera:
            return None, ""
        if TIKTOK in enabled:
            if w := has_word(boxes, TIKTOK_WORDS):
                return TIKTOK, f"ocr:{w}"
            if scores.get("meme", 0) >= t.meme_video:
                return TIKTOK, f"meme={scores['meme']:.2f}"
        if MESSAGING in enabled and (app := messaging_app(asset)):
            return MESSAGING, f"filename:{app}"
        return None, ""

    if RECEIPTS in enabled:
        r, b = scores.get("receipt", 0), scores.get("barcode", 0)
        # Camera photos: CLIP alone is not enough, the photo must also contain real text
        texty = not camera or len(_tokens(boxes)) >= CAMERA_MIN_OCR_WORDS
        if r >= t.receipt_with_ocr and texty and (w := has_word(boxes, RECEIPT_WORDS)):
            return RECEIPTS, f"receipt={r:.2f}+ocr:{w}"
        if r >= t.receipt and texty:
            return RECEIPTS, f"receipt={r:.2f}"
        if b >= t.barcode and texty:
            return RECEIPTS, f"barcode={b:.2f}"

    if camera:
        return camera_category(scores, boxes, enabled, t)

    if YOUTUBE in enabled:
        y = scores.get("youtube", 0)
        if y >= t.youtube:
            return YOUTUBE, f"youtube={y:.2f}"
        if y >= t.youtube_with_ocr and (w := has_word(boxes, YOUTUBE_WORDS)):
            return YOUTUBE, f"youtube={y:.2f}+ocr:{w}"

    if MEMES in enabled:
        if scores.get("meme", 0) >= t.meme:
            return MEMES, f"meme={scores['meme']:.2f}"
        if w := has_word(boxes, MEME_SITE_WORDS):
            return MEMES, f"ocr:{w}"
        if caption_band(boxes) and scores.get("caption", 0) >= t.caption:
            return MEMES, f"caption-band+caption={scores['caption']:.2f}"
        if size := is_web_size(asset):
            return MEMES, size

    if SCREENSHOTS in enabled:
        if is_native_screenshot(asset):
            return SCREENSHOTS, "native-png"
        if scores.get("screen", 0) >= t.screen:
            return SCREENSHOTS, f"screen={scores['screen']:.2f}"

    if MESSAGING in enabled and (app := messaging_app(asset)):
        return MESSAGING, f"filename:{app}"

    return None, ""
