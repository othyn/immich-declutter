import io

import numpy as np
from PIL import Image

from ml import image_stats


def webp(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buf, "WEBP", lossless=True)
    return buf.getvalue()


def test_image_stats_separates_sharp_from_flat_and_dark():
    checker = (np.indices((64, 64)).sum(axis=0) % 2) * 255
    sharp = image_stats(webp(checker))
    flat = image_stats(webp(np.full((64, 64), 128)))
    dark = image_stats(webp(np.full((64, 64), 3)))
    assert sharp["sharpness"] > 1000 and flat["sharpness"] < 1
    assert flat["contrast"] < 1 and dark["brightness"] < 5
