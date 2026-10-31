import media


def test_spatial_photo_is_not_portrait():
    meta = {"Copy1:AuxiliaryImageType": "disparity", ":Unknown_ster": "(Binary data)", ":FileType": "HEIC"}
    assert media.detect(meta) == {media.SPATIAL}


def test_portrait_from_depth():
    meta = {":AuxiliaryImageType": "urn:mpeg:hevc:2015:auxid:2", "Copy1:AuxiliaryImageType": "depth"}
    assert media.detect(meta) == {media.PORTRAIT}


def test_plain_photo_has_no_type():
    assert media.detect({":AuxiliaryImageType": "urn:com:apple:photo:2020:aux:hdrgainmap", ":ImageCaptureType": 12}) == set()


def test_videos():
    assert media.detect({":SpatialFormat-version": "1.0", ":VideoFrameRate": 29.997}) == {media.SPATIAL}
    assert media.detect({":Cinematic-video": "(Binary data)", ":VideoFrameRate": 23.997}) == {media.CINEMATIC}
    assert media.detect({":VideoFrameRate": 120.014}) == {media.SLOMO}


def test_burst_and_raw():
    assert media.detect({":BurstUUID": "22183F0D", ":ImageCaptureType": 1}) == {media.BURSTS, media.RAW}
    assert media.from_name({"originalFileName": "DSCF0001.RAF"}) == {media.RAW}
    assert media.from_name({"originalFileName": "IMG_1.HEIC"}) == set()


def test_unifi_exports_by_name():
    vid = {"type": "VIDEO", "exifInfo": {"make": None}}
    assert media.from_name(vid | {"originalFileName": "unifi_180524_2014_84.0s.mp4"}) == {media.UNIFI}
    assert media.from_name(vid | {"originalFileName": "Front Door 11-22-2025, 14.55.00 GMT - 11-22-2025, 15.55.00 GMT.mp4"}) == {media.UNIFI}
    assert media.from_name(vid | {"originalFileName": "IMG_1234.MOV"}) == set()
    assert media.from_name({"type": "VIDEO", "exifInfo": {"make": "Apple"}, "originalFileName": "unifi_180524_2014_84.0s.mp4"}) == set()


def test_only_apple_needs_exiftool():
    assert media.needs_exiftool({"exifInfo": {"make": "Apple"}})
    assert not media.needs_exiftool({"exifInfo": {"make": "FUJIFILM"}})
    assert not media.needs_exiftool({"exifInfo": None})
