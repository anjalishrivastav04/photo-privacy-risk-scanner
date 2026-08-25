"""
exif_privacy.py
Second, independent privacy check: a photo's metadata (EXIF) can leak your
exact GPS location, device model, and capture timestamp -- completely
separate from anything visible in the image itself. This is a real,
well-documented privacy leak, and it complements the fingerprint-exposure
check with a second, unrelated risk category under the same "check before
you post" idea.

IMPORTANT: call analyze_exif() on the image object right after
Image.open(), before any .convert()/.resize()/exif_transpose() call --
those operations create a new image and typically drop the metadata, so
checking too late will silently report "no metadata found" even when there
was some.
"""

from PIL import Image, ExifTags


def _to_degrees(value):
    d, m, s = value
    return float(d) + float(m) / 60.0 + float(s) / 3600.0


def _extract_gps(exif_data):
    """exif_data is a PIL Image.Exif object. The GPS block is a *sub*-IFD --
    exif_data[GPSInfo] on modern Pillow is just a raw IFD offset (an int),
    not the tag dict itself. get_ifd(IFD.GPSInfo) is the correct way to get
    the actual {tag_id: value} mapping for it."""
    if not exif_data:
        return None
    try:
        gps_info = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
    except (KeyError, AttributeError):
        return None
    if not gps_info:
        return None

    gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    try:
        lat = _to_degrees(gps_tags["GPSLatitude"])
        if gps_tags.get("GPSLatitudeRef") == "S":
            lat = -lat
        lon = _to_degrees(gps_tags["GPSLongitude"])
        if gps_tags.get("GPSLongitudeRef") == "W":
            lon = -lon
        return (lat, lon)
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def analyze_exif(pil_image):
    """Inspect a freshly-opened PIL Image for privacy-relevant EXIF fields.

    Returns:
      {
        "has_gps": bool, "gps": (lat, lon) or None,
        "device": "Apple iPhone 13" or None,
        "timestamp": "2026:08:25 10:15:00" or None,
        "risk": "high" | "medium" | "low",
        "findings": ["Exact GPS location embedded: 19.07600, 72.87770", ...],
      }
    """
    exif_data = pil_image.getexif()
    findings = []
    risk = "low"

    gps = _extract_gps(exif_data)
    if gps:
        findings.append(f"Exact GPS location embedded: {gps[0]:.5f}, {gps[1]:.5f}")
        risk = "high"

    tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()} if exif_data else {}

    device = " ".join(x for x in [tag_map.get("Make"), tag_map.get("Model")] if x) or None
    if device:
        findings.append(f"Device identified: {device}")
        if risk == "low":
            risk = "medium"

    timestamp = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
    if timestamp:
        findings.append(f"Original capture timestamp embedded: {timestamp}")
        if risk == "low":
            risk = "medium"

    if not findings:
        findings.append("No GPS, device, or timestamp metadata found in this file.")

    return {
        "has_gps": gps is not None,
        "gps": gps,
        "device": device,
        "timestamp": timestamp,
        "risk": risk,
        "findings": findings,
    }


def strip_exif(pil_image_rgb):
    """Return a copy of an RGB image with all metadata removed (re-saves
    pixel data only -- no EXIF/IPTC/XMP block survives)."""
    clean = Image.new(pil_image_rgb.mode, pil_image_rgb.size)
    clean.putdata(list(pil_image_rgb.getdata()))
    return clean