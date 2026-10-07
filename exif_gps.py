"""
exif_gps.py

Reads embedded GPS coordinates from an image's EXIF metadata, when present.

Many phone photos still carry a GPSInfo EXIF block even after resizing or
re-compression. When it's there, it's ground truth and far more reliable
than any visual geolocation model — so geolocate.py checks this first and
only falls back to GeoCLIP when no EXIF GPS is found.

Setup:
    pip install pillow

Usage:
    from exif_gps import get_exif_gps

    coords = get_exif_gps("photo.jpg")  # -> (lat, lon) or None
"""

from PIL import ExifTags, Image


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert EXIF (degrees, minutes, seconds) + hemisphere ref to decimal degrees."""
    degrees, minutes, seconds = (float(v) for v in dms)
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def get_exif_gps(image_path: str) -> tuple[float, float] | None:
    """
    Return (lat, lon) from an image's EXIF GPSInfo block, or None if the
    image has no GPS metadata (or no EXIF at all).
    """
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        return None

    if not gps_ifd:
        return None

    gps = {ExifTags.GPSTAGS.get(tag, tag): value for tag, value in gps_ifd.items()}
    lat_dms, lat_ref = gps.get("GPSLatitude"), gps.get("GPSLatitudeRef")
    lon_dms, lon_ref = gps.get("GPSLongitude"), gps.get("GPSLongitudeRef")

    if not (lat_dms and lon_dms and lat_ref and lon_ref):
        return None

    return _dms_to_decimal(lat_dms, lat_ref), _dms_to_decimal(lon_dms, lon_ref)
